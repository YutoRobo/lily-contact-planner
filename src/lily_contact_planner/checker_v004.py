import numpy as np
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError
from scipy.spatial.transform import Rotation,Slerp

from .analytic_ik import analytic_leg_ik_world
from .collision import segment_aabb_distance,segment_segment_distance,trim_segment_start
from .nlp_geometry_v002 import capsule_segments,state_points
from .v004_types import DenseCheckReportV004,V004Settings


def _point_in_support_hull(point_xy,support_xy,tol):
    point=np.asarray(point_xy,float); pts=np.asarray(support_xy,float)
    if len(pts)==0: return False,-np.inf
    unique=[]
    for p in pts:
        if not any(np.linalg.norm(p-old)<=tol for old in unique): unique.append(p)
    pts=np.asarray(unique,float)
    if len(pts)==1:
        d=float(np.linalg.norm(point-pts[0])); return d<=tol,-d
    centered=pts-pts[0]; rank=np.linalg.matrix_rank(centered,tol=max(tol,1e-12))
    if len(pts)==2 or rank<2:
        direction=pts[1]-pts[0]; length=float(np.linalg.norm(direction))
        if length<=tol:
            d=float(np.linalg.norm(point-pts[0])); return d<=tol,-d
        axis=direction/length; scalars=centered.dot(axis); lo,hi=float(np.min(scalars)),float(np.max(scalars))
        sp=float(np.dot(point-pts[0],axis)); perp=float(np.linalg.norm((point-pts[0])-sp*axis))
        inside=perp<=tol and sp>=lo-tol and sp<=hi+tol
        return (True,0.0) if inside else (False,-max(perp,lo-sp,sp-hi,0.0))
    try: hull=ConvexHull(pts)
    except QhullError: return False,-np.inf
    vals=hull.equations[:,:2].dot(point)+hull.equations[:,2]; normals=np.linalg.norm(hull.equations[:,:2],axis=1)
    margin=float(np.min(-vals/normals)); return bool(np.all(vals<=tol)),margin


def dense_projected_trajectory_v004(kin,state,solution,settings=V004Settings()):
    n=solution.body_pos.shape[0]; ks=np.linspace(0.,1.,n); ds=np.linspace(0.,1.,settings.checker_samples)
    t=np.column_stack([np.interp(ds,ks,solution.body_pos[:,j]) for j in range(3)])
    R=Slerp(ks,Rotation.from_matrix(solution.body_R))(ds).as_matrix()
    qf=solution.q.reshape(n,-1)
    q=np.column_stack([np.interp(ds,ks,qf[:,j]) for j in range(qf.shape[1])]).reshape(settings.checker_samples,solution.q.shape[1],3)
    old_support=[int(i) for i in np.where(np.asarray(state.contact,bool))[0]]
    is_switch=(solution.mode=='contact_switch')
    td=float(settings.touchdown_node)/float(settings.n_nodes-1) if is_switch else np.inf
    lo=float(settings.liftoff_node)/float(settings.n_nodes-1) if is_switch else np.inf
    tdleg=int(solution.candidate.touchdown_leg) if is_switch else None
    loleg=int(solution.candidate.liftoff_leg) if is_switch else None
    touchdown=np.array([solution.touchdown_xy[0],solution.touchdown_xy[1],0.]) if is_switch else None
    failures=0
    prev=np.asarray(state.q,float).copy()
    for f,p in enumerate(ds):
        anchors={leg:np.asarray(state.anchors_world[leg],float) for leg in old_support}
        active=list(old_support)
        if is_switch and p>=td-1e-12:
            active.append(tdleg); anchors[tdleg]=touchdown
        if is_switch and p>lo+1e-12 and loleg in active:
            active.remove(loleg)
        for leg in active:
            ref=prev[leg] if f>0 else np.asarray(state.q[leg],float)
            branches=analytic_leg_ik_world(kin,t[f],R[f],leg,anchors[leg],q_reference=ref)
            if not branches:
                failures+=1
                continue
            q[f,leg]=branches[0]
        prev=q[f].copy()
    max_step=0.0
    if len(q)>1:
        max_step=float(np.max(np.abs(np.diff(q,axis=0))))
    return ds,t,R,q,failures,max_step


def dense_check_solution_v004(kin,state,solution,target_body_pos,target_body_R,settings=V004Settings()):
    dense_s,t,R,q,ik_failures,max_dense_joint_step=dense_projected_trajectory_v004(kin,state,solution,settings)
    tol=float(settings.checker_tolerance)
    old_support=[int(i) for i in np.where(np.asarray(state.contact,bool))[0]]
    is_switch=(solution.mode=='contact_switch')
    if is_switch:
        tdleg=int(solution.candidate.touchdown_leg); loleg=int(solution.candidate.liftoff_leg)
        td=float(settings.touchdown_node)/float(settings.n_nodes-1); lo=float(settings.liftoff_node)/float(settings.n_nodes-1)
        touchdown=np.array([solution.touchdown_xy[0],solution.touchdown_xy[1],0.])
    else:
        tdleg=loleg=None; td=lo=np.inf; touchdown=None
    max_joint=max_support=max_td=0.; min_swing=min_margin=min_body=min_link=min_caps=min_body_link=np.inf; margin=float(settings.collision_margin_m)
    for f,p in enumerate(dense_s):
        low=np.maximum(kin.q_min-q[f],0.); high=np.maximum(q[f]-kin.q_max,0.); max_joint=max(max_joint,float(np.max(np.maximum(low,high))))
        roots,elbows,feet=state_points(kin,t[f],R[f],q[f])
        for leg in old_support:
            if is_switch and leg==loleg and p>lo+1e-12: continue
            max_support=max(max_support,float(np.linalg.norm(feet[leg]-np.asarray(state.anchors_world[leg],float))))
        support_now=list(old_support)
        if is_switch and p>=td-1e-12:
            max_td=max(max_td,float(np.linalg.norm(feet[tdleg]-touchdown))); support_now.append(tdleg)
        if is_switch and p>lo+1e-12 and loleg in support_now: support_now.remove(loleg)
        for leg in range(kin.n_legs):
            if leg not in support_now: min_swing=min(min_swing,float(feet[leg,2]))
        sxy=[]
        for leg in old_support:
            if is_switch and p>lo+1e-12 and leg==loleg: continue
            sxy.append(np.asarray(state.anchors_world[leg],float)[:2])
        if is_switch and p>=td-1e-12: sxy.append(touchdown[:2])
        inside,sm=_point_in_support_hull(t[f,:2],sxy,tol)
        if not inside: sm=min(sm,-10.*tol)
        min_margin=min(min_margin,float(sm))
        min_body=min(min_body,float(t[f,2]-kin.a*np.sum(np.abs(R[f,2,:]))))
        for leg in range(kin.n_legs): min_link=min(min_link,float(roots[leg,2]),float(elbows[leg,2]),float(feet[leg,2]))
        segs=capsule_segments(kin,roots,elbows,feet,settings)
        for ia,A in enumerate(segs):
            for ib in range(ia+1,len(segs)):
                B=segs[ib]
                if A[0]==B[0]: continue
                min_caps=min(min_caps,float(segment_segment_distance(A[2],A[3],B[2],B[3])-(A[4]+B[4]+margin)))
        for _,link,p0w,p1w,radius in segs:
            if link==0 and settings.root_ignore_m>0.: p0w,p1w=trim_segment_start(p0w,p1w,settings.root_ignore_m)
            p0b=R[f].T.dot(p0w-t[f]); p1b=R[f].T.dot(p1w-t[f])
            min_body_link=min(min_body_link,float(segment_aabb_distance(p0b,p1b,kin.a)-(radius+margin)))
    terr=float(np.linalg.norm(solution.body_pos[-1]-np.asarray(target_body_pos,float)))
    rerr=float(np.linalg.norm(Rotation.from_matrix(solution.body_R[-1].dot(np.asarray(target_body_R,float).T)).as_rotvec()))
    feasible=bool(ik_failures==0 and max_joint<=tol and max_support<=tol and max_td<=tol and min_swing>=-tol and min_margin>=-tol-settings.support_margin_m and min_body>=-tol and min_link>=-tol and min_caps>=-tol and min_body_link>=-tol and terr<=tol and rerr<=tol)
    return DenseCheckReportV004(feasible,float(max_joint),float(max_support),float(max_td),float(min_swing),float(min_margin),float(min_body),float(min_link),float(min_caps),float(min_body_link),terr,rerr,int(ik_failures),float(max_dense_joint_step))
