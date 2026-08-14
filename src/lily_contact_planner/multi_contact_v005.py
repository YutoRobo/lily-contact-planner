"""Recovered v0.0.5 multi-contact finite-horizon fallback.

The implementation preserves the archived Pitch45->Roll45 behavior: Sobol
contact seeds, 5->1 degree horizon fallback, (2,2)/(2,1)/(1,2) event patterns,
SLSQP continuous NLP, dense finite-thickness checking, and receding execution
only through the final liftoff node.
"""
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import qmc

from .analytic_ik import analytic_leg_ik_world
from .nlp_geometry_v002 import geometry_inequalities, capsule_segments, state_points
from .collision import segment_aabb_distance, segment_segment_distance, trim_segment_start
from .checker import _point_in_support_hull


@dataclass(frozen=True)
class V005MultiSettings:
    n_nodes: int = 11
    checker_samples: int = 101
    touchdown_seeds_per_leg: int = 5
    touchdown_seed_min_xy_distance_m: float = 0.05
    seed_samples_per_leg: int = 256
    l2_radius_m: float = 0.025
    l3_radius_m: float = 0.025
    collision_margin_m: float = 0.0
    support_margin_m: float = 0.0
    swing_ground_margin_m: float = 0.0
    root_attachment_ignore_m: float = 0.05
    weight_tracking_translation: float = 1.0
    weight_tracking_rotation: float = 1.0
    weight_joint_motion: float = 1.0e-3
    weight_joint_smoothness: float = 1.0e-4
    constraint_tolerance: float = 1.0e-6
    checker_tolerance: float = 1.0e-5
    maxiter: int = 300
    ftol: float = 1.0e-8
    initial_liftoff_clearance_m: float = 0.02
    initial_touchdown_arc_m: float = 0.02

    @property
    def root_ignore_m(self):
        return float(self.root_attachment_ignore_m)


@dataclass
class V005State:
    body_pos: np.ndarray
    body_R: np.ndarray
    q: np.ndarray
    contact: np.ndarray
    anchors_world: dict

    def validate(self, kin):
        if np.asarray(self.q).shape != (kin.n_legs, 3):
            raise ValueError("q shape mismatch")
        if np.asarray(self.contact).shape != (kin.n_legs,):
            raise ValueError("contact shape mismatch")


@dataclass(frozen=True)
class MultiContactCandidateV005:
    touchdown_legs: tuple
    touchdown_seed_xy: np.ndarray
    touchdown_nodes: tuple
    liftoff_legs: tuple
    liftoff_nodes: tuple


class _Layout:
    def __init__(self, n_nodes, n_legs, n_old, n_touch):
        o = 0
        self.n_nodes, self.n_legs = n_nodes, n_legs
        self.n_old, self.n_touch = n_old, n_touch
        self.body_pos = slice(o, o + 3*n_nodes); o += 3*n_nodes
        self.body_rotvec = slice(o, o + 3*n_nodes); o += 3*n_nodes
        self.q = slice(o, o + 3*n_legs*n_nodes); o += 3*n_legs*n_nodes
        self.touchdown_xy = slice(o, o + 2*n_touch); o += 2*n_touch
        self.support_weights = slice(o, o + n_nodes*(n_old+n_touch)); o += n_nodes*(n_old+n_touch)
        self.size = o

    def unpack(self, z):
        z = np.asarray(z, float)
        return (
            z[self.body_pos].reshape(self.n_nodes, 3),
            z[self.body_rotvec].reshape(self.n_nodes, 3),
            z[self.q].reshape(self.n_nodes, self.n_legs, 3),
            z[self.touchdown_xy].reshape(self.n_touch, 2),
            z[self.support_weights].reshape(self.n_nodes, self.n_old+self.n_touch),
        )


class MultiContactNLPV005:
    def __init__(self, kin, state, candidate, target_body_pos, target_body_R, settings):
        state.validate(kin)
        self.kin, self.state, self.candidate, self.settings = kin, state, candidate, settings
        self.target_body_pos = np.asarray(target_body_pos, float)
        self.target_body_R = np.asarray(target_body_R, float)
        self.support_idx = [int(i) for i in np.where(np.asarray(state.contact, bool))[0]]
        self.tdlegs = tuple(map(int, candidate.touchdown_legs))
        self.tdnodes = tuple(map(int, candidate.touchdown_nodes))
        self.lolegs = tuple(map(int, candidate.liftoff_legs))
        self.lonodes = tuple(map(int, candidate.liftoff_nodes))
        self.layout = _Layout(settings.n_nodes, kin.n_legs, len(self.support_idx), len(self.tdlegs))
        self.anchor_xy = np.asarray([state.anchors_world[i][:2] for i in self.support_idx], float)
        self.old_col = {l:i for i,l in enumerate(self.support_idx)}
        self.new_col = {l:len(self.support_idx)+i for i,l in enumerate(self.tdlegs)}
        self.delta_t = (self.target_body_pos-state.body_pos)/(settings.n_nodes-1)
        self.delta_r = Rotation.from_matrix(self.target_body_R.dot(state.body_R.T)).as_rotvec()/(settings.n_nodes-1)

    @staticmethod
    def _rot(rv):
        return Rotation.from_rotvec(rv).as_matrix()

    def support_set_at_node(self, k):
        s = set(self.support_idx)
        for leg,node in zip(self.tdlegs,self.tdnodes):
            if k >= node: s.add(leg)
        for leg,node in zip(self.lolegs,self.lonodes):
            if k >= node: s.discard(leg)
        return s

    def _active_cols(self, k):
        cols = []
        for leg in self.support_idx:
            node = self.lonodes[self.lolegs.index(leg)] if leg in self.lolegs else None
            if node is None or k < node: cols.append(self.old_col[leg])
        for leg,node in zip(self.tdlegs,self.tdnodes):
            if k >= node: cols.append(self.new_col[leg])
        return cols

    def _body_seed(self):
        n = self.settings.n_nodes
        t = np.zeros((n,3)); R = np.zeros((n,3,3))
        t[0], R[0] = self.state.body_pos, self.state.body_R
        dR = Rotation.from_rotvec(self.delta_r).as_matrix()
        for k in range(1,n):
            t[k] = t[k-1] + self.delta_t
            R[k] = dR.dot(R[k-1])
        return t,R

    def initial_guess(self):
        n = self.settings.n_nodes
        t,R = self._body_seed()
        rv = Rotation.from_matrix(R).as_rotvec()
        q = np.repeat(np.asarray(self.state.q,float)[None,:,:],n,axis=0)
        tdxy = np.asarray(self.candidate.touchdown_seed_xy,float).reshape(len(self.tdlegs),2)
        tdtargets = {l:np.r_[xy,0.] for l,xy in zip(self.tdlegs,tdxy)}
        start_feet = {l:self.kin.foot_world(t[0],R[0],l,q[0,l]).copy() for l in self.tdlegs}

        for k in range(1,n):
            for leg in self.support_idx:
                if leg in self.lolegs and k > self.lonodes[self.lolegs.index(leg)]:
                    continue
                br = analytic_leg_ik_world(self.kin,t[k],R[k],leg,self.state.anchors_world[leg],q_reference=q[k-1,leg])
                if br: q[k,leg] = br[0]
            for leg,node in zip(self.tdlegs,self.tdnodes):
                if k <= node:
                    s = k/float(node)
                    p = (1-s)*start_feet[leg] + s*tdtargets[leg]
                    p[2] += 4*self.settings.initial_touchdown_arc_m*s*(1-s)
                    br = analytic_leg_ik_world(self.kin,t[k],R[k],leg,p,q_reference=q[k-1,leg])
                else:
                    br = analytic_leg_ik_world(self.kin,t[k],R[k],leg,tdtargets[leg],q_reference=q[k-1,leg])
                if br: q[k,leg] = br[0]
            for leg,node in zip(self.lolegs,self.lonodes):
                if k > node:
                    alpha = (k-node)/float(max(1,n-1-node))
                    target = self.state.anchors_world[leg] + np.array([0.,0.,self.settings.initial_liftoff_clearance_m*alpha])
                    br = analytic_leg_ik_world(self.kin,t[k],R[k],leg,target,q_reference=q[k-1,leg])
                    if br: q[k,leg] = br[0]

        w = np.zeros((n,len(self.support_idx)+len(self.tdlegs)))
        for k in range(n):
            cols = self._active_cols(k)
            pts = [self.anchor_xy[c] if c < len(self.support_idx) else tdxy[c-len(self.support_idx)] for c in cols]
            pts = np.asarray(pts)
            A = np.vstack([pts.T,np.ones((1,len(pts)))])
            b = np.array([t[k,0],t[k,1],1.])
            wk = np.maximum(np.linalg.lstsq(A,b,rcond=None)[0],0.)
            wk = wk/wk.sum() if wk.sum()>1e-12 else np.full_like(wk,1/len(wk))
            for c,v in zip(cols,wk): w[k,c] = v

        z = np.empty(self.layout.size)
        z[self.layout.body_pos] = t.ravel(); z[self.layout.body_rotvec] = rv.ravel()
        z[self.layout.q] = q.ravel(); z[self.layout.touchdown_xy] = tdxy.ravel()
        z[self.layout.support_weights] = w.ravel()
        return z

    def bounds(self):
        lo = np.full(self.layout.size,-np.inf); hi = np.full(self.layout.size,np.inf)
        lo[self.layout.q] = np.tile(self.kin.q_min,self.settings.n_nodes*self.kin.n_legs)
        hi[self.layout.q] = np.tile(self.kin.q_max,self.settings.n_nodes*self.kin.n_legs)
        lo[self.layout.support_weights] = 0.; hi[self.layout.support_weights] = 1.
        W = len(self.support_idx)+len(self.tdlegs); start = self.layout.support_weights.start
        for k in range(self.settings.n_nodes):
            active = set(self._active_cols(k))
            for c in range(W):
                if c not in active: lo[start+k*W+c] = hi[start+k*W+c] = 0.
        return Bounds(lo,hi)

    def objective(self,z):
        t,rv,q,_,_ = self.layout.unpack(z); R = self._rot(rv); cost = 0.
        for k in range(self.settings.n_nodes-1):
            dp=t[k+1]-t[k]; dr=Rotation.from_matrix(R[k+1].dot(R[k].T)).as_rotvec(); dq=q[k+1]-q[k]
            et=dp-self.delta_t; er=dr-self.delta_r
            cost += self.settings.weight_tracking_translation*float(et@et)
            cost += self.settings.weight_tracking_rotation*float(er@er)
            cost += self.settings.weight_joint_motion*float(dq.ravel()@dq.ravel())
        for k in range(1,self.settings.n_nodes-1):
            ddq=q[k+1]-2*q[k]+q[k-1]
            cost += self.settings.weight_joint_smoothness*float(ddq.ravel()@ddq.ravel())
        return float(cost)

    def equality_constraints(self,z):
        t,rv,q,xy,w = self.layout.unpack(z); R=self._rot(rv); eq=[]; ns=len(self.support_idx)
        eq.extend(t[0]-self.state.body_pos)
        eq.extend(Rotation.from_matrix(R[0].dot(self.state.body_R.T)).as_rotvec())
        eq.extend((q[0]-self.state.q).ravel())
        td={l:np.r_[p,0.] for l,p in zip(self.tdlegs,xy)}
        for k in range(self.settings.n_nodes):
            if k>0:
                for leg in self.support_idx:
                    if leg in self.lolegs and k>self.lonodes[self.lolegs.index(leg)]: continue
                    eq.extend(self.kin.foot_world(t[k],R[k],leg,q[k,leg])-self.state.anchors_world[leg])
            for leg,node in zip(self.tdlegs,self.tdnodes):
                if k>=node: eq.extend(self.kin.foot_world(t[k],R[k],leg,q[k,leg])-td[leg])
            eq.append(float(w[k].sum()-1))
            sxy=np.sum(w[k,:ns,None]*self.anchor_xy,axis=0)
            for j in range(len(self.tdlegs)): sxy += w[k,ns+j]*xy[j]
            eq.extend(t[k,:2]-sxy)
        eq.extend(t[-1]-self.target_body_pos)
        eq.extend(Rotation.from_matrix(R[-1].dot(self.target_body_R.T)).as_rotvec())
        return np.asarray(eq,float)

    def inequality_constraints(self,z):
        t,rv,q,_,_ = self.layout.unpack(z); R=self._rot(rv); out=[]
        for k in range(self.settings.n_nodes):
            vals,_ = geometry_inequalities(self.kin,t[k],R[k],q[k],self.support_set_at_node(k),self.settings)
            out.extend(vals)
        return np.asarray(out,float)

    def solve(self):
        res=minimize(self.objective,self.initial_guess(),method="SLSQP",bounds=self.bounds(),constraints=[{"type":"eq","fun":self.equality_constraints},{"type":"ineq","fun":self.inequality_constraints}],options={"maxiter":self.settings.maxiter,"ftol":self.settings.ftol,"disp":False})
        t,rv,q,xy,_=self.layout.unpack(res.x); R=self._rot(rv)
        eq=float(np.max(np.abs(self.equality_constraints(res.x)))); ineq=float(np.min(self.inequality_constraints(res.x)))
        return {"success":bool(res.success and eq<=self.settings.constraint_tolerance and ineq>=-self.settings.constraint_tolerance),"objective":self.objective(res.x),"body_pos":t,"body_R":R,"q":q,"touchdown_xy":xy,"eq_max":eq,"ineq_min":ineq,"nit":int(getattr(res,"nit",-1)),"nfev":int(getattr(res,"nfev",-1)),"message":str(res.message)}


def _support_area(points):
    pts=np.asarray(points,float)
    if len(pts)<3 or np.linalg.matrix_rank(pts-pts[0])<2: return 0.
    try: return float(ConvexHull(pts).volume)
    except QhullError: return 0.


def touchdown_seed_map(kin,state,settings,seed):
    support=[int(i) for i in np.where(state.contact)[0]]
    support_xy=[state.anchors_world[i][:2] for i in support]
    out={}
    for leg0 in np.where(~state.contact)[0]:
        leg=int(leg0)
        sampler=qmc.Sobol(d=3,scramble=True,seed=int(seed)+7919*leg)
        power=int(np.ceil(np.log2(settings.seed_samples_per_leg)))
        samples=sampler.random_base2(power)[:settings.seed_samples_per_leg]
        qs=kin.q_min[None,:]+samples*(kin.q_max-kin.q_min)[None,:]
        qs=np.vstack([state.q[leg],qs]); raw=[]
        for ql in qs:
            f=kin.foot_world(state.body_pos,state.body_R,leg,ql); target=np.array([f[0],f[1],0.])
            br=analytic_leg_ik_world(kin,state.body_pos,state.body_R,leg,target,q_reference=state.q[leg],residual_tol=2e-6)
            if br: raw.append((_support_area(support_xy+[target[:2]]),target[:2].copy()))
        raw.sort(key=lambda x:x[0],reverse=True); selected=[]
        for area,xy in raw:
            if any(np.linalg.norm(xy-old[1])<settings.touchdown_seed_min_xy_distance_m for old in selected): continue
            selected.append((area,xy))
            if len(selected)>=settings.touchdown_seeds_per_leg: break
        if selected: out[leg]=selected
    return out


def event_nodes(nadd,nrem):
    return tuple(range(3,3+nadd)), tuple(range(3+nadd,3+nadd+nrem))


def dense_check_multi(kin,state,sol,cand,settings):
    nk=sol["body_pos"].shape[0]; knots=np.linspace(0,1,nk); dense=np.linspace(0,1,settings.checker_samples)
    bp=np.column_stack([np.interp(dense,knots,sol["body_pos"][:,j]) for j in range(3)])
    bR=Slerp(knots,Rotation.from_matrix(sol["body_R"]))(dense).as_matrix()
    qf=sol["q"].reshape(nk,-1); qq=np.column_stack([np.interp(dense,knots,qf[:,j]) for j in range(qf.shape[1])]).reshape(len(dense),kin.n_legs,3)
    tol=settings.checker_tolerance; old=[int(i) for i in np.where(state.contact)[0]]
    tdp={l:n/(settings.n_nodes-1) for l,n in zip(cand.touchdown_legs,cand.touchdown_nodes)}
    lop={l:n/(settings.n_nodes-1) for l,n in zip(cand.liftoff_legs,cand.liftoff_nodes)}
    tdtargets={l:np.r_[xy,0.] for l,xy in zip(cand.touchdown_legs,sol["touchdown_xy"])}
    maxj=maxsl=maxtd=0.; minswing=minsup=minbody=minlink=mincap=minbl=np.inf
    for i,p in enumerate(dense):
        maxj=max(maxj,float(np.max(np.maximum(np.maximum(kin.q_min-qq[i],0),np.maximum(qq[i]-kin.q_max,0)))))
        roots,els,feet=state_points(kin,bp[i],bR[i],qq[i]); supp=set(old)
        for leg in old:
            if leg in lop and p>lop[leg]+1e-12: continue
            maxsl=max(maxsl,float(np.linalg.norm(feet[leg]-state.anchors_world[leg])))
        for leg in cand.touchdown_legs:
            if p>=tdp[leg]-1e-12: maxtd=max(maxtd,float(np.linalg.norm(feet[leg]-tdtargets[leg]))); supp.add(leg)
        for leg in cand.liftoff_legs:
            if p>=lop[leg]-1e-12: supp.discard(leg)
        for leg in range(kin.n_legs):
            if leg not in supp: minswing=min(minswing,float(feet[leg,2]))
        sxy=[]
        for leg in old:
            if leg in lop and p>=lop[leg]-1e-12: continue
            sxy.append(state.anchors_world[leg][:2])
        for leg in cand.touchdown_legs:
            if p>=tdp[leg]-1e-12: sxy.append(tdtargets[leg][:2])
        inside,margin=_point_in_support_hull(bp[i,:2],np.asarray(sxy),tol=tol)
        minsup=min(minsup,float(margin if inside else min(margin,-10*tol)))
        minbody=min(minbody,float(bp[i,2]-kin.a*np.sum(np.abs(bR[i,2,:]))))
        minlink=min(minlink,float(np.min(roots[:,2])),float(np.min(els[:,2])),float(np.min(feet[:,2])))
        segs=capsule_segments(kin,roots,els,feet,settings)
        for a in range(len(segs)):
            for b in range(a+1,len(segs)):
                A,B=segs[a],segs[b]
                if A[0]==B[0]: continue
                mincap=min(mincap,float(segment_segment_distance(A[2],A[3],B[2],B[3])-(A[4]+B[4]+settings.collision_margin_m)))
        for _,link,p0,p1,r in segs:
            if link==0: p0,p1=trim_segment_start(p0,p1,settings.root_ignore_m)
            minbl=min(minbl,float(segment_aabb_distance(bR[i].T.dot(p0-bp[i]),bR[i].T.dot(p1-bp[i]),kin.a)-r-settings.collision_margin_m))
    feasible=bool(maxj<=tol and maxsl<=tol and maxtd<=tol and minswing>=-tol and minsup>=-tol-settings.support_margin_m and minbody>=-tol and minlink>=-tol and mincap>=-tol and minbl>=-tol)
    return {"feasible":feasible,"max_joint_violation_rad":maxj,"max_support_lock_error_m":maxsl,"max_touchdown_lock_error_m":maxtd,"min_swing_foot_height_m":minswing,"min_support_margin_m":minsup,"min_body_ground_clearance_m":minbody,"min_link_ground_height_m":minlink,"min_capsule_clearance_m":mincap,"min_body_link_clearance_m":minbl}


class V005MultiRecoveryMixin:
    """Recovered multi-contact search. Seed 1303 is the archived baseline seed."""
    v005_multi_settings=V005MultiSettings()
    v005_multi_seed=1303

    def _v005_state(self,angle,q,support,anchors):
        t,R=self._pose(angle); c=np.zeros(self.kin.n_legs,bool); c[list(support)]=True
        return V005State(t.copy(),R.copy(),q.copy(),c,{int(k):np.asarray(v,float).copy() for k,v in anchors.items()})

    def _v005_multi_recovery(self,angle,q,support,anchors,seed=None):
        seed=self.v005_multi_seed if seed is None else int(seed)
        st=self._v005_state(angle,q,support,anchors); cfg=self.v005_multi_settings
        cmap=touchdown_seed_map(self.kin,st,cfg,seed); patterns=((2,2),(2,1),(1,2))
        maxh=min(5,int(np.floor(self.max_roll_deg-angle+1e-9))); trials=[]
        for h in range(maxh,0,-1):
            tt,tR=self._pose(angle+h)
            for nadd,nrem in patterns:
                if len(cmap)<nadd or len(support)<nrem: continue
                tdnode,lonode=event_nodes(nadd,nrem); generated=[]
                for legs in itertools.combinations(sorted(cmap),nadd):
                    for picks in itertools.product(*[cmap[l] for l in legs]):
                        xy=np.asarray([x[1] for x in picks])
                        for rem in itertools.combinations(sorted(support),nrem):
                            generated.append(MultiContactCandidateV005(tuple(legs),xy.copy(),tdnode,tuple(rem),lonode))
                hull_valid=[]; terminal_valid=[]
                for cand in generated:
                    ns=tuple(sorted((set(support)|set(cand.touchdown_legs))-set(cand.liftoff_legs)))
                    na={int(k):np.asarray(v,float).copy() for k,v in anchors.items() if int(k) not in cand.liftoff_legs}
                    for leg,xy in zip(cand.touchdown_legs,cand.touchdown_seed_xy): na[leg]=np.r_[xy,0.]
                    inside,_=_point_in_support_hull(tt[:2],np.asarray([na[l][:2] for l in ns]),tol=1e-8)
                    if not inside: continue
                    hull_valid.append(cand)
                    if not all(analytic_leg_ik_world(self.kin,tt,tR,l,na[l],q_reference=q[l],residual_tol=2e-6) for l in ns): continue
                    terminal_valid.append((_support_area([na[l][:2] for l in ns]),cand,ns,na))
                terminal_valid.sort(key=lambda x:x[0],reverse=True)
                accepted=[]; solved_count=0
                for candidate_index,(_,cand,ns,_) in enumerate(terminal_valid):
                    sol=MultiContactNLPV005(self.kin,st,cand,tt,tR,cfg).solve()
                    if not sol["success"]: continue
                    solved_count+=1; chk=dense_check_multi(self.kin,st,sol,cand,cfg)
                    if not chk["feasible"]: continue
                    na={int(k):np.asarray(v,float).copy() for k,v in anchors.items() if int(k) not in cand.liftoff_legs}
                    for leg,xy in zip(cand.touchdown_legs,sol["touchdown_xy"]): na[leg]=np.r_[xy,0.]
                    accepted.append((sol["objective"],candidate_index,cand,sol,chk,ns,na))
                trials.append({"horizon_deg":h,"pattern":[nadd,nrem],"generated":len(generated),"terminal_hull_ok":len(hull_valid),"terminal_ik_ok":len(terminal_valid),"solved":solved_count,"accepted":len(accepted)})
                if accepted:
                    objective,candidate_index,cand,sol,chk,ns,na=min(accepted,key=lambda x:x[0])
                    exec_node=max(cand.liftoff_nodes); frac=exec_node/float(cfg.n_nodes-1)
                    return {"success":True,"seed":seed,"horizon_deg":float(h),"exec_node":int(exec_node),"exec_fraction":float(frac),"angle_after_deg":float(angle+h*frac),"q_after":np.asarray(sol["q"][exec_node],float).copy(),"support_after":tuple(ns),"anchors_after":na,"candidate_index":int(candidate_index),"candidate":cand,"solution":sol,"checker":chk,"trials":trials,"objective":float(objective)}
        return None
