import os,sys,json,time,importlib.util
import numpy as np
from scipy.spatial.transform import Rotation
ROOT='/mnt/data/Lily_v0.0.4_receding_horizon_draft'; sys.path.insert(0,ROOT+'/src')
from lily_contact_planner import LilyKinematics,V004Settings,V004State
from lily_contact_planner.analytic_ik import analytic_leg_ik_world
# load v005 search module
spec=importlib.util.spec_from_file_location('v005',ROOT+'/scripts/explore_pitch45_roll45_v005_resume.py')
v005=importlib.util.module_from_spec(spec); spec.loader.exec_module(v005)
base=v005.base
OUT='/mnt/data/Lily_v0.0.6_pitch45_roll45_search'; os.makedirs(OUT,exist_ok=True)
v005.OUTDIR=OUT; v005.base.OUTDIR=OUT
SETTINGS=v005.SETTINGS; kin=LilyKinematics(a=.15,L2=.30,L3=.30)

def checker_dict(ch): return v005.jsonable_checker(ch)

def append_frame(H,t,R,q,c):
    if H[0] and np.linalg.norm(t-H[0][-1])<1e-13 and np.linalg.norm(R-H[1][-1])<1e-13 and np.linalg.norm(q-H[2][-1])<1e-13 and np.array_equal(c,H[3][-1]): return
    H[0].append(t.copy());H[1].append(R.copy());H[2].append(q.copy());H[3].append(c.copy())

def make_static_event(H):
    # Start from v0.0.5 certified prefix: Pitch45 + Roll30 stop.
    d=np.load('/mnt/data/Lily_v0.0.5_pitch45_roll45_search/trajectory.npz')
    H[0].extend([x.copy() for x in d['body_pos']]);H[1].extend([x.copy() for x in d['body_R']]);H[2].extend([x.copy() for x in d['q']]);H[3].extend([x.astype(bool).copy() for x in d['contact']])
    t=H[0][-1].copy();R=H[1][-1].copy();q=H[2][-1].copy();c=H[3][-1].copy()
    before_support=np.where(c)[0].tolist()
    # PRM-certified static touchdown of leg4.
    p=np.load('/mnt/data/Lily_v0.0.5_pitch45_roll45_search/prm_leg4_static_path.npz'); path=p['q']; xy=p['goal_xy']
    for a,b in zip(path[:-1],path[1:]):
        for u in np.linspace(0,1,21)[1:]:
            qq=q.copy();qq[4]=(1-u)*a+u*b; append_frame(H,t,R,qq,c)
        q[4]=b.copy()
    # touchdown at exact final PRM state.
    c[4]=True; append_frame(H,t,R,q,c)
    # release leg6, then lift 50 mm vertically at static body pose.
    f6=kin.foot_world(t,R,6,q[6]); branches=analytic_leg_ik_world(kin,t,R,6,f6+np.array([0.,0.,0.05]),q_reference=q[6])
    if not branches: raise RuntimeError('leg6 static lift IK missing')
    q6goal=branches[0].copy(); c[6]=False; append_frame(H,t,R,q,c)
    q6start=q[6].copy()
    for u in np.linspace(0,1,51)[1:]:
        qq=q.copy();qq[6]=(1-u)*q6start+u*q6goal; append_frame(H,t,R,qq,c)
    q[6]=q6goal
    anchors={int(i):kin.foot_world(t,R,int(i),q[int(i)]) for i in np.where(c)[0]}
    st=V004State(t.copy(),R.copy(),q.copy(),c.copy(),anchors)
    ev={'version':'v0.0.6-static-event','phase':'roll','touchdown_legs':[4],'liftoff_legs':[6],
        'touchdown_xy':[xy.tolist()],'support_before':before_support,'support_after':np.where(c)[0].tolist(),
        'body_motion_deg':0.0,'note':'static-body PRM touchdown, then 50mm vertical liftoff; touchdown-before-liftoff'}
    return st,ev

def run():
    H=([],[],[],[]); state,static_ev=make_static_event(H)
    oldrep=json.load(open('/mnt/data/Lily_v0.0.5_pitch45_roll45_search/report.json'))
    cycles=list(oldrep.get('cycles',[])); events=list(oldrep.get('contact_events',[])); events.append(static_ev)
    finalR=Rotation.from_euler('x',45,degrees=True).as_matrix().dot(Rotation.from_euler('y',45,degrees=True).as_matrix())
    t_fixed=state.body_pos.copy(); seed=900; start=time.time(); cycle=0
    print('START after static event support',np.where(state.contact)[0].tolist(),'remaining',v005.orientation_error_deg(finalR,state.body_R),flush=True)
    while v005.orientation_error_deg(finalR,state.body_R)>1e-5:
        cycle+=1
        if cycle>80: raise RuntimeError('cycle limit')
        target_t,target_R,_=base.clipped_horizon_target(state,t_fixed,finalR,SETTINGS)
        rem=v005.orientation_error_deg(finalR,state.body_R); h=v005.orientation_error_deg(target_R,state.body_R)
        print(f'[roll-v006 c{cycle:02d}] rem={rem:.3f} h={h:.3f} support={np.where(state.contact)[0].tolist()}',flush=True)
        seed_ok,nlp,seed_sol,eq0,in0=base.fast_no_contact_seed(kin,state,target_t,target_R,SETTINGS); no_sol=seed_sol
        if not seed_ok:
            unreachable=[]
            for leg in np.where(state.contact)[0]:
                if not analytic_leg_ik_world(kin,target_t,target_R,int(leg),np.asarray(state.anchors_world[int(leg)],float),q_reference=state.q[int(leg)]): unreachable.append(int(leg))
            if not unreachable:
                no_sol=nlp.solve()
                if no_sol.success:no_sol.checker=base.dense_check_solution_v004(kin,state,no_sol,target_t,target_R,SETTINGS)
            else:no_sol.message+='; terminal support IK impossible '+str(unreachable)
        if seed_ok or no_sol.accepted:
            used=no_sol if no_sol.accepted else seed_sol; before=state; step=min(np.deg2rad(1.),np.deg2rad(rem)); state,seg,execfrac=base.no_contact_exec_state(kin,state,used,SETTINGS,step)
            ds,tt,RR,qq,fg=seg;base.append_segment(H[0],H[1],H[2],H[3],ds,tt,RR,qq,before,used,fg)
            cycles.append({'version':'v0.0.6','phase':'roll','cycle':cycle,'decision':'no_contact','remaining_before_deg':rem,'remaining_after_deg':v005.orientation_error_deg(finalR,state.body_R),'supports_before':np.where(before.contact)[0].tolist(),'supports_after':np.where(state.contact)[0].tolist(),'checker':checker_dict(used.checker)})
            continue
        before=state;best1=None;trials=[];selectedh=None
        maxh=max(1,int(np.ceil(h-1e-9)))
        for hdeg in range(maxh,0,-1):
            rfull=Rotation.from_matrix(finalR.dot(state.body_R.T)).as_rotvec();rang=float(np.linalg.norm(rfull));frac=1.0 if rang<=np.deg2rad(hdeg)+1e-12 else np.deg2rad(hdeg)/rang
            ctR=Rotation.from_rotvec(frac*rfull).as_matrix().dot(state.body_R);ctt=t_fixed.copy()
            rr,bb=v005.solve_all_contact_candidates_local(kin,state,ctt,ctR,SETTINGS,seed); trials.append({'h':hdeg,'solved':len(rr),'accepted':sum(1 for _,s,_ in rr if s.accepted)})
            if bb is not None:best1=bb;selectedh=hdeg;break
        seed+=1
        if best1 is not None:
            idx,sol,elapsed=best1;end=SETTINGS.liftoff_node/float(SETTINGS.n_nodes-1);state,seg0=base.event_state_from_solution(kin,state,sol,SETTINGS,end);ds,tt,RR,qq,nc=seg0;base.append_segment(H[0],H[1],H[2],H[3],ds,tt,RR,qq,before,sol,end,nc)
            ev={'version':'v0.0.4-1to1','phase':'roll','cycle_v006':cycle,'touchdown_legs':[int(sol.candidate.touchdown_leg)],'liftoff_legs':[int(sol.candidate.liftoff_leg)],'touchdown_xy':[np.asarray(sol.touchdown_xy,float).tolist()],'support_before':np.where(before.contact)[0].tolist(),'support_after':np.where(state.contact)[0].tolist(),'contact_horizon_deg':selectedh,'checker':checker_dict(sol.checker),'objective':float(sol.objective)};events.append(ev)
            print(' SWITCH 1to1',ev['touchdown_legs'],ev['liftoff_legs'],'->',ev['support_after'],flush=True);continue
        print(' 1to1 exhausted -> multi',flush=True)
        mbest,mtt,mtR,mh,mtrials,mdt=v005.solve_multi_fallback(kin,state,finalR,rem,seed,'rollv006',cycle);seed+=1
        if mbest is None:
            rep={'status':'stopped','stopped_phase':'roll','remaining_deg':rem,'elapsed_v006_s':time.time()-start,'cycles':cycles,'contact_events':events,'final_support':np.where(state.contact)[0].tolist(),'target_definition':'world Pitch +45 deg, then world Roll +45 deg; no translation'}
            json.dump(rep,open(OUT+'/report.json','w'),indent=2);np.savez_compressed(OUT+'/trajectory.npz',body_pos=np.asarray(H[0]),body_R=np.asarray(H[1]),q=np.asarray(H[2]),contact=np.asarray(H[3]));return rep
        midx,msol,melapsed=mbest;state,mseg=v005.event_state_multi(kin,state,msol);v005.append_multi(H[0],H[1],H[2],H[3],before,msol,mseg)
        ev={'version':'v0.0.5-multi','phase':'roll','cycle_v006':cycle,'touchdown_legs':list(msol.candidate.touchdown_legs),'liftoff_legs':list(msol.candidate.liftoff_legs),'touchdown_xy':np.asarray(msol.touchdown_xy,float).tolist(),'support_before':np.where(before.contact)[0].tolist(),'support_after':np.where(state.contact)[0].tolist(),'contact_horizon_deg':mh,'checker':checker_dict(msol.checker),'objective':float(msol.objective)};events.append(ev)
        print(' SWITCH MULTI',ev['touchdown_legs'],ev['liftoff_legs'],'->',ev['support_after'],flush=True)
    rep={'status':'completed','remaining_deg':v005.orientation_error_deg(finalR,state.body_R),'elapsed_v006_s':time.time()-start,'cycles':cycles,'contact_events':events,'final_support':np.where(state.contact)[0].tolist(),'target_definition':'world Pitch +45 deg, then world Roll +45 deg; no translation','note':'v0.0.6 exploration adds static-body PRM contact event at prior Roll30 stall'}
    json.dump(rep,open(OUT+'/report.json','w'),indent=2);np.savez_compressed(OUT+'/trajectory.npz',body_pos=np.asarray(H[0]),body_R=np.asarray(H[1]),q=np.asarray(H[2]),contact=np.asarray(H[3]));return rep
if __name__=='__main__':
 r=run();print(json.dumps({'status':r['status'],'remaining':r.get('remaining_deg'),'support':r['final_support'],'elapsed':r['elapsed_v006_s']},indent=2),flush=True)
