import numpy as np
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError
from scipy.stats import qmc
from .analytic_ik import analytic_leg_ik_world
from .v004_types import ContactCandidateV004, V004Settings


def support_area(points_xy):
    pts=np.asarray(points_xy,float)
    if len(pts)<3: return 0.0
    unique=[]
    for p in pts:
        if not any(np.linalg.norm(p-q)<1e-10 for q in unique): unique.append(p)
    pts=np.asarray(unique,float)
    if len(pts)<3 or np.linalg.matrix_rank(pts-pts[0])<2: return 0.0
    try: return float(ConvexHull(pts).volume)
    except QhullError: return 0.0


def generate_contact_candidates_v004(kin,state,settings=V004Settings(),seed=0):
    """Generate 1-touchdown x seed x 1-liftoff candidates; event nodes are fixed."""
    state.validate(kin)
    contact=np.asarray(state.contact,bool)
    support_idx=[int(i) for i in np.where(contact)[0]]
    swing_idx=[int(i) for i in np.where(~contact)[0]]
    support_xy=[np.asarray(state.anchors_world[i],float)[:2] for i in support_idx]
    out=[]
    for leg in swing_idx:
        sampler=qmc.Sobol(d=3,scramble=True,seed=int(seed)+7919*leg)
        n=max(int(settings.seed_samples_per_leg),settings.touchdown_seeds_per_leg)
        power=int(np.ceil(np.log2(max(n,1))))
        samples=sampler.random_base2(m=power)[:n]
        q_samples=kin.q_min[None,:]+samples*(kin.q_max-kin.q_min)[None,:]
        q_samples=np.vstack([np.asarray(state.q[leg],float),q_samples])
        raw=[]
        for q_leg in q_samples:
            foot=kin.foot_world(state.body_pos,state.body_R,leg,q_leg)
            target=np.array([foot[0],foot[1],0.0])
            branches=analytic_leg_ik_world(kin,state.body_pos,state.body_R,leg,target,
                                           q_reference=state.q[leg],residual_tol=2e-6)
            if branches:
                raw.append((support_area(support_xy+[target[:2]]),target[:2].copy()))
        raw.sort(key=lambda x:x[0],reverse=True)
        selected=[]
        for area,xy in raw:
            if any(np.linalg.norm(xy-old)<settings.touchdown_seed_min_xy_distance_m for _,old in selected):
                continue
            selected.append((area,xy))
            if len(selected)>=settings.touchdown_seeds_per_leg: break
        for area,xy in selected:
            for lo in support_idx:
                out.append(ContactCandidateV004(leg,np.asarray(xy,float),lo,float(area)))
    return out
