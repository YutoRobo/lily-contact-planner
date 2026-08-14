import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.spatial.transform import Rotation

from .analytic_ik import analytic_leg_ik_world
from .nlp_geometry_v002 import geometry_inequalities
from .v004_types import TrajectorySolutionV004, V004Settings


class _NoContactLayout:
    def __init__(self,n_nodes,n_legs,n_support):
        self.n_nodes=n_nodes; self.n_legs=n_legs; self.n_support=n_support; off=0
        self.body_pos=slice(off,off+3*n_nodes); off+=3*n_nodes
        self.body_rotvec=slice(off,off+3*n_nodes); off+=3*n_nodes
        self.q=slice(off,off+3*n_legs*n_nodes); off+=3*n_legs*n_nodes
        self.support_weights=slice(off,off+n_nodes*n_support); off+=n_nodes*n_support
        self.size=off
    def unpack(self,z):
        z=np.asarray(z,float)
        return (z[self.body_pos].reshape(self.n_nodes,3),
                z[self.body_rotvec].reshape(self.n_nodes,3),
                z[self.q].reshape(self.n_nodes,self.n_legs,3),
                z[self.support_weights].reshape(self.n_nodes,self.n_support))


class NoContactNLPV004:
    def __init__(self,kin,state,target_body_pos,target_body_R,settings=V004Settings()):
        state.validate(kin); self.kin=kin; self.state=state; self.settings=settings
        self.target_body_pos=np.asarray(target_body_pos,float); self.target_body_R=np.asarray(target_body_R,float)
        self.support_idx=[int(i) for i in np.where(np.asarray(state.contact,bool))[0]]
        self.layout=_NoContactLayout(settings.n_nodes,kin.n_legs,len(self.support_idx))
        self.anchor_xy=np.asarray([np.asarray(state.anchors_world[i],float)[:2] for i in self.support_idx])
        self.delta_t=(self.target_body_pos-np.asarray(state.body_pos,float))/float(settings.n_nodes-1)
        Rrel=self.target_body_R.dot(np.asarray(state.body_R,float).T); self.delta_r=Rotation.from_matrix(Rrel).as_rotvec()/float(settings.n_nodes-1)
    @staticmethod
    def _rotations(rv): return Rotation.from_rotvec(rv).as_matrix()
    def _body_seed(self):
        n=self.settings.n_nodes; t=np.zeros((n,3)); R=np.zeros((n,3,3)); t[0]=self.state.body_pos; R[0]=self.state.body_R
        dR=Rotation.from_rotvec(self.delta_r).as_matrix()
        for k in range(1,n): t[k]=t[k-1]+self.delta_t; R[k]=dR.dot(R[k-1])
        return t,R
    def initial_guess(self):
        n=self.settings.n_nodes; t,R=self._body_seed(); rv=Rotation.from_matrix(R).as_rotvec(); q=np.repeat(np.asarray(self.state.q,float)[None,:,:],n,axis=0)
        for k in range(1,n):
            for leg in self.support_idx:
                br=analytic_leg_ik_world(self.kin,t[k],R[k],leg,np.asarray(self.state.anchors_world[leg],float),q_reference=q[k-1,leg])
                if br: q[k,leg]=br[0]
        w=np.zeros((n,len(self.support_idx)))
        for k in range(n):
            A=np.vstack([self.anchor_xy.T,np.ones((1,len(self.support_idx)))]); b=np.array([t[k,0],t[k,1],1.])
            wk=np.linalg.lstsq(A,b,rcond=None)[0]; wk=np.maximum(wk,0.); wk=(wk/np.sum(wk)) if np.sum(wk)>1e-12 else np.full_like(wk,1./len(wk)); w[k]=wk
        z=np.empty(self.layout.size); z[self.layout.body_pos]=t.ravel(); z[self.layout.body_rotvec]=rv.ravel(); z[self.layout.q]=q.ravel(); z[self.layout.support_weights]=w.ravel(); return z
    def bounds(self):
        lo=np.full(self.layout.size,-np.inf); hi=np.full(self.layout.size,np.inf); lo[self.layout.q]=np.tile(self.kin.q_min,self.settings.n_nodes*self.kin.n_legs); hi[self.layout.q]=np.tile(self.kin.q_max,self.settings.n_nodes*self.kin.n_legs); lo[self.layout.support_weights]=0.; hi[self.layout.support_weights]=1.; return Bounds(lo,hi)
    def objective(self,z):
        t,rv,q,w=self.layout.unpack(z); R=self._rotations(rv); cost=0.
        for k in range(self.settings.n_nodes-1):
            dp=t[k+1]-t[k]; drot=Rotation.from_matrix(R[k+1].dot(R[k].T)).as_rotvec(); dq=q[k+1]-q[k]; et=dp-self.delta_t; er=drot-self.delta_r
            cost += self.settings.weight_tracking_translation*float(et@et)+self.settings.weight_tracking_rotation*float(er@er)+self.settings.weight_joint_motion*float(dq.ravel()@dq.ravel())
        for k in range(1,self.settings.n_nodes-1):
            ddq=q[k+1]-2*q[k]+q[k-1]; cost += self.settings.weight_joint_smoothness*float(ddq.ravel()@ddq.ravel())
        return float(cost)
    def equality_constraints(self,z):
        t,rv,q,w=self.layout.unpack(z); R=self._rotations(rv); eq=[]; eq.extend(t[0]-self.state.body_pos); eq.extend(Rotation.from_matrix(R[0].dot(self.state.body_R.T)).as_rotvec()); eq.extend((q[0]-self.state.q).ravel())
        for k in range(self.settings.n_nodes):
            if k>0:
                for leg in self.support_idx: eq.extend(self.kin.foot_world(t[k],R[k],leg,q[k,leg])-np.asarray(self.state.anchors_world[leg],float))
            eq.append(float(np.sum(w[k])-1.0)); eq.extend(t[k,:2]-np.sum(w[k,:,None]*self.anchor_xy,axis=0))
        eq.extend(t[-1]-self.target_body_pos); eq.extend(Rotation.from_matrix(R[-1].dot(self.target_body_R.T)).as_rotvec()); return np.asarray(eq,float)
    def inequality_constraints(self,z):
        t,rv,q,w=self.layout.unpack(z); R=self._rotations(rv); out=[]; support=set(self.support_idx)
        for k in range(self.settings.n_nodes):
            vals,_=geometry_inequalities(self.kin,t[k],R[k],q[k],support,self.settings); out.extend(vals)
        return np.asarray(out,float)
    def solve(self, initial_guess=None):
        z0=self.initial_guess() if initial_guess is None else np.asarray(initial_guess,float).copy()
        res=minimize(self.objective,z0,method='SLSQP',bounds=self.bounds(),constraints=[{'type':'eq','fun':self.equality_constraints},{'type':'ineq','fun':self.inequality_constraints}],options={'maxiter':int(self.settings.maxiter),'ftol':float(self.settings.ftol),'disp':False})
        t,rv,q,w=self.layout.unpack(res.x); R=self._rotations(rv); eqmax=float(np.max(np.abs(self.equality_constraints(res.x)))); inmin=float(np.min(self.inequality_constraints(res.x))); feasible=eqmax<=self.settings.constraint_tolerance and inmin>=-self.settings.constraint_tolerance; ok=bool(res.success and feasible); msg=f"{res.message}; solver_success={bool(res.success)}; eq_max={eqmax:.3e}; ineq_min={inmin:.3e}"
        return TrajectorySolutionV004('no_contact',ok,msg,float(self.objective(res.x)),t,R,q,scipy_result=res)


class _ContactLayout:
    def __init__(self,n_nodes,n_legs,n_support):
        self.n_nodes=n_nodes; self.n_legs=n_legs; self.n_support=n_support; off=0
        self.body_pos=slice(off,off+3*n_nodes); off+=3*n_nodes; self.body_rotvec=slice(off,off+3*n_nodes); off+=3*n_nodes; self.q=slice(off,off+3*n_legs*n_nodes); off+=3*n_legs*n_nodes; self.touchdown_xy=slice(off,off+2); off+=2; self.support_weights=slice(off,off+n_nodes*(n_support+1)); off+=n_nodes*(n_support+1); self.size=off
    def unpack(self,z):
        z=np.asarray(z,float); return (z[self.body_pos].reshape(self.n_nodes,3),z[self.body_rotvec].reshape(self.n_nodes,3),z[self.q].reshape(self.n_nodes,self.n_legs,3),z[self.touchdown_xy].copy(),z[self.support_weights].reshape(self.n_nodes,self.n_support+1))


class ContactSwitchNLPV004:
    def __init__(self,kin,state,candidate,target_body_pos,target_body_R,settings=V004Settings()):
        state.validate(kin); self.kin=kin; self.state=state; self.candidate=candidate; self.settings=settings; self.target_body_pos=np.asarray(target_body_pos,float); self.target_body_R=np.asarray(target_body_R,float); self.support_idx=[int(i) for i in np.where(np.asarray(state.contact,bool))[0]]; self.td=int(settings.touchdown_node); self.lo=int(settings.liftoff_node)
        self.layout=_ContactLayout(settings.n_nodes,kin.n_legs,len(self.support_idx)); self.anchor_xy=np.asarray([np.asarray(state.anchors_world[i],float)[:2] for i in self.support_idx]); self.lo_col=self.support_idx.index(int(candidate.liftoff_leg)); self.new_col=len(self.support_idx); self.delta_t=(self.target_body_pos-self.state.body_pos)/float(settings.n_nodes-1); self.delta_r=Rotation.from_matrix(self.target_body_R.dot(self.state.body_R.T)).as_rotvec()/float(settings.n_nodes-1)
    @staticmethod
    def _rotations(rv): return Rotation.from_rotvec(rv).as_matrix()
    def support_set_at_node(self,k):
        s=set(self.support_idx)
        if k>=self.td: s.add(int(self.candidate.touchdown_leg))
        if k>=self.lo: s.discard(int(self.candidate.liftoff_leg))
        return s
    def _active_cols(self,k):
        cols=list(range(len(self.support_idx)))
        if k>=self.lo: cols.remove(self.lo_col)
        if k>=self.td: cols.append(self.new_col)
        return cols
    def _body_seed(self):
        n=self.settings.n_nodes; t=np.zeros((n,3)); R=np.zeros((n,3,3)); t[0]=self.state.body_pos; R[0]=self.state.body_R; dR=Rotation.from_rotvec(self.delta_r).as_matrix()
        for k in range(1,n): t[k]=t[k-1]+self.delta_t; R[k]=dR.dot(R[k-1])
        return t,R
    def initial_guess(self):
        n=self.settings.n_nodes; t,R=self._body_seed(); rv=Rotation.from_matrix(R).as_rotvec(); q=np.repeat(self.state.q[None,:,:],n,axis=0); tdleg=int(self.candidate.touchdown_leg); loleg=int(self.candidate.liftoff_leg); touchdown=np.array([self.candidate.touchdown_seed_xy[0],self.candidate.touchdown_seed_xy[1],0.]); loanchor=np.asarray(self.state.anchors_world[loleg],float)
        for k in range(1,n):
            for leg in self.support_idx:
                if leg==loleg and k>self.lo: continue
                br=analytic_leg_ik_world(self.kin,t[k],R[k],leg,np.asarray(self.state.anchors_world[leg],float),q_reference=q[k-1,leg])
                if br: q[k,leg]=br[0]
            if k>=self.td:
                br=analytic_leg_ik_world(self.kin,t[k],R[k],tdleg,touchdown,q_reference=q[k-1,tdleg])
                if br: q[k,tdleg]=br[0]
            if k>self.lo:
                alpha=(k-self.lo)/float(max(1,n-1-self.lo)); target=loanchor+np.array([0,0,self.settings.initial_liftoff_clearance_m*alpha]); br=analytic_leg_ik_world(self.kin,t[k],R[k],loleg,target,q_reference=q[k-1,loleg])
                if br: q[k,loleg]=br[0]
        w=np.zeros((n,len(self.support_idx)+1))
        for k in range(n):
            cols=self._active_cols(k); pts=np.asarray([touchdown[:2] if c==self.new_col else self.anchor_xy[c] for c in cols]); A=np.vstack([pts.T,np.ones((1,len(pts)))]); b=np.array([t[k,0],t[k,1],1.]); wk=np.linalg.lstsq(A,b,rcond=None)[0]; wk=np.maximum(wk,0.); wk=wk/np.sum(wk) if np.sum(wk)>1e-12 else np.full_like(wk,1./len(wk));
            for c,val in zip(cols,wk): w[k,c]=val
        z=np.empty(self.layout.size); z[self.layout.body_pos]=t.ravel(); z[self.layout.body_rotvec]=rv.ravel(); z[self.layout.q]=q.ravel(); z[self.layout.touchdown_xy]=touchdown[:2]; z[self.layout.support_weights]=w.ravel(); return z
    def bounds(self):
        lo=np.full(self.layout.size,-np.inf); hi=np.full(self.layout.size,np.inf); lo[self.layout.q]=np.tile(self.kin.q_min,self.settings.n_nodes*self.kin.n_legs); hi[self.layout.q]=np.tile(self.kin.q_max,self.settings.n_nodes*self.kin.n_legs); lo[self.layout.support_weights]=0.; hi[self.layout.support_weights]=1.; width=len(self.support_idx)+1; start=self.layout.support_weights.start
        for k in range(self.settings.n_nodes):
            if k<self.td: lo[start+k*width+self.new_col]=hi[start+k*width+self.new_col]=0.
            if k>=self.lo: lo[start+k*width+self.lo_col]=hi[start+k*width+self.lo_col]=0.
        return Bounds(lo,hi)
    def objective(self,z):
        t,rv,q,xy,w=self.layout.unpack(z); R=self._rotations(rv); cost=0.
        for k in range(self.settings.n_nodes-1):
            dp=t[k+1]-t[k]; dr=Rotation.from_matrix(R[k+1].dot(R[k].T)).as_rotvec(); dq=q[k+1]-q[k]; et=dp-self.delta_t; er=dr-self.delta_r; cost+=self.settings.weight_tracking_translation*float(et@et)+self.settings.weight_tracking_rotation*float(er@er)+self.settings.weight_joint_motion*float(dq.ravel()@dq.ravel())
        for k in range(1,self.settings.n_nodes-1):
            ddq=q[k+1]-2*q[k]+q[k-1]; cost+=self.settings.weight_joint_smoothness*float(ddq.ravel()@ddq.ravel())
        return float(cost)
    def equality_constraints(self,z):
        t,rv,q,xy,w=self.layout.unpack(z); R=self._rotations(rv); eq=[]; eq.extend(t[0]-self.state.body_pos); eq.extend(Rotation.from_matrix(R[0].dot(self.state.body_R.T)).as_rotvec()); eq.extend((q[0]-self.state.q).ravel()); touchdown=np.array([xy[0],xy[1],0.]); tdleg=int(self.candidate.touchdown_leg); loleg=int(self.candidate.liftoff_leg); ns=len(self.support_idx)
        for k in range(self.settings.n_nodes):
            if k>0:
                for leg in self.support_idx:
                    if leg==loleg and k>self.lo: continue
                    eq.extend(self.kin.foot_world(t[k],R[k],leg,q[k,leg])-np.asarray(self.state.anchors_world[leg],float))
            if k>=self.td: eq.extend(self.kin.foot_world(t[k],R[k],tdleg,q[k,tdleg])-touchdown)
            eq.append(float(np.sum(w[k])-1.0)); sxy=np.sum(w[k,:ns,None]*self.anchor_xy,axis=0)+w[k,self.new_col]*xy; eq.extend(t[k,:2]-sxy)
        eq.extend(t[-1]-self.target_body_pos); eq.extend(Rotation.from_matrix(R[-1].dot(self.target_body_R.T)).as_rotvec()); return np.asarray(eq,float)
    def inequality_constraints(self,z):
        t,rv,q,xy,w=self.layout.unpack(z); R=self._rotations(rv); out=[]
        for k in range(self.settings.n_nodes):
            vals,_=geometry_inequalities(self.kin,t[k],R[k],q[k],self.support_set_at_node(k),self.settings); out.extend(vals)
        return np.asarray(out,float)
    def solve(self):
        res=minimize(self.objective,self.initial_guess(),method='SLSQP',bounds=self.bounds(),constraints=[{'type':'eq','fun':self.equality_constraints},{'type':'ineq','fun':self.inequality_constraints}],options={'maxiter':int(self.settings.maxiter),'ftol':float(self.settings.ftol),'disp':False}); t,rv,q,xy,w=self.layout.unpack(res.x); R=self._rotations(rv); eqmax=float(np.max(np.abs(self.equality_constraints(res.x)))); inmin=float(np.min(self.inequality_constraints(res.x))); feasible=eqmax<=self.settings.constraint_tolerance and inmin>=-self.settings.constraint_tolerance; ok=bool(res.success and feasible); msg=f"{res.message}; solver_success={bool(res.success)}; eq_max={eqmax:.3e}; ineq_min={inmin:.3e}"; return TrajectorySolutionV004('contact_switch',ok,msg,float(self.objective(res.x)),t,R,q,np.asarray(xy,float),self.candidate,scipy_result=res)
