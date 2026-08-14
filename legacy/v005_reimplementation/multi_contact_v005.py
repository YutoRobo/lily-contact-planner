"""Minimal reconstruction of the v0.0.5 multi-contact finite-horizon NLP.

This intentionally mirrors v0.0.4 ContactSwitchNLPV004 and only generalizes
one touchdown / one liftoff to multiple touchdown and liftoff events at fixed
NLP nodes. It is kept under legacy/ until behavior equivalence is locked down.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.spatial.transform import Rotation

from lily_contact_planner.analytic_ik import analytic_leg_ik_world
from lily_contact_planner.nlp_geometry_v002 import geometry_inequalities


@dataclass(frozen=True)
class MultiContactCandidateV005:
    touchdown_legs: tuple
    touchdown_seed_xy: np.ndarray
    touchdown_nodes: tuple
    liftoff_legs: tuple
    liftoff_nodes: tuple


class _Layout:
    def __init__(self, n_nodes, n_legs, n_old_support, n_touch):
        off = 0
        self.n_nodes = n_nodes
        self.n_legs = n_legs
        self.n_old_support = n_old_support
        self.n_touch = n_touch
        self.body_pos = slice(off, off + 3 * n_nodes); off += 3 * n_nodes
        self.body_rotvec = slice(off, off + 3 * n_nodes); off += 3 * n_nodes
        self.q = slice(off, off + 3 * n_legs * n_nodes); off += 3 * n_legs * n_nodes
        self.touchdown_xy = slice(off, off + 2 * n_touch); off += 2 * n_touch
        self.support_weights = slice(off, off + n_nodes * (n_old_support + n_touch)); off += n_nodes * (n_old_support + n_touch)
        self.size = off

    def unpack(self, z):
        z = np.asarray(z, float)
        return (
            z[self.body_pos].reshape(self.n_nodes, 3),
            z[self.body_rotvec].reshape(self.n_nodes, 3),
            z[self.q].reshape(self.n_nodes, self.n_legs, 3),
            z[self.touchdown_xy].reshape(self.n_touch, 2),
            z[self.support_weights].reshape(self.n_nodes, self.n_old_support + self.n_touch),
        )


class MultiContactNLPV005:
    """Finite-horizon multi-contact switch with fixed discrete event nodes."""

    def __init__(self, kin, state, candidate, target_body_pos, target_body_R, settings):
        state.validate(kin)
        self.kin = kin
        self.state = state
        self.candidate = candidate
        self.settings = settings
        self.target_body_pos = np.asarray(target_body_pos, float)
        self.target_body_R = np.asarray(target_body_R, float)
        self.support_idx = [int(i) for i in np.where(np.asarray(state.contact, bool))[0]]
        self.tdlegs = tuple(int(x) for x in candidate.touchdown_legs)
        self.tdnodes = tuple(int(x) for x in candidate.touchdown_nodes)
        self.lolegs = tuple(int(x) for x in candidate.liftoff_legs)
        self.lonodes = tuple(int(x) for x in candidate.liftoff_nodes)
        if len(self.tdlegs) != len(self.tdnodes) or len(self.lolegs) != len(self.lonodes):
            raise ValueError("event tuple length mismatch")
        if any(x in self.support_idx for x in self.tdlegs):
            raise ValueError("touchdown leg already support")
        if any(x not in self.support_idx for x in self.lolegs):
            raise ValueError("liftoff leg not support")
        if any(k <= 0 or k >= settings.n_nodes for k in self.tdnodes + self.lonodes):
            raise ValueError("event node out of range")
        self.layout = _Layout(settings.n_nodes, kin.n_legs, len(self.support_idx), len(self.tdlegs))
        self.anchor_xy = np.asarray([np.asarray(state.anchors_world[i], float)[:2] for i in self.support_idx])
        self.old_col = {leg: i for i, leg in enumerate(self.support_idx)}
        self.new_col = {leg: len(self.support_idx) + i for i, leg in enumerate(self.tdlegs)}
        self.delta_t = (self.target_body_pos - np.asarray(state.body_pos, float)) / float(settings.n_nodes - 1)
        self.delta_r = Rotation.from_matrix(self.target_body_R.dot(np.asarray(state.body_R, float).T)).as_rotvec() / float(settings.n_nodes - 1)

    @staticmethod
    def _rotations(rv):
        return Rotation.from_rotvec(rv).as_matrix()

    def support_set_at_node(self, k):
        support = set(self.support_idx)
        for leg, node in zip(self.tdlegs, self.tdnodes):
            if k >= node:
                support.add(leg)
        for leg, node in zip(self.lolegs, self.lonodes):
            if k >= node:
                support.discard(leg)
        return support

    def _active_cols(self, k):
        cols = []
        for leg in self.support_idx:
            node = self.lonodes[self.lolegs.index(leg)] if leg in self.lolegs else None
            if node is None or k < node:
                cols.append(self.old_col[leg])
        for leg, node in zip(self.tdlegs, self.tdnodes):
            if k >= node:
                cols.append(self.new_col[leg])
        return cols

    def _body_seed(self):
        n = self.settings.n_nodes
        t = np.zeros((n, 3))
        R = np.zeros((n, 3, 3))
        t[0] = self.state.body_pos
        R[0] = self.state.body_R
        dR = Rotation.from_rotvec(self.delta_r).as_matrix()
        for k in range(1, n):
            t[k] = t[k - 1] + self.delta_t
            R[k] = dR.dot(R[k - 1])
        return t, R

    def initial_guess(self):
        n = self.settings.n_nodes
        t, R = self._body_seed()
        rv = Rotation.from_matrix(R).as_rotvec()
        q = np.repeat(np.asarray(self.state.q, float)[None, :, :], n, axis=0)
        tdxy = np.asarray(self.candidate.touchdown_seed_xy, float).reshape(len(self.tdlegs), 2)
        tdtargets = {leg: np.r_[xy, 0.0] for leg, xy in zip(self.tdlegs, tdxy)}

        for k in range(1, n):
            for leg in self.support_idx:
                if leg in self.lolegs and k > self.lonodes[self.lolegs.index(leg)]:
                    continue
                branches = analytic_leg_ik_world(
                    self.kin, t[k], R[k], leg,
                    np.asarray(self.state.anchors_world[leg], float),
                    q_reference=q[k - 1, leg],
                )
                if branches:
                    q[k, leg] = branches[0]
            for leg, node in zip(self.tdlegs, self.tdnodes):
                if k >= node:
                    branches = analytic_leg_ik_world(
                        self.kin, t[k], R[k], leg, tdtargets[leg],
                        q_reference=q[k - 1, leg],
                    )
                    if branches:
                        q[k, leg] = branches[0]
            for leg, node in zip(self.lolegs, self.lonodes):
                if k > node:
                    anchor = np.asarray(self.state.anchors_world[leg], float)
                    alpha = (k - node) / float(max(1, n - 1 - node))
                    target = anchor + np.array([0.0, 0.0, self.settings.initial_liftoff_clearance_m * alpha])
                    branches = analytic_leg_ik_world(
                        self.kin, t[k], R[k], leg, target,
                        q_reference=q[k - 1, leg],
                    )
                    if branches:
                        q[k, leg] = branches[0]

        width = len(self.support_idx) + len(self.tdlegs)
        weights = np.zeros((n, width))
        for k in range(n):
            cols = self._active_cols(k)
            points = []
            for col in cols:
                if col < len(self.support_idx):
                    points.append(self.anchor_xy[col])
                else:
                    points.append(tdxy[col - len(self.support_idx)])
            points = np.asarray(points)
            A = np.vstack([points.T, np.ones((1, len(points)))])
            b = np.array([t[k, 0], t[k, 1], 1.0])
            wk = np.linalg.lstsq(A, b, rcond=None)[0]
            wk = np.maximum(wk, 0.0)
            wk = wk / np.sum(wk) if np.sum(wk) > 1e-12 else np.full_like(wk, 1.0 / len(wk))
            for col, value in zip(cols, wk):
                weights[k, col] = value

        z = np.empty(self.layout.size)
        z[self.layout.body_pos] = t.ravel()
        z[self.layout.body_rotvec] = rv.ravel()
        z[self.layout.q] = q.ravel()
        z[self.layout.touchdown_xy] = tdxy.ravel()
        z[self.layout.support_weights] = weights.ravel()
        return z

    def bounds(self):
        lo = np.full(self.layout.size, -np.inf)
        hi = np.full(self.layout.size, np.inf)
        lo[self.layout.q] = np.tile(self.kin.q_min, self.settings.n_nodes * self.kin.n_legs)
        hi[self.layout.q] = np.tile(self.kin.q_max, self.settings.n_nodes * self.kin.n_legs)
        lo[self.layout.support_weights] = 0.0
        hi[self.layout.support_weights] = 1.0
        width = len(self.support_idx) + len(self.tdlegs)
        start = self.layout.support_weights.start
        for k in range(self.settings.n_nodes):
            active = set(self._active_cols(k))
            for col in range(width):
                if col not in active:
                    lo[start + k * width + col] = 0.0
                    hi[start + k * width + col] = 0.0
        return Bounds(lo, hi)

    def objective(self, z):
        t, rv, q, _, _ = self.layout.unpack(z)
        R = self._rotations(rv)
        cost = 0.0
        for k in range(self.settings.n_nodes - 1):
            dp = t[k + 1] - t[k]
            dr = Rotation.from_matrix(R[k + 1].dot(R[k].T)).as_rotvec()
            dq = q[k + 1] - q[k]
            et = dp - self.delta_t
            er = dr - self.delta_r
            cost += self.settings.weight_tracking_translation * float(et @ et)
            cost += self.settings.weight_tracking_rotation * float(er @ er)
            cost += self.settings.weight_joint_motion * float(dq.ravel() @ dq.ravel())
        for k in range(1, self.settings.n_nodes - 1):
            ddq = q[k + 1] - 2 * q[k] + q[k - 1]
            cost += self.settings.weight_joint_smoothness * float(ddq.ravel() @ ddq.ravel())
        return float(cost)

    def equality_constraints(self, z):
        t, rv, q, xy, weights = self.layout.unpack(z)
        R = self._rotations(rv)
        eq = []
        ns = len(self.support_idx)
        eq.extend(t[0] - self.state.body_pos)
        eq.extend(Rotation.from_matrix(R[0].dot(self.state.body_R.T)).as_rotvec())
        eq.extend((q[0] - self.state.q).ravel())
        tdtargets = {leg: np.r_[point, 0.0] for leg, point in zip(self.tdlegs, xy)}
        for k in range(self.settings.n_nodes):
            if k > 0:
                for leg in self.support_idx:
                    if leg in self.lolegs and k > self.lonodes[self.lolegs.index(leg)]:
                        continue
                    eq.extend(self.kin.foot_world(t[k], R[k], leg, q[k, leg]) - np.asarray(self.state.anchors_world[leg], float))
            for leg, node in zip(self.tdlegs, self.tdnodes):
                if k >= node:
                    eq.extend(self.kin.foot_world(t[k], R[k], leg, q[k, leg]) - tdtargets[leg])
            eq.append(float(np.sum(weights[k]) - 1.0))
            support_xy = np.sum(weights[k, :ns, None] * self.anchor_xy, axis=0)
            for j in range(len(self.tdlegs)):
                support_xy += weights[k, ns + j] * xy[j]
            eq.extend(t[k, :2] - support_xy)
        eq.extend(t[-1] - self.target_body_pos)
        eq.extend(Rotation.from_matrix(R[-1].dot(self.target_body_R.T)).as_rotvec())
        return np.asarray(eq, float)

    def inequality_constraints(self, z):
        t, rv, q, _, _ = self.layout.unpack(z)
        R = self._rotations(rv)
        out = []
        for k in range(self.settings.n_nodes):
            values, _ = geometry_inequalities(
                self.kin, t[k], R[k], q[k], self.support_set_at_node(k), self.settings
            )
            out.extend(values)
        return np.asarray(out, float)

    def solve(self, initial_guess=None):
        z0 = self.initial_guess() if initial_guess is None else np.asarray(initial_guess, float).copy()
        result = minimize(
            self.objective,
            z0,
            method="SLSQP",
            bounds=self.bounds(),
            constraints=[
                {"type": "eq", "fun": self.equality_constraints},
                {"type": "ineq", "fun": self.inequality_constraints},
            ],
            options={
                "maxiter": int(self.settings.maxiter),
                "ftol": float(self.settings.ftol),
                "disp": False,
            },
        )
        t, rv, q, xy, _ = self.layout.unpack(result.x)
        R = self._rotations(rv)
        eq_max = float(np.max(np.abs(self.equality_constraints(result.x))))
        ineq_min = float(np.min(self.inequality_constraints(result.x)))
        feasible = eq_max <= self.settings.constraint_tolerance and ineq_min >= -self.settings.constraint_tolerance
        return {
            "success": bool(result.success and feasible),
            "message": f"{result.message}; solver_success={bool(result.success)}; eq_max={eq_max:.3e}; ineq_min={ineq_min:.3e}",
            "objective": float(self.objective(result.x)),
            "body_pos": t,
            "body_R": R,
            "q": q,
            "touchdown_xy": xy,
            "eq_max": eq_max,
            "ineq_min": ineq_min,
            "scipy_result": result,
        }
