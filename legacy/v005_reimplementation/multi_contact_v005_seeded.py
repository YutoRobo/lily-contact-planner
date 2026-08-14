"""Refined v0.0.5 reconstruction: restore the archived touchdown seed path.

This module does not change the NLP constraints, objective, event schedule, or
candidate ranking.  It only restores the initial-guess rule observed in the
verified v0.0.5 trajectory: each future touchdown foot follows a Cartesian
straight-line path to its touchdown point with a 20 mm parabolic lift,

    p(s) = (1-s) p0 + s p_goal + [0, 0, 4 h s (1-s)]

for NLP nodes before touchdown.  At and after touchdown, the foot is locked to
the touchdown point exactly as in MultiContactNLPV005.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from lily_contact_planner.analytic_ik import analytic_leg_ik_world
from multi_contact_v005 import MultiContactNLPV005


class MultiContactNLPV005Seeded(MultiContactNLPV005):
    """Same v0.0.5 NLP with the recovered Cartesian touchdown initial guess."""

    def initial_guess(self):
        n = self.settings.n_nodes
        t, R = self._body_seed()
        rv = Rotation.from_matrix(R).as_rotvec()
        q = np.repeat(np.asarray(self.state.q, float)[None, :, :], n, axis=0)

        tdxy = np.asarray(
            self.candidate.touchdown_seed_xy, float
        ).reshape(len(self.tdlegs), 2)
        tdtargets = {
            leg: np.r_[xy, 0.0]
            for leg, xy in zip(self.tdlegs, tdxy)
        }
        tdstarts = {
            leg: self.kin.foot_world(
                np.asarray(self.state.body_pos, float),
                np.asarray(self.state.body_R, float),
                leg,
                np.asarray(self.state.q[leg], float),
            ).copy()
            for leg in self.tdlegs
        }

        # v0.0.3 used 20 mm for the released-foot seed.  The archived v0.0.5
        # touchdown path independently reconstructs the same ~20 mm parabola.
        lift = float(getattr(self.settings, "initial_touchdown_lift_m", 0.02))

        for k in range(1, n):
            # Existing supports track their fixed anchors until their release.
            for leg in self.support_idx:
                if leg in self.lolegs and k > self.lonodes[self.lolegs.index(leg)]:
                    continue
                branches = analytic_leg_ik_world(
                    self.kin,
                    t[k],
                    R[k],
                    leg,
                    np.asarray(self.state.anchors_world[leg], float),
                    q_reference=q[k - 1, leg],
                )
                if branches:
                    q[k, leg] = branches[0]

            # Recovered touchdown seed: move the swing foot before contact.
            for leg, node in zip(self.tdlegs, self.tdnodes):
                goal = tdtargets[leg]
                if k < node:
                    s = float(k) / float(node)
                    target = (1.0 - s) * tdstarts[leg] + s * goal
                    target = target.copy()
                    target[2] += 4.0 * lift * s * (1.0 - s)
                else:
                    target = goal
                branches = analytic_leg_ik_world(
                    self.kin,
                    t[k],
                    R[k],
                    leg,
                    target,
                    q_reference=q[k - 1, leg],
                )
                if branches:
                    q[k, leg] = branches[0]

            # Same v0.0.3/v0.0.5 liftoff seed: after release, raise the old
            # support anchor gradually by the configured clearance (20 mm in
            # the archived setup).
            for leg, node in zip(self.lolegs, self.lonodes):
                if k > node:
                    anchor = np.asarray(self.state.anchors_world[leg], float)
                    alpha = (k - node) / float(max(1, n - 1 - node))
                    target = anchor + np.array([
                        0.0,
                        0.0,
                        self.settings.initial_liftoff_clearance_m * alpha,
                    ])
                    branches = analytic_leg_ik_world(
                        self.kin,
                        t[k],
                        R[k],
                        leg,
                        target,
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
            if np.sum(wk) > 1e-12:
                wk /= np.sum(wk)
            else:
                wk[:] = 1.0 / len(wk)
            for col, value in zip(cols, wk):
                weights[k, col] = value

        z = np.empty(self.layout.size)
        z[self.layout.body_pos] = t.ravel()
        z[self.layout.body_rotvec] = rv.ravel()
        z[self.layout.q] = q.ravel()
        z[self.layout.touchdown_xy] = tdxy.ravel()
        z[self.layout.support_weights] = weights.ravel()
        return z
