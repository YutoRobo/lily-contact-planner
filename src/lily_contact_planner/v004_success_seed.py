"""Successful v0.0.6 one-to-one seed refinement on top of v0.0.4.

Archived trajectories show that future touchdown feet approach the ground point
with linear Cartesian XY and a 20 mm sinusoidal Z arc before touchdown.  This
seed is what makes the later v0.0.4-labelled one-to-one events reproduce the
verified v0.0.6 run.

Candidate handling follows the recovered v0.0.4 planner semantics/specification:
Sobol touchdown seeds are ranked by support-polygon area and spatially
separated in ``candidate_v004``; every resulting touchdown/liftoff candidate is
then solved by the finite-horizon NLP and independently dense-checked.  The
lost v0.0.5 ``solve_all_contact_candidates_local`` source is not reconstructed
by inventing an extra prescreen here.
"""

import numpy as np

from .analytic_ik import analytic_leg_ik_world
from .candidate_v004 import generate_contact_candidates_v004
from .checker_v004 import dense_check_solution_v004
from .trajectory_nlp_v004 import ContactSwitchNLPV004
from .v004_receding import V004RecedingHorizonMixin


class _SuccessfulContactSwitchNLPV004(ContactSwitchNLPV004):
    def initial_guess(self):
        z = super().initial_guess()
        t, rv, q, xy, w = self.layout.unpack(z)
        R = self._rotations(rv)
        leg = int(self.candidate.touchdown_leg)
        p0 = self.kin.foot_world(t[0], R[0], leg, q[0, leg])
        p1 = np.array([xy[0], xy[1], 0.0], dtype=float)
        prev = q[0, leg].copy()
        for k in range(1, self.td):
            s = k / float(self.td)
            target = (1.0 - s) * p0 + s * p1
            target[2] += self.settings.initial_liftoff_clearance_m * np.sin(np.pi * s)
            branches = analytic_leg_ik_world(
                self.kin, t[k], R[k], leg, target,
                q_reference=prev, residual_tol=2e-6,
            )
            if branches:
                q[k, leg] = branches[0]
                prev = branches[0]
        z[self.layout.q] = q.ravel()
        return z


class V004SuccessfulSeedMixin(V004RecedingHorizonMixin):
    """Override only v0.0.4 contact-switch seed construction."""

    def _v004_contact_recovery(self, angle_deg, q, support, anchors):
        cfg = self._v004_settings()
        st = self._v004_state(angle_deg, q, support, anchors)
        seed = int(getattr(self, '_v004_contact_seed', 0))
        candidates = list(generate_contact_candidates_v004(
            self.kin, st, cfg, seed=seed
        ))
        self._v004_contact_seed = seed + 1

        phase_end = self._v004_phase_end(angle_deg)
        max_h = int(np.floor(min(5.0, phase_end - float(angle_deg)) + 1e-9))
        trials = []
        for h in range(max_h, 0, -1):
            target_t, target_R = self._v004_target(angle_deg, float(h))
            accepted = []
            solved = 0
            attempted = 0
            for candidate_index, cand in enumerate(candidates):
                attempted += 1
                sol = _SuccessfulContactSwitchNLPV004(
                    self.kin, st, cand, target_t, target_R, cfg
                ).solve()
                if not sol.success:
                    continue
                solved += 1
                sol.checker = dense_check_solution_v004(
                    self.kin, st, sol, target_t, target_R, cfg
                )
                if sol.accepted:
                    accepted.append((float(sol.objective), candidate_index, sol))

            trials.append({
                'horizon_deg': float(h),
                'candidate_count': int(len(candidates)),
                'attempted_candidates': int(attempted),
                'solved_candidates': int(solved),
                'accepted': int(len(accepted)),
            })
            if not accepted:
                continue

            objective, candidate_index, sol = min(accepted, key=lambda x: x[0])
            cand = sol.candidate
            exec_node = int(cfg.liftoff_node)
            exec_fraction = exec_node / float(cfg.n_nodes - 1)
            angle_after = float(angle_deg + h * exec_fraction)
            new_support = tuple(sorted(
                (set(support) | {int(cand.touchdown_leg)}) - {int(cand.liftoff_leg)}
            ))
            new_anchors = {
                int(k): np.asarray(v, float).copy()
                for k, v in anchors.items()
                if int(k) != int(cand.liftoff_leg)
            }
            new_anchors[int(cand.touchdown_leg)] = np.array([
                sol.touchdown_xy[0], sol.touchdown_xy[1], 0.0
            ])
            return {
                'success': True,
                'seed': seed,
                'candidate_index': int(candidate_index),
                'candidate': cand,
                'solution': sol,
                'objective': float(objective),
                'horizon_deg': float(h),
                'exec_node': exec_node,
                'exec_fraction': float(exec_fraction),
                'angle_after_deg': angle_after,
                'q_after': np.asarray(sol.q[exec_node], float).copy(),
                'support_after': new_support,
                'anchors_after': new_anchors,
                'checker': sol.checker,
                'trials': trials,
            }
        return None
