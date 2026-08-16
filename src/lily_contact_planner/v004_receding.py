"""Recovered v0.0.4 receding-horizon progression and one-to-one recovery.

This module preserves the successful v0.0.4 semantics used as the base of the
archived v0.0.6 Pitch45 -> Roll45 run:
- inspect up to 5 deg ahead;
- normally execute about 1 deg and replan;
- only launch 1-to-1 contact-switch NLP when the current contact set cannot
  certify the short horizon;
- search contact horizons from long to short;
- execute an accepted switch only through the liftoff node (node 6/10).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .analytic_ik import analytic_leg_ik_world
from .candidate_v004 import generate_contact_candidates_v004
from .checker_v004 import dense_check_solution_v004, dense_projected_trajectory_v004
from .trajectory_nlp_v004 import NoContactNLPV004, ContactSwitchNLPV004
from .v004_types import TrajectorySolutionV004, V004Settings, V004State


class V004RecedingHorizonMixin:
    """Helpers for the archived v0.0.4 receding-horizon policy."""

    def _v004_settings(self):
        return V004Settings(maxiter=120, ftol=1e-8)

    def _v004_phase_end(self, angle_deg):
        """Do not let one finite horizon cross a task-defined path corner.

        New multi-axis tasks expose their scalar-progress phase boundaries via
        ``task.phase_boundaries_deg``.  Pitch45ThenRoll45Task reports the same
        45-deg corner as the historical hard-coded behavior, so the regression
        task is unchanged.  Tasks without that property retain the old fallback.
        """
        angle = float(angle_deg)
        boundaries = getattr(self.task, "phase_boundaries_deg", None)
        if boundaries is not None:
            for boundary in boundaries:
                boundary = float(boundary)
                if angle < boundary - 1e-9:
                    return min(boundary, self.max_roll_deg)
            return self.max_roll_deg

        # Historical fallback for task classes that predate explicit phase
        # boundaries.  This preserves their existing behavior.
        if angle < 45.0 - 1e-9 and self.max_roll_deg > 45.0:
            return min(45.0, self.max_roll_deg)
        return self.max_roll_deg

    def _v004_state(self, angle_deg, q, support, anchors):
        t, R = self._pose(angle_deg)
        contact = np.zeros(self.kin.n_legs, dtype=bool)
        contact[list(support)] = True
        return V004State(
            np.asarray(t, float).copy(),
            np.asarray(R, float).copy(),
            np.asarray(q, float).copy(),
            contact,
            {int(k): np.asarray(v, float).copy() for k, v in anchors.items()},
        )

    def _v004_target(self, angle_deg, horizon_deg):
        return self._pose(float(angle_deg) + float(horizon_deg))

    def _v004_no_contact(self, angle_deg, q, support, anchors):
        """Certify the short horizon with the archived analytic seed first."""
        cfg = self._v004_settings()
        phase_end = self._v004_phase_end(angle_deg)
        horizon = min(5.0, phase_end - float(angle_deg))
        if horizon <= 1e-9:
            return None
        st = self._v004_state(angle_deg, q, support, anchors)
        target_t, target_R = self._v004_target(angle_deg, horizon)
        nlp = NoContactNLPV004(self.kin, st, target_t, target_R, cfg)
        z0 = nlp.initial_guess()
        eqmax = float(np.max(np.abs(nlp.equality_constraints(z0))))
        inmin = float(np.min(nlp.inequality_constraints(z0)))
        seed_feasible = (
            eqmax <= cfg.constraint_tolerance
            and inmin >= -cfg.constraint_tolerance
        )

        sol = None
        if seed_feasible:
            body_pos, rv, qk, _ = nlp.layout.unpack(z0)
            body_R = nlp._rotations(rv)
            sol = TrajectorySolutionV004(
                'no_contact', True, 'analytic seed feasible',
                float(nlp.objective(z0)), body_pos, body_R, qk,
            )
            sol.checker = dense_check_solution_v004(
                self.kin, st, sol, target_t, target_R, cfg
            )
        else:
            # Archived v0.0.6 behavior: if terminal support IK itself is
            # impossible, do not waste an SLSQP call. Otherwise allow the NLP
            # to repair a non-feasible seed.
            terminal_ik_possible = True
            for leg in support:
                if not analytic_leg_ik_world(
                    self.kin, target_t, target_R, int(leg),
                    np.asarray(anchors[int(leg)], float),
                    q_reference=np.asarray(q[int(leg)], float),
                ):
                    terminal_ik_possible = False
                    break
            if terminal_ik_possible:
                sol = nlp.solve()
                if sol.success:
                    sol.checker = dense_check_solution_v004(
                        self.kin, st, sol, target_t, target_R, cfg
                    )

        if sol is None or not sol.accepted:
            return {
                'accepted': False,
                'seed_feasible': bool(seed_feasible),
                'horizon_deg': float(horizon),
            }

        step = min(1.0, phase_end - float(angle_deg))
        frac = float(step / horizon)
        ds, _, _, qdense, _, _ = dense_projected_trajectory_v004(
            self.kin, st, sol, cfg
        )
        idx = int(np.argmin(np.abs(ds - frac)))
        return {
            'accepted': True,
            'seed_feasible': bool(seed_feasible),
            'horizon_deg': float(horizon),
            'exec_fraction': frac,
            'angle_after_deg': float(angle_deg + step),
            'q_after': np.asarray(qdense[idx], float).copy(),
            'solution': sol,
        }

    def _v004_contact_recovery(self, angle_deg, q, support, anchors):
        """Run the archived one-touchdown/one-liftoff horizon search."""
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
            for candidate_index, cand in enumerate(candidates):
                # Cheap terminal reachability prescreen used by the archived
                # exploration before invoking SLSQP.
                new_support = (set(support) | {int(cand.touchdown_leg)}) - {int(cand.liftoff_leg)}
                td_seed = np.array([
                    cand.touchdown_seed_xy[0],
                    cand.touchdown_seed_xy[1],
                    0.0,
                ])
                terminal_ok = True
                for leg in new_support:
                    anchor = td_seed if leg == int(cand.touchdown_leg) else anchors[int(leg)]
                    if not analytic_leg_ik_world(
                        self.kin, target_t, target_R, int(leg),
                        np.asarray(anchor, float), q_reference=q[int(leg)],
                    ):
                        terminal_ok = False
                        break
                if not terminal_ok:
                    continue

                sol = ContactSwitchNLPV004(
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
                'solved_candidates': int(solved),
                'accepted': int(len(accepted)),
            })
            if not accepted:
                continue

            accepted.sort(key=lambda x: x[0])
            objective, candidate_index, sol = accepted[0]
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
