"""Static-recovery candidate generation using PRM reachability.

This restores the missing v0.0.6 search link: use the archived v0.0.4 ground
contact candidate generator (seed=123), but judge touchdown reachability with
the same finite-thickness PRM used to execute static recovery. Static plans are
now ordered first by remaining support range of the liftoff leg, so a supporting
leg that is close to losing its fixed-anchor IK is tried before a leg with large
remaining range. Existing PRM feasibility and predicted-gain checks remain.
"""

import numpy as np

from .analytic_ik import analytic_leg_ik_world
from .candidate_v004 import generate_contact_candidates_v004
from .recovery_policy import RecoveryKind


class StaticPRMCandidateMixin:
    """Override candidate generation only for STATIC_RECONFIGURATION."""

    _v006_static_candidate_seed = 123

    def _plans_for_stage(self, angle_deg, q, support, anchors, stage):
        if stage.kind != RecoveryKind.STATIC_RECONFIGURATION:
            return super()._plans_for_stage(angle_deg, q, support, anchors, stage)

        t, R = self._pose(angle_deg)
        st = self._v004_state(angle_deg, q, support, anchors)
        cfg = self._v004_settings()
        raw = generate_contact_candidates_v004(
            self.kin, st, cfg, seed=self._v006_static_candidate_seed
        )
        liftoff_remaining = self._liftoff_remaining_ranges(
            angle_deg, q, support, anchors
        )

        plans = []
        path_cache = {}
        for cand in raw:
            td = int(cand.touchdown_leg)
            lo = int(cand.liftoff_leg)
            if td in support or lo not in support:
                continue

            new_support = tuple(sorted((set(support) | {td}) - {lo}))
            if len(new_support) < self.cfg.min_support_count:
                continue

            target = np.array([
                cand.touchdown_seed_xy[0], cand.touchdown_seed_xy[1], 0.0
            ], dtype=float)
            key = (td, round(float(target[0]), 10), round(float(target[1]), 10))

            goal = None
            if key in path_cache:
                goal = path_cache[key]
            else:
                branches = analytic_leg_ik_world(
                    self.kin, t, R, td, target,
                    q_reference=np.asarray(q[td], float), residual_tol=2e-6,
                )
                for q_goal in branches:
                    q_test = np.asarray(q, float).copy()
                    q_test[td] = q_goal
                    if not self._prm_static_leg_feasible(td, t, R, q_goal, q_test):
                        continue
                    path = self._prm_static_path(
                        td, t, R, np.asarray(q[td], float), q_goal,
                        np.asarray(q, float), seed=991
                    )
                    if path is not None:
                        goal = q_goal.copy()
                        break
                path_cache[key] = goal

            if goal is None:
                continue

            new_anchors = {
                int(k): np.asarray(v, float).copy()
                for k, v in anchors.items() if int(k) != lo
            }
            new_anchors[td] = target.copy()
            q_try = np.asarray(q, float).copy()
            q_try[td] = goal
            q_support = self._support_only(
                angle_deg, q_try, new_support, new_anchors
            )
            if q_support is None:
                continue

            gain = self._predict_gain(
                q_support, new_support, new_anchors, angle_deg,
                self.cfg.expanded_lookahead_deg,
            )
            if gain <= 0.0:
                continue

            add = {td: (target.copy(), goal.copy())}
            rem = [lo]
            score = gain - 0.12 * (len(add) + len(rem)) + 0.03 * len(new_support)
            plans.append((score, gain, add, rem, new_support, new_anchors, q_support))

        self._log(
            "V006 liftoff remaining", float(angle_deg), {
                int(k): float(v) for k, v in sorted(liftoff_remaining.items())
            },
        )

        # Urgent liftoff legs come first. Existing score/gain rank plans that
        # remove equally urgent legs. Geometrically distinct touchdown points
        # remain distinct so a PRM-reachable target is not collapsed early.
        plans.sort(key=lambda x: (
            self._liftoff_priority_key(x[3], liftoff_remaining),
            -float(x[0]),
            -float(x[1]),
        ))
        out = []
        seen = set()
        for plan in plans:
            _, _, add, rem, new_support, _, _ = plan
            td = next(iter(add))
            xy = add[td][0][:2]
            sig = (
                tuple(sorted(add)), tuple(rem), new_support,
                tuple(np.round(np.asarray(xy) / 1e-5).astype(int).tolist()),
            )
            if sig in seen:
                continue
            seen.add(sig)
            out.append(plan)
            if len(out) >= 12:
                break
        return out


__all__ = ["StaticPRMCandidateMixin"]
