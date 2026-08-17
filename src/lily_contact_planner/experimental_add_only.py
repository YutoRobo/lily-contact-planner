"""Opt-in experiment: allow one touchdown with no liftoff after V006 fails.

The production planner remains unchanged.  This module patches only
``V006StagedSearchMixin._try_v006_static`` for the current Python process:

1. run the original V006 static recovery unchanged;
2. only if every V006 branch fails, and the robot currently has exactly the
   configured minimum number of supports, try a v0.0.5-style ``(1, 0)`` event;
3. execute only through the touchdown node, then immediately recurse/replan.

This is intentionally narrower than a general support-set planner.  Its purpose
is to test the hypothesis that separating "gain a support" from "release an old
support" can recover states where the coupled contact switch has no solution.
"""

import numpy as np

from .analytic_ik import analytic_leg_ik_world
from .checker import _point_in_support_hull
from .multi_contact_v005 import (
    MultiContactCandidateV005,
    MultiContactNLPV005,
    _support_area,
    dense_check_multi,
    event_nodes,
    touchdown_seed_map,
)
from .recovery_policy import RecoveryKind
from .v004_success_seed import _CandidateSolveTimeout, _candidate_wall_clock_timeout


def _try_add_only_after_v006(self, angle, q_work, support, anchors, path, depth):
    """Try one touchdown and no liftoff, then replan from the touchdown state."""
    min_support = int(getattr(self.cfg, "min_support_count", 3))
    if len(support) != min_support:
        self._log(
            "ADD-ONLY skip", float(angle), "support", tuple(support),
            "reason", "support_count_not_minimum",
        )
        return None

    cfg = self.v005_multi_settings
    st = self._v005_state(angle, q_work, support, anchors)
    seed = int(self.v005_multi_seed)
    cmap = touchdown_seed_map(self.kin, st, cfg, seed)
    if not cmap:
        self._log(
            "ADD-ONLY failed", float(angle), "support", tuple(support),
            "reason", "no_touchdown_candidates",
        )
        return None

    phase_end = float(self._v004_phase_end(angle))
    maxh = int(np.floor(min(5.0, phase_end - float(angle)) + 1e-9))
    if maxh < 1:
        return None

    stats = getattr(self, "_search_stats", None)
    if stats is not None:
        stats.setdefault("add_only_entries", 0)
        stats.setdefault("add_only_nlp_attempted", 0)
        stats.setdefault("add_only_timeouts", 0)
        stats.setdefault("add_only_success", 0)
        stats["add_only_entries"] += 1

    self._log(
        "ADD-ONLY start", "angle", float(angle), "support", tuple(support),
        "horizons", tuple(range(maxh, 0, -1)), "seed", seed,
        "touchdown_legs", tuple(sorted(int(x) for x in cmap)),
    )

    trials = []
    for h in range(maxh, 0, -1):
        target_t, target_R = self._pose(float(angle) + float(h))
        touchdown_nodes, liftoff_nodes = event_nodes(1, 0)
        terminal_valid = []
        generated = 0
        hull_ok = 0

        # Rank-1 touchdown seed only, matching PRIMARY's bounded multi-start.
        for leg in sorted(cmap):
            if not cmap[leg]:
                continue
            generated += 1
            xy = np.asarray([cmap[leg][0][1]], dtype=float)
            cand = MultiContactCandidateV005(
                touchdown_legs=(int(leg),),
                touchdown_seed_xy=xy.copy(),
                touchdown_nodes=tuple(touchdown_nodes),
                liftoff_legs=(),
                liftoff_nodes=tuple(liftoff_nodes),
            )
            new_support = tuple(sorted(set(support) | {int(leg)}))
            new_anchors = {
                int(k): np.asarray(v, dtype=float).copy()
                for k, v in anchors.items()
            }
            new_anchors[int(leg)] = np.r_[xy[0], 0.0]

            inside, _ = _point_in_support_hull(
                np.asarray(target_t, dtype=float)[:2],
                np.asarray([new_anchors[l][:2] for l in new_support], dtype=float),
                tol=1e-8,
            )
            if not inside:
                continue
            hull_ok += 1

            if not all(
                analytic_leg_ik_world(
                    self.kin,
                    target_t,
                    target_R,
                    l,
                    new_anchors[l],
                    q_reference=np.asarray(q_work[l], dtype=float),
                    residual_tol=2e-6,
                )
                for l in new_support
            ):
                continue

            terminal_valid.append(
                (
                    _support_area([new_anchors[l][:2] for l in new_support]),
                    cand,
                    new_support,
                    new_anchors,
                )
            )

        terminal_valid.sort(key=lambda item: -float(item[0]))
        self._log(
            "ADD-ONLY candidates", "angle", float(angle), "horizon", int(h),
            "generated", int(generated), "hull_ok", int(hull_ok),
            "terminal_ik_ok", int(len(terminal_valid)),
        )

        attempted = 0
        solved_count = 0
        timed_out = 0
        accepted = None
        for candidate_index, (_, cand, new_support, _) in enumerate(terminal_valid):
            attempted += 1
            if stats is not None:
                stats["add_only_nlp_attempted"] += 1
                stats["v005_nlp_attempted"] += 1
            try:
                with _candidate_wall_clock_timeout(60.0):
                    sol = MultiContactNLPV005(
                        self.kin, st, cand, target_t, target_R, cfg
                    ).solve()
            except _CandidateSolveTimeout:
                timed_out += 1
                if stats is not None:
                    stats["add_only_timeouts"] += 1
                    stats["v005_timeouts"] += 1
                self._log(
                    "ADD-ONLY candidate timeout", "angle", float(angle),
                    "horizon", int(h), "candidate", int(candidate_index),
                    "limit_s", 60.0,
                )
                continue

            if not sol["success"]:
                continue
            solved_count += 1
            checker = dense_check_multi(self.kin, st, sol, cand, cfg)
            if not checker["feasible"]:
                continue

            optimized_anchors = {
                int(k): np.asarray(v, dtype=float).copy()
                for k, v in anchors.items()
            }
            td_leg = int(cand.touchdown_legs[0])
            optimized_anchors[td_leg] = np.r_[sol["touchdown_xy"][0], 0.0]
            accepted = (
                int(candidate_index), cand, sol, checker,
                tuple(new_support), optimized_anchors,
            )
            break

        trials.append(
            {
                "search_phase": "add_only_after_v006",
                "horizon_deg": int(h),
                "pattern": [1, 0],
                "generated": int(generated),
                "terminal_hull_ok": int(hull_ok),
                "terminal_ik_ok": int(len(terminal_valid)),
                "attempted": int(attempted),
                "solved": int(solved_count),
                "timed_out": int(timed_out),
                "accepted": int(accepted is not None),
                "candidate_timeout_s": 60.0,
            }
        )
        self._log(
            "ADD-ONLY result", "angle", float(angle), "horizon", int(h),
            "nlp_attempted", int(attempted), "nlp_solved", int(solved_count),
            "timed_out", int(timed_out), "accepted", int(accepted is not None),
        )

        if accepted is None:
            continue

        candidate_index, cand, sol, checker, new_support, new_anchors = accepted
        exec_node = max(int(x) for x in cand.touchdown_nodes)
        exec_fraction = exec_node / float(cfg.n_nodes - 1)
        angle_after = float(angle) + float(h) * exec_fraction
        q_after = np.asarray(sol["q"][exec_node], dtype=float).copy()

        transition_result = {
            "success": True,
            "seed": seed,
            "seed_rank": 1,
            "horizon_deg": float(h),
            "exec_node": int(exec_node),
            "exec_fraction": float(exec_fraction),
            "angle_after_deg": float(angle_after),
            "q_after": q_after,
            "support_after": tuple(new_support),
            "anchors_after": new_anchors,
            "candidate_index": int(candidate_index),
            "candidate": cand,
            "solution": sol,
            "checker": checker,
            "trials": list(trials),
            "objective": float(sol["objective"]),
            "search_phase": "add_only_after_v006",
        }

        # Preserve checkpoint/GIF capture when the trajectory mixin is present.
        if hasattr(self, "_v005_dense_frames") and hasattr(self, "_set_pending_transition"):
            frames = self._v005_dense_frames(
                angle, q_work, support, anchors, transition_result
            )
            self._set_pending_transition(
                frames, angle_after, q_after, new_support
            )

        event = {
            "angle_deg": float(angle),
            "version": "v0.0.5-add-only-experiment",
            "recovery_kind": RecoveryKind.MULTI_CONTACT.value,
            "search_phase": "add_only_after_v006",
            "body_progress_during_reconfiguration_deg": float(angle_after - angle),
            "contact_horizon_deg": float(h),
            "exec_node": int(exec_node),
            "exec_fraction": float(exec_fraction),
            "seed": seed,
            "seed_rank": 1,
            "candidate_index": int(candidate_index),
            "touchdown_legs": [int(x) for x in cand.touchdown_legs],
            "touchdown_nodes": [int(x) for x in cand.touchdown_nodes],
            "liftoff_legs": [],
            "liftoff_nodes": [],
            "add": [int(x) for x in cand.touchdown_legs],
            "remove": [],
            "support_before": [int(x) for x in support],
            "support_after": [int(x) for x in new_support],
            "anchors_added": {
                str(leg): new_anchors[int(leg)].tolist()
                for leg in cand.touchdown_legs
            },
            "touchdown_xy": np.asarray(sol["touchdown_xy"], dtype=float).tolist(),
            "objective": float(sol["objective"]),
            "checker": checker,
            "trials": list(trials),
            "touchdown_before_liftoff": True,
            "add_only": True,
        }
        if stats is not None:
            stats["add_only_success"] += 1
        self._log(
            "ADD-ONLY accepted", "angle", float(angle), "horizon", int(h),
            "candidate", int(candidate_index), "add", cand.touchdown_legs,
            "->", float(angle_after), tuple(new_support),
        )
        return self._dfs(
            angle_after,
            q_after,
            tuple(new_support),
            {k: v.copy() for k, v in new_anchors.items()},
            path + [event],
            depth + 1,
        )

    self._log(
        "ADD-ONLY failed", "angle", float(angle), "support", tuple(support),
        "horizons", tuple(range(maxh, 0, -1)),
    )
    return None


def enable_add_only_after_v006_experiment():
    """Enable the add-only fallback for the current Python process only."""
    from . import staged_search_v006

    cls = staged_search_v006.V006StagedSearchMixin
    if getattr(cls, "_add_only_after_v006_enabled", False):
        return {
            "enabled": True,
            "mode": "add_only_after_v006",
            "already_enabled": True,
        }

    original = cls._try_v006_static

    def patched_try_v006_static(self, angle, q_work, support, anchors, path, depth):
        result = original(self, angle, q_work, support, anchors, path, depth)
        if result is not None:
            return result
        return _try_add_only_after_v006(
            self, angle, q_work, support, anchors, path, depth
        )

    cls._try_v006_static = patched_try_v006_static
    cls._add_only_after_v006_enabled = True
    cls._add_only_after_v006_original = original
    return {
        "enabled": True,
        "mode": "add_only_after_v006",
        "activation": "only_after_all_original_v006_branches_fail",
        "support_count": "exactly min_support_count",
        "pattern": [1, 0],
        "seed_rank": 1,
        "candidate_timeout_s": 60.0,
    }


__all__ = [
    "enable_add_only_after_v006_experiment",
    "_try_add_only_after_v006",
]
