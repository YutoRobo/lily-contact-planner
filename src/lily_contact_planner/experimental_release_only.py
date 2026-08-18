"""Opt-in experiment: release one surplus support after existing recovery fails.

This module is intentionally layered on top of ``experimental_add_only``.
The production planner and the add-only A/B runner remain unchanged.

Activation order when both experiments are enabled:

1. original V006 static recovery;
2. add-only ``(1, 0)`` when support count is exactly the configured minimum;
3. release-only ``(0, 1)`` when support count is above the configured minimum.

Release candidates are ordered by remaining fixed-anchor support range, so the
most urgent support is tried first.  A candidate is retained only if the body
is supportable by the remaining contacts and those contacts predict positive
future task progress.  Execution reuses the existing V006 fixed-body 50 mm
vertical liftoff primitive and the existing final-state checks.
"""

import numpy as np

from .multi_contact_v005 import _support_area
from .recovery_policy import RecoveryKind


def _try_release_only_after_existing_recovery(
    self, angle, q_work, support, anchors, path, depth
):
    """Try one liftoff and no touchdown, then immediately replan."""
    min_support = int(getattr(self.cfg, "min_support_count", 3))
    if len(support) <= min_support:
        self._log(
            "RELEASE-ONLY skip", float(angle), "support", tuple(support),
            "reason", "no_surplus_support",
        )
        return None

    stats = getattr(self, "_search_stats", None)
    if stats is not None:
        stats.setdefault("release_only_entries", 0)
        stats.setdefault("release_only_candidates", 0)
        stats.setdefault("release_only_execution_failed", 0)
        stats.setdefault("release_only_success", 0)
        stats["release_only_entries"] += 1

    remaining = self._liftoff_remaining_ranges(
        angle, q_work, support, anchors
    )
    horizon = float(getattr(self.cfg, "expanded_lookahead_deg", 42.0))
    plans = []

    for leg0 in support:
        leg = int(leg0)
        new_support = tuple(int(x) for x in support if int(x) != leg)
        if len(new_support) < min_support:
            continue

        new_anchors = {
            int(k): np.asarray(v, dtype=float).copy()
            for k, v in anchors.items()
            if int(k) != leg
        }

        # This includes the support-polygon and fixed-anchor IK checks at the
        # current body pose.  No new feasibility semantics are introduced.
        q_support = self._support_only(
            angle, q_work, new_support, new_anchors
        )
        if q_support is None:
            continue

        gain = float(self._predict_gain(
            q_support, new_support, new_anchors, angle, horizon
        ))
        if gain <= 0.0:
            continue

        area = float(_support_area([
            new_anchors[x][:2] for x in new_support
        ]))
        urgency = float(remaining.get(leg, float("inf")))
        plans.append((urgency, -gain, -area, leg, gain, new_support, new_anchors))

    plans.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    if stats is not None:
        stats["release_only_candidates"] += int(len(plans))

    self._log(
        "RELEASE-ONLY start", "angle", float(angle),
        "support", tuple(support),
        "liftoff_remaining_deg", {
            int(k): float(v) for k, v in sorted(remaining.items())
        },
        "candidates", int(len(plans)),
    )

    for candidate_index, plan in enumerate(plans[: self.cfg.branch_width]):
        urgency, _, _, leg, gain, new_support, new_anchors = plan
        add = {}
        rem = [int(leg)]

        executed = self._execute_reconfiguration(
            angle,
            q_work,
            support,
            anchors,
            add,
            rem,
            new_support,
            new_anchors,
            RecoveryKind.STATIC_RECONFIGURATION,
        )
        if executed is None:
            if stats is not None:
                stats["release_only_execution_failed"] += 1
            self._log(
                "RELEASE-ONLY execution failed", "angle", float(angle),
                "candidate", int(candidate_index), "remove", int(leg),
            )
            continue

        q_after, trace = executed
        event = {
            "angle_deg": float(angle),
            "version": "v0.0.6-release-only-experiment",
            "recovery_kind": RecoveryKind.STATIC_RECONFIGURATION.value,
            "search_phase": "release_only_after_existing_recovery",
            "body_progress_during_reconfiguration_deg": 0.0,
            "add": [],
            "remove": [int(leg)],
            "support_before": [int(x) for x in support],
            "support_after": [int(x) for x in new_support],
            "anchors_added": {},
            "qgoal_added": {},
            "predicted_gain_deg": float(gain),
            "remaining_support_range_deg": float(urgency),
            "transition_frames": int(len(trace)),
            "touchdown_before_liftoff": False,
            "liftoff_vertical_m": 0.05,
            "release_only": True,
        }
        if stats is not None:
            stats["release_only_success"] += 1

        self._log(
            "RELEASE-ONLY accepted", "angle", float(angle),
            "candidate", int(candidate_index), "remove", int(leg),
            "remaining", float(urgency), "gain", float(gain),
            "->", tuple(new_support),
        )
        result = self._dfs(
            angle,
            q_after,
            tuple(new_support),
            {k: v.copy() for k, v in new_anchors.items()},
            path + [event],
            depth + 1,
        )
        if result is not None:
            return result

    self._log(
        "RELEASE-ONLY failed", "angle", float(angle),
        "support", tuple(support), "candidates", int(len(plans)),
    )
    return None


def enable_release_only_after_existing_recovery_experiment():
    """Layer release-only fallback on the currently installed V006 method."""
    from . import staged_search_v006

    cls = staged_search_v006.V006StagedSearchMixin
    if getattr(cls, "_release_only_experiment_enabled", False):
        return {
            "enabled": True,
            "mode": "release_only_after_existing_recovery",
            "already_enabled": True,
        }

    original = cls._try_v006_static

    def patched_try_v006_static(self, angle, q_work, support, anchors, path, depth):
        result = original(self, angle, q_work, support, anchors, path, depth)
        if result is not None:
            return result
        return _try_release_only_after_existing_recovery(
            self, angle, q_work, support, anchors, path, depth
        )

    cls._try_v006_static = patched_try_v006_static
    cls._release_only_experiment_enabled = True
    cls._release_only_experiment_original = original
    return {
        "enabled": True,
        "mode": "release_only_after_existing_recovery",
        "activation": "after_original_v006_and_add_only_return_no_solution",
        "support_count": "strictly_above_min_support_count",
        "pattern": [0, 1],
        "ranking": "remaining_support_range_then_predicted_gain_then_area",
        "execution": "existing_v006_static_50mm_liftoff",
    }


__all__ = [
    "enable_release_only_after_existing_recovery_experiment",
    "_try_release_only_after_existing_recovery",
]
