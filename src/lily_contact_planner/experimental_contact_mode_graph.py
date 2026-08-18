"""Opt-in experiment: generic contact-mode graph fallback.

This module deliberately avoids angle-specific or leg-specific rules.  A contact
mode is the support-leg set ``S`` and fallback transitions are generated from

    S_next = (S union A) \ R

subject to configurable cardinality bounds.  The current experiment uses two
primitive families after the original v0.0.6 static recovery has failed:

* acquire support: ``|A| = 1, |R| = 0`` while ``|S| < max_support_count``;
* release support: ``|A| = 0, 1 <= |R| <= max_release_per_transition`` while
  ``|S \ R| >= min_support_count``.

No leg index, task-progress angle, or special support pattern is hard-coded.
Candidate ordering uses only geometry/kinematics-derived quantities already
available to the planner: remaining fixed-anchor support range, predicted
future progress, and support-polygon area.

The production planner remains unchanged.  This is an opt-in research branch
used to evaluate whether explicit contact-mode search is a better abstraction
than fixed (n_add, n_remove) patterns.
"""

from dataclasses import dataclass
import itertools

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


@dataclass(frozen=True)
class ContactModeGraphSettings:
    """Search-complexity limits, not task- or leg-specific heuristics."""

    max_support_count: int = 5
    max_release_per_transition: int = 2
    add_horizon_max_deg: int = 5
    touchdown_seed_rank: int = 1
    candidate_timeout_s: float = 60.0


def _copy_anchors(anchors):
    return {
        int(k): np.asarray(v, dtype=float).copy()
        for k, v in anchors.items()
    }


def _try_acquire_support(self, angle, q_work, support, anchors, path, depth, settings):
    """Explore graph edges with one touchdown and no liftoff."""
    if len(support) >= int(settings.max_support_count):
        self._log(
            "CONTACT-GRAPH add skip", float(angle), "support", tuple(support),
            "reason", "support_count_at_max",
        )
        return None

    cfg = self.v005_multi_settings
    st = self._v005_state(angle, q_work, support, anchors)
    seed = int(self.v005_multi_seed)
    cmap = touchdown_seed_map(self.kin, st, cfg, seed)
    if not cmap:
        self._log(
            "CONTACT-GRAPH add failed", float(angle), "support", tuple(support),
            "reason", "no_touchdown_candidates",
        )
        return None

    phase_end = float(self._v004_phase_end(angle))
    maxh = int(np.floor(min(
        float(settings.add_horizon_max_deg), phase_end - float(angle)
    ) + 1e-9))
    if maxh < 1:
        return None

    stats = getattr(self, "_search_stats", None)
    if stats is not None:
        stats.setdefault("contact_graph_add_entries", 0)
        stats.setdefault("contact_graph_add_nlp_attempted", 0)
        stats.setdefault("contact_graph_add_timeouts", 0)
        stats.setdefault("contact_graph_add_success", 0)
        stats["contact_graph_add_entries"] += 1

    self._log(
        "CONTACT-GRAPH add start", "angle", float(angle),
        "support", tuple(support),
        "max_support", int(settings.max_support_count),
        "horizons", tuple(range(maxh, 0, -1)),
        "touchdown_legs", tuple(sorted(int(x) for x in cmap)),
    )

    rank_index = max(0, int(settings.touchdown_seed_rank) - 1)
    trials = []

    for h in range(maxh, 0, -1):
        target_t, target_R = self._pose(float(angle) + float(h))
        touchdown_nodes, liftoff_nodes = event_nodes(1, 0)
        terminal_valid = []
        generated = 0
        hull_ok = 0

        for leg in sorted(cmap):
            if len(cmap[leg]) <= rank_index:
                continue
            generated += 1
            xy = np.asarray([cmap[leg][rank_index][1]], dtype=float)
            cand = MultiContactCandidateV005(
                touchdown_legs=(int(leg),),
                touchdown_seed_xy=xy.copy(),
                touchdown_nodes=tuple(touchdown_nodes),
                liftoff_legs=(),
                liftoff_nodes=tuple(liftoff_nodes),
            )
            new_support = tuple(sorted(set(int(x) for x in support) | {int(leg)}))
            if len(new_support) > int(settings.max_support_count):
                continue

            new_anchors = _copy_anchors(anchors)
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

            terminal_valid.append((
                -float(_support_area([new_anchors[l][:2] for l in new_support])),
                int(leg),
                cand,
                new_support,
            ))

        terminal_valid.sort(key=lambda item: (item[0], item[1]))
        self._log(
            "CONTACT-GRAPH add candidates", "angle", float(angle),
            "horizon", int(h), "generated", int(generated),
            "hull_ok", int(hull_ok), "terminal_ik_ok", int(len(terminal_valid)),
        )

        attempted = 0
        solved_count = 0
        timed_out = 0
        accepted_count = 0

        for candidate_index, (_, _, cand, new_support) in enumerate(terminal_valid):
            attempted += 1
            if stats is not None:
                stats["contact_graph_add_nlp_attempted"] += 1
                stats["v005_nlp_attempted"] += 1
            try:
                with _candidate_wall_clock_timeout(float(settings.candidate_timeout_s)):
                    sol = MultiContactNLPV005(
                        self.kin, st, cand, target_t, target_R, cfg
                    ).solve()
            except _CandidateSolveTimeout:
                timed_out += 1
                if stats is not None:
                    stats["contact_graph_add_timeouts"] += 1
                    stats["v005_timeouts"] += 1
                self._log(
                    "CONTACT-GRAPH add timeout", "angle", float(angle),
                    "horizon", int(h), "candidate", int(candidate_index),
                    "limit_s", float(settings.candidate_timeout_s),
                )
                continue

            if not sol["success"]:
                continue
            solved_count += 1
            checker = dense_check_multi(self.kin, st, sol, cand, cfg)
            if not checker["feasible"]:
                continue
            accepted_count += 1

            optimized_anchors = _copy_anchors(anchors)
            td_leg = int(cand.touchdown_legs[0])
            optimized_anchors[td_leg] = np.r_[sol["touchdown_xy"][0], 0.0]

            exec_node = max(int(x) for x in cand.touchdown_nodes)
            exec_fraction = exec_node / float(cfg.n_nodes - 1)
            angle_after = float(angle) + float(h) * exec_fraction
            q_after = np.asarray(sol["q"][exec_node], dtype=float).copy()

            transition_result = {
                "success": True,
                "seed": seed,
                "seed_rank": int(settings.touchdown_seed_rank),
                "horizon_deg": float(h),
                "exec_node": int(exec_node),
                "exec_fraction": float(exec_fraction),
                "angle_after_deg": float(angle_after),
                "q_after": q_after,
                "support_after": tuple(new_support),
                "anchors_after": optimized_anchors,
                "candidate_index": int(candidate_index),
                "candidate": cand,
                "solution": sol,
                "checker": checker,
                "objective": float(sol["objective"]),
                "search_phase": "contact_mode_graph_add",
            }
            if hasattr(self, "_v005_dense_frames") and hasattr(self, "_set_pending_transition"):
                frames = self._v005_dense_frames(
                    angle, q_work, support, anchors, transition_result
                )
                self._set_pending_transition(
                    frames, angle_after, q_after, new_support
                )

            event = {
                "angle_deg": float(angle),
                "version": "contact-mode-graph-experiment",
                "recovery_kind": RecoveryKind.MULTI_CONTACT.value,
                "search_phase": "contact_mode_graph_add",
                "body_progress_during_reconfiguration_deg": float(angle_after - angle),
                "contact_horizon_deg": float(h),
                "exec_node": int(exec_node),
                "exec_fraction": float(exec_fraction),
                "add": [td_leg],
                "remove": [],
                "support_before": [int(x) for x in support],
                "support_after": [int(x) for x in new_support],
                "anchors_added": {str(td_leg): optimized_anchors[td_leg].tolist()},
                "objective": float(sol["objective"]),
                "checker": checker,
                "contact_mode_transition": True,
            }
            if stats is not None:
                stats["contact_graph_add_success"] += 1
            self._log(
                "CONTACT-GRAPH add accepted", "angle", float(angle),
                "horizon", int(h), "candidate", int(candidate_index),
                "add", (td_leg,), "->", float(angle_after), tuple(new_support),
            )

            result = self._dfs(
                angle_after,
                q_after,
                tuple(new_support),
                _copy_anchors(optimized_anchors),
                path + [event],
                depth + 1,
            )
            if result is not None:
                return result

        trials.append((int(h), attempted, solved_count, timed_out, accepted_count))
        self._log(
            "CONTACT-GRAPH add result", "angle", float(angle),
            "horizon", int(h), "nlp_attempted", int(attempted),
            "nlp_solved", int(solved_count), "timed_out", int(timed_out),
            "accepted", int(accepted_count),
        )

    self._log(
        "CONTACT-GRAPH add failed", "angle", float(angle),
        "support", tuple(support), "trials", trials,
    )
    return None


def _removal_priority(rem, remaining, gain, area):
    vals = sorted(float(remaining.get(int(leg), float("inf"))) for leg in rem)
    worst = max(vals) if vals else float("inf")
    mean = float(np.mean(vals)) if vals else float("inf")
    return (worst, mean, len(rem), -float(gain), -float(area), tuple(int(x) for x in rem))


def _try_release_support(self, angle, q_work, support, anchors, path, depth, settings):
    """Explore graph edges that release one or more surplus supports."""
    min_support = int(getattr(self.cfg, "min_support_count", 3))
    surplus = len(support) - min_support
    max_release = min(int(settings.max_release_per_transition), surplus)
    if max_release < 1:
        self._log(
            "CONTACT-GRAPH release skip", float(angle), "support", tuple(support),
            "reason", "no_surplus_support",
        )
        return None

    stats = getattr(self, "_search_stats", None)
    if stats is not None:
        stats.setdefault("contact_graph_release_entries", 0)
        stats.setdefault("contact_graph_release_candidates", 0)
        stats.setdefault("contact_graph_release_execution_failed", 0)
        stats.setdefault("contact_graph_release_success", 0)
        stats["contact_graph_release_entries"] += 1

    remaining = self._liftoff_remaining_ranges(angle, q_work, support, anchors)
    horizon = float(getattr(self.cfg, "expanded_lookahead_deg", 42.0))
    plans = []

    for nrem in range(1, max_release + 1):
        for rem0 in itertools.combinations(tuple(int(x) for x in support), nrem):
            rem = tuple(sorted(int(x) for x in rem0))
            new_support = tuple(int(x) for x in support if int(x) not in set(rem))
            if len(new_support) < min_support:
                continue
            new_anchors = {
                int(k): np.asarray(v, dtype=float).copy()
                for k, v in anchors.items()
                if int(k) not in set(rem)
            }

            q_support = self._support_only(angle, q_work, new_support, new_anchors)
            if q_support is None:
                continue
            gain = float(self._predict_gain(
                q_support, new_support, new_anchors, angle, horizon
            ))
            if gain <= 0.0:
                continue
            area = float(_support_area([new_anchors[x][:2] for x in new_support]))
            key = _removal_priority(rem, remaining, gain, area)
            plans.append((key, rem, gain, area, new_support, new_anchors))

    plans.sort(key=lambda item: item[0])
    if stats is not None:
        stats["contact_graph_release_candidates"] += int(len(plans))

    self._log(
        "CONTACT-GRAPH release start", "angle", float(angle),
        "support", tuple(support),
        "liftoff_remaining_deg", {
            int(k): float(v) for k, v in sorted(remaining.items())
        },
        "max_release", int(max_release), "candidates", int(len(plans)),
    )

    for candidate_index, (_, rem, gain, area, new_support, new_anchors) in enumerate(
        plans[: self.cfg.branch_width]
    ):
        executed = self._execute_reconfiguration(
            angle,
            q_work,
            support,
            anchors,
            {},
            list(rem),
            new_support,
            new_anchors,
            RecoveryKind.STATIC_RECONFIGURATION,
        )
        if executed is None:
            if stats is not None:
                stats["contact_graph_release_execution_failed"] += 1
            self._log(
                "CONTACT-GRAPH release execution failed", "angle", float(angle),
                "candidate", int(candidate_index), "remove", rem,
            )
            continue

        q_after, trace = executed
        event = {
            "angle_deg": float(angle),
            "version": "contact-mode-graph-experiment",
            "recovery_kind": RecoveryKind.STATIC_RECONFIGURATION.value,
            "search_phase": "contact_mode_graph_release",
            "body_progress_during_reconfiguration_deg": 0.0,
            "add": [],
            "remove": [int(x) for x in rem],
            "support_before": [int(x) for x in support],
            "support_after": [int(x) for x in new_support],
            "predicted_gain_deg": float(gain),
            "support_area_m2": float(area),
            "remaining_support_range_deg": {
                str(int(x)): float(remaining.get(int(x), float("inf")))
                for x in rem
            },
            "transition_frames": int(len(trace)),
            "liftoff_vertical_m": 0.05,
            "contact_mode_transition": True,
        }
        if stats is not None:
            stats["contact_graph_release_success"] += 1
        self._log(
            "CONTACT-GRAPH release accepted", "angle", float(angle),
            "candidate", int(candidate_index), "remove", rem,
            "gain", float(gain), "->", tuple(new_support),
        )

        result = self._dfs(
            angle,
            q_after,
            tuple(new_support),
            _copy_anchors(new_anchors),
            path + [event],
            depth + 1,
        )
        if result is not None:
            return result

    self._log(
        "CONTACT-GRAPH release failed", "angle", float(angle),
        "support", tuple(support), "candidates", int(len(plans)),
    )
    return None


def _try_contact_mode_graph(self, angle, q_work, support, anchors, path, depth, settings):
    """Explore neighboring contact modes without task/leg-specific cases.

    Acquisition is tried before release so the search may deliberately build
    temporary support redundancy before asking constrained supports to lift.
    Existing DFS memoization/backtracking prevents a successful local edge from
    being mistaken for a globally successful path.
    """
    result = _try_acquire_support(
        self, angle, q_work, support, anchors, path, depth, settings
    )
    if result is not None:
        return result
    return _try_release_support(
        self, angle, q_work, support, anchors, path, depth, settings
    )


def enable_contact_mode_graph_experiment(settings=None):
    """Patch only the current process; the production/main planner is untouched."""
    from . import staged_search_v006

    settings = settings or ContactModeGraphSettings()
    if int(settings.max_support_count) < 1:
        raise ValueError("max_support_count must be positive")
    if int(settings.max_release_per_transition) < 1:
        raise ValueError("max_release_per_transition must be positive")

    cls = staged_search_v006.V006StagedSearchMixin
    if getattr(cls, "_contact_mode_graph_enabled", False):
        cls._contact_mode_graph_settings = settings
        return {
            "enabled": True,
            "mode": "contact_mode_graph",
            "already_enabled": True,
            "settings": settings.__dict__,
        }

    original = cls._try_v006_static

    def patched_try_v006_static(self, angle, q_work, support, anchors, path, depth):
        result = original(self, angle, q_work, support, anchors, path, depth)
        if result is not None:
            return result
        cfg = getattr(self.__class__, "_contact_mode_graph_settings", settings)
        return _try_contact_mode_graph(
            self, angle, q_work, support, anchors, path, depth, cfg
        )

    cls._try_v006_static = patched_try_v006_static
    cls._contact_mode_graph_enabled = True
    cls._contact_mode_graph_original = original
    cls._contact_mode_graph_settings = settings
    return {
        "enabled": True,
        "mode": "contact_mode_graph",
        "state": "support_set_S",
        "transition": "S_next=(S_union_A)\\R",
        "min_support_count": "planner setting",
        "settings": settings.__dict__,
        "leg_specific_rules": False,
        "angle_specific_rules": False,
    }


__all__ = [
    "ContactModeGraphSettings",
    "enable_contact_mode_graph_experiment",
]
