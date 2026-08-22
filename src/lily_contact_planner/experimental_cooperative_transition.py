"""Opt-in experiment: cooperative contact-mode transition optimization.

A hybrid planner state contains continuous robot state x and support set S.
Neighboring contact modes are generated as

    S_next = (S union A) \\ R,

where A is a touchdown-leg set and R is a liftoff-leg set. Every candidate is
evaluated by the same V005-derived continuous NLP so body pose and all joints
may move cooperatively during both acquisition and release. No leg index or
task-progress angle is hard-coded.
"""

from dataclasses import dataclass
import itertools

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

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
class CooperativeTransitionSettings:
    """Generic search-complexity bounds, not gait-specific rules."""

    max_support_count: int = 5
    max_add_per_transition: int = 1
    max_release_per_transition: int = 2
    max_total_contact_changes: int = 3
    horizon_max_deg: int = 5
    touchdown_seed_rank: int = 1
    settling_nodes: int = 1
    liftoff_clearance_m: float = 0.02
    candidate_timeout_s: float = 60.0
    max_candidates_per_horizon: int = 24


class CooperativeTransitionNLP(MultiContactNLPV005):
    """V005 NLP plus receding-execution consistency and liftoff clearance.

    The existing V005 contact schedule and safety constraints are retained.
    In addition, the body pose at the executed node is fixed to the task
    reference at the corresponding progress, and released feet are required
    to rise smoothly to a configurable clearance by that node.
    """

    def __init__(
        self,
        kin,
        state,
        candidate,
        target_body_pos,
        target_body_R,
        settings,
        exec_node,
        exec_body_pos,
        exec_body_R,
        liftoff_clearance_m,
    ):
        super().__init__(
            kin, state, candidate, target_body_pos, target_body_R, settings
        )
        self.exec_node = int(exec_node)
        self.exec_body_pos = np.asarray(exec_body_pos, dtype=float)
        self.exec_body_R = np.asarray(exec_body_R, dtype=float)
        self.liftoff_clearance_m = float(liftoff_clearance_m)

    def equality_constraints(self, z):
        eq = list(super().equality_constraints(z))
        t, rv, _, _, _ = self.layout.unpack(z)
        R = self._rot(rv)
        eq.extend(t[self.exec_node] - self.exec_body_pos)
        eq.extend(
            Rotation.from_matrix(
                R[self.exec_node].dot(self.exec_body_R.T)
            ).as_rotvec()
        )
        return np.asarray(eq, dtype=float)

    def inequality_constraints(self, z):
        out = list(super().inequality_constraints(z))
        if not self.lolegs:
            return np.asarray(out, dtype=float)

        t, rv, q, _, _ = self.layout.unpack(z)
        R = self._rot(rv)
        for leg, node in zip(self.lolegs, self.lonodes):
            denom = max(1, self.exec_node - int(node))
            for k in range(int(node) + 1, self.settings.n_nodes):
                alpha = min(1.0, (k - int(node)) / float(denom))
                required = self.liftoff_clearance_m * alpha
                foot = self.kin.foot_world(
                    t[k], R[k], int(leg), q[k, int(leg)]
                )
                out.append(float(foot[2] - required))
        return np.asarray(out, dtype=float)


def _copy_anchors(anchors):
    return {
        int(k): np.asarray(v, dtype=float).copy()
        for k, v in anchors.items()
    }


def _event_exec_node(candidate, n_nodes, settling_nodes):
    events = [
        int(x)
        for x in tuple(candidate.touchdown_nodes) + tuple(candidate.liftoff_nodes)
    ]
    if not events:
        raise ValueError("contact transition must contain at least one event")
    return min(
        int(n_nodes) - 1,
        max(events) + max(0, int(settling_nodes)),
    )


def _support_after(support, candidate):
    return tuple(sorted(
        (set(int(x) for x in support) | set(int(x) for x in candidate.touchdown_legs))
        - set(int(x) for x in candidate.liftoff_legs)
    ))


def _candidate_seed_anchors(anchors, candidate):
    out = {
        int(k): np.asarray(v, dtype=float).copy()
        for k, v in anchors.items()
        if int(k) not in set(int(x) for x in candidate.liftoff_legs)
    }
    for leg, xy in zip(candidate.touchdown_legs, candidate.touchdown_seed_xy):
        out[int(leg)] = np.r_[np.asarray(xy, dtype=float), 0.0]
    return out


def _optimized_anchors(anchors, candidate, solution):
    out = {
        int(k): np.asarray(v, dtype=float).copy()
        for k, v in anchors.items()
        if int(k) not in set(int(x) for x in candidate.liftoff_legs)
    }
    for leg, xy in zip(candidate.touchdown_legs, solution["touchdown_xy"]):
        out[int(leg)] = np.r_[np.asarray(xy, dtype=float), 0.0]
    return out


def _candidate_priority(candidate, remaining, area):
    rem = tuple(int(x) for x in candidate.liftoff_legs)
    values = sorted(float(remaining.get(leg, float("inf"))) for leg in rem)
    worst = max(values) if values else float("inf")
    mean = float(np.mean(values)) if values else float("inf")
    nadd = len(candidate.touchdown_legs)
    nrem = len(candidate.liftoff_legs)
    return (
        worst,
        mean,
        nadd + nrem,
        nadd,
        -float(area),
        tuple(int(x) for x in candidate.touchdown_legs),
        rem,
    )


def _dense_liftoff_clearance_margin(
    kin, sol, cand, cfg, exec_node, clearance_m
):
    """Return minimum dense margin above the post-liftoff clearance profile."""
    if not cand.liftoff_legs:
        return float("inf")

    n = int(np.asarray(sol["body_pos"]).shape[0])
    knots = np.linspace(0.0, 1.0, n)
    dense = np.linspace(0.0, 1.0, int(cfg.checker_samples))
    bp = np.column_stack([
        np.interp(dense, knots, np.asarray(sol["body_pos"])[:, j])
        for j in range(3)
    ])
    bR = Slerp(
        knots, Rotation.from_matrix(np.asarray(sol["body_R"], dtype=float))
    )(dense).as_matrix()
    qflat = np.asarray(sol["q"], dtype=float).reshape(n, -1)
    qdense = np.column_stack([
        np.interp(dense, knots, qflat[:, j])
        for j in range(qflat.shape[1])
    ]).reshape(len(dense), kin.n_legs, 3)

    execp = float(exec_node) / float(cfg.n_nodes - 1)
    margin = float("inf")
    for leg, node in zip(cand.liftoff_legs, cand.liftoff_nodes):
        leg = int(leg)
        lop = float(node) / float(cfg.n_nodes - 1)
        denom = max(1e-12, execp - lop)
        for i, p0 in enumerate(dense):
            p = float(p0)
            if p <= lop + 1e-12:
                continue
            alpha = min(1.0, max(0.0, (p - lop) / denom))
            required = float(clearance_m) * alpha
            foot = kin.foot_world(bp[i], bR[i], leg, qdense[i, leg])
            margin = min(margin, float(foot[2] - required))
    return margin


def _enumerate_candidates(planner, angle, q_work, support, anchors, horizon, settings):
    cfg = planner.v005_multi_settings
    state = planner._v005_state(angle, q_work, support, anchors)
    cmap = touchdown_seed_map(
        planner.kin, state, cfg, int(planner.v005_multi_seed)
    )
    remaining = planner._liftoff_remaining_ranges(
        angle, q_work, support, anchors
    )
    min_support = int(getattr(planner.cfg, "min_support_count", 3))
    support_tuple = tuple(sorted(int(x) for x in support))
    non_support = tuple(sorted(int(x) for x in cmap))
    rank_index = max(0, int(settings.touchdown_seed_rank) - 1)

    max_add = min(
        int(settings.max_add_per_transition),
        len(non_support),
        max(0, int(settings.max_support_count) - len(support_tuple)),
    )
    max_rem = min(
        int(settings.max_release_per_transition),
        len(support_tuple),
    )

    records = []
    generated = hull_ok = terminal_ik_ok = 0

    for nadd in range(max_add + 1):
        add_sets = (
            itertools.combinations(non_support, nadd)
            if nadd > 0 else [()]
        )
        for add in add_sets:
            if any(len(cmap[leg]) <= rank_index for leg in add):
                continue
            xy = np.asarray(
                [cmap[leg][rank_index][1] for leg in add],
                dtype=float,
            ).reshape(nadd, 2)

            for nrem in range(max_rem + 1):
                if nadd == 0 and nrem == 0:
                    continue
                if nadd + nrem > int(settings.max_total_contact_changes):
                    continue
                if len(support_tuple) + nadd > int(settings.max_support_count):
                    continue
                final_count = len(support_tuple) + nadd - nrem
                if not (min_support <= final_count <= int(settings.max_support_count)):
                    continue

                for rem in itertools.combinations(support_tuple, nrem):
                    generated += 1
                    td_nodes, lo_nodes = event_nodes(nadd, nrem)
                    cand = MultiContactCandidateV005(
                        touchdown_legs=tuple(int(x) for x in add),
                        touchdown_seed_xy=xy.copy(),
                        touchdown_nodes=tuple(int(x) for x in td_nodes),
                        liftoff_legs=tuple(int(x) for x in rem),
                        liftoff_nodes=tuple(int(x) for x in lo_nodes),
                    )
                    exec_node = _event_exec_node(
                        cand, cfg.n_nodes, settings.settling_nodes
                    )
                    exec_fraction = exec_node / float(cfg.n_nodes - 1)
                    exec_angle = float(angle) + float(horizon) * exec_fraction
                    exec_t, exec_R = planner._pose(exec_angle)

                    new_support = _support_after(support_tuple, cand)
                    seed_anchors = _candidate_seed_anchors(anchors, cand)
                    pts = np.asarray(
                        [seed_anchors[l][:2] for l in new_support],
                        dtype=float,
                    )
                    inside, _ = _point_in_support_hull(
                        np.asarray(exec_t, dtype=float)[:2],
                        pts,
                        tol=1e-8,
                    )
                    if not inside:
                        continue
                    hull_ok += 1

                    if not all(
                        analytic_leg_ik_world(
                            planner.kin,
                            exec_t,
                            exec_R,
                            leg,
                            seed_anchors[leg],
                            q_reference=np.asarray(q_work[leg], dtype=float),
                            residual_tol=2e-6,
                        )
                        for leg in new_support
                    ):
                        continue
                    terminal_ik_ok += 1
                    area = float(_support_area(
                        [seed_anchors[l][:2] for l in new_support]
                    ))
                    records.append({
                        "key": _candidate_priority(cand, remaining, area),
                        "candidate": cand,
                        "exec_node": int(exec_node),
                        "exec_fraction": float(exec_fraction),
                        "exec_angle_deg": float(exec_angle),
                        "exec_body_pos": np.asarray(exec_t, dtype=float),
                        "exec_body_R": np.asarray(exec_R, dtype=float),
                        "support_after": tuple(new_support),
                        "support_area_seed_m2": area,
                    })

    records.sort(key=lambda x: x["key"])
    limit = max(1, int(settings.max_candidates_per_horizon))
    return records[:limit], {
        "generated": int(generated),
        "hull_ok": int(hull_ok),
        "terminal_ik_ok": int(terminal_ik_ok),
        "returned": int(min(len(records), limit)),
    }


def _solve_record(planner, angle, q_work, support, anchors, horizon, record, settings):
    cfg = planner.v005_multi_settings
    state = planner._v005_state(angle, q_work, support, anchors)
    target_t, target_R = planner._pose(float(angle) + float(horizon))
    cand = record["candidate"]

    sol = CooperativeTransitionNLP(
        planner.kin,
        state,
        cand,
        target_t,
        target_R,
        cfg,
        record["exec_node"],
        record["exec_body_pos"],
        record["exec_body_R"],
        settings.liftoff_clearance_m,
    ).solve()
    if not sol["success"]:
        return None, "nlp"

    checker = dense_check_multi(planner.kin, state, sol, cand, cfg)
    clearance_margin = _dense_liftoff_clearance_margin(
        planner.kin,
        sol,
        cand,
        cfg,
        record["exec_node"],
        settings.liftoff_clearance_m,
    )
    checker = dict(checker)
    checker["min_post_liftoff_clearance_margin_m"] = float(clearance_margin)
    if not checker["feasible"] or clearance_margin < -float(cfg.checker_tolerance):
        return None, "checker"

    new_support = _support_after(support, cand)
    new_anchors = _optimized_anchors(anchors, cand, sol)
    q_after = np.asarray(
        sol["q"][record["exec_node"]], dtype=float
    ).copy()
    gain = float(planner._predict_gain(
        q_after,
        new_support,
        new_anchors,
        record["exec_angle_deg"],
        float(getattr(planner.cfg, "expanded_lookahead_deg", 42.0)),
    ))
    if gain <= 0.0:
        return None, "predicted_gain"

    return {
        "success": True,
        "horizon_deg": float(horizon),
        "exec_node": int(record["exec_node"]),
        "exec_fraction": float(record["exec_fraction"]),
        "angle_after_deg": float(record["exec_angle_deg"]),
        "q_after": q_after,
        "support_after": tuple(new_support),
        "anchors_after": new_anchors,
        "candidate": cand,
        "solution": sol,
        "checker": checker,
        "objective": float(sol["objective"]),
        "predicted_gain_deg": gain,
        "search_phase": "cooperative_contact_transition",
    }, None


def solve_cooperative_transition_edges(
    planner,
    angle,
    q_work,
    support,
    anchors,
    settings=None,
    stop_after_first=True,
):
    """Solve generic neighboring edges without entering the global DFS."""
    settings = settings or CooperativeTransitionSettings()
    phase_end = float(planner._v004_phase_end(angle))
    maxh = int(np.floor(min(
        float(settings.horizon_max_deg),
        phase_end - float(angle),
    ) + 1e-9))
    accepted = []
    for h in range(maxh, 0, -1):
        records, _ = _enumerate_candidates(
            planner, angle, q_work, support, anchors, h, settings
        )
        for record in records:
            try:
                with _candidate_wall_clock_timeout(
                    float(settings.candidate_timeout_s)
                ):
                    result, _ = _solve_record(
                        planner, angle, q_work, support, anchors,
                        h, record, settings
                    )
            except _CandidateSolveTimeout:
                continue
            if result is None:
                continue
            accepted.append(result)
            if stop_after_first:
                return accepted
    return accepted


def _try_cooperative_transition(
    planner, angle, q_work, support, anchors, path, depth, settings
):
    phase_end = float(planner._v004_phase_end(angle))
    maxh = int(np.floor(min(
        float(settings.horizon_max_deg),
        phase_end - float(angle),
    ) + 1e-9))
    if maxh < 1:
        return None

    stats = getattr(planner, "_search_stats", None)
    if stats is not None:
        for key in (
            "cooperative_transition_entries",
            "cooperative_transition_candidates",
            "cooperative_transition_nlp_attempted",
            "cooperative_transition_timeouts",
            "cooperative_transition_checker_rejects",
            "cooperative_transition_gain_rejects",
            "cooperative_transition_success",
        ):
            stats.setdefault(key, 0)
        stats["cooperative_transition_entries"] += 1

    remaining = planner._liftoff_remaining_ranges(
        angle, q_work, support, anchors
    )
    planner._log(
        "COOP-TRANSITION start",
        "angle", float(angle),
        "support", tuple(int(x) for x in support),
        "liftoff_remaining_deg", {
            int(k): float(v) for k, v in sorted(remaining.items())
        },
        "horizons", tuple(range(maxh, 0, -1)),
    )

    for h in range(maxh, 0, -1):
        records, counts = _enumerate_candidates(
            planner, angle, q_work, support, anchors, h, settings
        )
        if stats is not None:
            stats["cooperative_transition_candidates"] += len(records)
        planner._log(
            "COOP-TRANSITION candidates",
            "angle", float(angle),
            "horizon", int(h),
            "generated", counts["generated"],
            "hull_ok", counts["hull_ok"],
            "terminal_ik_ok", counts["terminal_ik_ok"],
            "returned", counts["returned"],
        )

        for candidate_index, record in enumerate(records):
            cand = record["candidate"]
            if stats is not None:
                stats["cooperative_transition_nlp_attempted"] += 1
                stats["v005_nlp_attempted"] += 1
            try:
                with _candidate_wall_clock_timeout(
                    float(settings.candidate_timeout_s)
                ):
                    result, reject = _solve_record(
                        planner, angle, q_work, support, anchors,
                        h, record, settings
                    )
            except _CandidateSolveTimeout:
                if stats is not None:
                    stats["cooperative_transition_timeouts"] += 1
                    stats["v005_timeouts"] += 1
                planner._log(
                    "COOP-TRANSITION timeout",
                    "angle", float(angle),
                    "horizon", int(h),
                    "candidate", int(candidate_index),
                    "add", tuple(int(x) for x in cand.touchdown_legs),
                    "remove", tuple(int(x) for x in cand.liftoff_legs),
                )
                continue

            if result is None:
                if stats is not None and reject == "checker":
                    stats["cooperative_transition_checker_rejects"] += 1
                if stats is not None and reject == "predicted_gain":
                    stats["cooperative_transition_gain_rejects"] += 1
                continue

            if stats is not None:
                stats["cooperative_transition_success"] += 1

            new_support = result["support_after"]
            new_anchors = result["anchors_after"]
            q_after = result["q_after"]
            angle_after = result["angle_after_deg"]

            if hasattr(planner, "_v005_dense_frames") and hasattr(
                planner, "_set_pending_transition"
            ):
                frames = planner._v005_dense_frames(
                    angle, q_work, support, anchors, result
                )
                planner._set_pending_transition(
                    frames, angle_after, q_after, new_support
                )

            event = {
                "angle_deg": float(angle),
                "version": "cooperative-contact-transition-experiment",
                "recovery_kind": RecoveryKind.MULTI_CONTACT.value,
                "search_phase": "cooperative_contact_transition",
                "body_progress_during_reconfiguration_deg": float(
                    angle_after - float(angle)
                ),
                "contact_horizon_deg": float(h),
                "exec_node": int(result["exec_node"]),
                "exec_fraction": float(result["exec_fraction"]),
                "add": [int(x) for x in cand.touchdown_legs],
                "remove": [int(x) for x in cand.liftoff_legs],
                "support_before": [int(x) for x in support],
                "support_after": [int(x) for x in new_support],
                "anchors_added": {
                    str(int(leg)): np.asarray(new_anchors[int(leg)]).tolist()
                    for leg in cand.touchdown_legs
                },
                "predicted_gain_deg": float(result["predicted_gain_deg"]),
                "objective": float(result["objective"]),
                "checker": result["checker"],
                "contact_mode_transition": True,
                "cooperative_body_motion": True,
                "liftoff_clearance_m": float(settings.liftoff_clearance_m),
            }
            planner._log(
                "COOP-TRANSITION accepted",
                "angle", float(angle),
                "horizon", int(h),
                "candidate", int(candidate_index),
                "add", tuple(int(x) for x in cand.touchdown_legs),
                "remove", tuple(int(x) for x in cand.liftoff_legs),
                "->", float(angle_after), tuple(new_support),
                "gain", float(result["predicted_gain_deg"]),
            )

            child = planner._dfs(
                angle_after,
                q_after,
                tuple(new_support),
                _copy_anchors(new_anchors),
                path + [event],
                depth + 1,
            )
            if child is not None:
                return child

    planner._log(
        "COOP-TRANSITION failed",
        "angle", float(angle),
        "support", tuple(int(x) for x in support),
    )
    return None


def enable_cooperative_transition_experiment(settings=None):
    """Patch only this process; existing experiment branches remain unchanged."""
    from . import staged_search_v006

    settings = settings or CooperativeTransitionSettings()
    if int(settings.max_support_count) < 1:
        raise ValueError("max_support_count must be positive")
    if int(settings.max_add_per_transition) < 0:
        raise ValueError("max_add_per_transition must be nonnegative")
    if int(settings.max_release_per_transition) < 0:
        raise ValueError("max_release_per_transition must be nonnegative")
    if int(settings.max_total_contact_changes) < 1:
        raise ValueError("max_total_contact_changes must be positive")
    if int(settings.settling_nodes) < 0:
        raise ValueError("settling_nodes must be nonnegative")
    if float(settings.liftoff_clearance_m) < 0.0:
        raise ValueError("liftoff_clearance_m must be nonnegative")

    cls = staged_search_v006.V006StagedSearchMixin
    if getattr(cls, "_cooperative_transition_enabled", False):
        cls._cooperative_transition_settings = settings
        return {
            "enabled": True,
            "mode": "cooperative_contact_transition",
            "already_enabled": True,
            "settings": settings.__dict__,
        }

    original = cls._try_v006_static

    def patched_try_v006_static(
        self, angle, q_work, support, anchors, path, depth
    ):
        result = original(
            self, angle, q_work, support, anchors, path, depth
        )
        if result is not None:
            return result
        cfg = getattr(
            self.__class__, "_cooperative_transition_settings", settings
        )
        return _try_cooperative_transition(
            self, angle, q_work, support, anchors, path, depth, cfg
        )

    cls._try_v006_static = patched_try_v006_static
    cls._cooperative_transition_enabled = True
    cls._cooperative_transition_original = original
    cls._cooperative_transition_settings = settings

    return {
        "enabled": True,
        "mode": "cooperative_contact_transition",
        "state": "hybrid_(x,S)",
        "transition": "S_next=(S_union_A)\\R",
        "trajectory": "joint_body_contact_transition_NLP",
        "leg_specific_rules": False,
        "angle_specific_rules": False,
        "settings": settings.__dict__,
    }


__all__ = [
    "CooperativeTransitionSettings",
    "CooperativeTransitionNLP",
    "solve_cooperative_transition_edges",
    "enable_cooperative_transition_experiment",
]
