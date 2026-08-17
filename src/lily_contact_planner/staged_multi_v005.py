"""Horizon-selectable v0.0.5 multi-contact search.

The discrete touchdown/liftoff leg combinations remain exhaustive. Each leg
combination uses caller-selected ranked touchdown initial guesses instead of the
Cartesian product of all retained seeds. Rank 1 is used by PRIMARY; aligned
ranks 2..5 are available to DEEP FALLBACK. Liftoff combinations are now tried
in order of remaining support range: legs that can hold their current anchors
for fewer future degrees are preferred, with support area as the secondary
ranking. Terminal hull/IK prescreens, NLP formulation, dense Checker, and
pattern order are preserved.
"""

import itertools
import numpy as np

from .analytic_ik import analytic_leg_ik_world
from .checker import _point_in_support_hull
from .multi_contact_v005 import (
    MultiContactCandidateV005,
    MultiContactNLPV005,
    V005MultiRecoveryMixin,
    _support_area,
    dense_check_multi,
    event_nodes,
    touchdown_seed_map,
)
from .v004_success_seed import _CandidateSolveTimeout, _candidate_wall_clock_timeout


class V005StagedHorizonMixin(V005MultiRecoveryMixin):
    """Recovered v0.0.5 search with ranked-seed and horizon control."""

    def _v005_multi_recovery(
        self, angle, q, support, anchors, seed=None, horizons=None,
        touchdown_seed_ranks=(1,), candidate_timeout_s=60.0,
        search_phase="primary",
    ):
        seed = self.v005_multi_seed if seed is None else int(seed)
        st = self._v005_state(angle, q, support, anchors)
        cfg = self.v005_multi_settings
        cmap = touchdown_seed_map(self.kin, st, cfg, seed)
        patterns = ((2, 2), (2, 1), (1, 2))
        maxh = min(5, int(np.floor(self.max_roll_deg - angle + 1e-9)))
        if horizons is None:
            horizon_list = list(range(maxh, 0, -1))
        else:
            horizon_list = [int(h) for h in horizons if 1 <= int(h) <= maxh]
        ranks = tuple(sorted({int(r) for r in touchdown_seed_ranks if int(r) >= 1}))
        timeout_s = float(candidate_timeout_s)
        liftoff_remaining = self._liftoff_remaining_ranges(
            angle, q, support, anchors
        )

        seed_counts = {int(leg): int(len(items)) for leg, items in cmap.items()}
        self._log(
            "V005 start", "phase", str(search_phase),
            "angle", float(angle), "support", tuple(support),
            "horizons", tuple(horizon_list), "seed", int(seed),
            "seed_ranks", ranks, "timeout_s", timeout_s,
            "touchdown_legs", int(len(cmap)), "seed_counts", seed_counts,
            "liftoff_remaining_deg", {
                int(k): float(v) for k, v in sorted(liftoff_remaining.items())
            },
        )

        trials = []
        for h in horizon_list:
            tt, tR = self._pose(angle + h)
            for nadd, nrem in patterns:
                if len(cmap) < nadd or len(support) < nrem:
                    self._log(
                        "V005 skip pattern", "phase", str(search_phase),
                        "angle", float(angle), "horizon", int(h),
                        "pattern", (int(nadd), int(nrem)), "reason", "insufficient_legs",
                    )
                    continue

                tdnode, lonode = event_nodes(nadd, nrem)
                generated = []
                for rank in ranks:
                    idx = rank - 1
                    for legs in itertools.combinations(sorted(cmap), nadd):
                        if any(len(cmap[l]) <= idx for l in legs):
                            continue
                        picks = tuple(cmap[l][idx] for l in legs)
                        xy = np.asarray([x[1] for x in picks])
                        for rem in itertools.combinations(sorted(support), nrem):
                            generated.append((rank, MultiContactCandidateV005(
                                tuple(legs), xy.copy(), tdnode, tuple(rem), lonode
                            )))

                hull_valid = []
                terminal_valid = []
                for rank, cand in generated:
                    ns = tuple(sorted(
                        (set(support) | set(cand.touchdown_legs)) - set(cand.liftoff_legs)
                    ))
                    na = {
                        int(k): np.asarray(v, float).copy()
                        for k, v in anchors.items()
                        if int(k) not in cand.liftoff_legs
                    }
                    for leg, xy in zip(cand.touchdown_legs, cand.touchdown_seed_xy):
                        na[leg] = np.r_[xy, 0.0]
                    inside, _ = _point_in_support_hull(
                        tt[:2], np.asarray([na[l][:2] for l in ns]), tol=1e-8
                    )
                    if not inside:
                        continue
                    hull_valid.append((rank, cand))
                    if not all(
                        analytic_leg_ik_world(
                            self.kin, tt, tR, l, na[l],
                            q_reference=q[l], residual_tol=2e-6,
                        )
                        for l in ns
                    ):
                        continue
                    terminal_valid.append((
                        _support_area([na[l][:2] for l in ns]), rank, cand, ns, na
                    ))

                terminal_valid.sort(key=lambda x: (
                    self._liftoff_priority_key(
                        x[2].liftoff_legs, liftoff_remaining
                    ),
                    -float(x[0]),
                ))
                self._log(
                    "V005 candidates", "phase", str(search_phase),
                    "angle", float(angle), "horizon", int(h),
                    "pattern", (int(nadd), int(nrem)),
                    "generated", int(len(generated)),
                    "hull_ok", int(len(hull_valid)),
                    "terminal_ik_ok", int(len(terminal_valid)),
                )

                attempted = 0
                solved_count = 0
                timed_out = 0
                accepted = None
                for candidate_index, (_, rank, cand, ns, _) in enumerate(terminal_valid):
                    attempted += 1
                    if hasattr(self, '_search_stats'):
                        self._search_stats['v005_nlp_attempted'] += 1
                    try:
                        with _candidate_wall_clock_timeout(timeout_s):
                            sol = MultiContactNLPV005(self.kin, st, cand, tt, tR, cfg).solve()
                    except _CandidateSolveTimeout:
                        timed_out += 1
                        if hasattr(self, '_search_stats'):
                            self._search_stats['v005_timeouts'] += 1
                        self._log(
                            "V005 candidate timeout", "phase", str(search_phase),
                            "angle", float(angle), "horizon", int(h),
                            "pattern", (int(nadd), int(nrem)),
                            "candidate", int(candidate_index), "seed_rank", int(rank),
                            "limit_s", timeout_s,
                        )
                        continue
                    if not sol["success"]:
                        continue
                    solved_count += 1
                    chk = dense_check_multi(self.kin, st, sol, cand, cfg)
                    if not chk["feasible"]:
                        continue
                    na = {
                        int(k): np.asarray(v, float).copy()
                        for k, v in anchors.items()
                        if int(k) not in cand.liftoff_legs
                    }
                    for leg, xy in zip(cand.touchdown_legs, sol["touchdown_xy"]):
                        na[leg] = np.r_[xy, 0.0]
                    accepted = (
                        float(sol["objective"]), int(candidate_index), int(rank),
                        cand, sol, chk, ns, na,
                    )
                    break

                trial = {
                    "search_phase": str(search_phase),
                    "horizon_deg": h,
                    "pattern": [nadd, nrem],
                    "seed_ranks": [int(x) for x in ranks],
                    "generated": len(generated),
                    "terminal_hull_ok": len(hull_valid),
                    "terminal_ik_ok": len(terminal_valid),
                    "attempted": attempted,
                    "solved": solved_count,
                    "timed_out": timed_out,
                    "accepted": int(accepted is not None),
                    "candidate_timeout_s": timeout_s,
                    "liftoff_remaining_deg": {
                        int(k): float(v) for k, v in sorted(liftoff_remaining.items())
                    },
                }
                trials.append(trial)
                self._log(
                    "V005 result", "phase", str(search_phase),
                    "angle", float(angle), "horizon", int(h),
                    "pattern", (int(nadd), int(nrem)),
                    "nlp_attempted", int(attempted),
                    "nlp_solved", int(solved_count),
                    "timed_out", int(timed_out),
                    "accepted", int(accepted is not None),
                )

                if accepted is not None:
                    objective, candidate_index, rank, cand, sol, chk, ns, na = accepted
                    exec_node = max(cand.liftoff_nodes)
                    frac = exec_node / float(cfg.n_nodes - 1)
                    self._log(
                        "V005 accepted", "phase", str(search_phase),
                        "angle", float(angle), "horizon", int(h),
                        "pattern", (int(nadd), int(nrem)),
                        "candidate", int(candidate_index), "seed_rank", int(rank),
                        "add", tuple(int(x) for x in cand.touchdown_legs),
                        "remove", tuple(int(x) for x in cand.liftoff_legs),
                        "objective", float(objective),
                    )
                    return {
                        "success": True,
                        "seed": seed,
                        "seed_rank": int(rank),
                        "horizon_deg": float(h),
                        "exec_node": int(exec_node),
                        "exec_fraction": float(frac),
                        "angle_after_deg": float(angle + h * frac),
                        "q_after": np.asarray(sol["q"][exec_node], float).copy(),
                        "support_after": tuple(ns),
                        "anchors_after": na,
                        "candidate_index": int(candidate_index),
                        "candidate": cand,
                        "solution": sol,
                        "checker": chk,
                        "trials": trials,
                        "objective": float(objective),
                        "search_phase": str(search_phase),
                    }

        self._log(
            "V005 failed", "phase", str(search_phase),
            "angle", float(angle), "support", tuple(support),
            "horizons", tuple(horizon_list), "seed_ranks", ranks,
            "trials", int(len(trials)),
        )
        return None
