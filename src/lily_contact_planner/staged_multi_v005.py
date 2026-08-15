"""Horizon-selectable wrapper for the recovered v0.0.5 multi-contact search.

The underlying candidate generation, terminal hull/IK checks, NLP, dense Checker,
and objective comparison are unchanged.  This wrapper only lets the v0.0.6
staged policy choose which body-motion horizon is attempted at a given stage.
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


class V005StagedHorizonMixin(V005MultiRecoveryMixin):
    """Recovered v0.0.5 search with caller-controlled horizon order."""

    def _v005_multi_recovery(self, angle, q, support, anchors, seed=None, horizons=None):
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

        trials = []
        for h in horizon_list:
            tt, tR = self._pose(angle + h)
            for nadd, nrem in patterns:
                if len(cmap) < nadd or len(support) < nrem:
                    continue
                tdnode, lonode = event_nodes(nadd, nrem)
                generated = []
                for legs in itertools.combinations(sorted(cmap), nadd):
                    for picks in itertools.product(*[cmap[l] for l in legs]):
                        xy = np.asarray([x[1] for x in picks])
                        for rem in itertools.combinations(sorted(support), nrem):
                            generated.append(MultiContactCandidateV005(
                                tuple(legs), xy.copy(), tdnode, tuple(rem), lonode
                            ))

                hull_valid = []
                terminal_valid = []
                for cand in generated:
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
                    hull_valid.append(cand)
                    if not all(
                        analytic_leg_ik_world(
                            self.kin, tt, tR, l, na[l],
                            q_reference=q[l], residual_tol=2e-6,
                        )
                        for l in ns
                    ):
                        continue
                    terminal_valid.append((
                        _support_area([na[l][:2] for l in ns]), cand, ns, na
                    ))

                terminal_valid.sort(key=lambda x: x[0], reverse=True)
                accepted = []
                solved_count = 0
                for candidate_index, (_, cand, ns, _) in enumerate(terminal_valid):
                    sol = MultiContactNLPV005(self.kin, st, cand, tt, tR, cfg).solve()
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
                    accepted.append((
                        sol["objective"], candidate_index, cand, sol, chk, ns, na
                    ))

                trials.append({
                    "horizon_deg": h,
                    "pattern": [nadd, nrem],
                    "generated": len(generated),
                    "terminal_hull_ok": len(hull_valid),
                    "terminal_ik_ok": len(terminal_valid),
                    "solved": solved_count,
                    "accepted": len(accepted),
                })
                if accepted:
                    objective, candidate_index, cand, sol, chk, ns, na = min(
                        accepted, key=lambda x: x[0]
                    )
                    exec_node = max(cand.liftoff_nodes)
                    frac = exec_node / float(cfg.n_nodes - 1)
                    return {
                        "success": True,
                        "seed": seed,
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
                    }
        return None
