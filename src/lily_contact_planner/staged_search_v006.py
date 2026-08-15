"""v0.0.6 staged search with bounded primary multi-start.

At a stalled state the planner preserves exhaustive discrete leg choices but
uses only the top-ranked touchdown initial guess in PRIMARY. If v0.0.4 and
v0.0.5 PRIMARY fail, v0.0.6 static reconfiguration is tried before any extra
multi-start. Only after all static branches fail does DEEP FALLBACK use ranked
seeds 2..5, followed finally by shorter moving-body horizons.

When ``checkpoint_path`` is set on the planner, every new BEST state is written
atomically to JSON so an interrupted long run still retains its best trajectory
prefix and terminal state.
"""

import json
import os
from pathlib import Path

import numpy as np

from .planner_search import DfsSearchMixin
from .recovery_policy import RecoveryKind
from .v004_receding import V004RecedingHorizonMixin


def _checkpoint_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _checkpoint_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_jsonable(v) for v in value]
    return value


class V006StagedSearchMixin(V004RecedingHorizonMixin, DfsSearchMixin):
    """v0.0.4 normal progression -> PRIMARY -> static -> DEEP -> short horizon."""

    def _write_best_checkpoint(self, angle, q, support, anchors, path, depth):
        target = getattr(self, "checkpoint_path", None)
        if not target:
            return
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        payload = {
            "checkpoint": True,
            "success": False,
            "best_angle_deg": float(angle),
            "nodes": int(self.nodes),
            "depth": int(depth),
            "best_events": list(path),
            "events": list(path),
            "best_support": [int(x) for x in support],
            "best_q": np.asarray(q, float),
            "best_anchors": {
                str(k): np.asarray(v, float) for k, v in anchors.items()
            },
            "search_stats": dict(getattr(self, "_search_stats", {})),
            "fallback_free_so_far": bool(
                getattr(self, "_search_stats", {}).get("deep_fallback_entries", 0) == 0
                and getattr(self, "_search_stats", {}).get("short_horizon_entries", 0) == 0
            ),
            "candidate_type": "staged_v006_recovery",
            "same_algorithm_all_angles": True,
            "max_progress_deg": float(self.max_roll_deg),
        }
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(_checkpoint_jsonable(payload), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(target))
        self._log("CHECKPOINT", str(target), "best", float(angle), "nodes", self.nodes)

    def _try_v004_at_horizon(
        self, angle, q_work, support, anchors, path, depth,
        horizon_deg, seed, advance_seed,
        touchdown_seed_ranks=(1,), candidate_timeout_s=60.0,
        search_phase="primary",
    ):
        one = self._v004_contact_recovery(
            angle, q_work, support, anchors,
            horizons=(int(horizon_deg),),
            seed_override=int(seed),
            advance_seed=bool(advance_seed),
            touchdown_seed_ranks=touchdown_seed_ranks,
            candidate_timeout_s=float(candidate_timeout_s),
            search_phase=str(search_phase),
        )
        if one is None:
            return None

        cand = one['candidate']
        sol = one['solution']
        new_support = tuple(one['support_after'])
        new_anchors = one['anchors_after']
        event = {
            "angle_deg": float(angle),
            "version": "v0.0.4-1to1",
            "recovery_kind": RecoveryKind.ONE_TO_ONE.value,
            "search_phase": str(search_phase),
            "candidate_index": int(one['candidate_index']),
            "seed": int(one['seed']),
            "touchdown_leg": int(cand.touchdown_leg),
            "liftoff_leg": int(cand.liftoff_leg),
            "touchdown_xy": sol.touchdown_xy.tolist(),
            "add": [int(cand.touchdown_leg)],
            "remove": [int(cand.liftoff_leg)],
            "support_before": [int(x) for x in support],
            "support_after": [int(x) for x in new_support],
            "contact_horizon_deg": float(one['horizon_deg']),
            "exec_node": int(one['exec_node']),
            "exec_fraction": float(one['exec_fraction']),
            "body_progress_during_reconfiguration_deg": float(one['angle_after_deg'] - angle),
            "objective": float(one['objective']),
            "checker_feasible": bool(sol.checker.feasible),
            "trials": one['trials'],
            "touchdown_before_liftoff": True,
        }
        self._log(
            "V004 1to1", "phase", str(search_phase), angle,
            "horizon", one['horizon_deg'], "candidate", one['candidate_index'],
            "td", cand.touchdown_leg, "lo", cand.liftoff_leg,
            "->", one['angle_after_deg'], new_support,
        )
        return self._dfs(
            one['angle_after_deg'], one['q_after'], new_support,
            {k: v.copy() for k, v in new_anchors.items()},
            path + [event], depth + 1,
        )

    def _try_v005_at_horizon(
        self, angle, q_work, support, anchors, path, depth, horizon_deg,
        touchdown_seed_ranks=(1,), candidate_timeout_s=60.0,
        search_phase="primary",
    ):
        multi = self._v005_multi_recovery(
            angle, q_work, support, anchors,
            horizons=(int(horizon_deg),),
            touchdown_seed_ranks=touchdown_seed_ranks,
            candidate_timeout_s=float(candidate_timeout_s),
            search_phase=str(search_phase),
        )
        if multi is None:
            return None

        cand = multi['candidate']
        sol = multi['solution']
        new_support = tuple(multi['support_after'])
        new_anchors = multi['anchors_after']
        event = {
            "angle_deg": float(angle),
            "version": "v0.0.5-multi",
            "recovery_kind": RecoveryKind.MULTI_CONTACT.value,
            "search_phase": str(search_phase),
            "body_progress_during_reconfiguration_deg": float(multi['angle_after_deg'] - angle),
            "contact_horizon_deg": float(multi['horizon_deg']),
            "exec_node": int(multi['exec_node']),
            "exec_fraction": float(multi['exec_fraction']),
            "seed": int(multi['seed']),
            "seed_rank": int(multi.get('seed_rank', 1)),
            "candidate_index": int(multi['candidate_index']),
            "touchdown_legs": [int(x) for x in cand.touchdown_legs],
            "touchdown_nodes": [int(x) for x in cand.touchdown_nodes],
            "liftoff_legs": [int(x) for x in cand.liftoff_legs],
            "liftoff_nodes": [int(x) for x in cand.liftoff_nodes],
            "add": [int(x) for x in cand.touchdown_legs],
            "remove": [int(x) for x in cand.liftoff_legs],
            "support_before": [int(x) for x in support],
            "support_after": [int(x) for x in new_support],
            "anchors_added": {str(leg): new_anchors[leg].tolist() for leg in cand.touchdown_legs},
            "touchdown_xy": sol['touchdown_xy'].tolist(),
            "objective": float(multi['objective']),
            "checker": multi['checker'],
            "trials": multi['trials'],
            "touchdown_before_liftoff": True,
        }
        self._log(
            "V005 multi", "phase", str(search_phase), angle,
            "horizon", multi['horizon_deg'], "candidate", multi['candidate_index'],
            "add", cand.touchdown_legs, "rem", cand.liftoff_legs,
            "->", multi['angle_after_deg'], new_support,
        )
        return self._dfs(
            multi['angle_after_deg'], multi['q_after'], new_support,
            {k: v.copy() for k, v in new_anchors.items()},
            path + [event], depth + 1,
        )

    def _try_v006_static(self, angle, q_work, support, anchors, path, depth):
        stage = type('_Stage', (), {
            'kind': RecoveryKind.STATIC_RECONFIGURATION,
            'max_add': 2,
            'max_remove': 2,
        })()
        plans = self._plans_for_stage(angle, q_work, support, anchors, stage)
        self._log("V006 static", angle, "support", support, "plans", len(plans))
        for branch_index, plan in enumerate(plans[: self.cfg.branch_width]):
            score, gain, add, rem, new_support, new_anchors, _ = plan
            executed = self._execute_reconfiguration(
                angle, q_work, support, anchors, add, rem,
                new_support, new_anchors, RecoveryKind.STATIC_RECONFIGURATION,
            )
            if executed is None:
                continue
            q_after, trace = executed
            event = {
                "angle_deg": float(angle),
                "version": "v0.0.6-static-event",
                "recovery_kind": RecoveryKind.STATIC_RECONFIGURATION.value,
                "search_phase": "primary",
                "body_progress_during_reconfiguration_deg": 0.0,
                "add": [int(x) for x in sorted(add)],
                "remove": [int(x) for x in rem],
                "support_before": [int(x) for x in support],
                "support_after": [int(x) for x in new_support],
                "anchors_added": {str(leg): add[leg][0].tolist() for leg in add},
                "qgoal_added": {str(leg): add[leg][1].tolist() for leg in add},
                "predicted_gain_deg": float(gain),
                "transition_frames": int(len(trace)),
                "touchdown_before_liftoff": True,
                "liftoff_vertical_m": 0.05,
            }
            self._log(
                "V006 static branch", branch_index, "at", angle,
                "gain", gain, "add", sorted(add), "rem", rem,
                "->", new_support,
            )
            result = self._dfs(
                angle, q_after, new_support,
                {k: v.copy() for k, v in new_anchors.items()},
                path + [event], depth + 1,
            )
            if result is not None:
                return result
        return None

    def _dfs(self, angle_deg, q, support, anchors, path, depth=0):
        self.nodes += 1
        angle = float(angle_deg)
        q_work = q.copy()

        while angle < self.max_roll_deg - 1e-9:
            if angle > self.best_angle:
                self.best_angle = angle
                self.best_path = path.copy()
                self._log(
                    "BEST", self.best_angle, "depth", depth,
                    "support", support, "nodes", self.nodes,
                )
                self._write_best_checkpoint(
                    angle, q_work, support, anchors, path, depth
                )

            no = self._v004_no_contact(angle, q_work, support, anchors)
            if no is None or not no.get('accepted', False):
                break
            self._log(
                "V004 no_contact", angle, "->", no['angle_after_deg'],
                "horizon", no['horizon_deg'], "seed", no['seed_feasible'],
                "support", support,
            )
            angle = float(no['angle_after_deg'])
            q_work = no['q_after'].copy()

        if angle >= self.max_roll_deg - 1e-9:
            if angle > self.best_angle:
                self.best_angle = angle
                self.best_path = path.copy()
                self._write_best_checkpoint(
                    angle, q_work, support, anchors, path, depth
                )
            return {
                "angle": angle,
                "q": q_work,
                "support": tuple(support),
                "anchors": anchors,
                "events": path,
            }

        sig = self._state_signature(angle, support, anchors)
        if sig in self.memo:
            return None
        self.memo.add(sig)
        if depth >= self.cfg.max_depth or self.nodes >= self.cfg.max_nodes:
            return None

        phase_end = self._v004_phase_end(angle)
        primary_h = int(np.floor(min(5.0, phase_end - angle) + 1e-9))
        if primary_h < 1:
            return None

        v004_seed = int(getattr(self, '_v004_contact_seed', 0))

        self._log("PRIMARY recovery horizon", primary_h, "at", angle)
        result = self._try_v004_at_horizon(
            angle, q_work, support, anchors, path, depth,
            primary_h, v004_seed, True,
            touchdown_seed_ranks=(1,), candidate_timeout_s=60.0,
            search_phase="primary",
        )
        if result is not None:
            return result

        result = self._try_v005_at_horizon(
            angle, q_work, support, anchors, path, depth, primary_h,
            touchdown_seed_ranks=(1,), candidate_timeout_s=60.0,
            search_phase="primary",
        )
        if result is not None:
            return result

        result = self._try_v006_static(
            angle, q_work, support, anchors, path, depth,
        )
        if result is not None:
            return result

        self._search_stats['deep_fallback_entries'] += 1
        self._log("DEEP FALLBACK", "at", angle, "horizon", primary_h)
        result = self._try_v004_at_horizon(
            angle, q_work, support, anchors, path, depth,
            primary_h, v004_seed, False,
            touchdown_seed_ranks=(2, 3, 4, 5), candidate_timeout_s=180.0,
            search_phase="deep",
        )
        if result is not None:
            return result

        result = self._try_v005_at_horizon(
            angle, q_work, support, anchors, path, depth, primary_h,
            touchdown_seed_ranks=(2, 3, 4, 5), candidate_timeout_s=180.0,
            search_phase="deep",
        )
        if result is not None:
            return result

        for h in range(primary_h - 1, 0, -1):
            self._search_stats['short_horizon_entries'] += 1
            self._log("SHORT-HORIZON fallback", h, "at", angle)
            result = self._try_v004_at_horizon(
                angle, q_work, support, anchors, path, depth,
                h, v004_seed, False,
                touchdown_seed_ranks=(1,), candidate_timeout_s=60.0,
                search_phase="short_horizon",
            )
            if result is not None:
                return result

            result = self._try_v005_at_horizon(
                angle, q_work, support, anchors, path, depth, h,
                touchdown_seed_ranks=(1,), candidate_timeout_s=60.0,
                search_phase="short_horizon",
            )
            if result is not None:
                return result

        return None

    def plan(self, q0, support0):
        self._v004_contact_seed = 0
        self._search_stats = {
            'v004_nlp_attempted': 0,
            'v004_timeouts': 0,
            'v005_nlp_attempted': 0,
            'v005_timeouts': 0,
            'deep_fallback_entries': 0,
            'short_horizon_entries': 0,
        }
        result = super().plan(q0, support0)
        events = result.get('events', []) if isinstance(result, dict) else []
        self._search_stats.update({
            'v004_primary_successes': sum(
                1 for e in events
                if e.get('version') == 'v0.0.4-1to1' and e.get('search_phase') == 'primary'
            ),
            'v005_primary_successes': sum(
                1 for e in events
                if e.get('version') == 'v0.0.5-multi' and e.get('search_phase') == 'primary'
            ),
            'v006_successes': sum(
                1 for e in events if e.get('version') == 'v0.0.6-static-event'
            ),
            'deep_successes': sum(
                1 for e in events if e.get('search_phase') == 'deep'
            ),
            'short_horizon_successes': sum(
                1 for e in events if e.get('search_phase') == 'short_horizon'
            ),
        })
        if isinstance(result, dict):
            result['search_stats'] = dict(self._search_stats)
            result['fallback_free'] = bool(
                self._search_stats['deep_fallback_entries'] == 0
                and self._search_stats['short_horizon_entries'] == 0
            )
        return result
