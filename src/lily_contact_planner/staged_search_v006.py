"""v0.0.6 staged search with recovered v0.0.4/v0.0.5/v0.0.6 semantics."""

from .planner_search import DfsSearchMixin
from .recovery_policy import RecoveryKind
from .v004_receding import V004RecedingHorizonMixin


class V006StagedSearchMixin(V004RecedingHorizonMixin, DfsSearchMixin):
    """v0.0.4 normal progression -> v0.0.5 multi -> v0.0.6 static PRM."""

    def _dfs(self, angle_deg, q, support, anchors, path, depth=0):
        self.nodes += 1
        angle = float(angle_deg)
        q_work = q.copy()

        # v0.0.4 baseline: at every cycle inspect the short horizon first.
        # If it certifies, execute only ~1 deg and replan.
        while angle < self.max_roll_deg - 1e-9:
            if angle > self.best_angle:
                self.best_angle = angle
                self.best_path = path.copy()
                self._log("BEST", self.best_angle, "depth", depth, "support", support, "nodes", self.nodes)

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

        # Stage 1: exact v0.0.4 one-touchdown/one-liftoff finite-horizon NLP.
        one = self._v004_contact_recovery(angle, q_work, support, anchors)
        if one is not None:
            cand = one['candidate']
            sol = one['solution']
            new_support = tuple(one['support_after'])
            new_anchors = one['anchors_after']
            event = {
                "angle_deg": float(angle),
                "version": "v0.0.4-1to1",
                "recovery_kind": RecoveryKind.ONE_TO_ONE.value,
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
                "V004 1to1", angle, "horizon", one['horizon_deg'],
                "candidate", one['candidate_index'], "td", cand.touchdown_leg,
                "lo", cand.liftoff_leg, "->", one['angle_after_deg'], new_support,
            )
            result = self._dfs(
                one['angle_after_deg'], one['q_after'], new_support,
                {k: v.copy() for k, v in new_anchors.items()},
                path + [event], depth + 1,
            )
            if result is not None:
                return result

        # Stage 2: v0.0.5 moving-body multi-contact fallback.
        multi = self._v005_multi_recovery(angle, q_work, support, anchors)
        if multi is not None:
            cand = multi['candidate']; sol = multi['solution']
            new_support = tuple(multi['support_after']); new_anchors = multi['anchors_after']
            event = {
                "angle_deg": float(angle),
                "version": "v0.0.5-multi",
                "recovery_kind": RecoveryKind.MULTI_CONTACT.value,
                "body_progress_during_reconfiguration_deg": float(multi['angle_after_deg'] - angle),
                "contact_horizon_deg": float(multi['horizon_deg']),
                "exec_node": int(multi['exec_node']),
                "exec_fraction": float(multi['exec_fraction']),
                "seed": int(multi['seed']),
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
                "V005 multi", angle, "horizon", multi['horizon_deg'],
                "candidate", multi['candidate_index'], "add", cand.touchdown_legs,
                "rem", cand.liftoff_legs, "->", multi['angle_after_deg'], new_support,
            )
            result = self._dfs(
                multi['angle_after_deg'], multi['q_after'], new_support,
                {k: v.copy() for k, v in new_anchors.items()},
                path + [event], depth + 1,
            )
            if result is not None:
                return result

        # Stage 3: v0.0.6 fixed-body static reconfiguration using PRM.
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
            result = self._dfs(
                angle, q_after, new_support,
                {k: v.copy() for k, v in new_anchors.items()},
                path + [event], depth + 1,
            )
            if result is not None:
                return result
        return None

    def plan(self, q0, support0):
        self._v004_contact_seed = 0
        return super().plan(q0, support0)
