"""v0.0.6 staged DFS with the recovered moving-body v0.0.5 fallback."""

from .planner_search import DfsSearchMixin
from .recovery_policy import RecoveryKind, recovery_stages


class V006StagedSearchMixin(DfsSearchMixin):
    """Preserve v0.0.6 stage order while restoring v0.0.5 multi-contact semantics."""

    def _dfs(self, angle_deg, q, support, anchors, path, depth=0):
        self.nodes += 1
        stall_angle, q_stall = self._advance_to_stall(angle_deg, q, support, anchors)

        if stall_angle > self.best_angle:
            self.best_angle = stall_angle
            self.best_path = path.copy()
            self._log("BEST", self.best_angle, "depth", depth, "support", support, "nodes", self.nodes)
        if stall_angle >= self.max_roll_deg - 1e-9:
            return {
                "angle": stall_angle,
                "q": q_stall,
                "support": tuple(support),
                "anchors": anchors,
                "events": path,
            }

        sig = self._state_signature(stall_angle, support, anchors)
        if sig in self.memo:
            return None
        self.memo.add(sig)
        if depth >= self.cfg.max_depth or self.nodes >= self.cfg.max_nodes:
            return None

        for stage in recovery_stages():
            if stage.kind == RecoveryKind.MULTI_CONTACT:
                # v0.0.5 is not a fixed-body contact edit. It solves a short
                # moving-body finite-horizon NLP and executes only through the
                # final liftoff node before receding-horizon replanning.
                multi = self._v005_multi_recovery(
                    stall_angle, q_stall, support, anchors
                )
                if multi is None:
                    self._log(
                        "RECOVERY", stage.kind.value, "at", stall_angle,
                        "depth", depth, "support", support, "plans", 0,
                        "nodes", self.nodes,
                    )
                    continue

                cand = multi["candidate"]
                sol = multi["solution"]
                new_support = tuple(multi["support_after"])
                new_anchors = multi["anchors_after"]
                event = {
                    "angle_deg": float(stall_angle),
                    "version": "v0.0.5-multi",
                    "recovery_kind": stage.kind.value,
                    "body_progress_during_reconfiguration_deg": float(
                        multi["angle_after_deg"] - stall_angle
                    ),
                    "contact_horizon_deg": float(multi["horizon_deg"]),
                    "exec_node": int(multi["exec_node"]),
                    "exec_fraction": float(multi["exec_fraction"]),
                    "seed": int(multi["seed"]),
                    "candidate_index": int(multi["candidate_index"]),
                    "touchdown_legs": [int(x) for x in cand.touchdown_legs],
                    "touchdown_nodes": [int(x) for x in cand.touchdown_nodes],
                    "liftoff_legs": [int(x) for x in cand.liftoff_legs],
                    "liftoff_nodes": [int(x) for x in cand.liftoff_nodes],
                    "add": [int(x) for x in cand.touchdown_legs],
                    "remove": [int(x) for x in cand.liftoff_legs],
                    "support_before": [int(x) for x in support],
                    "support_after": [int(x) for x in new_support],
                    "anchors_added": {
                        str(leg): new_anchors[leg].tolist()
                        for leg in cand.touchdown_legs
                    },
                    "touchdown_xy": sol["touchdown_xy"].tolist(),
                    "objective": float(multi["objective"]),
                    "solver_iterations": int(sol["nit"]),
                    "solver_function_evaluations": int(sol["nfev"]),
                    "checker": multi["checker"],
                    "trials": multi["trials"],
                    "touchdown_before_liftoff": True,
                }
                self._log(
                    " TRY", stage.kind.value, "at", stall_angle,
                    "horizon", multi["horizon_deg"],
                    "candidate", multi["candidate_index"],
                    "add", cand.touchdown_legs, "rem", cand.liftoff_legs,
                    "->", new_support, "execute_to", multi["angle_after_deg"],
                )
                result = self._dfs(
                    multi["angle_after_deg"],
                    multi["q_after"],
                    new_support,
                    {k: v.copy() for k, v in new_anchors.items()},
                    path + [event],
                    depth + 1,
                )
                if result is not None:
                    return result
                # v0.0.5 selected the best accepted NLP, rather than branching
                # over every accepted multi candidate. If downstream fails,
                # continue to the v0.0.6 static fallback.
                continue

            plans = self._plans_for_stage(stall_angle, q_stall, support, anchors, stage)
            self._log(
                "RECOVERY", stage.kind.value, "at", stall_angle, "depth", depth,
                "support", support, "plans", len(plans), "nodes", self.nodes,
            )
            if not plans:
                continue

            for branch_index, plan in enumerate(plans[: self.cfg.branch_width]):
                score, gain, add, rem, new_support, new_anchors, q_support = plan
                executed = self._execute_reconfiguration(
                    stall_angle, q_stall, support, anchors, add, rem,
                    new_support, new_anchors, stage.kind,
                )
                if executed is None:
                    continue
                q_after, transition_trace = executed
                event = {
                    "angle_deg": float(stall_angle),
                    "recovery_kind": stage.kind.value,
                    "body_progress_during_reconfiguration_deg": 0.0,
                    "add": [int(x) for x in sorted(add)],
                    "remove": [int(x) for x in rem],
                    "support_before": [int(x) for x in support],
                    "support_after": [int(x) for x in new_support],
                    "anchors_added": {str(leg): add[leg][0].tolist() for leg in add},
                    "qgoal_added": {str(leg): add[leg][1].tolist() for leg in add},
                    "predicted_gain_deg": float(gain),
                    "transition_frames": int(len(transition_trace)),
                    "touchdown_before_liftoff": True,
                    "liftoff_vertical_m": (
                        0.05 if stage.kind == RecoveryKind.STATIC_RECONFIGURATION else None
                    ),
                }
                self._log(
                    " TRY", stage.kind.value, branch_index, "at", stall_angle,
                    "gain", gain, "add", sorted(add), "rem", rem, "->", new_support,
                    "transition_frames", len(transition_trace),
                )
                result = self._dfs(
                    stall_angle, q_after, new_support,
                    {k: v.copy() for k, v in new_anchors.items()},
                    path + [event], depth + 1,
                )
                if result is not None:
                    return result
        return None
