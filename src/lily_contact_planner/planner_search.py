"""DFS/backtracking layer for the unified Lily contact planner.

The v0.0.6 consolidation keeps the successful development order explicit:
continue with the current support set until it stalls, then try a one-to-one
contact exchange before widening to multi-contact recovery.  A final static
reconfiguration stage reuses the expanded local contact search at the stalled
body pose.  This is deliberately a conservative integration, not a new global
action optimizer.
"""

import numpy as np

from .recovery_policy import RecoveryKind, recovery_stages


class DfsSearchMixin:
    def _advance_to_stall(self, angle_deg, q, support, anchors):
        a = float(angle_deg)
        out = q.copy()
        while a < self.max_roll_deg - 1e-9:
            qn = self._actual(a + self.cfg.step_deg, out, support, anchors)
            if qn is None:
                break
            out = qn
            a += self.cfg.step_deg
        return a, out

    def _state_signature(self, angle_deg, support, anchors):
        t, _ = self._pose(angle_deg)
        rel = []
        for leg in support:
            rel.extend(np.round((anchors[leg][:2] - t[:2]) / 0.06).astype(int).tolist())
        return (int(round(angle_deg)), tuple(support), tuple(rel))

    @staticmethod
    def _plan_matches_stage(plan, stage):
        """Filter existing local candidates without changing their geometry."""
        _, _, add, rem, _, _, _ = plan
        nadd = len(add)
        nrem = len(rem)
        if stage.kind == RecoveryKind.ONE_TO_ONE:
            return nadd == 1 and nrem == 1
        if stage.kind == RecoveryKind.MULTI_CONTACT:
            return (
                nadd <= stage.max_add
                and nrem <= stage.max_remove
                and (nadd > 1 or nrem > 1)
            )
        if stage.kind == RecoveryKind.STATIC_RECONFIGURATION:
            return nadd <= stage.max_add and nrem <= stage.max_remove
        return False

    def _plans_for_stage(self, angle_deg, q, support, anchors, stage):
        expanded = stage.kind != RecoveryKind.ONE_TO_ONE
        plans = self._rank_plans(angle_deg, q, support, anchors, expanded)
        return [p for p in plans if self._plan_matches_stage(p, stage)]

    def _execute_reconfiguration(
        self,
        angle_deg,
        q_start,
        support_before,
        anchors_before,
        add,
        rem,
        new_support,
        new_anchors,
        stage_kind,
    ):
        """Execute the already-selected v0.0.6 contact edit at fixed body pose.

        This intentionally does not add new collision rules or a new optimizer.
        It preserves the successful ordering used during development:

        1. move each new swing leg continuously to its accepted touchdown state;
        2. only after touchdown, release the old support leg(s);
        3. for the v0.0.6 static fallback, lift each released foot vertically by
           50 mm before resuming body motion;
        4. for the earlier recovery stages, reuse the existing swing-clearance
           adjustment after release.

        The touchdown candidate generator has already required the complete
        joint-space segment to satisfy the legacy per-leg ground check.  We
        repeat that same check here so execution cannot silently jump to qgoal.
        """
        t, R = self._pose(angle_deg)
        q_work = q_start.copy()
        trace = []

        # Touchdown-before-liftoff.  Keep the body pose fixed throughout.
        support_during = list(support_before)
        for leg in sorted(add):
            _, q_goal = add[leg]
            q_from = q_work[leg].copy()
            if not self._segment_safe(leg, t, R, q_from, q_goal, n=50):
                return None
            for s in np.linspace(0.0, 1.0, 51)[1:]:
                h = s * s * (3.0 - 2.0 * s)
                q_frame = q_work.copy()
                q_frame[leg] = (1.0 - h) * q_from + h * q_goal
                trace.append(q_frame)
            q_work[leg] = q_goal.copy()
            if leg not in support_during:
                support_during.append(leg)

        # Re-solve all supports at the unchanged body pose before any liftoff.
        anchors_touch = {k: v.copy() for k, v in anchors_before.items()}
        for leg, (target, _) in add.items():
            anchors_touch[leg] = target.copy()
        q_touch = self._support_only(
            angle_deg, q_work, tuple(sorted(support_during)), anchors_touch
        )
        if q_touch is None:
            return None
        q_work = q_touch

        if stage_kind == RecoveryKind.STATIC_RECONFIGURATION:
            # Exact v0.0.6 recovery primitive: fixed body, then a 50 mm
            # vertical foot lift for each released support leg.
            for leg in rem:
                _, _, foot = self.kin.world_points(t, R, leg, q_work[leg])
                target = foot.copy()
                target[2] += 0.05
                ok, q_lift, _ = self._solve_leg_to_anchor(
                    t, R, leg, q_work[leg], target
                )
                if not ok:
                    return None
                if not self._segment_safe(leg, t, R, q_work[leg], q_lift, n=40):
                    return None
                q_from = q_work[leg].copy()
                for s in np.linspace(0.0, 1.0, 41)[1:]:
                    h = s * s * (3.0 - 2.0 * s)
                    q_frame = q_work.copy()
                    q_frame[leg] = (1.0 - h) * q_from + h * q_lift
                    trace.append(q_frame)
                q_work[leg] = q_lift
        else:
            # Earlier v0.0.4/v0.0.5 behavior: after release, use the existing
            # swing-clearance adjustment rather than introducing a new motion.
            q_clear = self._robust_swing(t, R, q_work, new_support)
            if q_clear is None:
                return None
            q_work = q_clear

        # Final state must still satisfy the legacy planner checks.  No new
        # constraints are introduced here; full-body improvements are deferred.
        q_after = self._actual(angle_deg, q_work, new_support, new_anchors)
        if q_after is None:
            return None
        return q_after, trace

    def _dfs(self, angle_deg, q, support, anchors, path, depth=0):
        self.nodes += 1
        stall_angle, q_stall = self._advance_to_stall(angle_deg, q, support, anchors)

        if stall_angle > self.best_angle:
            self.best_angle = stall_angle
            self.best_path = path.copy()
            self._log(
                "BEST", self.best_angle, "depth", depth, "support", support, "nodes", self.nodes
            )
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
                    stall_angle,
                    q_stall,
                    support,
                    anchors,
                    add,
                    rem,
                    new_support,
                    new_anchors,
                    stage.kind,
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
                    stall_angle,
                    q_after,
                    new_support,
                    {k: v.copy() for k, v in new_anchors.items()},
                    path + [event],
                    depth + 1,
                )
                if result is not None:
                    return result

            # Preserve the staged policy: only widen the recovery search after
            # every branch in the current stage has failed downstream.
        return None

    def plan(self, q0, support0):
        """Plan from the supplied initial state using the v0.0.6 staged policy."""
        q0 = np.asarray(q0, dtype=float).copy()
        support0 = tuple(int(i) for i in support0)
        t0, R0 = self._pose(0.0)
        anchors0 = {
            leg: self.kin.foot_world(t0, R0, leg, q0[leg]).copy()
            for leg in support0
        }
        for leg in anchors0:
            anchors0[leg][2] = 0.0

        self.memo.clear()
        self.nodes = 0
        self.best_angle = 0.0
        self.best_path = []

        result = self._dfs(0.0, q0, support0, anchors0, [], 0)
        return {
            "success": result is not None,
            "best_angle_deg": float(self.best_angle),
            "nodes": int(self.nodes),
            "best_events": self.best_path,
            "events": result["events"] if result is not None else [],
            "final_support": ([int(x) for x in result["support"]] if result is not None else None),
            "final_q": result["q"] if result is not None else None,
            "final_anchors": result["anchors"] if result is not None else None,
            "candidate_type": "staged_v006_recovery",
            "same_algorithm_all_angles": True,
            "recovery_policy": [stage.kind.value for stage in recovery_stages()],
        }
