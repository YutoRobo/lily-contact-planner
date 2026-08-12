"""DFS/backtracking layer for the unified Lily contact planner."""

import numpy as np


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

    def _dfs(self, angle_deg, q, support, anchors, path, depth=0):
        self.nodes += 1
        stall_angle, q_stall = self._advance_to_stall(angle_deg, q, support, anchors)

        if stall_angle > self.best_angle:
            self.best_angle = stall_angle
            self.best_path = path.copy()
            self._log(
                "BEST",
                self.best_angle,
                "depth",
                depth,
                "support",
                support,
                "nodes",
                self.nodes,
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

        plans = self._rank_plans(stall_angle, q_stall, support, anchors, False)
        if not plans:
            plans = self._rank_plans(stall_angle, q_stall, support, anchors, True)
        self._log(
            "STALL",
            stall_angle,
            "depth",
            depth,
            "support",
            support,
            "plans",
            len(plans),
            "nodes",
            self.nodes,
        )

        for branch_index, plan in enumerate(plans[: self.cfg.branch_width]):
            score, gain, add, rem, new_support, new_anchors, q_support = plan
            q_after = self._actual(stall_angle, q_support, new_support, new_anchors)
            if q_after is None:
                continue

            event = {
                "angle_deg": float(stall_angle),
                "add": [int(x) for x in sorted(add)],
                "remove": [int(x) for x in rem],
                "support_before": [int(x) for x in support],
                "support_after": [int(x) for x in new_support],
                "anchors_added": {
                    str(leg): add[leg][0].tolist() for leg in add
                },
                "qgoal_added": {
                    str(leg): add[leg][1].tolist() for leg in add
                },
                "predicted_gain_deg": float(gain),
            }
            self._log(
                " TRY",
                branch_index,
                "at",
                stall_angle,
                "gain",
                gain,
                "add",
                sorted(add),
                "rem",
                rem,
                "->",
                new_support,
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
        return None

    def plan(self, q0, support0):
        """Plan from the supplied initial state using one policy for the full horizon."""
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
            "final_support": (
                [int(x) for x in result["support"]] if result is not None else None
            ),
            "final_q": result["q"] if result is not None else None,
            "final_anchors": result["anchors"] if result is not None else None,
            "candidate_type": "directly_reachable_touchdowns",
            "same_algorithm_all_angles": True,
        }
