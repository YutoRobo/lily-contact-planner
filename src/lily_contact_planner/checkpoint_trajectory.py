"""Checkpoint trajectory capture for long Lily searches.

This mixin does not change candidate generation, solver settings, recovery order,
or acceptance rules.  It records the *executed portion* of accepted trajectories
so checkpoint_v2 can reproduce the planner motion instead of inventing it later.

Recording policy
----------------
- v0.0.4 no-contact: save the dense projected/checker trajectory up to the
  receding-horizon execution fraction.
- v0.0.4 contact: save the dense projected/checker trajectory up to liftoff.
- v0.0.5 contact: save the same 101-sample interpolation used by the dense
  checker, up to the final liftoff node.
- v0.0.6 static recovery: save the actual PRM/liftoff ``trace`` returned by the
  executor, plus its accepted terminal state.

Only recording is added; search semantics are delegated unchanged to ``super``.
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .checker_v004 import dense_projected_trajectory_v004


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class CheckpointTrajectoryMixin:
    """Add branch-consistent executed trajectory capture to BEST checkpoints."""

    def _checkpoint_frame_arrays(
        self,
        progress_deg,
        body_t,
        body_R,
        q,
        support,
        anchors,
        source,
        certified_state=True,
    ):
        return {
            "progress_deg": float(progress_deg),
            "body_t": np.asarray(body_t, dtype=float),
            "body_R": np.asarray(body_R, dtype=float),
            "q_rad": np.asarray(q, dtype=float),
            "support": [int(x) for x in support],
            "anchors": {
                str(k): np.asarray(v, dtype=float)
                for k, v in anchors.items()
            },
            "state_source": str(source),
            "certified_state": bool(certified_state),
        }

    def _checkpoint_frame(self, progress_deg, q, support, anchors, source):
        t, R = self._pose(float(progress_deg))
        return self._checkpoint_frame_arrays(
            progress_deg, t, R, q, support, anchors, source, True
        )

    @staticmethod
    def _same_keyframe(a, b):
        return (
            abs(float(a["progress_deg"]) - float(b["progress_deg"])) <= 1e-10
            and tuple(a["support"]) == tuple(b["support"])
            and np.allclose(
                np.asarray(a["q_rad"]), np.asarray(b["q_rad"]), atol=1e-12
            )
        )

    def _append_branch_frame(self, frame):
        frames = getattr(self, "_checkpoint_branch_frames", None)
        if frames is None:
            frames = []
            self._checkpoint_branch_frames = frames
        if frames and self._same_keyframe(frames[-1], frame):
            return
        frames.append(frame)

    def _append_branch_frames(self, frames):
        for frame in frames:
            self._append_branch_frame(frame)

    def _set_pending_transition(self, frames, angle, q, support):
        """Hold a speculative accepted transition until its child DFS is entered."""
        self._checkpoint_pending_transition = {
            "frames": list(frames),
            "angle": float(angle),
            "q": np.asarray(q, dtype=float).copy(),
            "support": tuple(int(x) for x in support),
        }

    def _consume_pending_transition(self, angle, q, support):
        pending = getattr(self, "_checkpoint_pending_transition", None)
        if pending is None:
            return
        matches = (
            abs(float(angle) - float(pending["angle"])) <= 1e-9
            and tuple(int(x) for x in support) == pending["support"]
            and np.allclose(
                np.asarray(q, dtype=float), pending["q"], atol=2e-8, rtol=0.0
            )
        )
        if matches:
            self._append_branch_frames(pending["frames"])
            self._checkpoint_pending_transition = None

    @staticmethod
    def _v004_support_and_anchors(state, solution, cfg, p):
        old = [int(i) for i in np.where(np.asarray(state.contact, bool))[0]]
        support = set(old)
        anchors = {
            int(k): np.asarray(v, dtype=float).copy()
            for k, v in state.anchors_world.items()
        }
        if solution.mode == "contact_switch":
            td = float(cfg.touchdown_node) / float(cfg.n_nodes - 1)
            lo = float(cfg.liftoff_node) / float(cfg.n_nodes - 1)
            tdleg = int(solution.candidate.touchdown_leg)
            loleg = int(solution.candidate.liftoff_leg)
            if p >= td - 1e-12:
                support.add(tdleg)
                anchors[tdleg] = np.array(
                    [solution.touchdown_xy[0], solution.touchdown_xy[1], 0.0],
                    dtype=float,
                )
            if p >= lo - 1e-12:
                support.discard(loleg)
                anchors.pop(loleg, None)
        return tuple(sorted(support)), anchors

    def _v004_dense_frames(
        self, angle, q, support, anchors, result, source
    ):
        sol = result.get("solution")
        if sol is None:
            return []
        cfg = self._v004_settings()
        state = self._v004_state(angle, q, support, anchors)
        ds, body_t, body_R, qdense, _, _ = dense_projected_trajectory_v004(
            self.kin, state, sol, cfg
        )
        exec_fraction = float(result.get("exec_fraction", 1.0))
        horizon = float(result.get("horizon_deg", 0.0))
        out = []
        for i, p in enumerate(ds):
            p = float(p)
            if p <= 1e-12:
                continue
            if p > exec_fraction + 1e-12:
                break
            frame_support, frame_anchors = self._v004_support_and_anchors(
                state, sol, cfg, p
            )
            out.append(
                self._checkpoint_frame_arrays(
                    float(angle) + horizon * p,
                    body_t[i],
                    body_R[i],
                    qdense[i],
                    frame_support,
                    frame_anchors,
                    source,
                    True,
                )
            )
        return out

    @staticmethod
    def _v005_support_and_anchors(state, cand, sol, cfg, p):
        old = [int(i) for i in np.where(np.asarray(state.contact, bool))[0]]
        support = set(old)
        anchors = {
            int(k): np.asarray(v, dtype=float).copy()
            for k, v in state.anchors_world.items()
        }
        td_targets = {
            int(leg): np.r_[np.asarray(xy, dtype=float), 0.0]
            for leg, xy in zip(cand.touchdown_legs, sol["touchdown_xy"])
        }
        for leg, node in zip(cand.touchdown_legs, cand.touchdown_nodes):
            threshold = float(node) / float(cfg.n_nodes - 1)
            if p >= threshold - 1e-12:
                leg = int(leg)
                support.add(leg)
                anchors[leg] = td_targets[leg].copy()
        for leg, node in zip(cand.liftoff_legs, cand.liftoff_nodes):
            threshold = float(node) / float(cfg.n_nodes - 1)
            if p >= threshold - 1e-12:
                leg = int(leg)
                support.discard(leg)
                anchors.pop(leg, None)
        return tuple(sorted(support)), anchors

    def _v005_dense_frames(self, angle, q, support, anchors, result):
        sol = result.get("solution")
        cand = result.get("candidate")
        if sol is None or cand is None:
            return []
        cfg = self.v005_multi_settings
        state = self._v005_state(angle, q, support, anchors)
        n = int(np.asarray(sol["body_pos"]).shape[0])
        knots = np.linspace(0.0, 1.0, n)
        dense = np.linspace(0.0, 1.0, int(cfg.checker_samples))
        body_t = np.column_stack(
            [
                np.interp(dense, knots, np.asarray(sol["body_pos"])[:, j])
                for j in range(3)
            ]
        )
        body_R = Slerp(
            knots, Rotation.from_matrix(np.asarray(sol["body_R"], dtype=float))
        )(dense).as_matrix()
        qflat = np.asarray(sol["q"], dtype=float).reshape(n, -1)
        qdense = np.column_stack(
            [np.interp(dense, knots, qflat[:, j]) for j in range(qflat.shape[1])]
        ).reshape(len(dense), self.kin.n_legs, 3)
        exec_fraction = float(result.get("exec_fraction", 1.0))
        horizon = float(result.get("horizon_deg", 0.0))
        out = []
        for i, p in enumerate(dense):
            p = float(p)
            if p <= 1e-12:
                continue
            if p > exec_fraction + 1e-12:
                break
            frame_support, frame_anchors = self._v005_support_and_anchors(
                state, cand, sol, cfg, p
            )
            out.append(
                self._checkpoint_frame_arrays(
                    float(angle) + horizon * p,
                    body_t[i],
                    body_R[i],
                    qdense[i],
                    frame_support,
                    frame_anchors,
                    "v005_dense_executed",
                    True,
                )
            )
        return out

    def _v006_trace_frames(
        self,
        angle_deg,
        q_start,
        support_before,
        anchors_before,
        add,
        rem,
        new_support,
        new_anchors,
        q_after,
        trace,
    ):
        t, R = self._pose(float(angle_deg))
        add_targets = {
            int(leg): np.asarray(add[leg][0], dtype=float)
            for leg in add
        }
        old_anchors = {
            int(k): np.asarray(v, dtype=float).copy()
            for k, v in anchors_before.items()
        }
        out = []
        for q_frame in trace:
            q_frame = np.asarray(q_frame, dtype=float)
            frame_support = set(int(x) for x in support_before)
            frame_anchors = {k: v.copy() for k, v in old_anchors.items()}

            # Infer the contact stage from the actual saved configuration.
            for leg, target in add_targets.items():
                foot = self.kin.foot_world(t, R, leg, q_frame[leg])
                if np.linalg.norm(foot - target) <= 2e-4:
                    frame_support.add(leg)
                    frame_anchors[leg] = target.copy()
            for leg0 in rem:
                leg = int(leg0)
                if leg in old_anchors:
                    foot = self.kin.foot_world(t, R, leg, q_frame[leg])
                    if np.linalg.norm(foot - old_anchors[leg]) > 2e-4:
                        frame_support.discard(leg)
                        frame_anchors.pop(leg, None)

            out.append(
                self._checkpoint_frame_arrays(
                    angle_deg,
                    t,
                    R,
                    q_frame,
                    tuple(sorted(frame_support)),
                    frame_anchors,
                    "v006_actual_trace",
                    False,
                )
            )

        out.append(
            self._checkpoint_frame_arrays(
                angle_deg,
                t,
                R,
                q_after,
                tuple(new_support),
                new_anchors,
                "v006_accepted_endpoint",
                True,
            )
        )
        return out

    # ------------------------------------------------------------------
    # Recording wrappers.  Each delegates the numerical/search work to the
    # historical implementation and only captures an accepted result.
    # ------------------------------------------------------------------
    def _v004_no_contact(self, angle_deg, q, support, anchors):
        result = super()._v004_no_contact(angle_deg, q, support, anchors)
        if result is not None and result.get("accepted", False):
            self._append_branch_frames(
                self._v004_dense_frames(
                    angle_deg, q, support, anchors, result,
                    "v004_no_contact_dense_executed",
                )
            )
        return result

    def _v004_contact_recovery(self, angle_deg, q, support, anchors, *args, **kwargs):
        result = super()._v004_contact_recovery(
            angle_deg, q, support, anchors, *args, **kwargs
        )
        if result is not None:
            frames = self._v004_dense_frames(
                angle_deg, q, support, anchors, result,
                "v004_contact_dense_executed",
            )
            self._set_pending_transition(
                frames,
                result["angle_after_deg"],
                result["q_after"],
                result["support_after"],
            )
        return result

    def _v005_multi_recovery(self, angle, q, support, anchors, *args, **kwargs):
        result = super()._v005_multi_recovery(
            angle, q, support, anchors, *args, **kwargs
        )
        if result is not None:
            frames = self._v005_dense_frames(angle, q, support, anchors, result)
            self._set_pending_transition(
                frames,
                result["angle_after_deg"],
                result["q_after"],
                result["support_after"],
            )
        return result

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
        result = super()._execute_reconfiguration(
            angle_deg,
            q_start,
            support_before,
            anchors_before,
            add,
            rem,
            new_support,
            new_anchors,
            stage_kind,
        )
        if result is not None:
            q_after, trace = result
            frames = self._v006_trace_frames(
                angle_deg,
                q_start,
                support_before,
                anchors_before,
                add,
                rem,
                new_support,
                new_anchors,
                q_after,
                trace,
            )
            self._set_pending_transition(
                frames, angle_deg, q_after, new_support
            )
        return result

    def _dfs(self, angle_deg, q, support, anchors, path, depth=0):
        """Observe recursion entry/exit while delegating all search work unchanged."""
        parent_frames = getattr(self, "_checkpoint_branch_frames", [])
        self._checkpoint_branch_frames = list(parent_frames)
        self._consume_pending_transition(angle_deg, q, support)
        self._append_branch_frame(
            self._checkpoint_frame(
                angle_deg, q, support, anchors, "recursive_entry"
            )
        )
        try:
            return super()._dfs(angle_deg, q, support, anchors, path, depth)
        finally:
            self._checkpoint_branch_frames = parent_frames

    def _write_best_checkpoint(self, angle, q, support, anchors, path, depth):
        """Replace the legacy summary-only checkpoint with checkpoint_v2."""
        target = getattr(self, "checkpoint_path", None)
        if not target:
            return

        self._append_branch_frame(
            self._checkpoint_frame(angle, q, support, anchors, "best_update")
        )
        trajectory = list(getattr(self, "_checkpoint_branch_frames", []))
        self.best_trajectory = list(trajectory)

        best_q = np.asarray(q, dtype=float)
        best_anchors = {
            str(k): np.asarray(v, dtype=float)
            for k, v in anchors.items()
        }
        stats = dict(getattr(self, "_search_stats", {}))
        payload = {
            "schema_version": "checkpoint_v2",
            "trajectory_storage": "executed_dense_and_actual_trace",
            "checkpoint": True,
            "success": False,
            "task": {
                "name": type(self.task).__name__,
                "max_progress_deg": float(self.max_roll_deg),
            },
            "best_summary": {
                "best_angle_deg": float(angle),
                "best_support": [int(x) for x in support],
                "best_q_rad": best_q,
                "best_q_deg": np.rad2deg(best_q),
                "best_anchors": best_anchors,
                "nodes": int(self.nodes),
                "depth": int(depth),
            },
            "best_events": list(path),
            "best_trajectory": trajectory,
            "search_stats": stats,
            "fallback_free_so_far": bool(
                stats.get("deep_fallback_entries", 0) == 0
                and stats.get("short_horizon_entries", 0) == 0
            ),
            "candidate_type": "staged_v006_recovery",
            "same_algorithm_all_angles": True,
            # Keep old top-level names for existing inspection commands.
            "best_angle_deg": float(angle),
            "nodes": int(self.nodes),
            "depth": int(depth),
            "events": list(path),
            "best_support": [int(x) for x in support],
            "best_q": best_q,
            "best_anchors": best_anchors,
            "max_progress_deg": float(self.max_roll_deg),
        }

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(target))
        self._log(
            "CHECKPOINT", str(target), "best", float(angle),
            "trajectory_frames", len(trajectory), "nodes", self.nodes,
        )

    def plan(self, q0, support0):
        self._checkpoint_branch_frames = []
        self._checkpoint_pending_transition = None
        self.best_trajectory = []
        result = super().plan(q0, support0)
        if isinstance(result, dict):
            result["best_trajectory"] = list(getattr(self, "best_trajectory", []))
        return result
