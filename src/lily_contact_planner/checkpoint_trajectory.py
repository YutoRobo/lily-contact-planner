"""Checkpoint-only trajectory capture for long Lily searches.

This mixin does not change candidate generation, solver settings, recovery order,
or acceptance rules.  It observes recursive planner entry states and BEST updates
and writes a branch-consistent keyframe history to the existing checkpoint path.
"""

import json
import os
from pathlib import Path

import numpy as np


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
    """Add renderable BEST-path keyframes without changing search semantics."""

    def _checkpoint_frame(self, progress_deg, q, support, anchors, source):
        t, R = self._pose(float(progress_deg))
        return {
            "progress_deg": float(progress_deg),
            "body_t": np.asarray(t, dtype=float),
            "body_R": np.asarray(R, dtype=float),
            "q_rad": np.asarray(q, dtype=float),
            "support": [int(x) for x in support],
            "anchors": {
                str(k): np.asarray(v, dtype=float)
                for k, v in anchors.items()
            },
            "state_source": str(source),
        }

    @staticmethod
    def _same_keyframe(a, b):
        return (
            abs(float(a["progress_deg"]) - float(b["progress_deg"])) <= 1e-10
            and tuple(a["support"]) == tuple(b["support"])
            and np.allclose(np.asarray(a["q_rad"]), np.asarray(b["q_rad"]), atol=1e-12)
        )

    def _append_branch_frame(self, frame):
        frames = getattr(self, "_checkpoint_branch_frames", None)
        if frames is None:
            frames = []
            self._checkpoint_branch_frames = frames
        if frames and self._same_keyframe(frames[-1], frame):
            return
        frames.append(frame)

    def _dfs(self, angle_deg, q, support, anchors, path, depth=0):
        """Observe recursion entry/exit while delegating all search work unchanged."""
        parent_frames = getattr(self, "_checkpoint_branch_frames", [])
        self._checkpoint_branch_frames = list(parent_frames)
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
            "keyframes", len(trajectory), "nodes", self.nodes,
        )

    def plan(self, q0, support0):
        self._checkpoint_branch_frames = []
        self.best_trajectory = []
        result = super().plan(q0, support0)
        if isinstance(result, dict):
            result["best_trajectory"] = list(getattr(self, "best_trajectory", []))
        return result
