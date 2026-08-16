"""Split lightweight BEST checkpoint metadata from dense trajectory samples.

This mixin changes storage only.  The underlying trajectory capture and search
semantics are provided by lower mixins in the MRO.
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


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def _trajectory_path_for_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    name = checkpoint_path.name
    suffix = "_checkpoint.json"
    if name.endswith(suffix):
        name = name[: -len(suffix)] + "_trajectory.json"
    else:
        name = checkpoint_path.stem + "_trajectory.json"
    return checkpoint_path.with_name(name)


class SplitCheckpointStorageMixin:
    """Write BEST summary and dense replay trajectory to separate JSON files."""

    def _write_best_checkpoint(self, angle, q, support, anchors, path, depth):
        checkpoint_target = getattr(self, "checkpoint_path", None)
        if not checkpoint_target:
            return

        # CheckpointTrajectoryMixin provides these capture helpers.  Appending the
        # BEST endpoint here preserves the same trajectory semantics as before.
        self._append_branch_frame(
            self._checkpoint_frame(angle, q, support, anchors, "best_update")
        )
        trajectory = list(getattr(self, "_checkpoint_branch_frames", []))
        self.best_trajectory = list(trajectory)

        checkpoint_target = Path(checkpoint_target)
        trajectory_target = _trajectory_path_for_checkpoint(checkpoint_target)

        best_q = np.asarray(q, dtype=float)
        best_anchors = {
            str(k): np.asarray(v, dtype=float)
            for k, v in anchors.items()
        }
        stats = dict(getattr(self, "_search_stats", {}))

        trajectory_payload = {
            "schema_version": "trajectory_v1",
            "trajectory_storage": "executed_dense_and_actual_trace",
            "task": {
                "name": type(self.task).__name__,
                "max_progress_deg": float(self.max_roll_deg),
            },
            "best_angle_deg": float(angle),
            "trajectory_frames": int(len(trajectory)),
            "best_trajectory": trajectory,
        }

        # Write the large file first.  The lightweight checkpoint is replaced
        # only after the trajectory file is complete, so its reference never
        # points to a half-written trajectory.
        _write_json_atomic(trajectory_target, trajectory_payload)

        checkpoint_payload = {
            "schema_version": "checkpoint_v3",
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
            "search_stats": stats,
            "fallback_free_so_far": bool(
                stats.get("deep_fallback_entries", 0) == 0
                and stats.get("short_horizon_entries", 0) == 0
            ),
            "trajectory_file": trajectory_target.name,
            "trajectory_schema_version": "trajectory_v1",
            "trajectory_frames": int(len(trajectory)),
            "candidate_type": "staged_v006_recovery",
            "same_algorithm_all_angles": True,
            # Keep legacy top-level summary names for existing inspection commands.
            "best_angle_deg": float(angle),
            "nodes": int(self.nodes),
            "depth": int(depth),
            "events": list(path),
            "best_support": [int(x) for x in support],
            "best_q": best_q,
            "best_anchors": best_anchors,
            "max_progress_deg": float(self.max_roll_deg),
        }
        _write_json_atomic(checkpoint_target, checkpoint_payload)
        self._log(
            "CHECKPOINT", str(checkpoint_target), "best", float(angle),
            "trajectory", str(trajectory_target),
            "trajectory_frames", len(trajectory), "nodes", self.nodes,
        )
