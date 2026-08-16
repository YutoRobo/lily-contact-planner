#!/usr/bin/env python3
"""Run Yaw45 -> translating Pitch145 -> translating Roll-145.

This harder validation keeps the current UnifiedContactPlanner search policy and
initial leg state unchanged.  Only the task path changes:
- in-place +yaw 45 deg about world +z;
- world +x translation while applying +pitch 145 deg about world +y;
- world +y translation while applying -roll 145 deg about world +x.

Translation follows the earlier world-frame experiment at 1/300 m per degree.
The total scalar planner progress is 335 deg.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lily_contact_planner.kinematics import LilyKinematics
from lily_contact_planner.tasks import Yaw45Pitch145Roll145WorldTask
from lily_contact_planner.unified_planner import UnifiedContactPlanner


def _serializable_result(result):
    serial = dict(result)
    if serial.get("final_q") is not None:
        serial["final_q"] = np.asarray(serial["final_q"]).tolist()
    if serial.get("final_anchors") is not None:
        serial["final_anchors"] = {
            str(k): np.asarray(v).tolist()
            for k, v in serial["final_anchors"].items()
        }
    # Dense trajectory data are persisted separately by the checkpoint storage
    # mixin; do not duplicate them in the final summary JSON.
    serial.pop("best_trajectory", None)
    return serial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="yaw45_pitch145_roll145_result.json",
        help="Final JSON output path",
    )
    parser.add_argument(
        "--checkpoint",
        default="yaw45_pitch145_roll145_checkpoint.json",
        help="Atomic BEST-state checkpoint JSON path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable planner progress logging",
    )
    args = parser.parse_args()

    kin = LilyKinematics(
        a=0.15,
        L2=0.30,
        L3=0.30,
        delta_top=0.0,
        delta_bottom=0.0,
        eps_top=+1.0,
        eps_bottom=-1.0,
    )
    task = Yaw45Pitch145Roll145WorldTask()
    planner = UnifiedContactPlanner(
        kin,
        task,
        max_roll_deg=task.total_progress_deg,
        verbose=not args.quiet,
    )
    planner.checkpoint_path = str(Path(args.checkpoint))

    # Keep the current Pitch45 -> Roll45 validation initial state unchanged so
    # differences primarily reflect the harder task path.
    q0 = np.tile(np.deg2rad([0.0, 20.0, -30.0]), (8, 1))
    support0 = (2, 4, 6)

    print(
        "TASK",
        json.dumps(
            {
                "yaw_deg": task.yaw_deg,
                "pitch_deg": task.pitch_deg,
                "roll_deg": -task.roll_deg,
                "total_progress_deg": task.total_progress_deg,
                "body_height_m": task.body_height_m,
                "translation_m_per_deg": task.forward_m_per_deg,
                "pitch_translation_x_m": task.forward_m_per_deg * task.pitch_deg,
                "roll_translation_y_m": task.forward_m_per_deg * task.roll_deg,
                "phase_boundaries_deg": list(task.phase_boundaries_deg),
            }
        ),
        flush=True,
    )
    print("CHECKPOINT_PATH", planner.checkpoint_path, flush=True)

    try:
        result = planner.plan(q0, support0)
    except KeyboardInterrupt:
        print(
            "INTERRUPTED checkpoint/trajectory retained at " + planner.checkpoint_path,
            flush=True,
        )
        return

    serial = _serializable_result(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2)

    summary = {
        "success": result["success"],
        "best_angle_deg": result["best_angle_deg"],
        "target_progress_deg": task.total_progress_deg,
        "nodes": result["nodes"],
        "n_events": len(result.get("events", [])),
        "final_support": result.get("final_support"),
        "event_versions": [e.get("version") for e in result.get("events", [])],
        "search_stats": result.get("search_stats"),
        "fallback_free": result.get("fallback_free"),
        "output": str(output),
        "checkpoint": planner.checkpoint_path,
    }
    print("FINAL_RESULT", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
