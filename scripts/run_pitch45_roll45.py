#!/usr/bin/env python3
"""Run the fresh Pitch45 -> Roll45 regression with UnifiedContactPlanner.

This is the single entry point used by both local Ubuntu runs and GitHub Actions.
The planner also writes an atomic BEST-state checkpoint whenever progress
improves so an interrupted long run can still be inspected and replayed.

Temporary CI diagnostic: when GITHUB_ACTIONS=true, run the Yaw45/Pitch145/Roll145
startup instead, dump the Python stack every 30 seconds, and interrupt after
180 seconds. This changes CI observation only; planner search semantics are not
modified. The diagnostic block is intended to be reverted after inspection.
"""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

# Prefer the source tree that belongs to this checkout.  This avoids accidentally
# importing an older lily_contact_planner previously installed in site-packages.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lily_contact_planner.kinematics import LilyKinematics
from lily_contact_planner.tasks import Pitch45ThenRoll45Task, Yaw45Pitch145Roll145WorldTask
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
    # Dense trajectory data are persisted separately by checkpoint storage.
    serial.pop("best_trajectory", None)
    return serial


def _diagnostic_alarm(signum, frame):
    raise KeyboardInterrupt("CI yaw startup diagnostic time limit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="full_v006_fresh_result.json",
        help="Final JSON output path",
    )
    parser.add_argument(
        "--checkpoint",
        default="full_v006_fresh_checkpoint.json",
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

    ci_yaw_diagnostic = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    if ci_yaw_diagnostic:
        task = Yaw45Pitch145Roll145WorldTask()
        max_progress = task.total_progress_deg
        print("CI_YAW_STARTUP_DIAGNOSTIC", True, flush=True)
        print("CI_DIAGNOSTIC_LIMIT_S", 180, flush=True)
        faulthandler.enable()
        faulthandler.dump_traceback_later(30.0, repeat=True)
        signal.signal(signal.SIGALRM, _diagnostic_alarm)
        signal.alarm(180)
    else:
        task = Pitch45ThenRoll45Task()
        max_progress = 90.0

    planner = UnifiedContactPlanner(
        kin,
        task,
        max_roll_deg=max_progress,
        verbose=not args.quiet,
    )
    planner.checkpoint_path = str(Path(args.checkpoint))

    q0 = np.tile(np.deg2rad([0.0, 20.0, -30.0]), (8, 1))
    support0 = (2, 4, 6)

    print("TASK", type(task).__name__, "MAX_PROGRESS", max_progress, flush=True)
    print("CHECKPOINT_PATH", planner.checkpoint_path, flush=True)
    try:
        result = planner.plan(q0, support0)
    except KeyboardInterrupt:
        print(
            "INTERRUPTED checkpoint retained at " + planner.checkpoint_path,
            flush=True,
        )
        if ci_yaw_diagnostic:
            print("CI_YAW_STARTUP_DIAGNOSTIC_COMPLETE", flush=True)
        return
    finally:
        if ci_yaw_diagnostic:
            signal.alarm(0)
            faulthandler.cancel_dump_traceback_later()

    serial = _serializable_result(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(serial, f, indent=2)

    summary = {
        "success": result["success"],
        "best_angle_deg": result["best_angle_deg"],
        "nodes": result["nodes"],
        "n_events": len(result.get("events", [])),
        "final_support": result.get("final_support"),
        "event_versions": [e.get("version") for e in result.get("events", [])],
        "output": str(output),
        "checkpoint": planner.checkpoint_path,
    }
    print("FINAL_RESULT", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
