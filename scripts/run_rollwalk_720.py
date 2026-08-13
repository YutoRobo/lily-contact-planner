#!/usr/bin/env python3
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lily_contact_planner import LilyKinematics
from lily_contact_planner.tasks import ForwardRollTask
from lily_contact_planner.unified_planner import UnifiedContactPlanner


# Initial condition only.  No future contact sequence or switch angle is provided.
Q0_DEG = np.array(
    [
        [-22.8, -47.6, -4.6],
        [-29.75, 95.0, 39.57],
        [12.13, 5.41, 137.75],
        [0.0, 0.0, 0.0],
        [22.8, -47.6, -4.6],
        [20.75, 85.17, 46.33],
        [12.65, -52.97, 93.10],
        [0.0, 0.0, 0.0],
    ],
    dtype=float,
)
INITIAL_SUPPORT = (1, 2, 5, 6)


def main():
    kin = LilyKinematics(
        a=0.15,
        L2=0.30,
        L3=0.30,
        delta_top=0.0,
        delta_bottom=0.0,
        eps_top=+1.0,
        eps_bottom=-1.0,
    )
    task = ForwardRollTask(body_height_m=0.35, forward_m_per_deg=1.0 / 300.0)
    planner = UnifiedContactPlanner(kin, task, max_roll_deg=720.0, verbose=True)
    result = planner.plan(np.deg2rad(Q0_DEG), INITIAL_SUPPORT)

    out = {
        key: value
        for key, value in result.items()
        if key not in ("final_q", "final_anchors")
    }
    out["task"] = {
        "type": "ForwardRollTask",
        "body_height_m": 0.35,
        "forward_m_per_deg": 1.0 / 300.0,
        "max_progress_deg": 720.0,
    }
    out["initial_condition"] = {
        "q0_deg": Q0_DEG.tolist(),
        "support": list(INITIAL_SUPPORT),
    }
    out["kinematics"] = {
        "a": 0.15,
        "L2": 0.30,
        "L3": 0.30,
        "delta_top": 0.0,
        "delta_bottom": 0.0,
        "eps_top": +1.0,
        "eps_bottom": -1.0,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / "latest_search_report.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["success", "best_angle_deg", "nodes", "final_support"]}, indent=2))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
