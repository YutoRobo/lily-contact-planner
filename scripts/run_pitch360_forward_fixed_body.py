#!/usr/bin/env python3
"""Run the Pitch360 +X task with the experimental fixed-body NLP bounds.

The standard ``scripts/run_pitch360_forward.py`` remains the free-body baseline.
Use this runner only for A/B comparison of whether fixing the body trajectory is
acceptable and computationally useful.
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lily_contact_planner.experimental_fixed_body import (
    enable_fixed_body_trajectory_experiment,
)


if __name__ == "__main__":
    info = enable_fixed_body_trajectory_experiment()
    print("BODY_TRAJECTORY_EXPERIMENT", json.dumps(info), flush=True)

    # Import only after enabling the experiment so the same standard runner and
    # task setup are reused without duplicating planner configuration.
    from run_pitch360_forward import main

    main()
