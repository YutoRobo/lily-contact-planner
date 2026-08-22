#!/usr/bin/env python3
"""Run Pitch+X with cooperative contact-mode transition optimization."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lily_contact_planner.experimental_cooperative_transition import (
    CooperativeTransitionSettings,
    enable_cooperative_transition_experiment,
)


if __name__ == "__main__":
    settings = CooperativeTransitionSettings(
        max_support_count=5,
        max_add_per_transition=1,
        max_release_per_transition=2,
        max_total_contact_changes=3,
        horizon_max_deg=5,
        touchdown_seed_rank=1,
        settling_nodes=1,
        liftoff_clearance_m=0.02,
        candidate_timeout_s=60.0,
        max_candidates_per_horizon=48,
    )
    info = enable_cooperative_transition_experiment(settings)
    print("COOPERATIVE_TRANSITION_EXPERIMENT", json.dumps(info), flush=True)

    from run_pitch360_forward import main

    main()
