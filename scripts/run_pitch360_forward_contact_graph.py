#!/usr/bin/env python3
"""Run Pitch+X with the generic contact-mode graph experiment."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lily_contact_planner.experimental_contact_mode_graph import (
    ContactModeGraphSettings,
    enable_contact_mode_graph_experiment,
)


if __name__ == "__main__":
    # These are experiment-complexity bounds, not task-angle or leg-specific rules.
    # They can be changed without changing the contact-mode formulation itself.
    settings = ContactModeGraphSettings(
        max_support_count=5,
        max_release_per_transition=2,
        add_horizon_max_deg=5,
        touchdown_seed_rank=1,
        candidate_timeout_s=60.0,
    )
    info = enable_contact_mode_graph_experiment(settings)
    print("CONTACT_MODE_GRAPH_EXPERIMENT", json.dumps(info), flush=True)

    from run_pitch360_forward import main

    main()
