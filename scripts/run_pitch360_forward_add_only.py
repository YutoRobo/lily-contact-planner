#!/usr/bin/env python3
"""Run Pitch+X with the opt-in add-only-after-V006 contact experiment."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lily_contact_planner.experimental_add_only import (
    enable_add_only_after_v006_experiment,
)


if __name__ == "__main__":
    info = enable_add_only_after_v006_experiment()
    print("CONTACT_MODE_EXPERIMENT", json.dumps(info), flush=True)

    # Reuse the standard task, initial posture, planner settings, checkpointing,
    # and CLI.  Only the opt-in contact-mode fallback above is different.
    from run_pitch360_forward import main

    main()
