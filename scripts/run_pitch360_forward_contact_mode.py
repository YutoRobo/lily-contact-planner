#!/usr/bin/env python3
"""Run Pitch+X with add-only and release-only contact-mode experiments."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lily_contact_planner.experimental_add_only import (
    enable_add_only_after_v006_experiment,
)
from lily_contact_planner.experimental_release_only import (
    enable_release_only_after_existing_recovery_experiment,
)


if __name__ == "__main__":
    add_info = enable_add_only_after_v006_experiment()
    release_info = enable_release_only_after_existing_recovery_experiment()
    print(
        "CONTACT_MODE_EXPERIMENT",
        json.dumps({"add_only": add_info, "release_only": release_info}),
        flush=True,
    )

    # Reuse the standard task, initial posture, planner settings, checkpointing,
    # and CLI.  Only the two opt-in contact-mode fallbacks above are different.
    from run_pitch360_forward import main

    main()
