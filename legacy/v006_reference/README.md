# v0.0.6 restoration reference

This directory preserves source recovered from the ChatGPT conversation/library artifacts used for the verified Pitch +45 deg -> Roll +45 deg Level-1 exploration.

## Recovered and smoke-tested

- v0.0.4 archived package: its original test suite passes unchanged (`3 passed`).
- contemporaneous legacy stack `lily_kinematics.py` + `lily_level1_checker.py` + `lily_hybrid_local_planner_v3.py` + `lily_hybrid_local_planner_v4.py`: import/compile succeeds, and a 2 deg short planning run with `HybridLocalPlannerV4` succeeds (`3 frames`, `0 contact events`).
- `resume_roll_after_static_prm_v006.py`: restored verbatim from `Lily_v0.0.6_pitch45_roll45_exploration.zip`.

## Verified v0.0.6 behavior preserved by the archived script

At the Roll ~30 deg stall of the Pitch45 -> Roll45 task, the body pose is held fixed, leg 4 follows a PRM-certified touchdown path, contact is established, leg 6 is released, and leg 6 is lifted vertically by 50 mm. The planner then returns to the ordinary short-horizon loop.

## Remaining restoration dependency

The archived v0.0.6 runner imports `scripts/explore_pitch45_roll45_v005_resume.py` and loads the v0.0.5 certified prefix (`trajectory.npz`, `report.json`, and `prm_leg4_static_path.npz`). Those exact v0.0.5 artifacts were referenced by the v0.0.6 archive but were not bundled inside it and have not yet been recovered as standalone files.

No replacement algorithm has been invented for these missing dependencies. Recovery work must either locate the exact v0.0.5 artifacts or stop for user approval before any behavioral reconstruction is attempted.
