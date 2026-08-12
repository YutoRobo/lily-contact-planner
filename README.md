# Lily Contact Planner

Research prototype for **hybrid geometric/kinematic contact planning** of the 8-legged robot Lily.

The current milestone is a **single unified contact-search algorithm** applied from 0° to 720° of forward + roll motion. The planner is not given switch angles or a gait/contact sequence. Starting from only the initial joint state and initial support set, it repeatedly advances the current support mode, detects loss of feasibility, generates continuously reachable touchdown candidates, searches local binary contact changes, and backtracks when a branch later dead-ends.

## Current task

The proof-of-concept task fixes the body geometric-center height at 0.35 m and imposes

- roll: 0° → 720°
- forward displacement: `x = roll_deg / 300`, so 720° corresponds to 2.4 m
- `y = 0`
- pitch = yaw = 0

This body path is a **task definition**, not a hard-coded gait. Contact switching remains autonomous.

## Unified algorithm

At every path sample the same rule is used:

1. Hold current support feet fixed in world coordinates and solve nonlinear leg IK for the next task sample.
2. Keep non-support links above the floor using configurations continuously reachable from their current joint state.
3. If the current support set cannot advance, sample touchdown candidates for swing legs.
4. Reject touchdown candidates that are not connected to the current swing configuration by a ground-safe continuous joint-space segment.
5. Form local discrete contact changes by adding/removing a small number of legs; no exhaustive enumeration of all `2^8` support modes is used.
6. Evaluate each candidate with support IK and finite look-ahead progress.
7. Search the contact-event tree with DFS/backtracking.
8. Continue until the requested horizon or search limits are reached.

There are **no angle-specific branches**, saved future states, or pre-programmed switch angles in the unified planner.

## Verified search result

The checked-in reference report `results/unified_rollwalk_720_search_report.json` was produced by the same policy across the entire 0°–720° horizon and reached 720° with 29 DFS nodes and 28 contact events.

Important limitation: this milestone certifies the **hybrid contact/search sequence and per-state Level-1 feasibility used inside the planner**. Dense continuous-trajectory certification between every numerical sample is a separate validation step. Self-collision is currently measured by the Level-1 checker but is intentionally not used to reject contact-search candidates while stepping logic is being developed.

## Structure

- `src/lily_contact_planner/kinematics.py` — Lily forward kinematics and Jacobians
- `src/lily_contact_planner/checker.py` — independent Level-1 geometric checker
- `src/lily_contact_planner/tasks.py` — task-path definitions
- `src/lily_contact_planner/unified_planner.py` — unified discrete-contact / nonlinear-IK search
- `scripts/run_rollwalk_720.py` — 0°→720° reproducibility entry point
- `docs/formulation.md` — mathematical Level-1 formulation
- `results/` — reference search report and terminal state

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_rollwalk_720.py
```

The full 720° search is intentionally computation-heavy; it is a research proof of concept, not a real-time planner.

## Status

This repository freezes the **first angle-independent contact-planning baseline**. The next stages are to make the continuous rollout/certification explicit, restore finite-thickness self-collision constraints, replace task-specific path progression with general joystick/task commands, and then investigate stronger continuous optimization inside each contact branch.
