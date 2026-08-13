# Lily Contact Planner

Research prototype for **hybrid geometric/kinematic contact planning** of the 8-legged robot Lily.

The current milestone is a **single unified contact-search algorithm** applied from 0° to 720° of forward + roll motion. The planner is not given switch angles or a gait/contact sequence. Starting from only the initial joint state and initial support set, it repeatedly advances the current support mode, detects loss of feasibility, generates continuously reachable touchdown candidates, searches local binary contact changes, and backtracks when a branch later dead-ends.

## Current benchmark task

The body geometric-center height is fixed at 0.35 m and the benchmark path is

- roll: 0° → 720°
- forward displacement: `x = roll_deg / 300`, so 720° corresponds to 2.4 m
- `y = 0`
- pitch = yaw = 0

This body path is a **task definition**, not a hard-coded gait. Contact switching remains autonomous.

## Defining arbitrary translation and rotation

The contact planner only needs a task path that returns the desired body pose

```text
(t(s), R(s))
```

for a path parameter `s`. Here `t(s) = [x, y, z]` is the body geometric-center position in the **world frame**, and `R(s)` is the body orientation in `SO(3)`.

### Arbitrary translation

Translation is specified directly in world coordinates. For example, a straight-line motion from `p0` with world-frame direction `d` is

```python
p = p0 + distance(s) * d / np.linalg.norm(d)
```

where `d` can point in any 3-D direction. A general curved path may simply define `x(s)`, `y(s)`, and `z(s)` independently.

Examples:

```python
# +x translation
p = np.array([distance, 0.0, 0.35])

# +y translation
p = np.array([0.0, distance, 0.35])

# diagonal xy translation
p = np.array([distance / np.sqrt(2), distance / np.sqrt(2), 0.35])
```

### Arbitrary rotation about a world-frame axis

Rotation is defined by a **world-frame rotation axis**

```text
nW = [nx, ny, nz],  ||nW|| = 1
```

and an angle `theta`. The orientation update is

```text
R(theta) = Exp([nW]x theta) R0
```

where `[nW]x` is the skew-symmetric matrix of `nW`. The important convention is that the incremental rotation is multiplied on the **left**, so the commanded axis remains fixed in the world frame.

With SciPy this can be written as

```python
from scipy.spatial.transform import Rotation

nW = np.asarray(nW, dtype=float)
nW = nW / np.linalg.norm(nW)
R_inc = Rotation.from_rotvec(theta_rad * nW).as_matrix()
R = R_inc @ R0
```

The usual yaw/pitch/roll commands are only special cases:

```text
+yaw   : nW = [ 0,  0,  1]
+pitch : nW = [ 0,  1,  0]
-roll  : nW = [-1,  0,  0]
```

For example, rotation about a 45° diagonal axis in the world `xy` plane uses

```python
nW = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
```

### Simultaneous arbitrary translation + rotation

Translation and rotation are independent parts of the task pose. A task may therefore combine any world-frame translation path with any world-frame rotation axis:

```python
def pose(s):
    # arbitrary world-frame translation
    p = p0 + distance(s) * direction_world

    # arbitrary world-frame rotation
    theta = angle(s)
    R_inc = Rotation.from_rotvec(theta * axis_world).as_matrix()
    R = R_inc @ R0
    return p, R
```

`direction_world` and `axis_world` do not need to be parallel. This allows, for example, diagonal translation while rotating about an unrelated 3-D axis.

### Piecewise motion commands

Longer maneuvers can be constructed by chaining pose segments. If segment `j` starts from orientation `R_start`, define

```text
R_j(theta) = Exp([nW_j]x theta) R_start
```

and use the terminal pose of that segment as the initial pose of the next segment. This is how the current experimental task represents

```text
in-place +yaw 45°
-> +x translation with +pitch 480°
-> +y translation with -roll 480°
```

while keeping every commanded rotation axis defined in the world frame.

For future joystick-style commands, the same convention can be expressed using desired world-frame translational and angular velocities

```text
v_des^W, omega_des^W
```

with a small-step orientation update

```text
R_{k+1} = Exp([omega_des^W]x Delta s) R_k.
```

The current planner still receives a prescribed body task path; it does **not yet optimize the body translation/orientation path itself**. It autonomously searches the contact sequence and leg configurations needed to follow the supplied task.

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

## What solver is actually used?

The checked-in 0°–720° baseline is **not one global NLP**, and it is **not a QP**. It uses two layers:

- **Discrete contact decisions:** depth-first search (DFS) with backtracking.
- **Continuous joint configurations:** bounded nonlinear least-squares inverse kinematics using `scipy.optimize.least_squares`.

In short:

```text
DFS / backtracking for contact decisions
+
nonlinear least-squares IK for continuous joint states
```

The main code locations are:

- `src/lily_contact_planner/planner_search.py`
  - `plan()` — planning entry point
  - `_dfs()` — DFS/backtracking over contact events
  - `_advance_to_stall()` — advance the current support mode until it can no longer progress
- `src/lily_contact_planner/planner_touchdown.py`
  - `_reachable_touchdowns()` — generate/refine reachable touchdown candidates
  - `_rank_plans()` — score local contact add/remove candidates using finite look-ahead
- `src/lily_contact_planner/planner_base.py`
  - `_solve_leg_to_anchor()` — bounded nonlinear least-squares support-foot IK
  - `_support_only()` — support-mode kinematic feasibility
  - `_actual()` — per-state Level-1 feasibility check
  - `_predict_gain()` — look-ahead progress estimate

A fuller explanation, including the current search score and the distinction from the earlier contact-implicit CasADi/IPOPT experiments, is in [`docs/algorithm_solver.md`](docs/algorithm_solver.md).

## Reference result

The checked-in baseline reached 720° with:

- 29 DFS nodes
- 28 contact events
- final support set `[0, 4, 7]`

See:

- `results/unified_rollwalk_720_search_summary.json` — validation scope and headline result
- `results/unified_rollwalk_720_contact_events.json` — contact-event sequence produced by the search
- `results/unified_rollwalk_720_terminal.npz` — terminal joint/support/anchor state

The event sequence is **output**, not an input gait schedule.

Important limitation: this milestone certifies the **hybrid contact/search sequence and per-state Level-1 feasibility used inside the planner**. Dense continuous-trajectory certification between every numerical sample is a separate validation step. Self-collision is currently measured by the Level-1 checker but is intentionally not used to reject contact-search candidates while stepping logic is being developed.

## Experimental progress — 2026-08-13

The original 720° result above remains the historical baseline. Current experiments keep the same DFS/backtracking contact-search architecture but have introduced several implementation changes: analytic 3-DOF leg IK for fast experimental solves, a requirement that a contact switch enable at least one real next task increment, expanded touchdown fallback after normal branches fail, and lighter search parameters.

The current multi-axis task defines rotations in the **world frame**:

- 0°–45°: in-place `+yaw` about world `+z`;
- 45°–525°: `+x` translation with `+pitch` about world `+y`;
- 525°–1005°: `+y` translation with `-roll` about world `+x`.

A reusable `YawPitchRollWorldTask` is now in `src/lily_contact_planner/tasks.py`. A fresh partial search reached total progress 255° = 45° yaw + 210° pitch, with 7 DFS nodes, 6 contact events, zero stored-state joint-limit violations, and support-region validity at all 256 stored 1° states. The roll phase has not yet been reached in this world-frame experiment.

For visualization, contact switches are displayed in the order `old support retained -> touchdown -> old+new dual support -> support transfer -> liftoff`. The reusable implementation is now checked in as `src/lily_contact_planner/visualization.py`, with the ordering documented in [`docs/visualization.md`](docs/visualization.md) and protected by `tests/test_visualization.py`. These inserted transition frames remain a **display-only reconstruction**, not yet an explicitly optimized finite-duration contact-transfer phase.

See [`docs/progress_20260813.md`](docs/progress_20260813.md) and `results/yaw45_pitch480_world_partial_255_summary.json` for the exact scope and limitations.

## Structure

- `src/lily_contact_planner/kinematics.py` — Lily forward kinematics and Jacobians
- `src/lily_contact_planner/checker.py` — independent Level-1 geometric checker
- `src/lily_contact_planner/tasks.py` — task-path definitions
- `src/lily_contact_planner/planner_base.py` — continuous kinematic feasibility layer
- `src/lily_contact_planner/planner_touchdown.py` — touchdown generation and local contact ranking
- `src/lily_contact_planner/planner_search.py` — DFS/backtracking contact-event search
- `src/lily_contact_planner/visualization.py` — display interpolation and touchdown-before-liftoff support-switch ordering
- `src/lily_contact_planner/unified_planner.py` — public unified planner class
- `scripts/run_rollwalk_720.py` — 0°→720° reproducibility entry point
- `docs/formulation.md` — mathematical Level-1 formulation
- `docs/algorithm_solver.md` — solver architecture and code map
- `docs/validated_baseline.md` — baseline and validation boundary
- `docs/progress_20260813.md` — current experimental changes and world-frame task
- `docs/visualization.md` — visualization-only interpolation convention and limitations
- `tests/test_visualization.py` — verifies new-foot touchdown/dual support occurs before old-foot liftoff
- `results/` — reference and experimental search summaries

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_rollwalk_720.py
```

The full 720° search is intentionally computation-heavy; it is a research proof of concept, not a real-time planner.

## Status

This repository preserves the **first angle-independent contact-planning baseline** while also recording the current world-frame multi-axis experiment. Next stages are dense continuous rollout/certification, explicit touchdown-before-liftoff transition planning, finite-thickness self-collision, general joystick/task commands, and stronger continuous optimization inside each discrete contact branch.
