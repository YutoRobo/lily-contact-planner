# Algorithm and solver architecture

This document states explicitly **what solves the current 0°–720° baseline** and where that logic lives in the code.

## Summary

The current baseline is **not** one monolithic optimization over the full 0°–720° horizon. It is also **not** a QP, and it is **not** the earlier contact-implicit CasADi/IPOPT formulation.

Instead, it combines two layers:

1. **Discrete contact-sequence search:** depth-first search (DFS) with backtracking.
2. **Continuous kinematic solves:** bounded nonlinear least-squares inverse kinematics using `scipy.optimize.least_squares`.

In short:

```text
DFS / backtracking for contact decisions
+
nonlinear least-squares IK for continuous joint configurations
```

The same policy is applied over the complete horizon; there are no angle-specific switch rules or stored future gait states.

## 1. Public planner entry point

The public planner class is:

```text
src/lily_contact_planner/unified_planner.py
```

`UnifiedContactPlanner` combines three implementation layers:

```python
class UnifiedContactPlanner(
    DfsSearchMixin,
    TouchdownSearchMixin,
    PlannerBaseMixin,
):
    pass
```

The actual algorithm is therefore split across:

```text
planner_search.py      discrete contact-event search
planner_touchdown.py   touchdown generation and contact-plan ranking
planner_base.py        continuous kinematic feasibility and nonlinear IK
```

The external planning call is:

```python
planner.plan(q0, support0)
```

which enters the DFS search from only the supplied initial joint state and initial support set.

## 2. Continuous support-leg solve

File:

```text
src/lily_contact_planner/planner_base.py
```

Main function:

```text
_solve_leg_to_anchor()
```

For a support leg, the world-frame support-foot anchor `a_i` is kept fixed while the body pose changes. The code solves approximately

\[
\min_{q_i}
\left\|
 p_i^W(T,q_i)-a_i
\right\|^2
\]

subject to the joint bounds.

The numerical solver is:

```python
scipy.optimize.least_squares
```

and the analytic leg Jacobian is supplied through

```python
self.kin.leg_jacobian_world(...)
```

The function `_support_only()` applies this solve to all current support legs after first checking the projected support-hull condition.

The function `_actual()` then also raises/checks swing legs and calls the independent `Level1Checker` for the current numerical state.

## 3. Advance until the current support mode stalls

File:

```text
src/lily_contact_planner/planner_search.py
```

Function:

```text
_advance_to_stall()
```

The planner advances the benchmark path in fixed increments (`step_deg = 1.0` in `PlannerSettings`). At each increment it calls `_actual()`.

Conceptually:

```text
while next 1° state is feasible:
    advance 1°

if next 1° state is infeasible:
    current contact mode has stalled
```

A stall is what triggers a search for a new contact mode.

## 4. Touchdown candidate generation

File:

```text
src/lily_contact_planner/planner_touchdown.py
```

Function:

```text
_reachable_touchdowns()
```

For each swing leg, the planner samples candidate joint configurations around the current state and, with lower probability, over the wider joint space.

Candidates are rejected unless the joint-space segment from the current swing configuration to the candidate remains ground-safe.

The surviving candidates are refined using bounded nonlinear least-squares so that the foot reaches the ground plane (`foot_z ≈ 0`) while maintaining the current Level-1 geometric requirements used by this routine.

Thus touchdown generation is approximately:

```text
random joint-space sampling
→ continuous ground-safe reachability check
→ nonlinear least-squares touchdown refinement
→ exact foot-anchor IK refinement
```

This is a heuristic candidate generator; it is not a global continuous optimizer over all possible footholds.

## 5. Contact add/remove candidate ranking

File:

```text
src/lily_contact_planner/planner_touchdown.py
```

Function:

```text
_rank_plans()
```

At a stall, local binary contact changes are formed by adding touchdown legs and removing existing support legs. The baseline does **not** enumerate all `2^8` support modes as a fundamental step.

Each new support mode is checked with `_support_only()` and then evaluated by a finite look-ahead using `_predict_gain()`.

The current ranking score is

\[
S = \Delta\theta_{\mathrm{gain}}
    -0.12(N_{\mathrm{add}}+N_{\mathrm{remove}})
    +0.03N_{\mathrm{support}}.
\]

This score is only a **search heuristic**. Level-1 validity is treated separately as a feasibility condition.

## 6. Discrete contact-sequence solver

File:

```text
src/lily_contact_planner/planner_search.py
```

Function:

```text
_dfs()
```

The contact sequence is selected with **depth-first search and backtracking**.

The procedure is:

```text
advance current support mode to stall
→ generate/rank local contact changes
→ try a high-ranked branch
→ recursively continue
→ if the branch later dead-ends, backtrack
→ try another branch
```

The current search is therefore best described as a **hybrid discrete-contact search with nonlinear kinematic solves inside each branch**.

## 7. What the current solver is not

The current checked-in 720° baseline is not:

- one global NLP over body pose, all 24 joint coordinates, and all contact variables;
- a QP;
- mixed-integer quadratic programming;
- the earlier relaxed contact-implicit CasADi/IPOPT formulation;
- exhaustive enumeration of all 256 contact masks at each step.

The earlier contact-implicit NLP work motivated the current separation of discrete and continuous decisions because relaxed contact variables could remain fractional and exploit big-M slack. The present baseline therefore keeps contact decisions discrete.

## 8. Compact code map

```text
UnifiedContactPlanner
│
├── planner_search.py
│   ├── plan()                  external planning call
│   ├── _dfs()                  DFS/backtracking contact search
│   └── _advance_to_stall()     advance current support mode
│
├── planner_touchdown.py
│   ├── _reachable_touchdowns() touchdown candidate generation
│   └── _rank_plans()           local add/remove ranking
│
└── planner_base.py
    ├── _solve_leg_to_anchor()  SciPy nonlinear least-squares IK
    ├── _support_only()         support-mode kinematic feasibility
    ├── _actual()               per-state Level-1 feasibility
    └── _predict_gain()         finite look-ahead progress
```

## 9. One-sentence description

If the current baseline must be summarized in one sentence:

> **The contact sequence is searched with DFS/backtracking, while each continuous support/swing configuration is obtained with bounded nonlinear least-squares kinematics and checked against the current Level-1 feasibility rules.**
