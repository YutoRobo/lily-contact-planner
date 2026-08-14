# v0.0.6 conservative integration

## Goal

Integrate the behavior that produced the Chat-saved Pitch +45 deg -> Roll +45 deg Level-1 result into one planner **without redesigning the algorithm**.

The completed result is recorded in `results/pitch45_roll45_v006_baseline.json` and is the regression target. The contact sequence in that file is evidence, not a schedule to replay or hard-code.

## Behavior to preserve

The planner remains hierarchical:

1. Try continuous body progress with the current contact set.
2. Only when progress stalls, search contact recovery.
3. Prefer the ordinary one-touchdown/one-liftoff recovery used by v0.0.4.
4. If that local recovery is insufficient, expand to the multi-contact recovery introduced by v0.0.5.
5. If forward body progress still cannot be recovered, permit the v0.0.6 static contact reconfiguration: body progress is zero while touchdown/liftoff changes the support set, then normal planning resumes.
6. Validate generated motion with the independent Level-1 checker.

This ordering is an implementation search hierarchy, not a new global action optimizer.

## Explicit non-goals

This integration must not yet:

- replace the hierarchy with a global A0/A1/A2/A3 action optimization;
- add a new objective function that changes the successful behavior;
- hard-code Pitch/Roll angles, leg IDs, or the 20-event successful sequence;
- optimize runtime;
- claim Level-2 force, friction, torque, impact, or real-robot robustness.

## Required regression

Before any later generalization, a fresh run from the initial state must satisfy all of the following:

- reach Pitch +45 deg and then Roll +45 deg;
- pass the independent Level-1 checker;
- have zero joint-limit violations;
- finish at the requested orientation to numerical tolerance;
- retain finite-thickness collision checking;
- log every contact recovery with its trigger/reason, support before/after, touchdown/liftoff set, and whether the recovery was ordinary, multi-contact, or static.

The historical run used 20 contact events and finished with support `{0,4,5}`. Those values are comparison diagnostics, not mandatory equality constraints if a fresh planner discovers another valid path.

## Why this step exists

The v0.0.4-v0.0.6 work demonstrated feasibility but accumulated recovery mechanisms while solving the task. The immediate engineering task is therefore consolidation and reproducibility. Generalizing the hybrid action space is intentionally deferred until this baseline is reproduced by one implementation.
