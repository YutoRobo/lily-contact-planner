# Validated baseline snapshot — 2026-08-13

The current unified search was run from the initial condition only, with no saved future contact states and no prescribed switch angles.

- target: 0° → 720° roll
- forward relation: `x = roll_deg / 300`
- body geometric-center height: 0.35 m
- one policy across the complete horizon
- result: reached 720°
- DFS nodes: 29
- contact events: 28
- final support set: `[0, 4, 7]`

The search invokes the same operations regardless of absolute angle: advance current support, generate continuously reachable touchdown candidates at a stall, locally alter the binary contact set, perform nonlinear support IK and finite look-ahead evaluation, and backtrack if necessary.

The contact-event sequence stored in `results/unified_rollwalk_720_contact_events.json` is **output**, not an input gait schedule.

## Validation boundary

This milestone establishes an angle-independent hybrid contact-search baseline. It does not yet claim dense certification of every interpolated point between numerical path samples. Finite-thickness self-collision is also intentionally deferred while contact switching is isolated.
