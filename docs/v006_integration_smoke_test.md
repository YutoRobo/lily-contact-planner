# v0.0.6 integration smoke-test checkpoint

Date: 2026-08-14
Branch: `agent/integrate-v006-chat-baseline`

## Purpose

Record the first executable checkpoint before running the full Pitch +45 deg -> world Roll +45 deg search.

## Source used for the check

The archived Chat artifacts were used without changing the planning algorithm:

- `Lily_v0.0.4_receding_horizon_draft_updated.zip` as the recoverable v0.0.4 code base.
- `Lily_v0.0.6_pitch45_roll45_exploration.zip` for the v0.0.6 static-event scripts and reference results.

The v0.0.6 archive is an extension package and does not contain the complete base package by itself, so its scripts were overlaid on the archived v0.0.4 base only for this local smoke check.

## Result

Command:

```bash
PYTHONPATH=src pytest -q
```

Result:

```text
3 passed in 2.81s
```

The recovered Python sources also pass syntax compilation.

## Scope / non-claims

This checkpoint confirms that the recovered base package and the v0.0.6 extension can coexist and that the archived unit tests execute successfully. It does **not** yet claim that the integrated GitHub planner reproduces the complete Pitch45 -> Roll45 trajectory or the 20 contact events.

No planner logic, constraints, objective, candidate-generation rule, or contact-switch policy was changed as part of this smoke check.

## Next verification

1. Restore the remaining v0.0.5 runtime dependencies used by the archived v0.0.6 resume script.
2. Run a short end-to-end planning segment.
3. Run the full Pitch45 -> Roll45 search.
4. Compare completion, final pose/support, contact events, and checker results against the archived v0.0.6 reference.

If reproducing the reference requires an algorithmic change, stop before making that change and obtain approval.
