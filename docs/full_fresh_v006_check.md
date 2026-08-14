# Full fresh v0.0.6 reproduction check

## Scope

Check whether one `UnifiedContactPlanner` can start from the archived v0.0.6 initial state and automatically reproduce the in-place world Pitch +45 deg -> world Roll +45 deg task without supplying saved future contact events.

The archived regression body height is temporarily set to `0.524575783 m` only for this reproduction check. This fixed regression constant must be removed after baseline verification/generalization.

## Confirmed before the blocker

- Initial state: all legs `[0, 20, -30] deg`, support `{2,4,6}`, body height approximately `0.524575783 m`.
- Fresh v0.0.4 no-contact progression advances automatically from Pitch 0 deg through Pitch 15 deg with the original support set.
- The first contact-search trigger occurs at Pitch 15 deg with support `{2,4,6}`, matching the archived successful run structure.
- Earlier local regression checks already reproduced the successful one-to-one candidates and the v0.0.5 multi-contact event independently.
- The v0.0.6 static PRM execution path itself was previously reproduced exactly when the historical touchdown goal was supplied.

## Newly identified blocker

The current static-reconfiguration stage obtains candidates from the generic expanded touchdown search. That candidate generator rejects a touchdown unless a direct joint-space segment from the current swing configuration to the touchdown configuration passes `_segment_safe`.

This conflicts with the purpose of the v0.0.6 static PRM fallback: the historical leg-4 touchdown at the Roll~30 deg stall specifically requires a non-straight PRM route. The direct segment is unsafe, while the recovered PRM path is safe.

At the archived static state (global task progress ~75 deg, support `{0,5,6}`), the current expanded generic ranking produced only two plans:

1. add leg 4 / remove leg 5
2. add leg 4 / remove leg 0

The historical successful add-leg-4 / remove-leg-6 event was not generated. Its touchdown point is approximately `[-0.371068, 0.291921, 0] m` and is PRM-only from the archived swing configuration.

Therefore the current implementation cannot yet claim a full fresh automatic reproduction of the archived v0.0.6 run. The PRM execution primitive is restored, but static candidate generation is still constrained by the earlier direct-segment reachability filter.

## Required next decision

To reproduce v0.0.6 automatically in one planner, the static-reconfiguration candidate generator needs a static-specific reachability rule that permits candidates whose connection is validated by PRM rather than requiring a direct segment first.

This changes search behavior, so it must not be implemented without explicit approval.
