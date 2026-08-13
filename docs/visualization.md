# Visualization convention

This repository distinguishes **planner/checker states** from **display-only interpolation**.

## Contact-switch ordering

Every visualized support switch must use the following order:

```text
old support retained
-> new foot approaches touchdown
-> new foot touches down
-> old + new feet support simultaneously
-> support transfer
-> old support foot lifts off
```

An old support foot must never be displayed as lifting before the newly added support foot is already on the ground.

The reusable implementation is in:

```text
src/lily_contact_planner/visualization.py
```

The main helper is:

```python
touchdown_first_switch_frames(...)
```

It constructs five display stages:

1. `pre-switch`
2. `touchdown approach` with the previous support set retained
3. `touchdown complete` with the union of old and newly added support legs active
4. `support transfer` with the union support set still active
5. `liftoff` of removed support legs

For intervals whose support set does not change, `interpolate_same_support(...)` can insert smooth display-only frames. Body orientation is interpolated by quaternion Slerp; body translation and joint coordinates use smoothstep interpolation.

## Important limitation

These inserted frames are **not planner output** and are **not automatically Level-1 certified**. They exist to make the discrete numerical contact solution easier to inspect visually.

The physically stronger next step is to include a finite-duration contact-transfer phase in the trajectory planner itself, then check touchdown, dual support, support transfer, and liftoff as actual trajectory states.
