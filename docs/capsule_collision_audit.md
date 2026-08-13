# Capsule collision audit

`src/lily_contact_planner/collision.py` provides the finite-thickness collision geometry used for diagnostic validation.

## Model

Each leg is represented by two capsules:

- `L2`: root -> J3
- `L3`: J3 -> foot

For every pair of different legs, all four combinations are checked:

- L2-L2
- L2-L3
- L3-L2
- L3-L3

The same-leg L2/L3 pair is exempted only because it intentionally shares J3.  In the current two-link distal model there is no non-adjacent same-leg pair.

Every capsule is also checked against the body cube.  Only a configurable proximal length of L2 may be ignored at its own root attachment; L3 receives no body-collision exemption.

Capsule clearance is

```text
segment_centerline_distance - radius_a - radius_b - margin
```

and body clearance is

```text
segment_to_body_box_distance - capsule_radius - margin
```

A negative value means overlap.

## Dense audit of the latest replay

First create the replay files if necessary:

```bash
python3 scripts/replay_latest.py
```

Then run the audit with the actual physical envelope radii of L2 and L3.  Example only (20 mm is not a measured Lily value):

```bash
python3 scripts/audit_capsule_collisions.py \
  --l2-radius-m 0.020 \
  --l3-radius-m 0.020 \
  --substeps 20
```

The output is:

```text
results/latest_capsule_collision_audit.json
```

The script reports the worst progress, link pair and signed clearance.  For example, a clearance of `-0.012` means 12 mm of finite-thickness overlap under the supplied capsule model.

`--root-attachment-ignore-m` controls only the intentional L2/body connection region.  If omitted, the diagnostic default is `2 * max(L2 radius, L3 radius)`.  This should eventually be replaced by a value derived from the actual joint/housing geometry.

## Important limitation

Dense interpolation is a diagnostic reconstruction, not a mathematical continuous-collision certificate and not yet a planner-generated swing trajectory.  The audit samples linear joint interpolation, linear body translation and quaternion Slerp body rotation between stored replay states.  Contact-switch pre/post states are also densely sampled.

The intended integration order is:

1. quantify collisions on an existing solution;
2. enable the same finite-thickness test as a planner candidate rejection condition;
3. if rejection alone destroys too many feasible branches, include collision clearance directly in a constrained nonlinear trajectory/IK optimization.
