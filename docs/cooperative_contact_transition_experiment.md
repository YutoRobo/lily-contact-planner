# Cooperative Contact-Mode Transition Experiment

## Purpose

This experiment evaluates a generic hybrid contact-planning formulation rather than a task-angle or leg-specific workaround.

The hybrid planner state is

\[
z = (x, S),
\]

where \(x\) contains the continuous robot state (body pose, joint angles, and world contact anchors) and \(S\) is the current support set.

A neighboring contact-mode edge is described by touchdown set \(A\) and liftoff set \(R\):

\[
S^+ = (S \cup A) \setminus R.
\]

The search is bounded only by generic cardinality parameters such as minimum and maximum support count, maximum additions, maximum releases, and maximum total contact changes per edge. No leg index or progress angle is hard-coded.

## Difference from the previous contact-mode graph

The previous experiment generalized the discrete support set but used two different continuous transition mechanisms:

- add-only edges: V005 multi-contact NLP;
- release edges: V006-style static-body reconfiguration/liftoff.

The Pitch60 experiment showed that 3->4 and 4->5 support acquisition can be feasible, while desired constrained-support releases can fail in the static reconfiguration executor even when redundant support exists.

The cooperative experiment therefore evaluates every `(A, R)` candidate with the same continuous finite-horizon NLP. Body pose and all joint angles may move cooperatively during both acquisition and release.

## Continuous transition NLP

For nodes \(k=0,\ldots,N-1\), the primary decision variables are

\[
p_k,\quad R_k,\quad q_k,
\]

plus touchdown locations for legs in \(A\) and support-polygon barycentric weights.

The existing V005 hard constraints are retained:

- joint limits;
- fixed-foot equality for active supports;
- touchdown lock after contact;
- body center inside the active support hull;
- link/foot ground constraints;
- capsule inter-link clearance;
- body-link clearance.

For each released leg, post-liftoff foot height is constrained to ramp to a configurable clearance value.

## Receding execution consistency

The optimization can look ahead to a task horizon \(H\), but only the executed prefix is committed.

Let

\[
k_e = \min(N-1,\; k_{\text{last event}} + n_{\text{settle}}).
\]

The body pose at \(k_e\) is hard constrained to the task reference pose at the corresponding scalar progress. Intermediate body nodes remain free.

This prevents an arbitrary optimized intermediate body pose from being treated as if it were exactly the task-reference pose at the next replanning state.

## Search structure

The experiment remains a discrete-outer / continuous-inner method rather than a MINLP:

1. Generate generic neighboring `(A, R)` contact modes.
2. Apply cheap support-hull and terminal-IK screening.
3. Solve a cooperative transition NLP for each candidate.
4. Run the existing dense safety checker plus dense post-liftoff clearance verification.
5. Reject edges with no predicted future progress.
6. Execute only through \(k_e\), then replan.

Candidate ordering uses current kinematic/geometric quantities, including remaining fixed-anchor support range and support area. Search limits are complexity parameters rather than gait definitions.

## Initial experimental bounds

The first runner uses:

- maximum support count: 5;
- maximum touchdown count per edge: 1;
- maximum liftoff count per edge: 2;
- maximum total contact changes per edge: 3;
- maximum task horizon: 5 deg;
- one settling node after the final contact event;
- post-liftoff clearance: 0.02 m.

These values are experiment settings, not part of the contact-mode formulation.

## Recommended first test

Before restarting a multi-day Pitch60 search, probe the saved BEST checkpoint near the current bottleneck:

```bash
python3 scripts/probe_cooperative_transition_checkpoint.py \
  --checkpoint pitch60_contactgraph_checkpoint.json
```

This tests neighboring cooperative edges from the saved hybrid state without entering the full DFS.

Only after a useful edge is demonstrated should the full experiment be run:

```bash
python3 scripts/run_pitch360_forward_cooperative_transition.py \
  --pitch-deg 60 \
  --checkpoint pitch60_cooperative_checkpoint.json \
  --output pitch60_cooperative_result.json
```

## Research interpretation

The intended research question is not whether a specific leg pair can be released near a specific angle. It is whether explicitly searching hybrid contact modes while optimizing the continuous mode-transition trajectory expands the feasible motion set compared with fixed contact-change patterns or static transition primitives.
