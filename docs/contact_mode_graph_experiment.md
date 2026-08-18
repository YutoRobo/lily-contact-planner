# Contact-mode graph experiment

## Purpose

This experiment evaluates contact scheduling as a discrete support-set search,
rather than as a fixed list of `(n_add, n_remove)` patterns.

The formulation is intentionally independent of a particular task angle or leg
number so that a successful result can later be evaluated as an algorithmic
contribution rather than as a case-specific recovery rule.

## Contact-mode state

Let

\[
S_k \subseteq \{0,\ldots,N_{leg}-1\}
\]

be the set of supporting legs at planning step `k`.

A contact-mode transition is represented by an acquisition set `A_k` and a
release set `R_k`:

\[
S_{k+1} = (S_k \cup A_k) \setminus R_k.
\]

The minimum-support requirement is

\[
|S_k| \ge m.
\]

For bounded search complexity an experimental upper support count `M` and a
maximum number of releases per transition `r_max` are introduced:

\[
m \le |S_k| \le M, \qquad |R_k| \le r_{max}.
\]

`M` and `r_max` are search-complexity parameters. They are not tied to a
specific leg combination or task-progress angle.

## Current graph-edge families

The present experiment evaluates two neighboring-mode families after the
existing v0.0.6 recovery has failed:

1. support acquisition: `|A_k|=1, |R_k|=0`;
2. support release: `|A_k|=0, 1<=|R_k|<=r_max`.

Acquisition is searched before release. This permits temporary support
redundancy before constrained supports are lifted.

The current Pitch+X runner uses `M=5` and `r_max=2` only as an initial
complexity-limited experiment. The implementation itself does not encode
`3->4->5->3`, leg 4/6, or any particular progress angle.

## Feasibility

A graph edge is accepted only through the existing planner feasibility layers.
Depending on the edge type these include:

- minimum support count;
- support-polygon containment;
- fixed-anchor IK for support legs;
- joint limits and ground constraints;
- collision checks through the existing dense checker / v0.0.6 executor;
- positive predicted task progress after the transition.

Touchdown edges use the existing v0.0.5 continuous NLP and dense checker.
Release edges use the existing v0.0.6 fixed-body vertical-liftoff primitive.

## Candidate ordering

No leg identity is privileged. Release candidates are ranked using quantities
computed from the current state:

1. remaining fixed-anchor support range of released legs (smaller first);
2. predicted future task progress (larger first);
3. remaining support-polygon area (larger first).

Touchdown candidates retain the existing geometric touchdown generation and are
ranked by resulting support-polygon area after terminal hull/IK screening.

## Research hypothesis

The hypothesis is not that a particular 5-support gait is necessary. The
hypothesis is:

> Separating contact-mode selection from continuous body/joint optimization,
> and allowing temporary support redundancy, enlarges the reachable feasible
> set compared with a planner restricted to preselected contact-count patterns.

A useful evaluation should therefore compare at least:

- success / maximum task progress;
- number and type of contact-mode transitions;
- NLP attempts and timeouts;
- computation time;
- minimum support count and geometric safety metrics;
- sensitivity to `M` and `r_max`;
- performance on multiple motion tasks rather than a single Pitch+X case.

This document describes an experiment, not yet a final algorithm or paper claim.
