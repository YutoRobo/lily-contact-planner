import numpy as np

from lily_contact_planner.experimental_cooperative_transition import (
    CooperativeTransitionSettings,
    _candidate_priority,
    _event_exec_node,
    _support_after,
)
from lily_contact_planner.multi_contact_v005 import (
    MultiContactCandidateV005,
    event_nodes,
)


def _candidate(add=(), remove=()):
    td, lo = event_nodes(len(add), len(remove))
    return MultiContactCandidateV005(
        touchdown_legs=tuple(add),
        touchdown_seed_xy=np.zeros((len(add), 2)),
        touchdown_nodes=td,
        liftoff_legs=tuple(remove),
        liftoff_nodes=lo,
    )


def test_cooperative_settings_are_generic_search_bounds():
    cfg = CooperativeTransitionSettings()
    assert cfg.max_support_count == 5
    assert cfg.max_add_per_transition == 1
    assert cfg.max_release_per_transition == 2
    assert cfg.max_total_contact_changes == 3
    assert cfg.settling_nodes == 1
    assert cfg.liftoff_clearance_m == 0.02


def test_release_only_executes_after_last_liftoff_node():
    cand = _candidate(remove=(4, 6))
    assert cand.touchdown_nodes == ()
    assert cand.liftoff_nodes == (3, 4)
    assert _event_exec_node(cand, n_nodes=11, settling_nodes=1) == 5
    assert _support_after((2, 4, 5, 6, 7), cand) == (2, 5, 7)


def test_mixed_transition_uses_touchdown_before_liftoff():
    cand = _candidate(add=(0,), remove=(4, 6))
    assert cand.touchdown_nodes == (3,)
    assert cand.liftoff_nodes == (4, 5)
    assert _event_exec_node(cand, n_nodes=11, settling_nodes=1) == 6
    assert _support_after((2, 4, 5, 6, 7), cand) == (0, 2, 5, 7)


def test_urgent_release_is_ranked_before_nonurgent_release():
    remaining = {2: 5.0, 4: 1.0, 5: 9.0, 6: 1.0, 7: 9.0}
    urgent = _candidate(remove=(4, 6))
    nonurgent = _candidate(remove=(5, 7))
    assert _candidate_priority(urgent, remaining, 0.4) < _candidate_priority(
        nonurgent, remaining, 0.4
    )
