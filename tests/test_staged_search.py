from lily_contact_planner.planner_search import DfsSearchMixin
from lily_contact_planner.recovery_policy import RecoveryKind, recovery_stages
from lily_contact_planner.v004_success_seed import V004SuccessfulSeedMixin


def _plan(nadd, nrem):
    add = {i: (None, None) for i in range(nadd)}
    rem = list(range(nrem))
    return (0.0, 1.0, add, rem, (), {}, None)


def test_recovery_stage_filtering_is_conservative():
    stages = {stage.kind: stage for stage in recovery_stages()}

    one = stages[RecoveryKind.ONE_TO_ONE]
    multi = stages[RecoveryKind.MULTI_CONTACT]
    static = stages[RecoveryKind.STATIC_RECONFIGURATION]

    assert DfsSearchMixin._plan_matches_stage(_plan(1, 1), one)
    assert not DfsSearchMixin._plan_matches_stage(_plan(2, 1), one)

    assert DfsSearchMixin._plan_matches_stage(_plan(2, 1), multi)
    assert DfsSearchMixin._plan_matches_stage(_plan(1, 2), multi)
    assert not DfsSearchMixin._plan_matches_stage(_plan(1, 1), multi)
    assert not DfsSearchMixin._plan_matches_stage(_plan(3, 1), multi)

    assert DfsSearchMixin._plan_matches_stage(_plan(1, 1), static)
    assert DfsSearchMixin._plan_matches_stage(_plan(2, 2), static)
    assert not DfsSearchMixin._plan_matches_stage(_plan(3, 1), static)


def test_liftoff_priority_prefers_leg_with_less_remaining_support_range():
    remaining = {0: 12.0, 2: 1.0, 4: 4.0, 6: 20.0}
    key = V004SuccessfulSeedMixin._liftoff_priority_key

    assert key((2,), remaining) < key((4,), remaining)
    assert key((4,), remaining) < key((0,), remaining)
    assert key((0,), remaining) < key((6,), remaining)


def test_liftoff_priority_for_two_leg_release_is_lexicographic_by_urgency():
    remaining = {0: 12.0, 2: 1.0, 4: 4.0, 6: 20.0}
    key = V004SuccessfulSeedMixin._liftoff_priority_key

    assert key((2, 6), remaining) < key((4, 6), remaining)
    assert key((2, 4), remaining) < key((2, 6), remaining)
