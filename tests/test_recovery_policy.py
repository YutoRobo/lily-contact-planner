from lily_contact_planner.recovery_policy import RecoveryKind, recovery_stages


def test_v006_recovery_order_is_conservative():
    stages = tuple(recovery_stages())
    assert [stage.kind for stage in stages] == [
        RecoveryKind.ONE_TO_ONE,
        RecoveryKind.MULTI_CONTACT,
        RecoveryKind.STATIC_RECONFIGURATION,
    ]


def test_only_static_stage_allows_zero_body_progress():
    stages = tuple(recovery_stages())
    assert stages[0].require_body_progress is True
    assert stages[1].require_body_progress is True
    assert stages[2].require_body_progress is False


def test_contact_edit_scope_does_not_expand_beyond_v006_consolidation():
    stages = tuple(recovery_stages())
    assert (stages[0].max_add, stages[0].max_remove) == (1, 1)
    assert (stages[1].max_add, stages[1].max_remove) == (2, 2)
    assert (stages[2].max_add, stages[2].max_remove) == (2, 2)
