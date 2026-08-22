import math

from lily_contact_planner.experimental_cooperative_transition import (
    CooperativeTransitionNLP,
)
from lily_contact_planner.experimental_cooperative_transition_numerics import (
    NumericallyConsistentCooperativeTransitionNLP,
    _required_liftoff_clearance,
)


def test_liftoff_seed_profile_reaches_clearance_at_exec_node():
    assert _required_liftoff_clearance(3, 3, 4, 0.02) == 0.0
    assert math.isclose(
        _required_liftoff_clearance(4, 3, 4, 0.02), 0.02,
        abs_tol=1e-12,
    )
    assert math.isclose(
        _required_liftoff_clearance(5, 3, 4, 0.02), 0.02,
        abs_tol=1e-12,
    )


def test_liftoff_seed_profile_ramps_when_settling_window_is_longer():
    assert math.isclose(
        _required_liftoff_clearance(4, 3, 5, 0.02), 0.01,
        abs_tol=1e-12,
    )
    assert math.isclose(
        _required_liftoff_clearance(5, 3, 5, 0.02), 0.02,
        abs_tol=1e-12,
    )


def test_numerical_patch_is_only_a_solver_subclass():
    assert issubclass(
        NumericallyConsistentCooperativeTransitionNLP,
        CooperativeTransitionNLP,
    )
