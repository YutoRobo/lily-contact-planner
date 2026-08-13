import numpy as np

from lily_contact_planner.collision import (
    CapsuleSegment,
    evaluate_capsules,
    segment_aabb_distance,
    segment_segment_distance,
)


def test_segment_segment_distance_parallel():
    a0 = np.array([0.0, 0.0, 0.0])
    a1 = np.array([1.0, 0.0, 0.0])
    b0 = np.array([0.0, 0.05, 0.0])
    b1 = np.array([1.0, 0.05, 0.0])
    assert abs(segment_segment_distance(a0, a1, b0, b1) - 0.05) < 1e-12


def test_segment_aabb_distance():
    # Cube [-0.5,0.5]^3 and segment parallel to x at y=0.8.
    p0 = np.array([-1.0, 0.8, 0.0])
    p1 = np.array([1.0, 0.8, 0.0])
    assert abs(segment_aabb_distance(p0, p1, 0.5) - 0.3) < 1e-12


def test_all_interleg_link_combinations_are_checked():
    # The closest pair is leg0 L3 vs leg1 L2.  If mixed link pairs were
    # accidentally omitted this collision would be missed.
    capsules = [
        CapsuleSegment(0, 0, np.array([0., 0., 1.]), np.array([1., 0., 1.]), 0.02),
        CapsuleSegment(0, 1, np.array([0., 0., 0.]), np.array([1., 0., 0.]), 0.03),
        CapsuleSegment(1, 0, np.array([0., 0.04, 0.]), np.array([1., 0.04, 0.]), 0.03),
        CapsuleSegment(1, 1, np.array([0., 1., 1.]), np.array([1., 1., 1.]), 0.02),
    ]
    rep = evaluate_capsules(
        capsules,
        body_t=np.array([10., 10., 10.]),
        body_R=np.eye(3),
        body_half_extent_m=0.15,
    )
    assert not rep.self_collision_ok
    assert rep.worst_capsule_pair == (0, 1, 1, 0)
    assert abs(rep.min_capsule_clearance_m + 0.02) < 1e-12


def test_same_leg_adjacent_joint_is_exempted():
    # Adjacent L2/L3 share their endpoint by design and must not make every
    # valid leg permanently collide with itself.
    capsules = [
        CapsuleSegment(0, 0, np.array([0., 0., 0.]), np.array([1., 0., 0.]), 0.02),
        CapsuleSegment(0, 1, np.array([1., 0., 0.]), np.array([1., 1., 0.]), 0.02),
    ]
    rep = evaluate_capsules(
        capsules,
        body_t=np.array([10., 10., 10.]),
        body_R=np.eye(3),
        body_half_extent_m=0.15,
    )
    assert rep.self_collision_ok


def test_capsule_body_collision_uses_radius():
    # Centerline stays 30 mm away from cube surface; a 40 mm capsule overlaps.
    capsules = [
        CapsuleSegment(
            0, 1,
            np.array([-0.5, 0.18, 0.0]),
            np.array([0.5, 0.18, 0.0]),
            0.04,
        )
    ]
    rep = evaluate_capsules(
        capsules,
        body_t=np.zeros(3),
        body_R=np.eye(3),
        body_half_extent_m=0.15,
    )
    assert not rep.link_body_collision_ok
    assert abs(rep.min_body_clearance_m + 0.01) < 1e-12
