import numpy as np
from scipy.spatial.transform import Rotation

from lily_contact_planner.tasks import (
    ForwardRollTask,
    PiecewiseWorldTask,
    Pitch45ThenRoll45Task,
    PitchForwardTask,
    WorldMotionPhase,
    Yaw45Pitch145Roll145WorldTask,
)


def test_forward_roll_task_end_pose():
    task = ForwardRollTask(body_height_m=0.35, forward_m_per_deg=1/300)
    t, R = task.pose(720.0)
    assert np.allclose(t, [2.4, 0.0, 0.35])
    assert np.allclose(R, np.eye(3), atol=1e-12)


def test_pitch45_then_roll45_is_in_place_and_world_frame():
    task = Pitch45ThenRoll45Task(body_height_m=0.35)

    t_pitch, R_pitch = task.pose(45.0)
    expected_pitch = Rotation.from_euler("y", 45.0, degrees=True).as_matrix()
    assert np.allclose(t_pitch, [0.0, 0.0, 0.35])
    assert np.allclose(R_pitch, expected_pitch, atol=1e-12)

    t_final, R_final = task.pose(90.0)
    expected_roll = Rotation.from_euler("x", 45.0, degrees=True).as_matrix()
    assert np.allclose(t_final, [0.0, 0.0, 0.35])
    assert np.allclose(R_final, expected_roll @ expected_pitch, atol=1e-12)
    assert task.phase_boundaries_deg == (45.0, 90.0)


def test_yaw45_pitch145_roll145_world_translation_and_rotation():
    task = Yaw45Pitch145Roll145WorldTask()
    d = 145.0 / 300.0

    assert task.total_progress_deg == 335.0
    assert task.phase_boundaries_deg == (45.0, 190.0, 335.0)

    t_yaw, R_yaw = task.pose(45.0)
    expected_yaw = Rotation.from_euler("z", 45.0, degrees=True).as_matrix()
    assert np.allclose(t_yaw, [0.0, 0.0, 0.35])
    assert np.allclose(R_yaw, expected_yaw, atol=1e-12)

    t_pitch, R_pitch = task.pose(190.0)
    expected_pitch = Rotation.from_euler("y", 145.0, degrees=True).as_matrix()
    assert np.allclose(t_pitch, [d, 0.0, 0.35])
    assert np.allclose(R_pitch, expected_pitch @ expected_yaw, atol=1e-12)

    t_final, R_final = task.pose(335.0)
    expected_roll = Rotation.from_euler("x", -145.0, degrees=True).as_matrix()
    assert np.allclose(t_final, [d, d, 0.35])
    assert np.allclose(
        R_final,
        expected_roll @ expected_pitch @ expected_yaw,
        atol=1e-12,
    )


def test_pitch_forward_360_end_pose():
    task = PitchForwardTask()
    t, R = task.pose(360.0)

    assert task.total_progress_deg == 360.0
    assert task.phase_boundaries_deg == (360.0,)
    assert np.allclose(t, [1.2, 0.0, 0.35])
    assert np.allclose(R, np.eye(3), atol=1e-12)


def test_piecewise_world_task_left_multiplies_world_rotations():
    task = PiecewiseWorldTask(
        body_height_m=0.35,
        phases=(
            WorldMotionPhase(
                progress_deg=45.0,
                rotation_axis="z",
                rotation_deg_per_progress=1.0,
            ),
            WorldMotionPhase(
                progress_deg=30.0,
                rotation_axis="y",
                rotation_deg_per_progress=1.0,
                translation_world_m_per_progress=(1.0 / 300.0, 0.0, 0.0),
            ),
            WorldMotionPhase(
                progress_deg=20.0,
                rotation_axis="x",
                rotation_deg_per_progress=-1.0,
                translation_world_m_per_progress=(0.0, 1.0 / 300.0, 0.0),
            ),
        ),
    )

    assert task.total_progress_deg == 95.0
    assert task.phase_boundaries_deg == (45.0, 75.0, 95.0)

    t, R = task.pose(95.0)
    Rz = Rotation.from_euler("z", 45.0, degrees=True).as_matrix()
    Ry = Rotation.from_euler("y", 30.0, degrees=True).as_matrix()
    Rx = Rotation.from_euler("x", -20.0, degrees=True).as_matrix()

    assert np.allclose(t, [30.0 / 300.0, 20.0 / 300.0, 0.35])
    assert np.allclose(R, Rx @ Ry @ Rz, atol=1e-12)
