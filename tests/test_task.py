import numpy as np
from scipy.spatial.transform import Rotation

from lily_contact_planner.tasks import ForwardRollTask, Pitch45ThenRoll45Task


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
