import numpy as np
from lily_contact_planner.tasks import ForwardRollTask


def test_forward_roll_task_end_pose():
    task = ForwardRollTask(body_height_m=0.35, forward_m_per_deg=1/300)
    t, R = task.pose(720.0)
    assert np.allclose(t, [2.4, 0.0, 0.35])
    assert np.allclose(R, np.eye(3), atol=1e-12)
