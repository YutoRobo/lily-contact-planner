import numpy as np

from lily_contact_planner.visualization import touchdown_first_switch_frames


def test_touchdown_is_displayed_before_liftoff():
    q_pre = np.zeros((8, 3), dtype=float)
    q_post = np.ones((8, 3), dtype=float)
    frames = touchdown_first_switch_frames(
        progress=59.0,
        body_t=np.array([0.0, 0.0, 0.35]),
        body_R=np.eye(3),
        q_pre=q_pre,
        q_post=q_post,
        support_before=[0, 2, 4, 6],
        support_after=[0, 1, 6],
        added_legs=[1],
        removed_legs=[2, 4],
    )

    masks = [set(np.flatnonzero(frame.support_mask)) for frame in frames]

    # Old supports are retained through touchdown approach.
    assert masks[0] == {0, 2, 4, 6}
    assert masks[1] == {0, 2, 4, 6}

    # New support is added before any old support is removed.
    assert masks[2] == {0, 1, 2, 4, 6}
    assert masks[3] == {0, 1, 2, 4, 6}

    # Only after dual-support / transfer do removed legs leave support.
    assert masks[4] == {0, 1, 6}
