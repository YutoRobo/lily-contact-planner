import numpy as np

from lily_contact_planner import LilyKinematics


def test_analytic_jacobian_matches_finite_difference():
    kin = LilyKinematics(a=0.15, L2=0.30, L3=0.30)
    q = np.deg2rad([35.0, -25.0, 70.0])
    R = np.eye(3)
    J = kin.leg_jacobian_world(R, 0, q)

    eps = 1e-7
    Jfd = np.zeros((3, 3))
    for j in range(3):
        qp = q.copy(); qm = q.copy()
        qp[j] += eps; qm[j] -= eps
        fp = kin.foot_world(np.zeros(3), R, 0, qp)
        fm = kin.foot_world(np.zeros(3), R, 0, qm)
        Jfd[:, j] = (fp - fm) / (2.0 * eps)

    assert np.max(np.abs(J - Jfd)) < 1e-5
