"""Analytic single-leg position IK for Lily's radial-1R + planar-2R leg."""

import numpy as np


def _wrap_q1_branches(base_q1, qmin, qmax):
    out = []
    two_pi = 2.0 * np.pi
    kmin = int(np.floor((qmin - base_q1) / two_pi)) - 1
    kmax = int(np.ceil((qmax - base_q1) / two_pi)) + 1
    for k in range(kmin, kmax + 1):
        q1 = base_q1 + two_pi * k
        if q1 >= qmin - 1e-10 and q1 <= qmax + 1e-10:
            out.append(float(q1))
    return out


def analytic_leg_ik_world(kin, body_pos, body_R, leg_index, target_world,
                          q_reference=None, residual_tol=1e-7):
    body_pos = np.asarray(body_pos, dtype=float)
    body_R = np.asarray(body_R, dtype=float)
    target_world = np.asarray(target_world, dtype=float)
    target_body = body_R.T.dot(target_world - body_pos)

    root = kin.root_body(leg_index)
    d = target_body - root
    er = kin.radial_axis_body(leg_index)
    ephi = kin.tangent_reference_body(leg_index)
    eperp = np.cross(er, ephi)
    dphi = float(np.dot(ephi, d))
    dperp = float(np.dot(eperp, d))

    if kin.sigmas[leg_index][2] > 0:
        delta = kin.delta_top
        eps = kin.eps_top
    else:
        delta = kin.delta_bottom
        eps = kin.eps_bottom

    if np.hypot(dphi, dperp) <= 1e-10 and q_reference is not None:
        theta0 = float(delta + eps * np.asarray(q_reference, dtype=float)[0])
    else:
        theta0 = float(np.arctan2(-dphi, dperp))

    rho2 = float(np.dot(d, d))
    c3 = (rho2 - kin.L2 * kin.L2 - kin.L3 * kin.L3) / (2.0 * kin.L2 * kin.L3)
    if c3 < -1.0 - 1e-10 or c3 > 1.0 + 1e-10:
        return []
    c3 = float(np.clip(c3, -1.0, 1.0))

    branches = []
    for theta in (theta0, theta0 + np.pi):
        base_q1 = (theta - delta) / eps
        for q1 in _wrap_q1_branches(base_q1, kin.q_min[0], kin.q_max[0]):
            n = kin.pitch_axis_body(leg_index, q1)
            y_axis = np.cross(n, er)
            x = float(np.dot(er, d))
            y = float(np.dot(y_axis, d))
            alpha = float(np.arccos(c3))
            for q3 in (alpha, -alpha):
                q2 = float(
                    np.arctan2(y, x)
                    - np.arctan2(
                        kin.L3 * np.sin(q3),
                        kin.L2 + kin.L3 * np.cos(q3),
                    )
                )
                q = np.array([q1, q2, q3], dtype=float)
                if np.any(q < kin.q_min - 1e-9) or np.any(q > kin.q_max + 1e-9):
                    continue
                foot = kin.foot_world(body_pos, body_R, leg_index, q)
                if np.linalg.norm(foot - target_world) > residual_tol:
                    continue
                if not any(np.linalg.norm(q - old) < 1e-8 for old in branches):
                    branches.append(q)

    if q_reference is not None and branches:
        qref = np.asarray(q_reference, dtype=float)
        branches.sort(key=lambda q: float(np.linalg.norm(q - qref)))
    return branches
