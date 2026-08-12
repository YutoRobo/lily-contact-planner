
import numpy as np
from itertools import product


def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 0.0:
        raise ValueError("zero-length vector")
    return v / n


def _rodrigues(v, axis, angle):
    """Rotate vector v about 'axis' by angle [rad]."""
    v = np.asarray(v, dtype=float)
    k = _normalize(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)


def _skew(v):
    x, y, z = np.asarray(v, dtype=float)
    return np.array([
        [0.0, -z,  y],
        [z,   0.0, -x],
        [-y,   x,  0.0],
    ])


class LilyKinematics:
    """
    URDF-free kinematic model for the current Lily assumptions.

    Confirmed geometry
    ------------------
    - Cube body, half side length = a
    - 8 leg roots at the 8 vertices
    - 3 revolute joints / leg
    - L1 = 0, so J1 and J2 are co-located at the root
    - J1 axis is radial from body center to the root
    - J2 and J3 axes are parallel
    - q=[0,0,0] => L2 and L3 are collinear and radial outward

    Parameters intentionally left open
    ----------------------------------
    delta_top, delta_bottom:
        Joint-1 zero-phase offsets [rad]
    eps_top, eps_bottom:
        Joint-1 positive-direction signs (+1/-1)

    Leg ordering
    ------------
    self.sigmas is the Cartesian product:
      (-1,-1,-1), (-1,-1,+1), (-1,+1,-1), (-1,+1,+1),
      (+1,-1,-1), (+1,-1,+1), (+1,+1,-1), (+1,+1,+1)
    """

    def __init__(
        self,
        a=1.0,
        L2=0.85,
        L3=0.75,
        delta_top=0.0,
        delta_bottom=0.0,
        eps_top=+1.0,
        eps_bottom=-1.0,
        q_min_deg=(-360.0, -95.0, -150.0),
        q_max_deg=(+360.0, +95.0, +150.0),
    ):
        self.a = float(a)
        self.L2 = float(L2)
        self.L3 = float(L3)

        self.delta_top = float(delta_top)
        self.delta_bottom = float(delta_bottom)
        self.eps_top = float(eps_top)
        self.eps_bottom = float(eps_bottom)

        self.q_min = np.deg2rad(np.asarray(q_min_deg, dtype=float))
        self.q_max = np.deg2rad(np.asarray(q_max_deg, dtype=float))

        self.sigmas = list(product([-1, 1], repeat=3))

    @property
    def n_legs(self):
        return 8

    def root_body(self, leg_index):
        return self.a * np.asarray(self.sigmas[leg_index], dtype=float)

    def radial_axis_body(self, leg_index):
        return _normalize(np.asarray(self.sigmas[leg_index], dtype=float))

    def tangent_reference_body(self, leg_index):
        """
        A deterministic reference axis perpendicular to the radial J1 axis.
        This only defines the zero-phase reference of J2/J3.
        """
        er = self.radial_axis_body(leg_index)
        ez = np.array([0.0, 0.0, 1.0])
        return _normalize(np.cross(ez, er))

    def joint1_angle_internal(self, leg_index, q1):
        sz = self.sigmas[leg_index][2]
        if sz > 0:
            return self.delta_top + self.eps_top * q1
        return self.delta_bottom + self.eps_bottom * q1

    def joint1_sign(self, leg_index):
        return self.eps_top if self.sigmas[leg_index][2] > 0 else self.eps_bottom

    def pitch_axis_body(self, leg_index, q1):
        """
        Common J2/J3 axis in the body frame.
        """
        er = self.radial_axis_body(leg_index)
        ephi = self.tangent_reference_body(leg_index)
        theta1 = self.joint1_angle_internal(leg_index, q1)
        return _rodrigues(ephi, er, theta1)

    def leg_points_body(self, leg_index, q_leg):
        """
        Returns root, J3 position (elbow), foot in body frame.
        q_leg = [q1,q2,q3] [rad]
        """
        q1, q2, q3 = np.asarray(q_leg, dtype=float)

        root = self.root_body(leg_index)
        er = self.radial_axis_body(leg_index)
        n = self.pitch_axis_body(leg_index, q1)

        u2 = _rodrigues(er, n, q2)
        u3 = _rodrigues(er, n, q2 + q3)

        elbow = root + self.L2 * u2
        foot = elbow + self.L3 * u3
        return root, elbow, foot

    def joint_axes_body(self, leg_index, q_leg):
        """
        Returns (J1_axis, J2_axis, J3_axis) in body frame.
        J2 and J3 are parallel by construction.
        """
        q1 = float(q_leg[0])
        er = self.radial_axis_body(leg_index)
        n = self.pitch_axis_body(leg_index, q1)
        return er, n, n

    def leg_jacobian_body(self, leg_index, q_leg):
        """
        Analytic translational Jacobian of the foot wrt [q1,q2,q3],
        expressed in the body frame.

        dp_foot^B / dq = J_leg^B
        """
        q1, q2, q3 = np.asarray(q_leg, dtype=float)

        root, elbow, foot = self.leg_points_body(leg_index, q_leg)
        er, n, _ = self.joint_axes_body(leg_index, q_leg)

        eps = self.joint1_sign(leg_index)

        J1 = eps * np.cross(er, foot - root)
        J2 = np.cross(n, foot - root)
        J3 = np.cross(n, foot - elbow)

        return np.column_stack((J1, J2, J3))

    def world_points(self, body_pos, body_R, leg_index, q_leg):
        """
        Returns root, elbow, foot in world coordinates.
        """
        body_pos = np.asarray(body_pos, dtype=float)
        body_R = np.asarray(body_R, dtype=float)

        root_B, elbow_B, foot_B = self.leg_points_body(leg_index, q_leg)
        return (
            body_pos + body_R @ root_B,
            body_pos + body_R @ elbow_B,
            body_pos + body_R @ foot_B,
        )

    def world_joint_axes(self, body_R, leg_index, q_leg):
        body_R = np.asarray(body_R, dtype=float)
        return tuple(body_R @ a for a in self.joint_axes_body(leg_index, q_leg))

    def leg_jacobian_world(self, body_R, leg_index, q_leg):
        return np.asarray(body_R, dtype=float) @ self.leg_jacobian_body(leg_index, q_leg)

    def foot_world(self, body_pos, body_R, leg_index, q_leg):
        return self.world_points(body_pos, body_R, leg_index, q_leg)[2]

    def whole_body_foot_jacobian(self, body_R, leg_index, q_leg):
        """
        Jacobian for world foot velocity wrt:
            [v_body_world(3), omega_body_world(3), qdot_leg(3)]

        p_dot = J_whole [v, omega, qdot]^T

        Convention:
          omega is expressed in the world frame.
        """
        body_R = np.asarray(body_R, dtype=float)
        _, _, foot_B = self.leg_points_body(leg_index, q_leg)
        r_world = body_R @ foot_B

        J_body_translation = np.eye(3)
        J_body_rotation = -_skew(r_world)
        J_leg = self.leg_jacobian_world(body_R, leg_index, q_leg)

        return np.hstack((J_body_translation, J_body_rotation, J_leg))

    def check_joint_limits(self, q):
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.q_min) and np.all(q <= self.q_max))

    def numerical_jacobian_error(self, leg_index, q_leg, h=1e-7):
        """
        Finite-difference check of the analytic leg Jacobian.
        Returns max absolute element error.
        """
        q_leg = np.asarray(q_leg, dtype=float)
        J_num = np.zeros((3, 3))

        for j in range(3):
            qp = q_leg.copy()
            qm = q_leg.copy()
            qp[j] += h
            qm[j] -= h
            fp = self.leg_points_body(leg_index, qp)[2]
            fm = self.leg_points_body(leg_index, qm)[2]
            J_num[:, j] = (fp - fm) / (2.0 * h)

        J_ana = self.leg_jacobian_body(leg_index, q_leg)
        return float(np.max(np.abs(J_ana - J_num)))

    def lower_leg_indices(self):
        return [i for i, s in enumerate(self.sigmas) if s[2] < 0]

    def upper_leg_indices(self):
        return [i for i, s in enumerate(self.sigmas) if s[2] > 0]
