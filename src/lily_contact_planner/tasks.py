from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ForwardRollTask:
    """Task path used by the original Level-1 proof of concept.

    The body geometric center stays at a fixed height, translates along +x,
    and rolls about +x. The contact planner does not receive any prescribed
    switch angles or contact sequence.
    """

    body_height_m: float = 0.35
    forward_m_per_deg: float = 1.0 / 300.0

    def pose(self, roll_deg: float):
        t = np.array(
            [self.forward_m_per_deg * float(roll_deg), 0.0, self.body_height_m],
            dtype=float,
        )
        R = Rotation.from_euler("x", float(roll_deg), degrees=True).as_matrix()
        return t, R


@dataclass(frozen=True)
class Pitch45ThenRoll45Task:
    """In-place world pitch +45 deg followed by world roll +45 deg.

    This is the regression task used for the successful Chat v0.0.6 baseline.
    The scalar planner progress ``s_deg`` runs from 0 to 90 deg. Translation
    is intentionally zero. During the second phase, world-frame roll is
    left-multiplied onto the completed pitch orientation.

    ``body_height_m`` is temporarily fixed to the archived successful v0.0.6
    regression height. This constant is a reproduction aid only and must be
    removed when the task is generalized after baseline verification.
    """

    body_height_m: float = 0.524575783
    pitch_deg: float = 45.0
    roll_deg: float = 45.0

    @property
    def total_progress_deg(self):
        return float(self.pitch_deg + self.roll_deg)

    def pose(self, s_deg: float):
        s = float(np.clip(s_deg, 0.0, self.total_progress_deg))
        t = np.array([0.0, 0.0, self.body_height_m], dtype=float)
        if s <= self.pitch_deg:
            R = Rotation.from_euler("y", s, degrees=True).as_matrix()
            return t, R

        r = s - self.pitch_deg
        Ry = Rotation.from_euler("y", self.pitch_deg, degrees=True).as_matrix()
        Rx = Rotation.from_euler("x", r, degrees=True).as_matrix()
        return t, Rx @ Ry


@dataclass(frozen=True)
class YawPitchRollWorldTask:
    """World-frame task used by the 2026-08-13 multi-axis experiment.

    Progress ``s_deg`` is split into three phases:

    1. 0..45 deg: in-place +yaw about world +z.
    2. 45..525 deg: +x translation while applying +pitch about world +y.
    3. 525..1005 deg: +y translation while applying -roll about world +x.

    New world-frame rotations are left-multiplied. Thus the pitch and roll
    directions do not rotate with the robot body after the preceding phase.
    """

    body_height_m: float = 0.35
    forward_m_per_deg: float = 1.0 / 300.0
    yaw_deg: float = 45.0
    pitch_deg: float = 480.0
    roll_deg: float = 480.0

    def pose(self, s_deg: float):
        s = float(s_deg)
        yaw_end = self.yaw_deg
        pitch_end = yaw_end + self.pitch_deg

        Rz_yaw = Rotation.from_euler("z", self.yaw_deg, degrees=True).as_matrix()

        if s <= yaw_end:
            t = np.array([0.0, 0.0, self.body_height_m], dtype=float)
            R = Rotation.from_euler("z", s, degrees=True).as_matrix()
            return t, R

        if s <= pitch_end:
            p = s - yaw_end
            t = np.array(
                [self.forward_m_per_deg * p, 0.0, self.body_height_m],
                dtype=float,
            )
            Ry = Rotation.from_euler("y", p, degrees=True).as_matrix()
            return t, Ry @ Rz_yaw

        r = s - pitch_end
        t = np.array(
            [
                self.forward_m_per_deg * self.pitch_deg,
                self.forward_m_per_deg * r,
                self.body_height_m,
            ],
            dtype=float,
        )
        Ry_pitch = Rotation.from_euler(
            "y", self.pitch_deg, degrees=True
        ).as_matrix()
        Rx_roll = Rotation.from_euler("x", -r, degrees=True).as_matrix()
        return t, Rx_roll @ Ry_pitch @ Rz_yaw
