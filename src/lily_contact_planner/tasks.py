from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ForwardRollTask:
    """Task path used by the current Level-1 proof of concept.

    The body geometric center stays at a fixed height, translates along +x,
    and rolls about +x.  The contact planner does not receive any prescribed
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
