"""Display-only trajectory reconstruction helpers.

The contact planner currently produces discrete contact events and numerical
keyframes.  This module reconstructs additional frames for visualization.
It does NOT turn those display frames into planner-certified states.

The most important convention is that a support switch is displayed as

    old support retained
    -> new foot touchdown
    -> old + new feet simultaneously supporting
    -> support transfer
    -> old foot liftoff

Thus an old support foot is never visually released before the newly added
support foot has touched the ground.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class DisplayFrame:
    progress: float
    body_t: np.ndarray
    body_R: np.ndarray
    joint_q: np.ndarray
    support_mask: np.ndarray
    note: str = ""


def support_mask(support: Iterable[int], n_legs: int = 8) -> np.ndarray:
    mask = np.zeros(n_legs, dtype=np.uint8)
    mask[list(support)] = 1
    return mask


def _interp_rotation(R0: np.ndarray, R1: np.ndarray, u: float) -> np.ndarray:
    rotations = Rotation.from_matrix(np.stack([R0, R1], axis=0))
    return Slerp([0.0, 1.0], rotations)([float(u)]).as_matrix()[0]


def _smoothstep(u: float) -> float:
    u = float(u)
    return u * u * (3.0 - 2.0 * u)


def interpolate_same_support(
    a0: float,
    t0: np.ndarray,
    R0: np.ndarray,
    q0: np.ndarray,
    a1: float,
    t1: np.ndarray,
    R1: np.ndarray,
    q1: np.ndarray,
    mask: np.ndarray,
    n_mid: int = 2,
) -> List[DisplayFrame]:
    """Insert display-only smooth frames when the support set is unchanged."""
    out: List[DisplayFrame] = []
    for j in range(1, n_mid + 1):
        raw = j / float(n_mid + 1)
        u = _smoothstep(raw)
        out.append(
            DisplayFrame(
                progress=(1.0 - u) * a0 + u * a1,
                body_t=(1.0 - u) * t0 + u * t1,
                body_R=_interp_rotation(R0, R1, u),
                joint_q=(1.0 - u) * q0 + u * q1,
                support_mask=np.asarray(mask, dtype=np.uint8).copy(),
                note="display interpolation; support unchanged",
            )
        )
    return out


def touchdown_first_switch_frames(
    progress: float,
    body_t: np.ndarray,
    body_R: np.ndarray,
    q_pre: np.ndarray,
    q_post: np.ndarray,
    support_before: Sequence[int],
    support_after: Sequence[int],
    added_legs: Sequence[int],
    removed_legs: Sequence[int],
    touchdown_fraction: float = 0.30,
    transfer_fraction: float = 0.65,
) -> List[DisplayFrame]:
    """Construct the display sequence for one discrete contact switch.

    This routine intentionally orders the display transition as

        pre-switch
        -> touchdown approach while old support is retained
        -> touchdown complete with union(old,new) support
        -> support transfer with union support
        -> liftoff of removed legs

    q_pre and q_post should be the numerical states immediately before and
    after the discrete contact event at the same task progress.  Intermediate
    configurations are visualization-only interpolants and are not certified
    by the Level-1 checker unless checked separately.
    """
    before = tuple(int(x) for x in support_before)
    after = tuple(int(x) for x in support_after)
    added = tuple(int(x) for x in added_legs)
    removed = tuple(int(x) for x in removed_legs)
    union = tuple(sorted(set(before) | set(added)))

    frames: List[DisplayFrame] = []
    frames.append(
        DisplayFrame(
            progress=float(progress),
            body_t=np.asarray(body_t).copy(),
            body_R=np.asarray(body_R).copy(),
            joint_q=np.asarray(q_pre).copy(),
            support_mask=support_mask(before),
            note=f"pre-switch add={list(added)} remove={list(removed)}",
        )
    )

    q_touch_approach = np.asarray(q_pre).copy()
    for leg in added:
        q_touch_approach[leg] = (
            (1.0 - touchdown_fraction) * q_pre[leg]
            + touchdown_fraction * q_post[leg]
        )
    frames.append(
        DisplayFrame(
            progress=float(progress) + 0.12,
            body_t=np.asarray(body_t).copy(),
            body_R=np.asarray(body_R).copy(),
            joint_q=q_touch_approach,
            support_mask=support_mask(before),
            note=f"touchdown approach add={list(added)}; old support retained",
        )
    )

    q_touch = np.asarray(q_pre).copy()
    for leg in added:
        q_touch[leg] = q_post[leg]
    frames.append(
        DisplayFrame(
            progress=float(progress) + 0.24,
            body_t=np.asarray(body_t).copy(),
            body_R=np.asarray(body_R).copy(),
            joint_q=q_touch,
            support_mask=support_mask(union),
            note="touchdown complete; old + new support simultaneously active",
        )
    )

    q_transfer = (1.0 - transfer_fraction) * q_touch + transfer_fraction * q_post
    frames.append(
        DisplayFrame(
            progress=float(progress) + 0.36,
            body_t=np.asarray(body_t).copy(),
            body_R=np.asarray(body_R).copy(),
            joint_q=q_transfer,
            support_mask=support_mask(union),
            note="support transfer; new feet already down",
        )
    )

    frames.append(
        DisplayFrame(
            progress=float(progress) + 0.48,
            body_t=np.asarray(body_t).copy(),
            body_R=np.asarray(body_R).copy(),
            joint_q=np.asarray(q_post).copy(),
            support_mask=support_mask(after),
            note=f"liftoff old={list(removed)} after touchdown",
        )
    )
    return frames
