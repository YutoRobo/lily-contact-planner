"""Finite-thickness collision geometry for Lily.

The current Lily kinematic model has two physical link segments per leg:
root->elbow (L2) and elbow->foot (L3).  This module models every such segment
as a capsule and provides exact segment-segment clearance plus segment-vs-box
clearance for the body cube.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def segment_segment_distance(p1, q1, p2, q2):
    """Minimum Euclidean distance between two closed 3-D line segments."""
    p1 = np.asarray(p1, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    eps = 1e-14
    D = a * c - b * b

    if a <= eps and c <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        t = np.clip(e / c, 0.0, 1.0)
        return float(np.linalg.norm(p1 - (p2 + t * v)))
    if c <= eps:
        s = np.clip(-d / a, 0.0, 1.0)
        return float(np.linalg.norm((p1 + s * u) - p2))

    sN, sD = 0.0, D
    tN, tD = 0.0, D
    if D < eps:
        sN, sD = 0.0, 1.0
        tN, tD = e, c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN, tD = e, c
        elif sN > sD:
            sN = sD
            tN, tD = e + b, c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN, sD = -d, a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN, sD = (-d + b), a

    sc = 0.0 if abs(sN) < eps else sN / sD
    tc = 0.0 if abs(tN) < eps else tN / tD
    return float(np.linalg.norm(w + sc * u - tc * v))


def _point_aabb_sq_distance(p, half_extent):
    p = np.asarray(p, dtype=float)
    a = float(half_extent)
    excess = np.maximum(np.abs(p) - a, 0.0)
    return float(np.dot(excess, excess))


def segment_aabb_distance(p0, p1, half_extent):
    """Exact minimum distance between a segment and axis-aligned cube.

    The squared point-to-box distance along a segment is convex piecewise
    quadratic.  Breakpoints occur only where a coordinate crosses +/-a, so
    each piece can be minimized analytically.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    a = float(half_extent)

    breaks = [0.0, 1.0]
    eps = 1e-14
    for j in range(3):
        if abs(d[j]) <= eps:
            continue
        for bound in (-a, a):
            t = (bound - p0[j]) / d[j]
            if 0.0 < t < 1.0:
                breaks.append(float(t))
    breaks = sorted(set(breaks))

    best_sq = min(_point_aabb_sq_distance(p0, a), _point_aabb_sq_distance(p1, a))
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        mid = 0.5 * (lo + hi)
        pm = p0 + mid * d
        A = 0.0
        B = 0.0
        for j in range(3):
            if pm[j] < -a:
                bound = -a
            elif pm[j] > a:
                bound = a
            else:
                continue
            alpha = p0[j] - bound
            beta = d[j]
            A += beta * beta
            B += alpha * beta

        candidates = [lo, hi]
        if A > eps:
            candidates.append(float(np.clip(-B / A, lo, hi)))
        for t in candidates:
            best_sq = min(best_sq, _point_aabb_sq_distance(p0 + t * d, a))

    return float(np.sqrt(max(best_sq, 0.0)))


def trim_segment_start(p0, p1, trim_m):
    """Trim a physical length from p0; used only for intentional body attachment."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L <= 1e-14:
        return p1.copy(), p1.copy()
    h = min(max(float(trim_m), 0.0) / L, 1.0)
    return p0 + h * v, p1.copy()


@dataclass(frozen=True)
class CapsuleSegment:
    leg: int
    link: int  # 0=L2/root->elbow, 1=L3/elbow->foot
    p0: np.ndarray
    p1: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class CollisionSample:
    self_collision_ok: bool
    link_body_collision_ok: bool
    min_capsule_clearance_m: float
    min_body_clearance_m: float
    worst_capsule_pair: Optional[Tuple[int, int, int, int]]
    worst_body_link: Optional[Tuple[int, int]]


def build_capsules(roots, elbows, feet, radii_m):
    r2, r3 = [float(x) for x in radii_m]
    out: List[CapsuleSegment] = []
    for leg in range(len(roots)):
        out.append(CapsuleSegment(leg, 0, roots[leg], elbows[leg], r2))
        out.append(CapsuleSegment(leg, 1, elbows[leg], feet[leg], r3))
    return out


def evaluate_capsules(
    capsules: Sequence[CapsuleSegment],
    body_t,
    body_R,
    body_half_extent_m,
    margin_m=0.0,
    root_attachment_ignore_m=0.0,
):
    """Evaluate all required capsule pairs and every capsule against body.

    Pair rule:
    - every pair of different legs is checked (L2-L2, L2-L3, L3-L2, L3-L3)
    - same-leg adjacent L2/L3 is omitted because they intentionally share J3
      (the current model contains no non-adjacent same-leg pair)

    Body rule:
    - every capsule is checked against the body cube
    - only the proximal part of L2 may be ignored to represent its intentional
      mechanical attachment at the root; L3 is never exempted.
    """
    margin = float(margin_m)
    min_clear = np.inf
    worst_pair = None
    self_ok = True

    for ia in range(len(capsules)):
        A = capsules[ia]
        for ib in range(ia + 1, len(capsules)):
            B = capsules[ib]
            if A.leg == B.leg:
                # In the current 2-link model this is exactly the adjacent
                # L2/L3 pair sharing J3, which is an intentional joint.
                continue
            center_dist = segment_segment_distance(A.p0, A.p1, B.p0, B.p1)
            clearance = center_dist - (A.radius_m + B.radius_m + margin)
            if clearance < min_clear:
                min_clear = clearance
                worst_pair = (A.leg, A.link, B.leg, B.link)
            if clearance < 0.0:
                self_ok = False

    t = np.asarray(body_t, dtype=float)
    R = np.asarray(body_R, dtype=float)
    min_body = np.inf
    worst_body = None
    body_ok = True
    for C in capsules:
        p0w, p1w = C.p0, C.p1
        if C.link == 0 and root_attachment_ignore_m > 0.0:
            p0w, p1w = trim_segment_start(p0w, p1w, root_attachment_ignore_m)
        p0b = R.T @ (p0w - t)
        p1b = R.T @ (p1w - t)
        center_dist = segment_aabb_distance(p0b, p1b, body_half_extent_m)
        clearance = center_dist - (C.radius_m + margin)
        if clearance < min_body:
            min_body = clearance
            worst_body = (C.leg, C.link)
        if clearance < 0.0:
            body_ok = False

    return CollisionSample(
        self_collision_ok=self_ok,
        link_body_collision_ok=body_ok,
        min_capsule_clearance_m=float(min_clear),
        min_body_clearance_m=float(min_body),
        worst_capsule_pair=worst_pair,
        worst_body_link=worst_body,
    )
