"""Geometry helpers used by the v0.0.2-v0.0.5 finite-horizon NLPs."""

import numpy as np

from .collision import segment_aabb_distance, segment_segment_distance, trim_segment_start


def state_points(kin, body_pos, body_R, q_frame):
    roots, elbows, feet = [], [], []
    for leg in range(kin.n_legs):
        root, elbow, foot = kin.world_points(body_pos, body_R, leg, q_frame[leg])
        roots.append(root)
        elbows.append(elbow)
        feet.append(foot)
    return np.asarray(roots), np.asarray(elbows), np.asarray(feet)


def capsule_segments(kin, roots, elbows, feet, settings):
    out = []
    for leg in range(kin.n_legs):
        out.append((leg, 0, roots[leg], elbows[leg], float(settings.l2_radius_m)))
        out.append((leg, 1, elbows[leg], feet[leg], float(settings.l3_radius_m)))
    return out


def geometry_inequalities(kin, body_pos, body_R, q_frame, support_now, settings):
    """Return all geometry inequalities in g(x) >= 0 form for one NLP node."""
    roots, elbows, feet = state_points(kin, body_pos, body_R, q_frame)
    out = []

    for leg in range(kin.n_legs):
        if leg not in support_now:
            out.append(float(feet[leg, 2] - settings.swing_ground_margin_m))
        out.append(float(min(roots[leg, 2], elbows[leg, 2])))
        out.append(float(min(elbows[leg, 2], feet[leg, 2])))

    body_min_z = float(body_pos[2] - kin.a * np.sum(np.abs(body_R[2, :])))
    out.append(body_min_z)

    segments = capsule_segments(kin, roots, elbows, feet, settings)
    margin = float(settings.collision_margin_m)

    for ia in range(len(segments)):
        A = segments[ia]
        for ib in range(ia + 1, len(segments)):
            B = segments[ib]
            if A[0] == B[0]:
                continue
            center_dist = segment_segment_distance(A[2], A[3], B[2], B[3])
            out.append(float(center_dist - (A[4] + B[4] + margin)))

    for _, link, p0w, p1w, radius in segments:
        if link == 0 and settings.root_ignore_m > 0.0:
            p0w, p1w = trim_segment_start(p0w, p1w, settings.root_ignore_m)
        p0b = body_R.T.dot(p0w - body_pos)
        p1b = body_R.T.dot(p1w - body_pos)
        center_dist = segment_aabb_distance(p0b, p1b, kin.a)
        out.append(float(center_dist - (radius + margin)))

    return np.asarray(out, dtype=float), (roots, elbows, feet)
