
import numpy as np
from dataclasses import dataclass, asdict
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError
except ImportError:
    # Compatibility with older SciPy versions where QhullError was not
    # re-exported from scipy.spatial.
    from scipy.spatial.qhull import QhullError
from scipy.spatial.transform import Rotation


def _segment_segment_distance(p1, q1, p2, q2):
    """
    Minimum Euclidean distance between 3D line segments p1-q1 and p2-q2.
    Robust implementation based on closest points on two segments.
    """
    p1 = np.asarray(p1, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    u = q1 - p1
    v = q2 - p2
    w = p1 - p2

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    eps = 1e-14
    D = a * c - b * b

    sN = 0.0
    sD = D
    tN = 0.0
    tD = D

    if D < eps:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < eps else sN / sD
    tc = 0.0 if abs(tN) < eps else tN / tD

    dp = w + sc * u - tc * v
    return float(np.linalg.norm(dp))


def _segment_aabb_intersection_interval(p0, p1, half_extent):
    """
    Segment vs axis-aligned box [-a,+a]^3.
    Returns (intersects, t_enter, t_exit) for p(t)=p0+t(p1-p0), t in [0,1].
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    a = float(half_extent)

    tmin = 0.0
    tmax = 1.0
    eps = 1e-14

    for j in range(3):
        if abs(d[j]) < eps:
            if p0[j] < -a or p0[j] > a:
                return False, None, None
        else:
            inv = 1.0 / d[j]
            t1 = (-a - p0[j]) * inv
            t2 = (+a - p0[j]) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False, None, None

    return True, tmin, tmax


def _point_in_support_hull(point_xy, support_xy, tol=1e-9):
    """
    Works for 1, 2, collinear, or full 2D support sets.
    Returns (inside, margin).
    margin:
      >0  inside a 2D polygon
       0  degenerate support (point/line) when feasible
      <0  outside
    """
    p = np.asarray(point_xy, dtype=float)
    pts = np.asarray(support_xy, dtype=float)

    if len(pts) == 0:
        return False, -np.inf

    # Remove duplicate points.
    unique = []
    for x in pts:
        if not any(np.linalg.norm(x - y) <= tol for y in unique):
            unique.append(x)
    pts = np.asarray(unique)

    if len(pts) == 1:
        d = np.linalg.norm(p - pts[0])
        return d <= tol, -d

    centered = pts - pts[0]
    rank = np.linalg.matrix_rank(centered, tol=max(tol, 1e-12))

    if len(pts) == 2 or rank < 2:
        direction = pts[1] - pts[0]
        n = np.linalg.norm(direction)
        if n <= tol:
            d = np.linalg.norm(p - pts[0])
            return d <= tol, -d

        u = direction / n
        scalar = (pts - pts[0]) @ u
        lo = scalar.min()
        hi = scalar.max()
        sp = np.dot(p - pts[0], u)
        perp = np.linalg.norm((p - pts[0]) - sp * u)
        inside = (perp <= tol) and (sp >= lo - tol) and (sp <= hi + tol)
        if inside:
            return True, 0.0
        endpoint_violation = max(lo - sp, sp - hi, 0.0)
        return False, -max(perp, endpoint_violation)

    try:
        hull = ConvexHull(pts)
    except QhullError:
        direction = pts[np.argmax(np.linalg.norm(pts - pts[0], axis=1))] - pts[0]
        n = np.linalg.norm(direction)
        if n <= tol:
            d = np.linalg.norm(p - pts[0])
            return d <= tol, -d
        u = direction / n
        scalar = (pts - pts[0]) @ u
        lo, hi = scalar.min(), scalar.max()
        sp = np.dot(p - pts[0], u)
        perp = np.linalg.norm((p - pts[0]) - sp * u)
        inside = (perp <= tol) and (sp >= lo - tol) and (sp <= hi + tol)
        return inside, 0.0 if inside else -max(perp, max(lo-sp, sp-hi, 0.0))

    vals = hull.equations[:, :2] @ p + hull.equations[:, 2]
    norms = np.linalg.norm(hull.equations[:, :2], axis=1)
    signed_inside_dist = -vals / norms
    margin = float(np.min(signed_inside_dist))
    return bool(np.all(vals <= tol)), margin


@dataclass
class Level1Report:
    feasible: bool
    n_frames: int

    joint_limits_ok: bool
    support_foot_lock_ok: bool
    support_contact_height_ok: bool
    swing_foot_ground_ok: bool
    support_region_ok: bool
    body_ground_ok: bool
    link_ground_ok: bool
    self_collision_ok: bool
    link_body_collision_ok: bool

    max_joint_limit_violation_rad: float
    max_support_foot_lock_error_m: float
    max_support_contact_height_error_m: float
    min_swing_foot_height_m: float
    min_support_margin_m: float
    min_body_ground_clearance_m: float
    min_link_ground_height_m: float
    min_interleg_segment_distance_m: float

    worst_joint_frame: int
    worst_support_lock_frame: int
    worst_support_height_frame: int
    worst_swing_ground_frame: int
    worst_support_region_frame: int
    worst_body_ground_frame: int
    worst_link_ground_frame: int
    worst_self_collision_frame: int
    worst_link_body_frame: int

    def to_dict(self):
        return asdict(self)


class Level1Checker:
    """
    Independent Level-1 geometric/kinematic trajectory checker.

    Current collision representation
    --------------------------------
    - Body: exact cube with half-side a
    - Links: zero-radius line segments
    - Self-collision: centerline crossing / near-crossing, using collision_tol
    - Link-body: exact segment-vs-body-cube intersection
    - Ground: z=0

    Therefore this is a centerline collision model, not yet a finite-thickness
    hardware collision model.
    """

    def __init__(
        self,
        kinematics,
        joint_tol=1e-8,
        contact_tol=1e-7,
        ground_tol=1e-8,
        support_tol=1e-8,
        collision_tol=1e-6,
        root_ignore_t=1e-6,
    ):
        self.kin = kinematics
        self.joint_tol = float(joint_tol)
        self.contact_tol = float(contact_tol)
        self.ground_tol = float(ground_tol)
        self.support_tol = float(support_tol)
        self.collision_tol = float(collision_tol)
        self.root_ignore_t = float(root_ignore_t)

    def check(self, body_pos, body_R, q, contact):
        q = np.asarray(q, dtype=float)
        body_R = np.asarray(body_R, dtype=float)
        contact = np.asarray(contact).astype(bool)

        F = q.shape[0]

        body_pos = np.asarray(body_pos, dtype=float)
        if body_pos.shape == (3,):
            body_pos = np.repeat(body_pos[None, :], F, axis=0)

        if body_R.shape != (F, 3, 3):
            raise ValueError("body_R must have shape (F,3,3)")
        if q.shape != (F, self.kin.n_legs, 3):
            raise ValueError("q must have shape (F,8,3)")
        if contact.shape != (F, self.kin.n_legs):
            raise ValueError("contact must have shape (F,8)")
        if body_pos.shape != (F, 3):
            raise ValueError("body_pos must have shape (F,3) or (3,)")

        max_joint_violation = 0.0
        max_lock_error = 0.0
        max_contact_z_err = 0.0
        min_swing_z = np.inf
        min_support_margin = np.inf
        min_body_clear = np.inf
        min_link_z = np.inf
        min_interleg_dist = np.inf

        worst_joint_frame = -1
        worst_lock_frame = -1
        worst_contact_z_frame = -1
        worst_swing_frame = -1
        worst_support_frame = -1
        worst_body_ground_frame = -1
        worst_link_ground_frame = -1
        worst_self_collision_frame = -1
        worst_link_body_frame = -1

        joint_limits_ok = True
        support_foot_lock_ok = True
        support_contact_height_ok = True
        swing_foot_ground_ok = True
        support_region_ok = True
        body_ground_ok = True
        link_ground_ok = True
        self_collision_ok = True
        link_body_collision_ok = True

        previous_feet = None
        previous_contact = None

        for f in range(F):
            R = body_R[f]
            t = body_pos[f]
            qf = q[f]
            cf = contact[f]

            low_violation = np.maximum(self.kin.q_min - qf, 0.0)
            high_violation = np.maximum(qf - self.kin.q_max, 0.0)
            frame_joint_violation = float(np.max(np.maximum(low_violation, high_violation)))

            if frame_joint_violation > max_joint_violation:
                max_joint_violation = frame_joint_violation
                worst_joint_frame = f

            if frame_joint_violation > self.joint_tol:
                joint_limits_ok = False

            roots = []
            elbows = []
            feet = []
            for i in range(self.kin.n_legs):
                root, elbow, foot = self.kin.world_points(t, R, i, qf[i])
                roots.append(root)
                elbows.append(elbow)
                feet.append(foot)

            roots = np.asarray(roots)
            elbows = np.asarray(elbows)
            feet = np.asarray(feet)

            support_idx = np.where(cf)[0]
            swing_idx = np.where(~cf)[0]

            if len(support_idx):
                frame_contact_z = float(np.max(np.abs(feet[support_idx, 2])))
                if frame_contact_z > max_contact_z_err:
                    max_contact_z_err = frame_contact_z
                    worst_contact_z_frame = f
                if frame_contact_z > self.contact_tol:
                    support_contact_height_ok = False

            if len(swing_idx):
                frame_min_swing = float(np.min(feet[swing_idx, 2]))
                if frame_min_swing < min_swing_z:
                    min_swing_z = frame_min_swing
                    worst_swing_frame = f
                if frame_min_swing < -self.ground_tol:
                    swing_foot_ground_ok = False

            if previous_feet is not None:
                keep = cf & previous_contact
                if np.any(keep):
                    frame_lock = float(
                        np.max(np.linalg.norm(feet[keep] - previous_feet[keep], axis=1))
                    )
                    if frame_lock > max_lock_error:
                        max_lock_error = frame_lock
                        worst_lock_frame = f
                    if frame_lock > self.contact_tol:
                        support_foot_lock_ok = False

            previous_feet = feet.copy()
            previous_contact = cf.copy()

            if len(support_idx) == 0:
                support_region_ok = False
                frame_margin = -np.inf
            else:
                inside, frame_margin = _point_in_support_hull(
                    t[:2],
                    feet[support_idx, :2],
                    tol=self.support_tol,
                )
                if not inside:
                    support_region_ok = False

            if frame_margin < min_support_margin:
                min_support_margin = frame_margin
                worst_support_frame = f

            body_min_z = float(t[2] - self.kin.a * np.sum(np.abs(R[2, :])))
            if body_min_z < min_body_clear:
                min_body_clear = body_min_z
                worst_body_ground_frame = f
            if body_min_z < -self.ground_tol:
                body_ground_ok = False

            frame_min_link_z = np.inf
            for i in range(self.kin.n_legs):
                frame_min_link_z = min(
                    frame_min_link_z,
                    float(roots[i, 2]),
                    float(elbows[i, 2]),
                )
                frame_min_link_z = min(
                    frame_min_link_z,
                    float(elbows[i, 2]),
                    float(feet[i, 2]),
                )

            if frame_min_link_z < min_link_z:
                min_link_z = frame_min_link_z
                worst_link_ground_frame = f

            if frame_min_link_z < -self.ground_tol:
                link_ground_ok = False

            segments = []
            for i in range(self.kin.n_legs):
                segments.append((i, 0, roots[i], elbows[i]))
                segments.append((i, 1, elbows[i], feet[i]))

            for ia in range(len(segments)):
                lega, seg_a, p1, q1 = segments[ia]
                for ib in range(ia + 1, len(segments)):
                    legb, seg_b, p2, q2 = segments[ib]
                    if lega == legb:
                        continue
                    dist = _segment_segment_distance(p1, q1, p2, q2)
                    if dist < min_interleg_dist:
                        min_interleg_dist = dist
                        worst_self_collision_frame = f
                    if dist < self.collision_tol:
                        self_collision_ok = False

            for i in range(self.kin.n_legs):
                for p0w, p1w in ((roots[i], elbows[i]), (elbows[i], feet[i])):
                    p0b = R.T @ (p0w - t)
                    p1b = R.T @ (p1w - t)
                    hit, t_enter, t_exit = _segment_aabb_intersection_interval(
                        p0b,
                        p1b,
                        self.kin.a,
                    )
                    if hit:
                        if t_exit is not None and t_exit > self.root_ignore_t:
                            link_body_collision_ok = False
                            worst_link_body_frame = f

        feasible = all(
            [
                joint_limits_ok,
                support_foot_lock_ok,
                support_contact_height_ok,
                swing_foot_ground_ok,
                support_region_ok,
                body_ground_ok,
                link_ground_ok,
                self_collision_ok,
                link_body_collision_ok,
            ]
        )

        return Level1Report(
            feasible=feasible,
            n_frames=F,
            joint_limits_ok=joint_limits_ok,
            support_foot_lock_ok=support_foot_lock_ok,
            support_contact_height_ok=support_contact_height_ok,
            swing_foot_ground_ok=swing_foot_ground_ok,
            support_region_ok=support_region_ok,
            body_ground_ok=body_ground_ok,
            link_ground_ok=link_ground_ok,
            self_collision_ok=self_collision_ok,
            link_body_collision_ok=link_body_collision_ok,
            max_joint_limit_violation_rad=max_joint_violation,
            max_support_foot_lock_error_m=max_lock_error,
            max_support_contact_height_error_m=max_contact_z_err,
            min_swing_foot_height_m=min_swing_z,
            min_support_margin_m=min_support_margin,
            min_body_ground_clearance_m=min_body_clear,
            min_link_ground_height_m=min_link_z,
            min_interleg_segment_distance_m=min_interleg_dist,
            worst_joint_frame=worst_joint_frame,
            worst_support_lock_frame=worst_lock_frame,
            worst_support_height_frame=worst_contact_z_frame,
            worst_swing_ground_frame=worst_swing_frame,
            worst_support_region_frame=worst_support_frame,
            worst_body_ground_frame=worst_body_ground_frame,
            worst_link_ground_frame=worst_link_ground_frame,
            worst_self_collision_frame=worst_self_collision_frame,
            worst_link_body_frame=worst_link_body_frame,
        )
