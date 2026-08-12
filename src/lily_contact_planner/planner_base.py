"""Continuous kinematic layer for the unified Lily contact planner."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .checker import Level1Checker, _point_in_support_hull
from .kinematics import LilyKinematics
from .tasks import ForwardRollTask


@dataclass
class PlannerSettings:
    step_deg: float = 1.0
    swing_clearance_m: float = 0.025
    normal_touchdown_samples: int = 420
    expanded_touchdown_samples: int = 900
    normal_touchdowns_per_leg: int = 4
    expanded_touchdowns_per_leg: int = 6
    normal_lookahead_deg: float = 28.0
    expanded_lookahead_deg: float = 42.0
    max_depth: int = 55
    max_nodes: int = 220
    branch_width: int = 8
    min_support_count: int = 3


class PlannerBaseMixin:
    def __init__(
        self,
        kinematics: LilyKinematics,
        task: ForwardRollTask,
        max_roll_deg: float,
        settings: Optional[PlannerSettings] = None,
        verbose: bool = True,
    ):
        self.kin = kinematics
        self.task = task
        self.max_roll_deg = float(max_roll_deg)
        self.cfg = settings or PlannerSettings()
        self.verbose = bool(verbose)
        self.checker = Level1Checker(self.kin)

        self.seed_lib = [
            np.deg2rad([q1, q2, q3])
            for q1 in [-270, -180, -90, 0, 90, 180, 270]
            for q2, q3 in [
                (-85, 135),
                (85, -135),
                (-60, 110),
                (60, -110),
                (-30, 80),
                (30, -80),
                (0, 120),
                (0, -120),
            ]
        ]

        self.memo = set()
        self.nodes = 0
        self.best_angle = 0.0
        self.best_path: List[dict] = []

    def _log(self, *args):
        if self.verbose:
            print(*args, flush=True)

    def _pose(self, angle_deg: float):
        return self.task.pose(angle_deg)

    def _solve_leg_to_anchor(self, t, R, leg, q_seed, anchor):
        res = least_squares(
            lambda x: self.kin.foot_world(t, R, leg, x) - anchor,
            q_seed,
            jac=lambda x: self.kin.leg_jacobian_world(R, leg, x),
            bounds=(self.kin.q_min, self.kin.q_max),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=120,
        )
        err = float(np.linalg.norm(res.fun))
        return bool(res.success and err <= 2e-6), res.x.copy(), err

    def _leg_safe(self, leg, t, R, q_leg):
        _, elbow, foot = self.kin.world_points(t, R, leg, q_leg)
        return min(elbow[2], foot[2]) >= -1e-8

    def _segment_safe(self, leg, t, R, qa, qb, n=24):
        for s in np.linspace(0.0, 1.0, n):
            h = s * s * (3.0 - 2.0 * s)
            if not self._leg_safe(leg, t, R, (1.0 - h) * qa + h * qb):
                return False
        return True

    def _robust_swing(self, t, R, q, support):
        """Raise low swing links without jumping between disconnected branches."""
        out = q.copy()
        S = set(support)
        clear = self.cfg.swing_clearance_m
        roll_key = int(round(np.rad2deg(Rotation.from_matrix(R).as_rotvec()[0])))

        for leg in range(self.kin.n_legs):
            if leg in S:
                continue
            qcur = out[leg].copy()
            _, elbow, foot = self.kin.world_points(t, R, leg, qcur)
            if min(elbow[2], foot[2]) >= clear:
                continue

            rng = np.random.default_rng(300007 + roll_key * 31 + leg * 197)
            seeds = [qcur.copy()]
            sig = np.deg2rad(np.array([50.0, 35.0, 55.0]))
            for _ in range(45):
                seeds.append(
                    np.clip(qcur + rng.normal(0.0, sig), self.kin.q_min, self.kin.q_max)
                )
            seeds += self.seed_lib[::8]

            best = None
            for seed in seeds:
                def residual(x):
                    _, e, f = self.kin.world_points(t, R, leg, x)
                    pe = max(clear - e[2], 0.0)
                    pf = max(clear - f[2], 0.0)
                    rel = (x - qcur) / (self.kin.q_max - self.kin.q_min)
                    return np.r_[30.0 * pe, 30.0 * pf, 1.5 * (f[2] - 0.08), 0.05 * rel]

                rr = least_squares(
                    residual,
                    seed,
                    bounds=(self.kin.q_min, self.kin.q_max),
                    max_nfev=55,
                )
                _, e, f = self.kin.world_points(t, R, leg, rr.x)
                margin = min(e[2], f[2])
                if margin < -1e-8:
                    continue
                if not self._segment_safe(leg, t, R, qcur, rr.x, n=28):
                    continue
                score = margin - 0.008 * np.linalg.norm(
                    (rr.x - qcur) / (self.kin.q_max - self.kin.q_min)
                )
                if best is None or score > best[0]:
                    best = (score, rr.x.copy())

            if best is None:
                return None
            out[leg] = best[1]
        return out

    def _support_only(self, angle_deg, q, support, anchors):
        t, R = self._pose(angle_deg)
        support = tuple(support)
        xy = np.array([anchors[i][:2] for i in support])
        inside, _ = _point_in_support_hull(t[:2], xy, tol=1e-8)
        if not inside:
            return None

        out = q.copy()
        for leg in support:
            ok, q_leg, _ = self._solve_leg_to_anchor(
                t, R, leg, out[leg], anchors[leg]
            )
            out[leg] = q_leg
            if not ok:
                return None
            _, elbow, foot = self.kin.world_points(t, R, leg, out[leg])
            if min(elbow[2], foot[2]) < -1e-7:
                return None
        return out

    def _actual(self, angle_deg, q, support, anchors):
        t, R = self._pose(angle_deg)
        out = self._support_only(angle_deg, q, support, anchors)
        if out is None:
            return None
        out = self._robust_swing(t, R, out, support)
        if out is None:
            return None

        C = np.zeros((1, self.kin.n_legs), dtype=bool)
        C[0, list(support)] = True
        rep = self.checker.check(t[None, :], R[None, :, :], out[None, :, :], C)
        basic = (
            rep.joint_limits_ok
            and rep.support_contact_height_ok
            and rep.swing_foot_ground_ok
            and rep.support_region_ok
            and rep.body_ground_ok
            and rep.link_ground_ok
        )
        return out if basic else None

    def _predict_gain(self, q, support, anchors, angle0, horizon_deg):
        out = q.copy()
        a = float(angle0)
        while a + self.cfg.step_deg <= min(
            self.max_roll_deg, angle0 + horizon_deg
        ) + 1e-9:
            qn = self._support_only(a + self.cfg.step_deg, out, support, anchors)
            if qn is None:
                break
            out = qn
            a += self.cfg.step_deg
        return a - angle0
