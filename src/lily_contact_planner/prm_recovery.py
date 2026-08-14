"""v0.0.6 static-body PRM touchdown recovery.

This module restores the successful v0.0.6 static reconfiguration primitive
without changing the ordinary one-to-one or multi-contact search policy.
"""

import heapq

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .collision import segment_aabb_distance, segment_segment_distance, trim_segment_start
from .recovery_policy import RecoveryKind


class PRMStaticRecoveryMixin:
    """Override only the static-reconfiguration execution path with v0.0.6 PRM."""

    _prm_l2_radius_m = 0.025
    _prm_l3_radius_m = 0.025
    _prm_root_ignore_m = 0.05

    def _prm_static_leg_feasible(self, leg, t, R, q_leg, q_all):
        root, elbow, foot = self.kin.world_points(t, R, leg, q_leg)
        if min(root[2], elbow[2], foot[2]) < -1e-9:
            return False

        moving = [
            (0, root, elbow, self._prm_l2_radius_m),
            (1, elbow, foot, self._prm_l3_radius_m),
        ]

        for other_leg in range(self.kin.n_legs):
            if other_leg == leg:
                continue
            oroot, oelbow, ofoot = self.kin.world_points(
                t, R, other_leg, q_all[other_leg]
            )
            other = [
                (0, oroot, oelbow, self._prm_l2_radius_m),
                (1, oelbow, ofoot, self._prm_l3_radius_m),
            ]
            for _, p0, p1, rad in moving:
                for _, r0, r1, orad in other:
                    if segment_segment_distance(p0, p1, r0, r1) < rad + orad - 1e-9:
                        return False

        for link, p0, p1, rad in moving:
            if link == 0:
                p0, p1 = trim_segment_start(p0, p1, self._prm_root_ignore_m)
            p0b = R.T @ (p0 - t)
            p1b = R.T @ (p1 - t)
            if segment_aabb_distance(p0b, p1b, self.kin.a) < rad - 1e-9:
                return False
        return True

    def _prm_static_edge_safe(self, leg, t, R, qa, qb, q_all, n=21):
        for u in np.linspace(0.0, 1.0, n):
            q_test = q_all.copy()
            q_test[leg] = (1.0 - u) * qa + u * qb
            if not self._prm_static_leg_feasible(leg, t, R, q_test[leg], q_test):
                return False
        return True

    def _prm_static_path(self, leg, t, R, qa, qb, q_all, seed=991):
        """Corrected lazy A* PRM used by the successful v0.0.6 recovery."""
        if self._prm_static_edge_safe(leg, t, R, qa, qb, q_all, n=41):
            return [qa.copy(), qb.copy()]

        sampler = qmc.Sobol(d=3, scramble=True, seed=int(seed))
        samples = sampler.random_base2(13)  # 8192, matching archived v0.0.6 diagnostic
        qs = self.kin.q_min + samples * (self.kin.q_max - self.kin.q_min)

        good = [qa.copy()]
        for x in qs:
            q_test = q_all.copy()
            q_test[leg] = x
            if self._prm_static_leg_feasible(leg, t, R, x, q_test):
                good.append(x.copy())

        good.append(qb.copy())
        goal = len(good) - 1
        Q = np.asarray(good)
        scale = np.array([2.0 * np.pi, np.deg2rad(190.0), np.deg2rad(300.0)])
        X = Q / scale
        tree = cKDTree(X)

        start = 0
        k = 30
        dist = {start: 0.0}
        prev = {}
        edge_cache = {}
        pq = [(float(np.linalg.norm(X[start] - X[goal])), 0.0, start)]
        found = None

        while pq:
            _, gcur, u = heapq.heappop(pq)
            if gcur > dist[u] + 1e-12:
                continue
            if u == goal:
                found = u
                break
            _, ids = tree.query(X[u], k=min(k, len(Q)))
            for v in np.atleast_1d(ids):
                v = int(v)
                if v == u:
                    continue
                key = (min(u, v), max(u, v))
                ok = edge_cache.get(key)
                if ok is None:
                    ok = self._prm_static_edge_safe(
                        leg, t, R, Q[u], Q[v], q_all, n=7
                    )
                    edge_cache[key] = ok
                if not ok:
                    continue
                w = float(np.linalg.norm(X[v] - X[u]))
                nd = gcur + w
                if nd < dist.get(v, 1e99):
                    dist[v] = nd
                    prev[v] = u
                    h = float(np.linalg.norm(X[v] - X[goal]))
                    heapq.heappush(pq, (nd + h, nd, v))

        if found is None:
            return None

        ids = []
        u = found
        while True:
            ids.append(u)
            if u == start:
                break
            u = prev[u]
        ids.reverse()
        path = [Q[i].copy() for i in ids]

        if not all(
            self._prm_static_edge_safe(leg, t, R, a, b, q_all, n=101)
            for a, b in zip(path[:-1], path[1:])
        ):
            return None
        return path

    def _execute_reconfiguration(
        self,
        angle_deg,
        q_start,
        support_before,
        anchors_before,
        add,
        rem,
        new_support,
        new_anchors,
        stage_kind,
    ):
        if stage_kind != RecoveryKind.STATIC_RECONFIGURATION:
            return super()._execute_reconfiguration(
                angle_deg,
                q_start,
                support_before,
                anchors_before,
                add,
                rem,
                new_support,
                new_anchors,
                stage_kind,
            )

        t, R = self._pose(angle_deg)
        q_work = q_start.copy()
        trace = []
        support_during = list(support_before)
        anchors_touch = {k: v.copy() for k, v in anchors_before.items()}

        # v0.0.6: hold body fixed and touchdown each added leg along a
        # finite-thickness collision-checked PRM path.
        for leg in sorted(add):
            target, q_goal = add[leg]
            path = self._prm_static_path(
                leg, t, R, q_work[leg].copy(), q_goal.copy(), q_work, seed=991
            )
            if path is None:
                return None
            for qa, qb in zip(path[:-1], path[1:]):
                # Archived successful trajectory used 20 interpolated frames per edge.
                for u in np.linspace(0.0, 1.0, 21)[1:]:
                    q_frame = q_work.copy()
                    q_frame[leg] = (1.0 - u) * qa + u * qb
                    trace.append(q_frame)
                q_work[leg] = qb.copy()
            anchors_touch[leg] = target.copy()
            if leg not in support_during:
                support_during.append(leg)

        q_touch = self._support_only(
            angle_deg, q_work, tuple(sorted(support_during)), anchors_touch
        )
        if q_touch is None:
            return None
        q_work = q_touch

        # Touchdown before liftoff, then the exact v0.0.6 50 mm vertical lift.
        for leg in rem:
            _, _, foot = self.kin.world_points(t, R, leg, q_work[leg])
            target = foot.copy()
            target[2] += 0.05
            ok, q_lift, _ = self._solve_leg_to_anchor(
                t, R, leg, q_work[leg], target
            )
            if not ok:
                return None
            if not self._segment_safe(leg, t, R, q_work[leg], q_lift, n=40):
                return None
            q_from = q_work[leg].copy()
            for u in np.linspace(0.0, 1.0, 51)[1:]:
                q_frame = q_work.copy()
                q_frame[leg] = (1.0 - u) * q_from + u * q_lift
                trace.append(q_frame)
            q_work[leg] = q_lift.copy()

        q_after = self._actual(angle_deg, q_work, new_support, new_anchors)
        if q_after is None:
            return None
        return q_after, trace
