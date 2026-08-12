"""Touchdown candidate generation and local contact-mode ranking."""

import itertools

import numpy as np
from scipy.optimize import least_squares


class TouchdownSearchMixin:
    def _reachable_touchdowns(self, angle_deg, q, leg, K=4, expanded=False):
        """Generate touchdown states connected to the current swing state.

        A candidate is accepted only when a ground-safe continuous joint-space
        segment exists from the current swing configuration to the touchdown.
        """
        t, R = self._pose(angle_deg)
        qcur = q[leg].copy()
        rng = np.random.default_rng(
            100003 + int(round(angle_deg)) * 97 + leg * 7919
        )
        n_samples = (
            self.cfg.expanded_touchdown_samples
            if expanded
            else self.cfg.normal_touchdown_samples
        )

        raw = [qcur.copy()]
        sig = np.deg2rad(np.array([70.0, 45.0, 75.0]))
        for _ in range(n_samples):
            if rng.random() < 0.78:
                x = np.clip(
                    qcur + rng.normal(0.0, sig), self.kin.q_min, self.kin.q_max
                )
            else:
                x = self.kin.q_min + (self.kin.q_max - self.kin.q_min) * rng.random(3)
            raw.append(x)

        scored = []
        for x in raw:
            if not self._leg_safe(leg, t, R, x):
                continue
            if not self._segment_safe(leg, t, R, qcur, x, n=12):
                continue
            _, _, foot = self.kin.world_points(t, R, leg, x)
            margin = float(np.min(np.minimum(x - self.kin.q_min, self.kin.q_max - x)))
            score = (
                abs(foot[2])
                + 0.002 * np.linalg.norm((x - qcur) / (self.kin.q_max - self.kin.q_min))
                - 0.0005 * margin
            )
            scored.append((score, x.copy()))
        scored.sort(key=lambda z: z[0])

        candidates = []
        refine_count = 80 if expanded else 45
        for _, seed in scored[:refine_count]:
            def residual(x):
                _, elbow, foot = self.kin.world_points(t, R, leg, x)
                rel = (x - seed) / (self.kin.q_max - self.kin.q_min)
                return np.r_[
                    100.0 * foot[2],
                    15.0 * max(0.015 - elbow[2], 0.0),
                    0.018 * rel,
                ]

            rr = least_squares(
                residual,
                seed,
                bounds=(self.kin.q_min, self.kin.q_max),
                max_nfev=100,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
            )
            qg = rr.x.copy()
            _, elbow, foot = self.kin.world_points(t, R, leg, qg)
            if abs(foot[2]) > 4e-5 or min(elbow[2], foot[2]) < -1e-7:
                continue
            if not self._segment_safe(leg, t, R, qcur, qg, n=40):
                continue

            target = foot.copy()
            target[2] = 0.0
            ex = least_squares(
                lambda x: self.kin.foot_world(t, R, leg, x) - target,
                qg,
                jac=lambda x: self.kin.leg_jacobian_world(R, leg, x),
                bounds=(self.kin.q_min, self.kin.q_max),
                max_nfev=100,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            if np.linalg.norm(ex.fun) > 3e-6:
                continue
            qg = ex.x.copy()
            if not self._leg_safe(leg, t, R, qg):
                continue
            if not self._segment_safe(leg, t, R, qcur, qg, n=50):
                continue
            if any(np.linalg.norm(target[:2] - old[0][:2]) < 0.05 for old in candidates):
                continue
            candidates.append((target.copy(), qg.copy()))
            if len(candidates) >= K:
                break
        return candidates

    def _rank_plans(self, angle_deg, q, support, anchors, expanded=False):
        swing = [i for i in range(self.kin.n_legs) if i not in support]
        K = (
            self.cfg.expanded_touchdowns_per_leg
            if expanded
            else self.cfg.normal_touchdowns_per_leg
        )
        cmap = {
            leg: self._reachable_touchdowns(angle_deg, q, leg, K, expanded)
            for leg in swing
        }
        cmap = {leg: vals for leg, vals in cmap.items() if vals}

        ranked = []
        nadds = (1, 2, 3) if expanded else (1, 2)
        for nadd in nadds:
            if len(cmap) < nadd:
                continue
            for legs in itertools.combinations(sorted(cmap), nadd):
                combos = list(itertools.product(*[cmap[leg] for leg in legs]))
                if expanded and len(combos) > 50:
                    combos = combos[:50]
                for chosen in combos:
                    add = {leg: value for leg, value in zip(legs, chosen)}
                    for nrem in (1, 2, 3):
                        if nrem > len(support):
                            continue
                        for rem in itertools.combinations(support, nrem):
                            new_support = tuple(sorted((set(support) - set(rem)).union(legs)))
                            if len(new_support) < self.cfg.min_support_count:
                                continue

                            new_anchors = {
                                i: anchors[i].copy() for i in support if i not in rem
                            }
                            q_try = q.copy()
                            for leg, (target, qgoal) in add.items():
                                new_anchors[leg] = target.copy()
                                q_try[leg] = qgoal.copy()

                            q_support = self._support_only(
                                angle_deg, q_try, new_support, new_anchors
                            )
                            if q_support is None:
                                continue
                            horizon = (
                                self.cfg.expanded_lookahead_deg
                                if expanded
                                else self.cfg.normal_lookahead_deg
                            )
                            gain = self._predict_gain(
                                q_support, new_support, new_anchors, angle_deg, horizon
                            )
                            if gain <= 0.0:
                                continue
                            score = gain - 0.12 * (nadd + nrem) + 0.03 * len(new_support)
                            ranked.append(
                                (
                                    score,
                                    gain,
                                    add,
                                    list(rem),
                                    new_support,
                                    new_anchors,
                                    q_support,
                                )
                            )

        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out = []
        seen = set()
        for plan in ranked:
            _, _, add, rem, new_support, _, _ = plan
            sig = (tuple(sorted(add)), tuple(rem), new_support)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(plan)
            if len(out) >= 12:
                break
        return out
