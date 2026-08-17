"""Successful v0.0.6 one-to-one seed refinement on top of v0.0.4.

Touchdown seeds retain the recovered Sobol generation and support-area ranking.
The staged planner may now choose which ranked touchdown initial guesses are
attempted and may override the per-candidate wall-clock limit. Liftoff choices
are ordered by how much farther each currently supporting leg can keep its
present ground anchor; legs with less remaining support range are tried first.
NLP formulation, dense Checker, and first-feasible acceptance are unchanged.
"""

from contextlib import contextmanager
import signal
import threading

import numpy as np

from .analytic_ik import analytic_leg_ik_world
from .candidate_v004 import generate_contact_candidates_v004
from .checker_v004 import dense_check_solution_v004
from .trajectory_nlp_v004 import ContactSwitchNLPV004
from .v004_receding import V004RecedingHorizonMixin


class _CandidateSolveTimeout(RuntimeError):
    pass


def _raise_candidate_timeout(signum, frame):
    raise _CandidateSolveTimeout("contact candidate NLP timed out")


@contextmanager
def _candidate_wall_clock_timeout(seconds):
    """Interrupt one SLSQP solve after ``seconds`` on POSIX main threads."""
    seconds = float(seconds)
    if seconds <= 0.0:
        yield
        return
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise RuntimeError("candidate wall-clock timeout requires POSIX SIGALRM/setitimer")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("candidate wall-clock timeout requires the main thread")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_candidate_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


class _SuccessfulContactSwitchNLPV004(ContactSwitchNLPV004):
    def initial_guess(self):
        z = super().initial_guess()
        t, rv, q, xy, w = self.layout.unpack(z)
        R = self._rotations(rv)
        leg = int(self.candidate.touchdown_leg)
        p0 = self.kin.foot_world(t[0], R[0], leg, q[0, leg])
        p1 = np.array([xy[0], xy[1], 0.0], dtype=float)
        prev = q[0, leg].copy()
        for k in range(1, self.td):
            s = k / float(self.td)
            target = (1.0 - s) * p0 + s * p1
            target[2] += self.settings.initial_liftoff_clearance_m * np.sin(np.pi * s)
            branches = analytic_leg_ik_world(
                self.kin, t[k], R[k], leg, target,
                q_reference=prev, residual_tol=2e-6,
            )
            if branches:
                q[k, leg] = branches[0]
                prev = branches[0]
        z[self.layout.q] = q.ravel()
        return z


class V004SuccessfulSeedMixin(V004RecedingHorizonMixin):
    """v0.0.4 contact recovery with ranked seeds and first-feasible stop."""

    def _liftoff_remaining_ranges(self, angle_deg, q, support, anchors):
        """Estimate how many more degrees each support leg can hold its anchor.

        This is intentionally a cheap kinematic priority metric, not a new hard
        constraint. For each currently supporting leg we advance the task in
        ``cfg.step_deg`` increments, keep that leg's world contact point fixed,
        and require an analytic IK branch that remains above the ground. The
        check stops at the current phase boundary or expanded look-ahead limit.

        A small value means the leg is close to losing support feasibility and
        should therefore be *tried earlier* as a liftoff leg. Existing support,
        collision and trajectory checks still decide whether that liftoff is
        actually allowed.
        """
        angle0 = float(angle_deg)
        support = tuple(int(x) for x in support)
        step = max(float(self.cfg.step_deg), 1e-6)
        phase_end = float(self._v004_phase_end(angle0))
        lookahead_end = min(
            float(self.max_roll_deg),
            phase_end,
            angle0 + float(self.cfg.expanded_lookahead_deg),
        )

        cache_key = (
            round(angle0, 9),
            tuple(
                (
                    int(leg),
                    tuple(np.round(np.asarray(q[int(leg)], float), 7)),
                    tuple(np.round(np.asarray(anchors[int(leg)], float), 7)),
                )
                for leg in support
            ),
            round(lookahead_end, 9),
        )
        cache = getattr(self, "_liftoff_remaining_range_cache", None)
        if cache is None:
            cache = {}
            self._liftoff_remaining_range_cache = cache
        if cache_key in cache:
            return dict(cache[cache_key])

        remaining = {}
        for leg in support:
            q_ref = np.asarray(q[leg], float).copy()
            anchor = np.asarray(anchors[leg], float)
            a = angle0
            last = 0.0
            while a + step <= lookahead_end + 1e-9:
                a_next = min(a + step, lookahead_end)
                t_next, R_next = self._pose(a_next)
                branches = analytic_leg_ik_world(
                    self.kin,
                    t_next,
                    R_next,
                    leg,
                    anchor,
                    q_reference=q_ref,
                    residual_tol=2e-6,
                )
                q_next = None
                for branch in branches:
                    _, elbow, foot = self.kin.world_points(
                        t_next, R_next, leg, branch
                    )
                    if min(float(elbow[2]), float(foot[2])) >= -1e-7:
                        q_next = np.asarray(branch, float).copy()
                        break
                if q_next is None:
                    break
                q_ref = q_next
                a = a_next
                last = a - angle0
            remaining[leg] = float(last)

        if len(cache) >= 256:
            cache.clear()
        cache[cache_key] = dict(remaining)
        return remaining

    @staticmethod
    def _liftoff_priority_key(liftoff_legs, remaining_ranges):
        """Smaller remaining support range sorts first; existing order breaks ties."""
        return tuple(sorted(
            float(remaining_ranges.get(int(leg), float("inf")))
            for leg in liftoff_legs
        ))

    def _v004_contact_recovery(
        self,
        angle_deg,
        q,
        support,
        anchors,
        horizons=None,
        seed_override=None,
        advance_seed=True,
        touchdown_seed_ranks=None,
        candidate_timeout_s=None,
        search_phase="primary",
    ):
        cfg = self._v004_settings()
        st = self._v004_state(angle_deg, q, support, anchors)
        current_seed = int(getattr(self, '_v004_contact_seed', 0))
        seed = current_seed if seed_override is None else int(seed_override)
        candidates = list(generate_contact_candidates_v004(
            self.kin, st, cfg, seed=seed,
            touchdown_seed_ranks=touchdown_seed_ranks,
        ))
        if advance_seed:
            self._v004_contact_seed = current_seed + 1

        liftoff_remaining = self._liftoff_remaining_ranges(
            angle_deg, q, support, anchors
        )
        candidates.sort(key=lambda cand: float(
            liftoff_remaining.get(int(cand.liftoff_leg), float("inf"))
        ))

        timeout_s = (
            float(cfg.candidate_timeout_s)
            if candidate_timeout_s is None else float(candidate_timeout_s)
        )
        phase_end = self._v004_phase_end(angle_deg)
        max_h = int(np.floor(min(5.0, phase_end - float(angle_deg)) + 1e-9))
        if horizons is None:
            horizon_list = list(range(max_h, 0, -1))
        else:
            horizon_list = [int(h) for h in horizons if 1 <= int(h) <= max_h]

        self._log(
            'V004 contact start', 'phase', str(search_phase),
            'angle', float(angle_deg), 'horizons', tuple(horizon_list),
            'seed_ranks', None if touchdown_seed_ranks is None else tuple(touchdown_seed_ranks),
            'candidates', int(len(candidates)), 'timeout_s', timeout_s,
            'liftoff_remaining_deg', {
                int(k): float(v) for k, v in sorted(liftoff_remaining.items())
            },
        )

        trials = []
        for h in horizon_list:
            target_t, target_R = self._v004_target(angle_deg, float(h))
            solved = 0
            attempted = 0
            timed_out = 0
            timeout_candidate_indices = []
            chosen = None

            for candidate_index, cand in enumerate(candidates):
                attempted += 1
                if hasattr(self, '_search_stats'):
                    self._search_stats['v004_nlp_attempted'] += 1
                try:
                    with _candidate_wall_clock_timeout(timeout_s):
                        sol = _SuccessfulContactSwitchNLPV004(
                            self.kin, st, cand, target_t, target_R, cfg
                        ).solve()
                except _CandidateSolveTimeout:
                    timed_out += 1
                    if hasattr(self, '_search_stats'):
                        self._search_stats['v004_timeouts'] += 1
                    timeout_candidate_indices.append(int(candidate_index))
                    self._log(
                        'v0.0.4 candidate timeout',
                        'phase', str(search_phase),
                        'angle', float(angle_deg),
                        'horizon', float(h),
                        'candidate', int(candidate_index),
                        'limit_s', timeout_s,
                    )
                    continue

                if not sol.success:
                    continue

                solved += 1
                sol.checker = dense_check_solution_v004(
                    self.kin, st, sol, target_t, target_R, cfg
                )
                if not sol.accepted:
                    continue

                chosen = (float(sol.objective), int(candidate_index), sol)
                self._log(
                    'v0.0.4 first feasible candidate',
                    'phase', str(search_phase),
                    'angle', float(angle_deg),
                    'horizon', float(h),
                    'candidate', int(candidate_index),
                    'attempted', int(attempted),
                    'objective', float(sol.objective),
                )
                break

            trials.append({
                'search_phase': str(search_phase),
                'horizon_deg': float(h),
                'candidate_count': int(len(candidates)),
                'attempted_candidates': int(attempted),
                'solved_candidates': int(solved),
                'accepted': int(chosen is not None),
                'candidate_timeout_s': timeout_s,
                'timed_out_candidates': int(timed_out),
                'timeout_candidate_indices': timeout_candidate_indices,
                'touchdown_seed_ranks': (
                    None if touchdown_seed_ranks is None
                    else [int(x) for x in touchdown_seed_ranks]
                ),
                'early_stop': bool(chosen is not None and attempted < len(candidates)),
                'remaining_candidates_skipped': int(len(candidates) - attempted) if chosen is not None else 0,
                'liftoff_remaining_deg': {
                    int(k): float(v) for k, v in sorted(liftoff_remaining.items())
                },
            })

            if chosen is None:
                continue

            objective, candidate_index, sol = chosen
            cand = sol.candidate
            exec_node = int(cfg.liftoff_node)
            exec_fraction = exec_node / float(cfg.n_nodes - 1)
            angle_after = float(angle_deg + h * exec_fraction)
            new_support = tuple(sorted(
                (set(support) | {int(cand.touchdown_leg)}) - {int(cand.liftoff_leg)}
            ))
            new_anchors = {
                int(k): np.asarray(v, float).copy()
                for k, v in anchors.items()
                if int(k) != int(cand.liftoff_leg)
            }
            new_anchors[int(cand.touchdown_leg)] = np.array([
                sol.touchdown_xy[0], sol.touchdown_xy[1], 0.0
            ])
            return {
                'success': True,
                'seed': seed,
                'candidate_index': int(candidate_index),
                'candidate': cand,
                'solution': sol,
                'objective': float(objective),
                'horizon_deg': float(h),
                'exec_node': exec_node,
                'exec_fraction': float(exec_fraction),
                'angle_after_deg': angle_after,
                'q_after': np.asarray(sol.q[exec_node], float).copy(),
                'support_after': new_support,
                'anchors_after': new_anchors,
                'checker': sol.checker,
                'trials': trials,
                'search_phase': str(search_phase),
            }
        return None
