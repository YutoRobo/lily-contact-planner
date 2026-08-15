"""Successful v0.0.6 one-to-one seed refinement on top of v0.0.4.

Archived trajectories show that future touchdown feet approach the ground point
with linear Cartesian XY and a 20 mm sinusoidal Z arc before touchdown.  This
seed is what makes the later v0.0.4-labelled one-to-one events reproduce the
verified v0.0.6 run.

Candidate handling follows the recovered v0.0.4 candidate generation semantics:
Sobol touchdown seeds are ranked by support-polygon area and spatially
separated in ``candidate_v004``.  Candidates are attempted in that existing
order.  Each finite-horizon NLP has a wall-clock limit; a numerically stalled
candidate is treated as failed.  As soon as one candidate passes both the NLP
and the independent dense Checker, the remaining candidates at that horizon
are not solved.  No additional ranking score or prescreen is introduced.
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
    raise _CandidateSolveTimeout("v0.0.4 contact candidate NLP timed out")


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
    """v0.0.4 contact recovery with timeout and first-feasible early stop."""

    def _v004_contact_recovery(
        self,
        angle_deg,
        q,
        support,
        anchors,
        horizons=None,
        seed_override=None,
        advance_seed=True,
    ):
        cfg = self._v004_settings()
        st = self._v004_state(angle_deg, q, support, anchors)
        current_seed = int(getattr(self, '_v004_contact_seed', 0))
        seed = current_seed if seed_override is None else int(seed_override)
        candidates = list(generate_contact_candidates_v004(
            self.kin, st, cfg, seed=seed
        ))
        if advance_seed:
            self._v004_contact_seed = current_seed + 1

        phase_end = self._v004_phase_end(angle_deg)
        max_h = int(np.floor(min(5.0, phase_end - float(angle_deg)) + 1e-9))
        if horizons is None:
            horizon_list = list(range(max_h, 0, -1))
        else:
            horizon_list = [int(h) for h in horizons if 1 <= int(h) <= max_h]

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
                try:
                    with _candidate_wall_clock_timeout(cfg.candidate_timeout_s):
                        sol = _SuccessfulContactSwitchNLPV004(
                            self.kin, st, cand, target_t, target_R, cfg
                        ).solve()
                except _CandidateSolveTimeout:
                    timed_out += 1
                    timeout_candidate_indices.append(int(candidate_index))
                    self._log(
                        'v0.0.4 candidate timeout',
                        'angle', float(angle_deg),
                        'horizon', float(h),
                        'candidate', int(candidate_index),
                        'limit_s', float(cfg.candidate_timeout_s),
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
                    'angle', float(angle_deg),
                    'horizon', float(h),
                    'candidate', int(candidate_index),
                    'attempted', int(attempted),
                    'objective', float(sol.objective),
                )
                break

            trials.append({
                'horizon_deg': float(h),
                'candidate_count': int(len(candidates)),
                'attempted_candidates': int(attempted),
                'solved_candidates': int(solved),
                'accepted': int(chosen is not None),
                'candidate_timeout_s': float(cfg.candidate_timeout_s),
                'timed_out_candidates': int(timed_out),
                'timeout_candidate_indices': timeout_candidate_indices,
                'early_stop': bool(chosen is not None and attempted < len(candidates)),
                'remaining_candidates_skipped': int(len(candidates) - attempted) if chosen is not None else 0,
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
            }
        return None
