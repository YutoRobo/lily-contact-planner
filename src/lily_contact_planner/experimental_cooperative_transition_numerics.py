"""Numerical consistency patch for the cooperative transition experiment.

This module intentionally does not change contact-mode candidates, objective
weights, hard safety constraints, or acceptance rules.  It only aligns the NLP
initial guess with constraints that were added by the cooperative-transition
experiment and exposes solver diagnostics for controlled A/B evaluation.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .analytic_ik import analytic_leg_ik_world
from . import experimental_cooperative_transition as _base


_LAST_DIAGNOSTICS = None
_ORIGINAL_CLASS = _base.CooperativeTransitionNLP


def _required_liftoff_clearance(k, liftoff_node, exec_node, clearance_m):
    """Clearance profile used by both the hard constraint and the seed."""
    k = int(k)
    liftoff_node = int(liftoff_node)
    exec_node = int(exec_node)
    if k <= liftoff_node:
        return 0.0
    denom = max(1, exec_node - liftoff_node)
    alpha = min(1.0, (k - liftoff_node) / float(denom))
    return float(clearance_m) * alpha


class NumericallyConsistentCooperativeTransitionNLP(_ORIGINAL_CLASS):
    """Same cooperative NLP with a constraint-consistent liftoff seed."""

    def initial_guess(self):
        z = super().initial_guess()
        if not self.lolegs:
            return z

        t, rv, q, _, _ = self.layout.unpack(z)
        body_R = self._rot(rv)
        q = np.asarray(q, dtype=float).copy()

        for leg0, node0 in zip(self.lolegs, self.lonodes):
            leg = int(leg0)
            node = int(node0)
            anchor = np.asarray(self.state.anchors_world[leg], dtype=float)
            q_ref = np.asarray(q[node, leg], dtype=float).copy()

            for k in range(node + 1, self.settings.n_nodes):
                clearance = _required_liftoff_clearance(
                    k,
                    node,
                    self.exec_node,
                    self.liftoff_clearance_m,
                )
                target = anchor + np.array([0.0, 0.0, clearance], dtype=float)
                branches = analytic_leg_ik_world(
                    self.kin,
                    t[k],
                    body_R[k],
                    leg,
                    target,
                    q_reference=q_ref,
                    residual_tol=2e-6,
                )
                if branches:
                    q[k, leg] = np.asarray(branches[0], dtype=float)
                    q_ref = q[k, leg].copy()

        z[self.layout.q] = q.ravel()
        return z

    def solve(self):
        global _LAST_DIAGNOSTICS

        z0 = self.initial_guess()
        eq0 = np.asarray(self.equality_constraints(z0), dtype=float)
        ineq0 = np.asarray(self.inequality_constraints(z0), dtype=float)
        t0, rv0, _, _, _ = self.layout.unpack(z0)
        R0 = self._rot(rv0)
        exec_pos_error = float(np.linalg.norm(
            t0[self.exec_node] - self.exec_body_pos
        ))
        exec_rot_error = float(np.linalg.norm(
            Rotation.from_matrix(
                R0[self.exec_node].dot(self.exec_body_R.T)
            ).as_rotvec()
        ))

        initial_diag = {
            "success": None,
            "message": "solver_running_or_timed_out",
            "eq_max": None,
            "ineq_min": None,
            "nit": None,
            "nfev": None,
            "objective": None,
            "initial_eq_max": float(np.max(np.abs(eq0))) if eq0.size else 0.0,
            "initial_ineq_min": float(np.min(ineq0)) if ineq0.size else float("inf"),
            "initial_exec_body_pos_error_m": exec_pos_error,
            "initial_exec_body_rot_error_rad": exec_rot_error,
            "seed_policy": "liftoff_clearance_matches_hard_constraint",
        }
        _LAST_DIAGNOSTICS = initial_diag

        result = super().solve()
        _LAST_DIAGNOSTICS = {
            **initial_diag,
            "success": bool(result.get("success", False)),
            "message": str(result.get("message", "")),
            "eq_max": float(result.get("eq_max", float("nan"))),
            "ineq_min": float(result.get("ineq_min", float("nan"))),
            "nit": int(result.get("nit", -1)),
            "nfev": int(result.get("nfev", -1)),
            "objective": float(result.get("objective", float("nan"))),
        }
        return result


def reset_cooperative_nlp_diagnostics():
    global _LAST_DIAGNOSTICS
    _LAST_DIAGNOSTICS = None


def get_cooperative_nlp_diagnostics():
    if _LAST_DIAGNOSTICS is None:
        return None
    return dict(_LAST_DIAGNOSTICS)


def enable_consistent_cooperative_numerics():
    """Install the seed/diagnostic patch in the current Python process only."""
    if _base.CooperativeTransitionNLP is NumericallyConsistentCooperativeTransitionNLP:
        return {
            "enabled": True,
            "already_enabled": True,
            "semantics_changed": False,
            "seed_policy": "liftoff_clearance_matches_hard_constraint",
        }

    if _base.CooperativeTransitionNLP is not _ORIGINAL_CLASS:
        raise RuntimeError(
            "CooperativeTransitionNLP was already replaced by an unknown patch"
        )

    _base.CooperativeTransitionNLP = NumericallyConsistentCooperativeTransitionNLP
    return {
        "enabled": True,
        "already_enabled": False,
        "semantics_changed": False,
        "seed_policy": "liftoff_clearance_matches_hard_constraint",
    }


__all__ = [
    "NumericallyConsistentCooperativeTransitionNLP",
    "_required_liftoff_clearance",
    "enable_consistent_cooperative_numerics",
    "reset_cooperative_nlp_diagnostics",
    "get_cooperative_nlp_diagnostics",
]
