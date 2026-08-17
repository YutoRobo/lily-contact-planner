"""Opt-in fixed-body-trajectory experiment for the finite-horizon NLPs.

This module deliberately leaves the production/free-body NLP classes unchanged.
Calling :func:`enable_fixed_body_trajectory_experiment` replaces only the module-
local class references used by the planner with subclasses whose body position
and rotation-vector bounds are fixed to the existing NLP body seed trajectory.

For the current PitchForwardTask, that seed trajectory is exactly the requested
linear +X translation and constant-rate world +Y pitch over each horizon.

The experiment is therefore reversible by process/runner choice:
- normal runner: original free-body NLPs
- experimental runner: fixed-body bounds enabled for this process only
"""

import numpy as np
from scipy.optimize import Bounds
from scipy.spatial.transform import Rotation


def _fixed_body_bounds(nlp, base_bounds):
    """Return bounds with every body-pose node fixed to ``nlp._body_seed()``."""
    lo = np.asarray(base_bounds.lb, dtype=float).copy()
    hi = np.asarray(base_bounds.ub, dtype=float).copy()

    body_pos, body_R = nlp._body_seed()
    body_rv = Rotation.from_matrix(np.asarray(body_R, float)).as_rotvec()

    body_pos_flat = np.asarray(body_pos, float).ravel()
    body_rv_flat = np.asarray(body_rv, float).ravel()

    lo[nlp.layout.body_pos] = body_pos_flat
    hi[nlp.layout.body_pos] = body_pos_flat
    lo[nlp.layout.body_rotvec] = body_rv_flat
    hi[nlp.layout.body_rotvec] = body_rv_flat
    return Bounds(lo, hi)


def enable_fixed_body_trajectory_experiment():
    """Enable fixed-body bounds for v0.0.4/v0.0.5 NLPs in this Python process.

    Existing classes are not edited.  Instead, the module globals referenced by
    the staged planner are redirected to subclasses that only override bounds().
    A fresh Python process running the normal runner therefore uses the original
    free-body implementation automatically.
    """
    from . import multi_contact_v005
    from . import staged_multi_v005
    from . import trajectory_nlp_v004
    from . import v004_receding
    from . import v004_success_seed

    original_no_contact = trajectory_nlp_v004.NoContactNLPV004
    original_contact = trajectory_nlp_v004.ContactSwitchNLPV004
    original_success_contact = v004_success_seed._SuccessfulContactSwitchNLPV004
    original_multi = multi_contact_v005.MultiContactNLPV005

    class FixedBodyNoContactNLPV004(original_no_contact):
        def bounds(self):
            return _fixed_body_bounds(self, super().bounds())

    class FixedBodyContactSwitchNLPV004(original_contact):
        def bounds(self):
            return _fixed_body_bounds(self, super().bounds())

    class FixedBodySuccessfulContactSwitchNLPV004(original_success_contact):
        def bounds(self):
            return _fixed_body_bounds(self, super().bounds())

    class FixedBodyMultiContactNLPV005(original_multi):
        def bounds(self):
            return _fixed_body_bounds(self, super().bounds())

    # v0.0.4 no-contact / legacy contact references.
    v004_receding.NoContactNLPV004 = FixedBodyNoContactNLPV004
    v004_receding.ContactSwitchNLPV004 = FixedBodyContactSwitchNLPV004

    # Current staged v0.0.4 successful-seed path.
    v004_success_seed._SuccessfulContactSwitchNLPV004 = (
        FixedBodySuccessfulContactSwitchNLPV004
    )

    # Current staged v0.0.5 path and its legacy module reference.
    staged_multi_v005.MultiContactNLPV005 = FixedBodyMultiContactNLPV005
    multi_contact_v005.MultiContactNLPV005 = FixedBodyMultiContactNLPV005

    return {
        "enabled": True,
        "mode": "fixed_body_bounds",
        "v004_no_contact": FixedBodyNoContactNLPV004.__name__,
        "v004_contact": FixedBodySuccessfulContactSwitchNLPV004.__name__,
        "v005_multi": FixedBodyMultiContactNLPV005.__name__,
    }


__all__ = [
    "enable_fixed_body_trajectory_experiment",
    "_fixed_body_bounds",
]
