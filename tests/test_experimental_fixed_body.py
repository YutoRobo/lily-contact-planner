import numpy as np
from scipy.optimize import Bounds

from lily_contact_planner.experimental_fixed_body import _fixed_body_bounds


class _Layout:
    body_pos = slice(0, 6)
    body_rotvec = slice(6, 12)
    q = slice(12, 15)


class _DummyNLP:
    layout = _Layout()

    def _body_seed(self):
        t = np.array([[0.0, 0.0, 0.35], [0.1, 0.0, 0.35]])
        R = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
        return t, R


def test_fixed_body_bounds_only_fix_body_variables():
    nlp = _DummyNLP()
    lo = np.full(15, -10.0)
    hi = np.full(15, +10.0)
    base = Bounds(lo, hi)

    fixed = _fixed_body_bounds(nlp, base)
    t, _ = nlp._body_seed()

    assert np.allclose(fixed.lb[nlp.layout.body_pos], t.ravel())
    assert np.allclose(fixed.ub[nlp.layout.body_pos], t.ravel())
    assert np.allclose(fixed.lb[nlp.layout.body_rotvec], 0.0)
    assert np.allclose(fixed.ub[nlp.layout.body_rotvec], 0.0)

    assert np.allclose(fixed.lb[nlp.layout.q], -10.0)
    assert np.allclose(fixed.ub[nlp.layout.q], +10.0)
