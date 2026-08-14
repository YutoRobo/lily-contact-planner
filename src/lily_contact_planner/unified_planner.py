"""Unified Level-1 contact search for Lily.

The same algorithm is applied at every path sample. There are no switch-angle
special cases and no saved future contact states. Contact decisions remain
discrete; continuous support/swing motion is solved with nonlinear kinematics.
"""

from .planner_base import PlannerBaseMixin, PlannerSettings
from .planner_touchdown import TouchdownSearchMixin
from .staged_search_v006 import V006StagedSearchMixin
from .prm_recovery import PRMStaticRecoveryMixin
from .multi_contact_v005 import V005MultiRecoveryMixin


class UnifiedContactPlanner(
    PRMStaticRecoveryMixin,
    V005MultiRecoveryMixin,
    V006StagedSearchMixin,
    TouchdownSearchMixin,
    PlannerBaseMixin,
):
    """Angle-independent hybrid discrete-contact / nonlinear-IK planner."""


__all__ = ["UnifiedContactPlanner", "PlannerSettings"]
