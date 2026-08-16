"""Unified Level-1 contact search for Lily.

The same algorithm is applied at every path sample. There are no switch-angle
special cases and no saved future contact states. Contact decisions remain
discrete; continuous support/swing motion is solved with nonlinear kinematics.
"""

from .planner_base import PlannerBaseMixin, PlannerSettings
from .planner_touchdown import TouchdownSearchMixin
from .staged_search_v006 import V006StagedSearchMixin
from .prm_recovery import PRMStaticRecoveryMixin
from .static_prm_candidates import StaticPRMCandidateMixin
from .staged_multi_v005 import V005StagedHorizonMixin
from .v004_success_seed import V004SuccessfulSeedMixin
from .checkpoint_trajectory import CheckpointTrajectoryMixin
from .checkpoint_storage import SplitCheckpointStorageMixin


class UnifiedContactPlanner(
    SplitCheckpointStorageMixin,
    CheckpointTrajectoryMixin,
    StaticPRMCandidateMixin,
    PRMStaticRecoveryMixin,
    V005StagedHorizonMixin,
    V004SuccessfulSeedMixin,
    V006StagedSearchMixin,
    TouchdownSearchMixin,
    PlannerBaseMixin,
):
    """Recovered v0.0.4 -> v0.0.5 -> v0.0.6 staged planner."""


__all__ = ["UnifiedContactPlanner", "PlannerSettings"]
