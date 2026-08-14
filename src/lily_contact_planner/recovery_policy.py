"""Recovery staging for the v0.0.6 consolidation.

This module deliberately does not introduce a new global action optimizer.
It only makes the recovery order accumulated through v0.0.4-v0.0.6 explicit
and testable so the successful Chat baseline can be reproduced before further
generalization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple


class RecoveryKind(str, Enum):
    ONE_TO_ONE = "one_to_one"
    MULTI_CONTACT = "multi_contact"
    STATIC_RECONFIGURATION = "static_reconfiguration"


@dataclass(frozen=True)
class RecoveryStage:
    kind: RecoveryKind
    max_add: int
    max_remove: int
    require_body_progress: bool


# Search hierarchy preserved from the successful development path.
DEFAULT_RECOVERY_STAGES: Tuple[RecoveryStage, ...] = (
    RecoveryStage(RecoveryKind.ONE_TO_ONE, 1, 1, True),
    RecoveryStage(RecoveryKind.MULTI_CONTACT, 2, 2, True),
    RecoveryStage(RecoveryKind.STATIC_RECONFIGURATION, 2, 2, False),
)


def recovery_stages() -> Iterable[RecoveryStage]:
    """Return the conservative v0.0.6 recovery order.

    Continuous progress with the current support set is intentionally not a
    recovery stage: the caller must try that first. Static reconfiguration is
    last and permits zero body progress only while changing contact state.
    """
    return DEFAULT_RECOVERY_STAGES


__all__ = [
    "RecoveryKind",
    "RecoveryStage",
    "DEFAULT_RECOVERY_STAGES",
    "recovery_stages",
]
