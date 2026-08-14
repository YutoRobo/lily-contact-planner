from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class V004Settings:
    n_nodes: int = 11
    checker_samples: int = 101
    max_horizon_rotation_rad: float = np.deg2rad(5.0)
    max_horizon_translation_m: float = 0.05
    execution_fraction: float = 0.2
    touchdown_node: int = 5
    liftoff_node: int = 6
    touchdown_seeds_per_leg: int = 5
    touchdown_seed_min_xy_distance_m: float = 0.05
    seed_samples_per_leg: int = 256
    l2_radius_m: float = 0.025
    l3_radius_m: float = 0.025
    collision_margin_m: float = 0.0
    support_margin_m: float = 0.0
    swing_ground_margin_m: float = 0.0
    root_attachment_ignore_m: Optional[float] = None
    weight_tracking_translation: float = 1.0
    weight_tracking_rotation: float = 1.0
    weight_joint_motion: float = 1.0e-3
    weight_joint_smoothness: float = 1.0e-4
    constraint_tolerance: float = 1.0e-6
    checker_tolerance: float = 1.0e-5
    maxiter: int = 300
    ftol: float = 1.0e-8
    initial_liftoff_clearance_m: float = 0.02

    def __post_init__(self):
        if self.n_nodes < 3:
            raise ValueError("n_nodes must be >= 3")
        if self.checker_samples < self.n_nodes:
            raise ValueError("checker_samples must be >= n_nodes")
        if not (0 < self.touchdown_node < self.liftoff_node < self.n_nodes):
            raise ValueError("require 0 < touchdown_node < liftoff_node < n_nodes")
        if not (0.0 < self.execution_fraction <= 1.0):
            raise ValueError("execution_fraction must be in (0,1]")

    @property
    def root_ignore_m(self):
        if self.root_attachment_ignore_m is not None:
            return float(self.root_attachment_ignore_m)
        return 2.0 * max(float(self.l2_radius_m), float(self.l3_radius_m))


@dataclass(frozen=True)
class V004State:
    body_pos: np.ndarray
    body_R: np.ndarray
    q: np.ndarray
    contact: np.ndarray
    anchors_world: Dict[int, np.ndarray]

    def validate(self, kin):
        if np.asarray(self.body_pos, float).shape != (3,): raise ValueError("body_pos shape")
        if np.asarray(self.body_R, float).shape != (3,3): raise ValueError("body_R shape")
        if np.asarray(self.q, float).shape != (kin.n_legs,3): raise ValueError("q shape")
        c=np.asarray(self.contact,bool)
        if c.shape != (kin.n_legs,): raise ValueError("contact shape")
        support=np.where(c)[0]
        if len(support)==0: raise ValueError("at least one support required")
        for leg in support:
            if int(leg) not in self.anchors_world: raise ValueError("missing support anchor")


@dataclass(frozen=True)
class ContactCandidateV004:
    touchdown_leg: int
    touchdown_seed_xy: np.ndarray
    liftoff_leg: int
    support_area_m2: float


@dataclass
class DenseCheckReportV004:
    feasible: bool
    max_joint_violation_rad: float
    max_support_lock_error_m: float
    max_touchdown_lock_error_m: float
    min_swing_foot_height_m: float
    min_support_margin_m: float
    min_body_ground_clearance_m: float
    min_link_ground_height_m: float
    min_capsule_clearance_m: float
    min_body_link_clearance_m: float
    terminal_position_error_m: float
    terminal_rotation_error_rad: float
    support_projection_ik_failures: int = 0
    max_dense_joint_step_rad: float = 0.0


@dataclass
class TrajectorySolutionV004:
    mode: str
    success: bool
    message: str
    objective: float
    body_pos: np.ndarray
    body_R: np.ndarray
    q: np.ndarray
    touchdown_xy: Optional[np.ndarray] = None
    candidate: Optional[ContactCandidateV004] = None
    checker: Optional[DenseCheckReportV004] = None
    scipy_result: object = None

    @property
    def accepted(self):
        return bool(self.success and self.checker is not None and self.checker.feasible)


@dataclass
class CycleResultV004:
    mode: str
    best: Optional[TrajectorySolutionV004]
    no_contact: TrajectorySolutionV004
    contact_solutions: List[TrajectorySolutionV004]
    trigger_reason: str
