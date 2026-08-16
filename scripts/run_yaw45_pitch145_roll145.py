#!/usr/bin/env python3
"""Run Yaw45 -> translating Pitch145 -> translating Roll-145.

The current UnifiedContactPlanner search policy is unchanged.  This runner uses
an initial posture that is rebuilt for the task's 0.35 m body height; the
Pitch45 -> Roll45 regression runner and its archived initial condition are not
modified.
"""

import argparse
import faulthandler
import json
from pathlib import Path
import sys

import numpy as np

# Prefer the source tree that belongs to this checkout.  This avoids accidentally
# importing an older lily_contact_planner previously installed in site-packages.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lily_contact_planner.analytic_ik import analytic_leg_ik_world
from lily_contact_planner.kinematics import LilyKinematics
from lily_contact_planner.nlp_geometry_v002 import geometry_inequalities
from lily_contact_planner.tasks import (
    Pitch45ThenRoll45Task,
    Yaw45Pitch145Roll145WorldTask,
)
from lily_contact_planner.unified_planner import UnifiedContactPlanner
from lily_contact_planner.v004_types import V004Settings


REFERENCE_Q_DEG = np.array([0.0, 20.0, -30.0], dtype=float)
INITIAL_SUPPORT = (2, 4, 6)


def _serializable_result(result):
    serial = dict(result)
    if serial.get("final_q") is not None:
        serial["final_q"] = np.asarray(serial["final_q"]).tolist()
    if serial.get("final_anchors") is not None:
        serial["final_anchors"] = {
            str(k): np.asarray(v).tolist()
            for k, v in serial["final_anchors"].items()
        }
    # Dense trajectory data are persisted separately by the checkpoint storage
    # mixin; do not duplicate them in the final summary JSON.
    serial.pop("best_trajectory", None)
    return serial


def _build_height_consistent_initial_q(kin, task):
    """Rebuild q0 for the 0.35 m task without touching the 45->45 baseline.

    The old regression posture is used only to define the horizontal foot
    layout.  At the new body height every leg is solved again with analytic IK
    to the same foot XY and z=0.  This removes the previous 0.1746 m ground
    penetration caused by reusing the regression joint angles at a lower body.
    """
    q_reference = np.tile(np.deg2rad(REFERENCE_Q_DEG), (kin.n_legs, 1))

    reference_task = Pitch45ThenRoll45Task()
    tref, Rref = reference_task.pose(0.0)
    t0, R0 = task.pose(0.0)

    q0 = np.empty_like(q_reference)
    target_feet = []
    for leg in range(kin.n_legs):
        target = kin.foot_world(tref, Rref, leg, q_reference[leg]).copy()
        target[2] = 0.0
        target_feet.append(target.copy())

        branches = analytic_leg_ik_world(
            kin,
            t0,
            R0,
            leg,
            target,
            q_reference=q_reference[leg],
            residual_tol=2e-6,
        )
        safe = []
        for q_leg in branches:
            root, elbow, foot = kin.world_points(t0, R0, leg, q_leg)
            min_z = float(min(root[2], elbow[2], foot[2]))
            if min_z >= -1e-8:
                safe.append((float(np.linalg.norm(q_leg - q_reference[leg])), q_leg))
        if not safe:
            raise RuntimeError(
                "cannot construct a ground-safe 0.35 m initial IK posture "
                "for leg {} at target {}".format(leg, target.tolist())
            )
        safe.sort(key=lambda item: item[0])
        q0[leg] = safe[0][1]

    # Reject an invalid initial state before entering SLSQP.  This is runner-side
    # validation only; planner search semantics are unchanged.
    cfg = V004Settings()
    geom, _ = geometry_inequalities(
        kin, t0, R0, q0, set(INITIAL_SUPPORT), cfg
    )
    geom_min = float(np.min(geom))
    if geom_min < -1e-8:
        raise RuntimeError(
            "constructed 0.35 m initial posture is globally infeasible: "
            "geometry_min={:.9g}".format(geom_min)
        )

    foot_z = []
    elbow_z = []
    for leg in range(kin.n_legs):
        _, elbow, foot = kin.world_points(t0, R0, leg, q0[leg])
        elbow_z.append(float(elbow[2]))
        foot_z.append(float(foot[2]))

    diagnostics = {
        "q0_deg": np.rad2deg(q0).tolist(),
        "target_feet_world_m": np.asarray(target_feet).tolist(),
        "support0": list(INITIAL_SUPPORT),
        "min_foot_z_m": float(min(foot_z)),
        "min_elbow_z_m": float(min(elbow_z)),
        "geometry_min": geom_min,
    }
    return q0, INITIAL_SUPPORT, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="yaw45_pitch145_roll145_result.json",
        help="Final JSON output path",
    )
    parser.add_argument(
        "--checkpoint",
        default="yaw45_pitch145_roll145_checkpoint.json",
        help="Atomic BEST-state checkpoint JSON path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable planner progress logging",
    )
    parser.add_argument(
        "--stack-dump-interval-s",
        type=float,
        default=30.0,
        help=(
            "Diagnostic only: dump the current Python stack every N seconds "
            "while planning. Set 0 to disable. This does not alter search semantics."
        ),
    )
    parser.add_argument(
        "--probe-init-only",
        action="store_true",
        help="Construct and validate q0, print diagnostics, then exit.",
    )
    args = parser.parse_args()

    kin = LilyKinematics(
        a=0.15,
        L2=0.30,
        L3=0.30,
        delta_top=0.0,
        delta_bottom=0.0,
        eps_top=+1.0,
        eps_bottom=-1.0,
    )
    task = Yaw45Pitch145Roll145WorldTask()
    planner = UnifiedContactPlanner(
        kin,
        task,
        max_roll_deg=task.total_progress_deg,
        verbose=not args.quiet,
    )
    planner.checkpoint_path = str(Path(args.checkpoint))

    q0, support0, init_diag = _build_height_consistent_initial_q(kin, task)

    print(
        "TASK",
        json.dumps(
            {
                "yaw_deg": task.yaw_deg,
                "pitch_deg": task.pitch_deg,
                "roll_deg": -task.roll_deg,
                "total_progress_deg": task.total_progress_deg,
                "body_height_m": task.body_height_m,
                "translation_m_per_deg": task.forward_m_per_deg,
                "pitch_translation_x_m": task.forward_m_per_deg * task.pitch_deg,
                "roll_translation_y_m": task.forward_m_per_deg * task.roll_deg,
                "phase_boundaries_deg": list(task.phase_boundaries_deg),
            }
        ),
        flush=True,
    )
    print("INITIAL_STATE", json.dumps(init_diag), flush=True)
    print("CHECKPOINT_PATH", planner.checkpoint_path, flush=True)

    if args.probe_init_only:
        print("INITIAL_STATE_PROBE_OK", flush=True)
        return

    stack_interval = max(0.0, float(args.stack_dump_interval_s))
    faulthandler.enable()
    if stack_interval > 0.0:
        print("STACK_DUMP_INTERVAL_S", stack_interval, flush=True)
        faulthandler.dump_traceback_later(stack_interval, repeat=True)

    try:
        try:
            result = planner.plan(q0, support0)
        except KeyboardInterrupt:
            print(
                "INTERRUPTED checkpoint/trajectory retained at " + planner.checkpoint_path,
                flush=True,
            )
            return
    finally:
        if stack_interval > 0.0:
            faulthandler.cancel_dump_traceback_later()

    serial = _serializable_result(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2)

    summary = {
        "success": result["success"],
        "best_angle_deg": result["best_angle_deg"],
        "target_progress_deg": task.total_progress_deg,
        "nodes": result["nodes"],
        "n_events": len(result.get("events", [])),
        "final_support": result.get("final_support"),
        "event_versions": [e.get("version") for e in result.get("events", [])],
        "search_stats": result.get("search_stats"),
        "fallback_free": result.get("fallback_free"),
        "output": str(output),
        "checkpoint": planner.checkpoint_path,
    }
    print("FINAL_RESULT", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
