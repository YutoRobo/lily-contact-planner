#!/usr/bin/env python3
"""Run Pitch45 -> Roll45 locally; inspect Yaw startup feasibility in Actions."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lily_contact_planner.kinematics import LilyKinematics
from lily_contact_planner.nlp_geometry_v002 import geometry_inequalities
from lily_contact_planner.tasks import Pitch45ThenRoll45Task, Yaw45Pitch145Roll145WorldTask
from lily_contact_planner.trajectory_nlp_v004 import NoContactNLPV004
from lily_contact_planner.unified_planner import UnifiedContactPlanner


def _serializable_result(result):
    serial = dict(result)
    if serial.get("final_q") is not None:
        serial["final_q"] = np.asarray(serial["final_q"]).tolist()
    if serial.get("final_anchors") is not None:
        serial["final_anchors"] = {
            str(k): np.asarray(v).tolist()
            for k, v in serial["final_anchors"].items()
        }
    serial.pop("best_trajectory", None)
    return serial


def _ci_probe(kin, q0, support0):
    print("CI_INITIAL_STATE_FEASIBILITY_PROBE", flush=True)
    for task in (Pitch45ThenRoll45Task(), Yaw45Pitch145Roll145WorldTask()):
        planner = UnifiedContactPlanner(
            kin, task, max_roll_deg=task.total_progress_deg, verbose=False
        )
        cfg = planner._v004_settings()
        t0, R0 = task.pose(0.0)
        roots, elbows, feet = [], [], []
        for leg in range(kin.n_legs):
            r, e, f = kin.world_points(t0, R0, leg, q0[leg])
            roots.append(r); elbows.append(e); feet.append(f)
        roots = np.asarray(roots); elbows = np.asarray(elbows); feet = np.asarray(feet)

        g0, _ = geometry_inequalities(
            kin, t0, R0, q0, set(support0), cfg
        )
        anchors0 = {
            leg: kin.foot_world(t0, R0, leg, q0[leg]).copy()
            for leg in support0
        }
        for leg in anchors0:
            anchors0[leg][2] = 0.0
        st = planner._v004_state(0.0, q0, support0, anchors0)
        target_t, target_R = task.pose(5.0)
        nlp = NoContactNLPV004(kin, st, target_t, target_R, cfg)

        tic = time.perf_counter()
        z0 = nlp.initial_guess()
        initial_guess_s = time.perf_counter() - tic
        tic = time.perf_counter()
        eq = nlp.equality_constraints(z0)
        eq_s = time.perf_counter() - tic
        tic = time.perf_counter()
        ineq = nlp.inequality_constraints(z0)
        ineq_s = time.perf_counter() - tic

        print("TASK_PROBE", type(task).__name__, flush=True)
        print(" body_height_m", float(t0[2]), flush=True)
        print(" support_foot_z_m", [float(feet[i,2]) for i in support0], flush=True)
        print(" min_all_foot_z_m", float(np.min(feet[:,2])), flush=True)
        print(" min_all_elbow_z_m", float(np.min(elbows[:,2])), flush=True)
        print(" initial_geometry_min", float(np.min(g0)), flush=True)
        print(" nlp_variables", int(nlp.layout.size), flush=True)
        print(" equality_count", int(eq.size), "eq_max", float(np.max(np.abs(eq))), flush=True)
        print(" inequality_count", int(ineq.size), "ineq_min", float(np.min(ineq)), flush=True)
        print(" seed_feasible", bool(
            np.max(np.abs(eq)) <= cfg.constraint_tolerance
            and np.min(ineq) >= -cfg.constraint_tolerance
        ), flush=True)
        print(" timing_s initial_guess", initial_guess_s, "eq", eq_s, "ineq", ineq_s, flush=True)
    print("CI_INITIAL_STATE_FEASIBILITY_PROBE_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="full_v006_fresh_result.json")
    parser.add_argument("--checkpoint", default="full_v006_fresh_checkpoint.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    kin = LilyKinematics(
        a=0.15, L2=0.30, L3=0.30,
        delta_top=0.0, delta_bottom=0.0,
        eps_top=+1.0, eps_bottom=-1.0,
    )
    q0 = np.tile(np.deg2rad([0.0, 20.0, -30.0]), (8, 1))
    support0 = (2, 4, 6)

    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        _ci_probe(kin, q0, support0)
        return

    task = Pitch45ThenRoll45Task()
    planner = UnifiedContactPlanner(
        kin, task, max_roll_deg=90.0, verbose=not args.quiet
    )
    planner.checkpoint_path = str(Path(args.checkpoint))
    print("CHECKPOINT_PATH", planner.checkpoint_path, flush=True)
    try:
        result = planner.plan(q0, support0)
    except KeyboardInterrupt:
        print("INTERRUPTED checkpoint retained at " + planner.checkpoint_path, flush=True)
        return

    serial = _serializable_result(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(serial, f, indent=2)
    print("FINAL_RESULT", json.dumps({
        "success": result["success"],
        "best_angle_deg": result["best_angle_deg"],
        "nodes": result["nodes"],
        "n_events": len(result.get("events", [])),
        "final_support": result.get("final_support"),
        "output": str(output),
        "checkpoint": planner.checkpoint_path,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
