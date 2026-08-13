#!/usr/bin/env python3
"""Reconstruct a 1-deg trajectory from results/latest_search_report.json.

This script does not search for a new contact sequence.  It replays the saved
contact events from the report and solves the continuous kinematic states again
with the same checked-in planner implementation.

The current script targets the checked-in ForwardRollTask / run_rollwalk_720.py
benchmark.  Future task runners should save task metadata in their reports so
this replay entry point can be generalized without guessing the task.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lily_contact_planner import LilyKinematics
from lily_contact_planner.tasks import ForwardRollTask
from lily_contact_planner.unified_planner import UnifiedContactPlanner


Q0_DEG = np.array(
    [
        [-22.8, -47.6, -4.6],
        [-29.75, 95.0, 39.57],
        [12.13, 5.41, 137.75],
        [0.0, 0.0, 0.0],
        [22.8, -47.6, -4.6],
        [20.75, 85.17, 46.33],
        [12.65, -52.97, 93.10],
        [0.0, 0.0, 0.0],
    ],
    dtype=float,
)
INITIAL_SUPPORT = (1, 2, 5, 6)


def build_kinematics():
    return LilyKinematics(
        a=0.15,
        L2=0.30,
        L3=0.30,
        delta_top=0.0,
        delta_bottom=0.0,
        eps_top=+1.0,
        eps_bottom=-1.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results" / "latest_search_report.json",
    )
    parser.add_argument(
        "--trajectory-out",
        type=Path,
        default=ROOT / "results" / "latest_trajectory.npz",
    )
    parser.add_argument(
        "--switch-out",
        type=Path,
        default=ROOT / "results" / "latest_switch_states.npz",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not report.get("success", False):
        raise RuntimeError(
            "report success=false. Replay expects a completed saved solution."
        )

    best = float(report["best_angle_deg"])
    events = sorted(report.get("events", []), key=lambda e: float(e["angle_deg"]))

    kin = build_kinematics()
    task = ForwardRollTask(body_height_m=0.35, forward_m_per_deg=1.0 / 300.0)
    planner = UnifiedContactPlanner(kin, task, max_roll_deg=best, verbose=False)

    q = np.deg2rad(Q0_DEG).copy()
    support = tuple(INITIAL_SUPPORT)
    t0, R0 = task.pose(0.0)
    anchors = {
        leg: kin.foot_world(t0, R0, leg, q[leg]).copy() for leg in support
    }
    for leg in anchors:
        anchors[leg][2] = 0.0

    event_by_angle = {int(round(float(e["angle_deg"]))): e for e in events}

    angles = np.arange(0.0, best + 1e-9, planner.cfg.step_deg)
    body_t = []
    body_R = []
    joint_q = []
    support_mask = []

    switch_progress = []
    switch_q_pre = []
    switch_q_post = []
    switch_support_before = []
    switch_support_after = []
    switch_added = []
    switch_removed = []

    def mask_of(support_set):
        m = np.zeros(kin.n_legs, dtype=np.uint8)
        m[list(support_set)] = 1
        return m

    for idx, angle in enumerate(angles):
        ai = int(round(float(angle)))

        if idx > 0:
            qn = planner._actual(float(angle), q, support, anchors)
            if qn is None:
                raise RuntimeError(
                    "Replay failed before saved event at angle " + str(float(angle))
                )
            q = qn

        if ai in event_by_angle:
            e = event_by_angle[ai]
            q_pre = q.copy()
            support_before = tuple(int(x) for x in e["support_before"])
            support_after = tuple(int(x) for x in e["support_after"])
            added = tuple(int(x) for x in e.get("add", []))
            removed = tuple(int(x) for x in e.get("remove", []))

            if tuple(support) != support_before:
                raise RuntimeError(
                    "Replay support mismatch at %.1f deg: current=%s report=%s"
                    % (angle, support, support_before)
                )

            new_anchors = {
                leg: anchors[leg].copy()
                for leg in support_after
                if leg in anchors
            }
            for leg in added:
                key = str(leg)
                new_anchors[leg] = np.asarray(e["anchors_added"][key], dtype=float)

            q_seed = q.copy()
            for leg in added:
                key = str(leg)
                q_seed[leg] = np.asarray(e["qgoal_added"][key], dtype=float)

            q_after = planner._actual(
                float(angle), q_seed, support_after, new_anchors
            )
            if q_after is None:
                raise RuntimeError(
                    "Replay failed while applying saved contact event at %.1f deg"
                    % angle
                )

            switch_progress.append(float(angle))
            switch_q_pre.append(q_pre)
            switch_q_post.append(q_after.copy())
            switch_support_before.append(mask_of(support_before))
            switch_support_after.append(mask_of(support_after))
            switch_added.append(mask_of(added))
            switch_removed.append(mask_of(removed))

            q = q_after
            support = support_after
            anchors = new_anchors

        t, R = task.pose(float(angle))
        body_t.append(t)
        body_R.append(R)
        joint_q.append(q.copy())
        support_mask.append(mask_of(support))

    args.trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.trajectory_out,
        angles_deg=np.asarray(angles, dtype=float),
        body_t=np.asarray(body_t, dtype=float),
        body_R=np.asarray(body_R, dtype=float),
        joint_q=np.asarray(joint_q, dtype=float),
        support_mask=np.asarray(support_mask, dtype=np.uint8),
    )

    np.savez_compressed(
        args.switch_out,
        progress=np.asarray(switch_progress, dtype=float),
        q_pre=np.asarray(switch_q_pre, dtype=float),
        q_post=np.asarray(switch_q_post, dtype=float),
        support_before=np.asarray(switch_support_before, dtype=np.uint8),
        support_after=np.asarray(switch_support_after, dtype=np.uint8),
        added=np.asarray(switch_added, dtype=np.uint8),
        removed=np.asarray(switch_removed, dtype=np.uint8),
    )

    print("replay states:", len(angles))
    print("contact switches:", len(switch_progress))
    print("trajectory:", args.trajectory_out)
    print("switch states:", args.switch_out)


if __name__ == "__main__":
    main()
