#!/usr/bin/env python3
"""Dense finite-thickness collision audit for a replayed Lily trajectory.

This is intentionally independent of the contact-search scoring.  It answers:
where, which two links, and by how many metres does a replayed path violate
finite-thickness capsule clearance?

The audit checks:
- every L2/L3 capsule pair between different legs
- every capsule against the body cube
- dense samples between stored replay states
- dense samples through each stored contact-switch pre/post state

The current Lily model has only two links per leg, so the only same-leg pair is
adjacent L2/L3 sharing J3 and is intentionally exempted at that joint.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lily_contact_planner import LilyKinematics
from lily_contact_planner.collision import build_capsules, evaluate_capsules


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


def interp_R(R0, R1, u):
    rots = Rotation.from_matrix(np.stack([R0, R1], axis=0))
    return Slerp([0.0, 1.0], rots)([float(u)]).as_matrix()[0]


def evaluate_state(kin, progress, t, R, q, radii, margin, root_ignore, source):
    roots, elbows, feet = [], [], []
    for leg in range(kin.n_legs):
        root, elbow, foot = kin.world_points(t, R, leg, q[leg])
        roots.append(root)
        elbows.append(elbow)
        feet.append(foot)
    capsules = build_capsules(np.asarray(roots), np.asarray(elbows), np.asarray(feet), radii)
    sample = evaluate_capsules(
        capsules,
        body_t=t,
        body_R=R,
        body_half_extent_m=kin.a,
        margin_m=margin,
        root_attachment_ignore_m=root_ignore,
    )
    return {
        "progress_deg": float(progress),
        "source": source,
        "self_collision_ok": bool(sample.self_collision_ok),
        "link_body_collision_ok": bool(sample.link_body_collision_ok),
        "min_capsule_clearance_m": float(sample.min_capsule_clearance_m),
        "min_body_clearance_m": float(sample.min_body_clearance_m),
        "worst_capsule_pair": None if sample.worst_capsule_pair is None else list(sample.worst_capsule_pair),
        "worst_body_link": None if sample.worst_body_link is None else list(sample.worst_body_link),
    }


def link_name(link):
    return "L2(root-J3)" if int(link) == 0 else "L3(J3-foot)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=ROOT / "results" / "latest_trajectory.npz",
    )
    parser.add_argument(
        "--switches",
        type=Path,
        default=ROOT / "results" / "latest_switch_states.npz",
    )
    parser.add_argument("--l2-radius-m", type=float, required=True)
    parser.add_argument("--l3-radius-m", type=float, required=True)
    parser.add_argument("--margin-m", type=float, default=0.0)
    parser.add_argument(
        "--root-attachment-ignore-m",
        type=float,
        default=None,
        help=(
            "Length of proximal L2 ignored only for body attachment. "
            "Default: 2*max(link radius)."
        ),
    )
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "latest_capsule_collision_audit.json",
    )
    args = parser.parse_args()

    if not args.trajectory.exists():
        raise FileNotFoundError(
            "Replay trajectory not found. Run: python3 scripts/replay_latest.py"
        )

    radii = (float(args.l2_radius_m), float(args.l3_radius_m))
    if min(radii) < 0.0:
        raise ValueError("capsule radii must be >= 0")
    root_ignore = (
        2.0 * max(radii)
        if args.root_attachment_ignore_m is None
        else float(args.root_attachment_ignore_m)
    )
    substeps = max(1, int(args.substeps))

    traj = np.load(args.trajectory)
    angles = np.asarray(traj["angles_deg"], dtype=float)
    body_t = np.asarray(traj["body_t"], dtype=float)
    body_R = np.asarray(traj["body_R"], dtype=float)
    joint_q = np.asarray(traj["joint_q"], dtype=float)
    kin = build_kinematics()

    worst_self = None
    worst_body = None
    collision_samples = 0
    body_collision_samples = 0
    n_samples = 0

    def consume(rec):
        nonlocal worst_self, worst_body, collision_samples, body_collision_samples, n_samples
        n_samples += 1
        if not rec["self_collision_ok"]:
            collision_samples += 1
        if not rec["link_body_collision_ok"]:
            body_collision_samples += 1
        if worst_self is None or rec["min_capsule_clearance_m"] < worst_self["min_capsule_clearance_m"]:
            worst_self = rec
        if worst_body is None or rec["min_body_clearance_m"] < worst_body["min_body_clearance_m"]:
            worst_body = rec

    # Dense samples between all stored replay states.  The interpolation is a
    # diagnostic reconstruction: body rotation uses Slerp and q is linear.
    for k in range(len(angles) - 1):
        for j in range(substeps):
            u = j / float(substeps)
            progress = (1.0 - u) * angles[k] + u * angles[k + 1]
            t = (1.0 - u) * body_t[k] + u * body_t[k + 1]
            R = interp_R(body_R[k], body_R[k + 1], u)
            q = (1.0 - u) * joint_q[k] + u * joint_q[k + 1]
            consume(
                evaluate_state(
                    kin, progress, t, R, q, radii, args.margin_m,
                    root_ignore, "between-replay-states"
                )
            )
    consume(
        evaluate_state(
            kin, angles[-1], body_t[-1], body_R[-1], joint_q[-1], radii,
            args.margin_m, root_ignore, "terminal-replay-state"
        )
    )

    # Explicitly audit switch pre->post configurations at fixed body pose.
    # This detects pass-through in the numerical switch reconstruction even
    # though a future planner should replace it with a certified swing path.
    switch_count = 0
    if args.switches.exists():
        switches = np.load(args.switches)
        for i in range(len(switches["progress"])):
            a = float(switches["progress"][i])
            # use task pose stored nearest to this progress
            k = int(np.argmin(np.abs(angles - a)))
            q0 = np.asarray(switches["q_pre"][i], dtype=float)
            q1 = np.asarray(switches["q_post"][i], dtype=float)
            for j in range(substeps + 1):
                u = j / float(substeps)
                q = (1.0 - u) * q0 + u * q1
                consume(
                    evaluate_state(
                        kin, a + 1e-3 * u, body_t[k], body_R[k], q, radii,
                        args.margin_m, root_ignore, "contact-switch-pre-post"
                    )
                )
            switch_count += 1

    result = {
        "model": {
            "L2_capsule_radius_m": radii[0],
            "L3_capsule_radius_m": radii[1],
            "clearance_margin_m": float(args.margin_m),
            "root_attachment_ignore_m": float(root_ignore),
            "body": "cube; exact segment-to-AABB centerline distance minus capsule radius",
            "pair_rule": "all inter-leg L2/L3 segment pairs; same-leg adjacent L2/L3 joint exempted",
        },
        "sampling": {
            "substeps_per_replay_interval": substeps,
            "trajectory_interpolation": "linear q + linear t + quaternion Slerp R (diagnostic)",
            "contact_switch_interpolation": "linear q_pre->q_post at fixed body pose (diagnostic)",
            "n_samples": n_samples,
            "n_switches": switch_count,
        },
        "self_collision_ok": bool(collision_samples == 0),
        "link_body_collision_ok": bool(body_collision_samples == 0),
        "self_collision_sample_count": int(collision_samples),
        "link_body_collision_sample_count": int(body_collision_samples),
        "worst_self_collision": worst_self,
        "worst_link_body_collision": worst_body,
        "interpretation": (
            "Negative clearance means finite-thickness overlap.  This audit is "
            "dense diagnostic validation of an interpolated replay, not yet a "
            "continuous-time collision certificate or a planner-generated swing trajectory."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("self_collision_ok:", result["self_collision_ok"])
    print("link_body_collision_ok:", result["link_body_collision_ok"])
    if worst_self is not None:
        pair = worst_self["worst_capsule_pair"]
        print("worst capsule clearance [m]:", worst_self["min_capsule_clearance_m"])
        print("worst progress [deg]:", worst_self["progress_deg"])
        if pair is not None:
            print(
                "worst pair: leg%d %s vs leg%d %s"
                % (pair[0], link_name(pair[1]), pair[2], link_name(pair[3]))
            )
        print("worst source:", worst_self["source"])
    if worst_body is not None:
        bl = worst_body["worst_body_link"]
        print("worst body clearance [m]:", worst_body["min_body_clearance_m"])
        print("worst body progress [deg]:", worst_body["progress_deg"])
        if bl is not None:
            print("worst body link: leg%d %s" % (bl[0], link_name(bl[1])))
        print("worst body source:", worst_body["source"])
    print("report:", args.out)


if __name__ == "__main__":
    main()
