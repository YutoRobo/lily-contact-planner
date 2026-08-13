#!/usr/bin/env python3
"""Render results/latest_trajectory.npz as a diagnostic GIF.

If the replay files do not exist, run scripts/replay_latest.py first.
Contact switches use the repository visualization convention:
old support retained -> touchdown -> dual support -> transfer -> liftoff.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lily_contact_planner import LilyKinematics
from lily_contact_planner.visualization import (
    DisplayFrame,
    touchdown_first_switch_frames,
)


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


def cube_edges():
    corners = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=float,
    )
    edges = []
    for i, a in enumerate(corners):
        for j, b in enumerate(corners):
            if j <= i:
                continue
            if np.sum(a != b) == 1:
                edges.append((i, j))
    return corners, edges


def indices_from_mask(mask):
    return tuple(int(i) for i in np.where(np.asarray(mask).astype(bool))[0])


def build_display_frames(traj, switches, stride):
    angles = traj["angles_deg"]
    body_t = traj["body_t"]
    body_R = traj["body_R"]
    joint_q = traj["joint_q"]
    support_mask = traj["support_mask"]

    switch_map = {
        int(round(float(p))): i for i, p in enumerate(switches["progress"])
    }

    frames = []
    for k in range(len(angles)):
        a = float(angles[k])
        ai = int(round(a))

        if ai in switch_map:
            j = switch_map[ai]
            before = indices_from_mask(switches["support_before"][j])
            after = indices_from_mask(switches["support_after"][j])
            added = indices_from_mask(switches["added"][j])
            removed = indices_from_mask(switches["removed"][j])
            frames.extend(
                touchdown_first_switch_frames(
                    progress=a,
                    body_t=body_t[k],
                    body_R=body_R[k],
                    q_pre=switches["q_pre"][j],
                    q_post=switches["q_post"][j],
                    support_before=before,
                    support_after=after,
                    added_legs=added,
                    removed_legs=removed,
                )
            )
            continue

        if k % stride != 0 and k != len(angles) - 1:
            continue

        frames.append(
            DisplayFrame(
                progress=a,
                body_t=body_t[k].copy(),
                body_R=body_R[k].copy(),
                joint_q=joint_q[k].copy(),
                support_mask=support_mask[k].copy(),
                note="planner replay state",
            )
        )
    return frames


def draw_frame(ax, kin, frame, half_window):
    ax.cla()
    t = frame.body_t
    R = frame.body_R
    q = frame.joint_q
    support = frame.support_mask.astype(bool)

    # Ground reference.
    x0, y0 = t[0], t[1]
    grid = np.linspace(-half_window, half_window, 9)
    for g in grid:
        ax.plot(
            [x0 - half_window, x0 + half_window],
            [y0 + g, y0 + g],
            [0.0, 0.0],
            linewidth=0.4,
            alpha=0.25,
        )
        ax.plot(
            [x0 + g, x0 + g],
            [y0 - half_window, y0 + half_window],
            [0.0, 0.0],
            linewidth=0.4,
            alpha=0.25,
        )

    # Body cube.
    unit_corners, edges = cube_edges()
    corners_w = t[None, :] + (R @ (kin.a * unit_corners).T).T
    for i, j in edges:
        p0, p1 = corners_w[i], corners_w[j]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            linewidth=2.0,
        )

    # Legs and feet.
    for leg in range(kin.n_legs):
        root, elbow, foot = kin.world_points(t, R, leg, q[leg])
        pts = np.vstack([root, elbow, foot])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0)
        if support[leg]:
            ax.scatter([foot[0]], [foot[1]], [foot[2]], marker="x", s=45)
        else:
            ax.scatter([foot[0]], [foot[1]], [foot[2]], marker="o", s=12)

    # World axes at body center.
    axis_len = 0.18
    for j, label in enumerate(("Xw", "Yw", "Zw")):
        end = t + axis_len * np.eye(3)[j]
        ax.plot([t[0], end[0]], [t[1], end[1]], [t[2], end[2]], linewidth=2.0)
        ax.text(end[0], end[1], end[2], label)

    # Body axes.
    for j, label in enumerate(("Xb", "Yb", "Zb")):
        end = t + axis_len * R[:, j]
        ax.plot([t[0], end[0]], [t[1], end[1]], [t[2], end[2]], linestyle="--")
        ax.text(end[0], end[1], end[2], label)

    ax.set_xlim(t[0] - half_window, t[0] + half_window)
    ax.set_ylim(t[1] - half_window, t[1] + half_window)
    ax.set_zlim(-0.05, max(0.85, t[2] + 0.55))
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_zlabel("world z [m]")
    ax.view_init(elev=22, azim=38)
    ax.set_box_aspect((1.0, 1.0, 0.85))
    ax.set_title(
        "Lily replay  progress=%.2f deg\nsupport=%s\n%s"
        % (frame.progress, list(np.where(support)[0]), frame.note)
    )


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
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "latest.gif",
    )
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--half-window", type=float, default=0.80)
    args = parser.parse_args()

    if not args.trajectory.exists() or not args.switches.exists():
        raise FileNotFoundError(
            "Replay files not found. Run: python3 scripts/replay_latest.py"
        )

    traj = np.load(args.trajectory)
    switches = np.load(args.switches)
    frames = build_display_frames(traj, switches, max(1, int(args.stride)))

    kin = build_kinematics()
    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    writer = PillowWriter(fps=float(args.fps))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(args.out), dpi=100):
        for i, frame in enumerate(frames):
            draw_frame(ax, kin, frame, float(args.half_window))
            writer.grab_frame()
            if i % 100 == 0:
                print("render", i, "/", len(frames), flush=True)

    plt.close(fig)
    print("frames:", len(frames))
    print("gif:", args.out)


if __name__ == "__main__":
    main()
