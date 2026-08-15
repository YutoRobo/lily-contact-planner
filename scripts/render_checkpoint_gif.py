#!/usr/bin/env python3
"""Render a checkpoint_v2 BEST trajectory directly to GIF.

Planner keyframes come from the checkpoint. Extra frames inserted here are
visualization-only interpolation and are not planner-certified states.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lily_contact_planner.kinematics import LilyKinematics
from lily_contact_planner.visualization import (
    DisplayFrame,
    interpolate_same_support,
    support_mask,
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


def load_checkpoint(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "checkpoint_v2":
        raise RuntimeError(
            "checkpoint_v2 required. Re-run the planner with the trajectory checkpoint update."
        )
    trajectory = data.get("best_trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError("best_trajectory is missing or empty")
    return data


def keyframe(item):
    support = tuple(int(x) for x in item["support"])
    return DisplayFrame(
        progress=float(item["progress_deg"]),
        body_t=np.asarray(item["body_t"], dtype=float),
        body_R=np.asarray(item["body_R"], dtype=float),
        joint_q=np.asarray(item["q_rad"], dtype=float),
        support_mask=support_mask(support),
        note="planner keyframe: " + str(item.get("state_source", "unknown")),
    )


def indices_from_mask(mask):
    return tuple(int(i) for i in np.where(np.asarray(mask).astype(bool))[0])


def _interp_rotation(R0, R1, u):
    rotations = Rotation.from_matrix(np.stack([R0, R1], axis=0))
    return Slerp([0.0, 1.0], rotations)([float(u)]).as_matrix()[0]


def _smoothstep(u):
    u = float(u)
    return u * u * (3.0 - 2.0 * u)


def moving_support_switch_frames(prev, cur, n_mid=4):
    """Display-only interpolation for a contact change while body progress moves."""
    before = indices_from_mask(prev.support_mask)
    after = indices_from_mask(cur.support_mask)
    added = set(after) - set(before)
    union = tuple(sorted(set(before) | set(after)))
    out = []
    for j in range(1, max(1, int(n_mid)) + 1):
        raw = j / float(max(1, int(n_mid)) + 1)
        u = _smoothstep(raw)
        if raw < 0.45:
            frame_support = before
            note = "display interpolation; old support retained"
        elif raw < 0.72 and added:
            frame_support = union
            note = "display interpolation; touchdown before liftoff"
        else:
            frame_support = after
            note = "display interpolation; support transfer"
        out.append(DisplayFrame(
            progress=(1.0 - u) * prev.progress + u * cur.progress,
            body_t=(1.0 - u) * prev.body_t + u * cur.body_t,
            body_R=_interp_rotation(prev.body_R, cur.body_R, u),
            joint_q=(1.0 - u) * prev.joint_q + u * cur.joint_q,
            support_mask=support_mask(frame_support),
            note=note,
        ))
    return out


def build_display_frames(trajectory, same_support_midframes):
    keys = [keyframe(item) for item in trajectory]
    out = [keys[0]]
    n_mid = max(0, int(same_support_midframes))

    for prev, cur in zip(keys[:-1], keys[1:]):
        before = indices_from_mask(prev.support_mask)
        after = indices_from_mask(cur.support_mask)

        if abs(cur.progress - prev.progress) <= 1e-9 and before != after:
            added = tuple(sorted(set(after) - set(before)))
            removed = tuple(sorted(set(before) - set(after)))
            switch = touchdown_first_switch_frames(
                progress=float(cur.progress),
                body_t=cur.body_t,
                body_R=cur.body_R,
                q_pre=prev.joint_q,
                q_post=cur.joint_q,
                support_before=before,
                support_after=after,
                added_legs=added,
                removed_legs=removed,
            )
            out.extend(switch[1:])
            continue

        if before == after:
            if n_mid > 0:
                out.extend(interpolate_same_support(
                    prev.progress,
                    prev.body_t,
                    prev.body_R,
                    prev.joint_q,
                    cur.progress,
                    cur.body_t,
                    cur.body_R,
                    cur.joint_q,
                    prev.support_mask,
                    n_mid=n_mid,
                ))
        elif n_mid > 0:
            out.extend(moving_support_switch_frames(prev, cur, n_mid=max(2, n_mid)))
        out.append(cur)

    return out


def draw_frame(ax, kin, frame, best_angle, half_window):
    ax.cla()
    t = frame.body_t
    R = frame.body_R
    q = frame.joint_q
    support = frame.support_mask.astype(bool)

    grid = np.linspace(-half_window, half_window, 9)
    for g in grid:
        ax.plot(
            [t[0] - half_window, t[0] + half_window],
            [t[1] + g, t[1] + g],
            [0.0, 0.0],
            linewidth=0.4,
            alpha=0.25,
        )
        ax.plot(
            [t[0] + g, t[0] + g],
            [t[1] - half_window, t[1] + half_window],
            [0.0, 0.0],
            linewidth=0.4,
            alpha=0.25,
        )

    unit_corners, edges = cube_edges()
    corners_w = t[None, :] + (R @ (kin.a * unit_corners).T).T
    for i, j in edges:
        p0, p1 = corners_w[i], corners_w[j]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=2.0
        )

    for leg in range(kin.n_legs):
        root, elbow, foot = kin.world_points(t, R, leg, q[leg])
        pts = np.vstack([root, elbow, foot])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0)
        if support[leg]:
            ax.scatter([foot[0]], [foot[1]], [foot[2]], marker="x", s=45)
        else:
            ax.scatter([foot[0]], [foot[1]], [foot[2]], marker="o", s=12)
        ax.text(foot[0], foot[1], foot[2] + 0.02, str(leg))

    axis_len = 0.18
    for j, label in enumerate(("Xw", "Yw", "Zw")):
        end = t + axis_len * np.eye(3)[j]
        ax.plot([t[0], end[0]], [t[1], end[1]], [t[2], end[2]], linewidth=2.0)
        ax.text(end[0], end[1], end[2], label)
    for j, label in enumerate(("Xb", "Yb", "Zb")):
        end = t + axis_len * R[:, j]
        ax.plot(
            [t[0], end[0]], [t[1], end[1]], [t[2], end[2]], linestyle="--"
        )
        ax.text(end[0], end[1], end[2], label)

    ax.set_xlim(t[0] - half_window, t[0] + half_window)
    ax.set_ylim(t[1] - half_window, t[1] + half_window)
    ax.set_zlim(-0.05, max(0.90, t[2] + 0.45))
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_zlabel("world z [m]")
    ax.view_init(elev=22, azim=38)
    ax.set_box_aspect((1.0, 1.0, 0.85))
    ax.set_title(
        "Lily checkpoint replay  progress=%.2f / %.2f deg\n"
        "support=%s\n%s"
        % (frame.progress, best_angle, list(np.where(support)[0]), frame.note)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "full_v006_fresh_checkpoint.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "checkpoint.gif",
    )
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--half-window", type=float, default=0.80)
    parser.add_argument(
        "--same-support-midframes",
        type=int,
        default=2,
        help="Display-only frames inserted between planner keyframes",
    )
    args = parser.parse_args()

    data = load_checkpoint(args.checkpoint)
    trajectory = data["best_trajectory"]
    best_summary = data.get("best_summary", {})
    best_angle = float(best_summary.get("best_angle_deg", data["best_angle_deg"]))
    frames = build_display_frames(trajectory, args.same_support_midframes)

    kin = build_kinematics()
    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    writer = PillowWriter(fps=float(args.fps))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(args.output), dpi=100):
        for i, frame in enumerate(frames):
            draw_frame(ax, kin, frame, best_angle, float(args.half_window))
            writer.grab_frame()
            if i % 100 == 0:
                print("render", i, "/", len(frames), flush=True)

    plt.close(fig)
    print("planner keyframes:", len(trajectory))
    print("display frames:", len(frames))
    print("best_angle_deg:", best_angle)
    print("gif:", args.output)


if __name__ == "__main__":
    main()
