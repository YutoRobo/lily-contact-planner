#!/usr/bin/env python3
"""Render a checkpoint_v2 BEST trajectory directly to GIF.

By default the GIF replays the saved planner trajectory exactly. Optional
visualization-only interpolation can be requested with
``--same-support-midframes > 0``; those inserted frames are not planner states.
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
    source = str(item.get("state_source", "unknown"))
    certified = item.get("certified_state", None)
    note = "saved trajectory: " + source
    if certified is not None:
        note += "  certified=%s" % bool(certified)
    return DisplayFrame(
        progress=float(item["progress_deg"]),
        body_t=np.asarray(item["body_t"], dtype=float),
        body_R=np.asarray(item["body_R"], dtype=float),
        joint_q=np.asarray(item["q_rad"], dtype=float),
        support_mask=support_mask(support),
        note=note,
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
    n_mid = max(0, int(same_support_midframes))

    # Exact replay mode: no invented state is inserted between stored planner
    # trajectory samples. This is the default for checkpoint inspection.
    if n_mid == 0:
        return keys

    out = [keys[0]]
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
        else:
            out.extend(moving_support_switch_frames(prev, cur, n_mid=max(2, n_mid)))
        out.append(cur)

    return out


def _adaptive_axis_limits(points, body_t, half_window, axis_padding):
    """Keep the legacy view as a minimum, but expand to contain the full robot."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    t = np.asarray(body_t, dtype=float)
    pad = max(0.0, float(axis_padding))
    half = max(0.01, float(half_window))

    xlo = min(float(t[0] - half), float(np.min(pts[:, 0]) - pad))
    xhi = max(float(t[0] + half), float(np.max(pts[:, 0]) + pad))
    ylo = min(float(t[1] - half), float(np.min(pts[:, 1]) - pad))
    yhi = max(float(t[1] + half), float(np.max(pts[:, 1]) + pad))

    # Preserve the old ground context while allowing large rotated limbs above
    # or below the former fixed [-0.05, 0.90] z range.
    zlo = min(-0.05, float(np.min(pts[:, 2]) - pad))
    zhi = max(0.90, float(np.max(pts[:, 2]) + pad))
    return (xlo, xhi), (ylo, yhi), (zlo, zhi)


def draw_frame(ax, kin, frame, best_angle, half_window, axis_padding=0.08):
    ax.cla()
    t = frame.body_t
    R = frame.body_R
    q = frame.joint_q
    support = frame.support_mask.astype(bool)

    unit_corners, edges = cube_edges()
    corners_w = t[None, :] + (R @ (kin.a * unit_corners).T).T

    leg_points = []
    for leg in range(kin.n_legs):
        root, elbow, foot = kin.world_points(t, R, leg, q[leg])
        leg_points.append(np.vstack([root, elbow, foot]))

    axis_len = 0.18
    world_axis_ends = [t + axis_len * np.eye(3)[j] for j in range(3)]
    body_axis_ends = [t + axis_len * R[:, j] for j in range(3)]

    subject_points = np.vstack(
        [corners_w]
        + leg_points
        + [np.asarray(world_axis_ends), np.asarray(body_axis_ends)]
    )
    xlim, ylim, zlim = _adaptive_axis_limits(
        subject_points, t, half_window, axis_padding
    )

    # Ground grid follows the actual visible horizontal window.
    xgrid = np.linspace(xlim[0], xlim[1], 9)
    ygrid = np.linspace(ylim[0], ylim[1], 9)
    for yg in ygrid:
        ax.plot(
            [xlim[0], xlim[1]],
            [yg, yg],
            [0.0, 0.0],
            linewidth=0.4,
            alpha=0.25,
        )
    for xg in xgrid:
        ax.plot(
            [xg, xg],
            [ylim[0], ylim[1]],
            [0.0, 0.0],
            linewidth=0.4,
            alpha=0.25,
        )

    for i, j in edges:
        p0, p1 = corners_w[i], corners_w[j]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=2.0
        )

    for leg, pts in enumerate(leg_points):
        foot = pts[-1]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0)
        if support[leg]:
            ax.scatter([foot[0]], [foot[1]], [foot[2]], marker="x", s=45)
        else:
            ax.scatter([foot[0]], [foot[1]], [foot[2]], marker="o", s=12)
        ax.text(foot[0], foot[1], foot[2] + 0.02, str(leg))

    for end, label in zip(world_axis_ends, ("Xw", "Yw", "Zw")):
        ax.plot([t[0], end[0]], [t[1], end[1]], [t[2], end[2]], linewidth=2.0)
        ax.text(end[0], end[1], end[2], label)
    for end, label in zip(body_axis_ends, ("Xb", "Yb", "Zb")):
        ax.plot(
            [t[0], end[0]], [t[1], end[1]], [t[2], end[2]], linestyle="--"
        )
        ax.text(end[0], end[1], end[2], label)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_zlabel("world z [m]")
    ax.view_init(elev=22, azim=38)

    # Match the visual box aspect to the actual data spans so the robot is not
    # distorted when z has to expand for a rotated limb.
    spans = np.array(
        [xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]],
        dtype=float,
    )
    ax.set_box_aspect(tuple(spans))
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
    parser.add_argument(
        "--half-window",
        type=float,
        default=0.80,
        help="Minimum horizontal half-window around the body; expands automatically.",
    )
    parser.add_argument(
        "--axis-padding",
        type=float,
        default=0.08,
        help="Extra margin [m] around robot geometry when auto-expanding axes.",
    )
    parser.add_argument(
        "--same-support-midframes",
        type=int,
        default=0,
        help=(
            "Optional display-only interpolation between stored trajectory samples. "
            "0 (default) replays checkpoint samples exactly."
        ),
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
            draw_frame(
                ax,
                kin,
                frame,
                best_angle,
                float(args.half_window),
                float(args.axis_padding),
            )
            writer.grab_frame()
            if i % 100 == 0:
                print("render", i, "/", len(frames), flush=True)

    plt.close(fig)
    print("saved trajectory frames:", len(trajectory))
    print("display frames:", len(frames))
    print("best_angle_deg:", best_angle)
    print("gif:", args.output)


if __name__ == "__main__":
    main()
