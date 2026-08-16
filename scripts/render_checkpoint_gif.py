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


def _frame_geometry(kin, frame):
    t = frame.body_t
    R = frame.body_R
    q = frame.joint_q

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
    return corners_w, edges, leg_points, world_axis_ends, body_axis_ends, subject_points


def fixed_equal_axis_limits(frames, kin, half_window, axis_padding):
    """Compute one fixed, equal-scale view that contains the complete GIF trajectory."""
    if not frames:
        raise ValueError("frames must not be empty")

    point_clouds = []
    for frame in frames:
        point_clouds.append(_frame_geometry(kin, frame)[-1])

    pts = np.vstack(point_clouds)
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)

    # Keep a small amount of ground below z=0 visible for contact interpretation.
    mins[2] = min(float(mins[2]), -0.05)
    maxs[2] = max(float(maxs[2]), 0.0)

    centers = 0.5 * (mins + maxs)
    required_span = float(np.max(maxs - mins))
    minimum_span = 2.0 * max(0.01, float(half_window))
    pad = max(0.0, float(axis_padding))
    span = max(required_span + 2.0 * pad, minimum_span)
    half_span = 0.5 * span

    return (
        (float(centers[0] - half_span), float(centers[0] + half_span)),
        (float(centers[1] - half_span), float(centers[1] + half_span)),
        (float(centers[2] - half_span), float(centers[2] + half_span)),
    )


def draw_frame(
    ax,
    kin,
    frame,
    best_angle,
    half_window,
    axis_padding=0.08,
    fixed_limits=None,
):
    ax.cla()
    t = frame.body_t
    R = frame.body_R
    support = frame.support_mask.astype(bool)

    (
        corners_w,
        edges,
        leg_points,
        world_axis_ends,
        body_axis_ends,
        subject_points,
    ) = _frame_geometry(kin, frame)

    if fixed_limits is None:
        # Compatibility path for direct callers: make a single-frame equal-scale view.
        xlim, ylim, zlim = fixed_equal_axis_limits(
            [frame], kin, half_window, axis_padding
        )
    else:
        xlim, ylim, zlim = fixed_limits

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
    ax.set_box_aspect((1.0, 1.0, 1.0))
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
        help="Minimum equal-axis half-span for the fixed GIF view.",
    )
    parser.add_argument(
        "--axis-padding",
        type=float,
        default=0.08,
        help="Extra margin [m] around the complete trajectory before fixing the view.",
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
    fixed_limits = fixed_equal_axis_limits(
        frames, kin, float(args.half_window), float(args.axis_padding)
    )

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
                fixed_limits=fixed_limits,
            )
            writer.grab_frame()
            if i % 100 == 0:
                print("render", i, "/", len(frames), flush=True)

    plt.close(fig)
    print("saved trajectory frames:", len(trajectory))
    print("display frames:", len(frames))
    print("fixed xlim:", fixed_limits[0])
    print("fixed ylim:", fixed_limits[1])
    print("fixed zlim:", fixed_limits[2])
    print("best_angle_deg:", best_angle)
    print("gif:", args.output)


if __name__ == "__main__":
    main()
