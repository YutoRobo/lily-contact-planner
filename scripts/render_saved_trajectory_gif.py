#!/usr/bin/env python3
"""Render a split Lily BEST trajectory to GIF with playback-speed control.

The planner keeps all dense trajectory samples in a separate trajectory JSON.
This renderer may decimate frames for fast playback, but never modifies the
saved trajectory data itself.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_checkpoint_gif as replay


def load_split_files(checkpoint_path, trajectory_override=None):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    if checkpoint.get("schema_version") == "checkpoint_v2":
        # Backward compatibility with the short-lived embedded format.
        trajectory = checkpoint.get("best_trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise RuntimeError("checkpoint_v2 has no embedded best_trajectory")
        return checkpoint, trajectory, checkpoint_path

    if checkpoint.get("schema_version") != "checkpoint_v3":
        raise RuntimeError("checkpoint_v3 (or legacy checkpoint_v2) required")

    if trajectory_override is not None:
        trajectory_path = Path(trajectory_override)
    else:
        trajectory_name = checkpoint.get("trajectory_file")
        if not trajectory_name:
            raise RuntimeError("checkpoint_v3 has no trajectory_file reference")
        trajectory_path = checkpoint_path.parent / trajectory_name

    trajectory_data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if trajectory_data.get("schema_version") != "trajectory_v1":
        raise RuntimeError("trajectory_v1 required")
    trajectory = trajectory_data.get("best_trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise RuntimeError("trajectory file has no best_trajectory frames")
    return checkpoint, trajectory, trajectory_path


def speed_plan(frames, base_fps, speed):
    """Return playback frames and FPS while avoiding impractically high GIF FPS."""
    if speed <= 0.0:
        raise ValueError("--speed must be > 0")
    if base_fps <= 0.0:
        raise ValueError("--fps must be > 0")

    # For speed-up, decimate first and then compensate residual ratio with FPS.
    # This preserves approximately the requested overall duration while keeping
    # GIF frame timing in a practical range.
    stride = max(1, int(math.floor(speed))) if speed >= 1.0 else 1
    sampled = list(frames[::stride])
    if sampled[-1] is not frames[-1]:
        sampled.append(frames[-1])
    effective_fps = float(base_fps) * float(speed) / float(stride)
    return sampled, effective_fps, stride


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "full_v006_fresh_checkpoint.json",
        help="Lightweight checkpoint_v3 JSON",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Optional trajectory JSON override; default follows checkpoint trajectory_file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "checkpoint.gif",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Base playback FPS at --speed 1.0",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier, e.g. 2.0 or 4.0; saved data is unchanged",
    )
    parser.add_argument("--half-window", type=float, default=0.80)
    parser.add_argument(
        "--same-support-midframes",
        type=int,
        default=0,
        help="Optional display-only interpolation; 0 replays saved trajectory samples",
    )
    args = parser.parse_args()

    checkpoint, trajectory, trajectory_path = load_split_files(
        args.checkpoint, args.trajectory
    )
    best_summary = checkpoint.get("best_summary", {})
    best_angle = float(best_summary.get("best_angle_deg", checkpoint["best_angle_deg"]))

    display_frames = replay.build_display_frames(
        trajectory, args.same_support_midframes
    )
    playback_frames, effective_fps, stride = speed_plan(
        display_frames, args.fps, args.speed
    )

    kin = replay.build_kinematics()
    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    writer = PillowWriter(fps=effective_fps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(args.output), dpi=100):
        for i, frame in enumerate(playback_frames):
            replay.draw_frame(ax, kin, frame, best_angle, float(args.half_window))
            writer.grab_frame()
            if i % 100 == 0:
                print("render", i, "/", len(playback_frames), flush=True)

    plt.close(fig)
    print("trajectory file:", trajectory_path)
    print("saved trajectory frames:", len(trajectory))
    print("display frames before speed-up:", len(display_frames))
    print("GIF frames:", len(playback_frames))
    print("speed multiplier:", float(args.speed))
    print("frame stride:", stride)
    print("effective fps:", effective_fps)
    print("best_angle_deg:", best_angle)
    print("gif:", args.output)


if __name__ == "__main__":
    main()
