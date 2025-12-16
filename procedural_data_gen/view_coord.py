#!/usr/bin/env python3
"""
Quick viewer for coord.npy (+ optional color.npy) using matplotlib (3D scatter).

Usage:
  python procedural_data_gen/view_coord.py --in-root /tmp/sq_scene --show-box
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-root", required=True, help="Folder containing coord.npy (and optionally color.npy, meta.json)")
    p.add_argument("--max-points", type=int, default=200_000, help="Downsample for visualization")
    p.add_argument("--seed", type=int, default=0, help="Seed for visualization downsampling")
    p.add_argument("--elev", type=float, default=20.0, help="Camera elevation")
    p.add_argument("--azim", type=float, default=-60.0, help="Camera azimuth")
    p.add_argument("--show-box", action="store_true", help="Draw room bounding box (requires meta.json)")
    return p.parse_args()


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def draw_room_box(ax, L: float, W: float, H: float):
    corners = np.array([
        [0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0],
        [0, 0, H], [L, 0, H], [L, W, H], [0, W, H],
    ], dtype=np.float32)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]])


def main() -> None:
    args = parse_args()
    in_root = Path(args.in_root)

    coord_path = in_root / "coord.npy"
    if not coord_path.exists():
        raise FileNotFoundError(f"Missing: {coord_path}")
    xyz = np.load(coord_path)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"coord.npy must be (N,3), got {xyz.shape}")

    color_path = in_root / "color.npy"
    rgb = None
    if color_path.exists():
        rgb = np.load(color_path)
        if rgb.shape != xyz.shape:
            raise ValueError(f"color.npy shape {rgb.shape} must match coord.npy shape {xyz.shape}")
        if rgb.dtype != np.uint8:
            # still support float colors, but uint8 is recommended
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # downsample (apply same indices to xyz and rgb)
    n = xyz.shape[0]
    if n > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, size=args.max_points, replace=False)
        xyz_vis = xyz[idx]
        rgb_vis = rgb[idx] if rgb is not None else None
    else:
        xyz_vis = xyz
        rgb_vis = rgb

    # optional room dims from meta.json
    meta_path = in_root / "meta.json"
    room = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            room = meta.get("room", None)
        except Exception:
            room = None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)

    if rgb_vis is not None:
        colors01 = rgb_vis.astype(np.float32) / 255.0
        ax.scatter(xyz_vis[:, 0], xyz_vis[:, 1], xyz_vis[:, 2], s=1, marker=".", c=colors01)
    else:
        ax.scatter(xyz_vis[:, 0], xyz_vis[:, 1], xyz_vis[:, 2], s=1, marker=".")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    if args.show_box and room is not None:
        L, W, H = float(room["L"]), float(room["W"]), float(room["H"])
        draw_room_box(ax, L, W, H)

    if room is not None:
        L, W, H = float(room["L"]), float(room["W"]), float(room["H"])
        ax.set_xlim(0, L)
        ax.set_ylim(0, W)
        ax.set_zlim(0, H)
    else:
        mins = xyz_vis.min(axis=0)
        maxs = xyz_vis.max(axis=0)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

    set_axes_equal(ax)
    title = f"{coord_path.name} ({xyz.shape[0]} pts)"
    if rgb is not None:
        title += " + color.npy"
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()
