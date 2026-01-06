#!/usr/bin/env python3
"""
viz_primitives.py

Simple viewer for .npy primitive point clouds produced by generate_primitives.py.

- Input: .npy with shape (N, 6) where:
    xyz = data[:, 0:3]
    nrm = data[:, 3:6]
- Shows:
    * scatter plot of points
    * optional subset of normals as arrows (3D quiver)
    * optional coloring by normal direction

Usage examples:
    python viz_primitives.py --file data/primitives_test/primitive_000000.npy

    # Color by normal direction
    python viz_primitives.py --file data/primitives_test/primitive_000000.npy --color-by-normals

    # From a directory, pick by index, color by normals, no arrows
    python viz_primitives.py --data-dir data/primitives_test --index 2 --color-by-normals --no-arrows
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


def load_npy(file_path: Path) -> np.ndarray:
    arr = np.load(file_path)
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(
            f"Expected shape (N, 3) or (N, 6), got {arr.shape} for {file_path}"
        )
    return arr


def choose_file_from_dir(data_dir: Path, index: int) -> Path:
    npy_files = sorted(data_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")
    if index < 0 or index >= len(npy_files):
        raise IndexError(
            f"Index {index} out of range (found {len(npy_files)} .npy files)."
        )
    return npy_files[index]


def plot_pointcloud_with_normals(
    points: np.ndarray,
    normals: np.ndarray,
    max_arrows: int = 500,
    color_by_normals: bool = False,
    show_arrows: bool = True,
):
    """
    points:  (N, 3)
    normals: (N, 3), unit length
    """
    N = points.shape[0]

    # Colors: either map normal directions to RGB, or use a single color
    if color_by_normals:
        # Ensure normals are unit length
        nrm = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
        # Map from [-1, 1] to [0, 1] for RGB
        colors = 0.5 * (nrm + 1.0)
    else:
        colors = None  # matplotlib will use default color

    # Subsample for arrows so we don't draw 8000+ quivers
    if max_arrows is not None and N > max_arrows:
        idx = np.random.choice(N, size=max_arrows, replace=False)
    else:
        idx = np.arange(N)

    pts_arrow = points[idx]
    nrm_arrow = normals[idx]

    # Set up figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter all points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1,
        alpha=0.6,
        depthshade=False,
        c=colors,
    )

    # Compute a reasonable arrow length
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    max_range = bbox_size.max()
    arrow_length = max_range * 0.1  # 10% of largest dimension

    # Quiver for normals (optional)
    if show_arrows:
        ax.quiver(
            pts_arrow[:, 0], pts_arrow[:, 1], pts_arrow[:, 2],
            nrm_arrow[:, 0], nrm_arrow[:, 1], nrm_arrow[:, 2],
            length=arrow_length,
            normalize=True,
            linewidth=0.5,
        )

    # Equal aspect ratio
    mid = (bbox_min + bbox_max) / 2.0
    for axis, m in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        axis(m - max_range / 2.0, m + max_range / 2.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = "Primitive with Normals"
    if color_by_normals:
        title += " (colored by normal direction)"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize primitive .npy with normals.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single .npy file. If set, --data-dir and --index are ignored.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing .npy files (used if --file is not provided).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of .npy file within --data-dir (sorted by name).",
    )
    parser.add_argument(
        "--max-arrows",
        type=int,
        default=500,
        help="Max number of normals to draw as arrows (subsampled).",
    )
    parser.add_argument(
        "--color-by-normals",
        action="store_true",
        help="Color points by normal direction (maps nx,ny,nz in [-1,1] to RGB in [0,1]).",
    )
    parser.add_argument(
        "--no-arrows",
        action="store_true",
        help="Disable normal arrows (quiver), show only colored points.",
    )

    args = parser.parse_args()

    if args.file is not None:
        file_path = Path(args.file)
    else:
        if args.data_dir is None:
            raise ValueError("Either --file or --data-dir must be provided.")
        data_dir = Path(args.data_dir)
        file_path = choose_file_from_dir(data_dir, args.index)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[INFO] Loading {file_path}")
    arr = load_npy(file_path)

    if arr.shape[1] == 3:
        points = arr
        normals = None
        print("[WARN] Array has shape (N, 3) — no normals present, arrows/colors by normals will be skipped.")
    else:
        points = arr[:, 0:3]
        normals = arr[:, 3:6]

    if normals is None:
        # Just scatter
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=1, alpha=0.5, depthshade=False)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Primitive (no normals)")
        plt.tight_layout()
        plt.show()
    else:
        plot_pointcloud_with_normals(
            points,
            normals,
            max_arrows=args.max_arrows,
            color_by_normals=args.color_by_normals,
            show_arrows=not args.no_arrows,
        )


if __name__ == "__main__":
    main()
