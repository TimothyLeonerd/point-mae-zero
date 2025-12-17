#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load_npy(root: Path, name: str):
    p = root / name
    if not p.exists():
        return None
    return np.load(p)


def _subsample_indices(rng: np.random.Generator, N: int, max_n: int):
    if max_n <= 0 or N <= max_n:
        return np.arange(N, dtype=np.int64)
    return rng.choice(N, size=max_n, replace=False).astype(np.int64)


def _draw_box(ax, L, W, H, m=0.0):
    # draw inner box [m, L-m] x [m, W-m] x [m, H-m]
    xs = [m, L - m]
    ys = [m, W - m]
    zs = [m, H - m]

    # 12 edges
    edges = [
        # bottom rectangle (z=zs[0])
        ((xs[0], ys[0], zs[0]), (xs[1], ys[0], zs[0])),
        ((xs[1], ys[0], zs[0]), (xs[1], ys[1], zs[0])),
        ((xs[1], ys[1], zs[0]), (xs[0], ys[1], zs[0])),
        ((xs[0], ys[1], zs[0]), (xs[0], ys[0], zs[0])),
        # top rectangle (z=zs[1])
        ((xs[0], ys[0], zs[1]), (xs[1], ys[0], zs[1])),
        ((xs[1], ys[0], zs[1]), (xs[1], ys[1], zs[1])),
        ((xs[1], ys[1], zs[1]), (xs[0], ys[1], zs[1])),
        ((xs[0], ys[1], zs[1]), (xs[0], ys[0], zs[1])),
        # vertical edges
        ((xs[0], ys[0], zs[0]), (xs[0], ys[0], zs[1])),
        ((xs[1], ys[0], zs[0]), (xs[1], ys[0], zs[1])),
        ((xs[1], ys[1], zs[0]), (xs[1], ys[1], zs[1])),
        ((xs[0], ys[1], zs[0]), (xs[0], ys[1], zs[1])),
    ]
    for (x0, y0, z0), (x1, y1, z1) in edges:
        ax.plot([x0, x1], [y0, y1], [z0, z1], linewidth=1.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-root", required=True, help="Directory containing coord.npy (+ optional color.npy/normal.npy/meta.json)")
    p.add_argument("--max-points", type=int, default=200_000, help="Max points to draw (scatter).")
    p.add_argument("--seed", type=int, default=0, help="Seed for subsampling.")
    p.add_argument("--point-size", type=float, default=1.0, help="Scatter marker size.")
    p.add_argument("--show-box", action="store_true", help="Draw room box if meta.json exists.")
    p.add_argument("--show-normals", action="store_true", help="Draw normals (requires normal.npy).")
    p.add_argument("--max-normals", type=int, default=5_000, help="Max normal arrows to draw.")
    p.add_argument("--normal-scale", type=float, default=0.08, help="Arrow length scale in meters.")
    p.add_argument("--normal-alpha", type=float, default=0.8, help="Arrow alpha.")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.in_root)

    coord = _load_npy(root, "coord.npy")
    if coord is None:
        raise FileNotFoundError(f"Missing coord.npy in {root}")

    color = _load_npy(root, "color.npy")  # optional
    normal = _load_npy(root, "normal.npy")  # optional

    rng = np.random.default_rng(args.seed)

    # subsample points for scatter
    idx = _subsample_indices(rng, coord.shape[0], args.max_points)
    P = coord[idx].astype(np.float32, copy=False)

    if color is not None and color.shape[0] == coord.shape[0]:
        C = color[idx].astype(np.float32) / 255.0
    else:
        C = None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=args.point_size, c=C)

    # normals (quiver) — draw a smaller subsample
    if args.show_normals:
        if normal is None:
            print("WARNING: --show-normals set but normal.npy not found.")
        elif normal.shape[0] != coord.shape[0]:
            print("WARNING: normal.npy length does not match coord.npy; skipping normals.")
        else:
            idxn = _subsample_indices(rng, coord.shape[0], args.max_normals)
            Pn = coord[idxn].astype(np.float32, copy=False)
            Nn = normal[idxn].astype(np.float32, copy=False)

            # normalize just in case
            nrm = np.linalg.norm(Nn, axis=1, keepdims=True)
            Nn = Nn / np.maximum(nrm, 1e-20)

            ax.quiver(
                Pn[:, 0], Pn[:, 1], Pn[:, 2],
                Nn[:, 0], Nn[:, 1], Nn[:, 2],
                length=args.normal_scale,
                normalize=True,
                linewidth=0.8,
                alpha=args.normal_alpha,
            )

    # optional room box from meta.json if present
    if args.show_box:
        meta_path = root / "meta.json"
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
            room = meta.get("sampled_room", None)
            cfg = meta.get("scene_config", None)
            if room is not None:
                L, W, H = room["L"], room["W"], room["H"]
                m = 0.0
                if cfg is not None and "room_margin" in cfg:
                    m = float(cfg["room_margin"])
                _draw_box(ax, L, W, H, m=m)
        else:
            print("WARNING: --show-box set but meta.json not found.")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    # Keep axes roughly equal (simple heuristic)
    mins = coord.min(axis=0)
    maxs = coord.max(axis=0)
    center = 0.5 * (mins + maxs)
    size = float(np.max(maxs - mins))
    half = 0.5 * size
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
