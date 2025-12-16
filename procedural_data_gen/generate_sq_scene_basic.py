#!/usr/bin/env python3
"""
Generate ONE basic indoor-like scene (room cuboid + random objects + clip)
and write coord.npy + color.npy into --out-root.

Color rules:
1) same superquadric -> identical color
2) same object -> similar colors (SQ colors are small variations of base object color)
3) different objects -> very different colors (distinct hues)

Usage:
  python procedural_data_gen/generate_sq_scene_basic.py --out-root /tmp/sq_scene
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np

from sample_SQs import _make_cloud_once  # returns points4 Nx4, last col = SQ id


# ---------- hardcoded MVP defaults ----------
ROOM_MARGIN = 0.05
L_RANGE = (3.0, 8.0)
W_RANGE = (3.0, 8.0)
H_RANGE = (2.2, 3.5)

N_OBJ_RANGE = (6, 20)

# Object sampler params
N_SQ_MAX = 9
POINTS_PER_OBJECT = 20000
ALPHA = 2.0
GROWTH = 1.3
MAX_ROUNDS = 6
# ------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", required=True, help="Directory to write coord.npy + color.npy (+ meta.json).")
    p.add_argument("--seed", type=int, default=42, help="Seed for reproducibility. Use -1 for OS entropy.")
    p.add_argument("--write-meta", action="store_true", default=True, help="Write meta.json next to coord.npy.")
    p.add_argument("--no-write-meta", dest="write_meta", action="store_false")
    return p.parse_args()


def sample_room(rng: np.random.Generator) -> Tuple[float, float, float]:
    L = float(rng.uniform(*L_RANGE))
    W = float(rng.uniform(*W_RANGE))
    H = float(rng.uniform(*H_RANGE))
    return L, W, H


def place_object_randomly(
    xyz: np.ndarray,
    rng: np.random.Generator,
    L: float,
    W: float,
    H: float,
) -> np.ndarray:
    """Shift object so its centroid is uniformly sampled inside room interior."""
    centroid = xyz.mean(axis=0)
    target = np.array(
        [
            rng.uniform(ROOM_MARGIN, max(ROOM_MARGIN, L - ROOM_MARGIN)),
            rng.uniform(ROOM_MARGIN, max(ROOM_MARGIN, W - ROOM_MARGIN)),
            rng.uniform(ROOM_MARGIN, max(ROOM_MARGIN, H - ROOM_MARGIN)),
        ],
        dtype=xyz.dtype,
    )
    return xyz + (target - centroid)


def clip_to_room(xyz: np.ndarray, L: float, W: float, H: float) -> np.ndarray:
    m = ROOM_MARGIN
    keep = (
        (xyz[:, 0] >= m) & (xyz[:, 0] <= (L - m)) &
        (xyz[:, 1] >= m) & (xyz[:, 1] <= (W - m)) &
        (xyz[:, 2] >= m) & (xyz[:, 2] <= (H - m))
    )
    return keep


def _sample_distinct_hues(rng: np.random.Generator, n: int, min_dist: float = 0.12) -> List[float]:
    """
    Sample n hues in [0,1) such that they're separated by at least min_dist on the circle.
    For n up to ~20 this is fine with rejection sampling.
    """
    hues: List[float] = []
    tries = 0
    while len(hues) < n:
        tries += 1
        if tries > 50_000:
            # fallback: evenly spaced
            return [((i / max(n, 1)) % 1.0) for i in range(n)]
        h = float(rng.random())
        ok = True
        for h2 in hues:
            d = abs(h - h2)
            d = min(d, 1.0 - d)
            if d < min_dist:
                ok = False
                break
        if ok:
            hues.append(h)
    return hues


def _rgb01_to_uint8(rgb01: Tuple[float, float, float]) -> np.ndarray:
    rgb = np.clip(np.array(rgb01, dtype=np.float32), 0.0, 1.0)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _make_object_sq_colors(
    rng: np.random.Generator,
    sq_ids: np.ndarray,
    base_rgb_u8: np.ndarray,
) -> np.ndarray:
    """
    Make per-point colors for an object:
      - base color per object
      - per SQ: small brightness/saturation variation so SQs are "similar" within the object
    """
    sq_ids_int = sq_ids.astype(np.int64)
    n_sqs = int(sq_ids_int.max()) + 1 if sq_ids_int.size > 0 else 0

    # base in HSV to vary "slightly" and stay similar
    base_rgb01 = (base_rgb_u8.astype(np.float32) / 255.0).tolist()
    base_h, base_s, base_v = colorsys.rgb_to_hsv(*base_rgb01)

    sq_color_lut = np.zeros((n_sqs, 3), dtype=np.uint8)
    for k in range(n_sqs):
        # Keep hue nearly fixed, vary S/V a bit
        dh = float(rng.uniform(-0.015, 0.015))  # tiny hue wiggle
        ds = float(rng.uniform(-0.08, 0.08))
        dv = float(rng.uniform(-0.12, 0.12))
        h = (base_h + dh) % 1.0
        s = float(np.clip(base_s + ds, 0.55, 0.98))
        v = float(np.clip(base_v + dv, 0.45, 0.98))
        rgb01 = colorsys.hsv_to_rgb(h, s, v)
        sq_color_lut[k] = _rgb01_to_uint8(rgb01)

    colors = sq_color_lut[sq_ids_int]  # (N,3)
    return colors


def main() -> None:
    args = parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(None if args.seed == -1 else args.seed)
    scene_seed = int(rng_master.integers(0, 2**32 - 1, dtype=np.uint32))
    rng = np.random.default_rng(scene_seed)

    L, W, H = sample_room(rng)
    n_obj = int(rng.integers(N_OBJ_RANGE[0], N_OBJ_RANGE[1] + 1))

    # distinct base hues per object
    hues = _sample_distinct_hues(rng, n_obj, min_dist=0.12)

    all_xyz = []
    all_rgb = []
    pts_before = 0

    for j in range(n_obj):
        points4, _sq_params = _make_cloud_once(
            N_SQ_MAX,
            POINTS_PER_OBJECT,
            rng=rng,
            alpha=ALPHA,
            growth=GROWTH,
            max_rounds=MAX_ROUNDS,
        )

        xyz = points4[:, :3].astype(np.float32, copy=False)
        sq_ids = points4[:, 3]  # float64 ids (0..n_sqs-1)

        # base object color from distinct hue
        # (fixed-ish saturation/value so objects are vivid and distinct)
        base_rgb01 = colorsys.hsv_to_rgb(hues[j], 0.85, 0.90)
        base_rgb_u8 = _rgb01_to_uint8(base_rgb01)

        rgb = _make_object_sq_colors(rng, sq_ids, base_rgb_u8)  # uint8 Nx3

        xyz = place_object_randomly(xyz, rng, L, W, H)

        pts_before += xyz.shape[0]
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    xyz_scene = np.concatenate(all_xyz, axis=0) if all_xyz else np.zeros((0, 3), dtype=np.float32)
    rgb_scene = np.concatenate(all_rgb, axis=0) if all_rgb else np.zeros((0, 3), dtype=np.uint8)

    keep = clip_to_room(xyz_scene, L, W, H)
    xyz_scene = xyz_scene[keep]
    rgb_scene = rgb_scene[keep]

    np.save(out_root / "coord.npy", xyz_scene.astype(np.float32, copy=False))
    np.save(out_root / "color.npy", rgb_scene.astype(np.uint8, copy=False))

    if args.write_meta:
        meta = {
            "seed": scene_seed,
            "room": {"L": L, "W": W, "H": H, "margin": ROOM_MARGIN},
            "n_objects": n_obj,
            "object_sampler": {
                "n_SQ_max": N_SQ_MAX,
                "points_per_object": POINTS_PER_OBJECT,
                "alpha": ALPHA,
                "growth": GROWTH,
                "max_rounds": MAX_ROUNDS,
            },
            "points_before_clip": int(pts_before),
            "points_after_clip": int(xyz_scene.shape[0]),
            "color": {"dtype": "uint8", "range": "[0..255]"},
        }
        (out_root / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote: {(out_root / 'coord.npy')} and {(out_root / 'color.npy')}")
    print(f"Room (L,W,H)=({L:.2f},{W:.2f},{H:.2f})  n_obj={n_obj}  points={xyz_scene.shape[0]}")


if __name__ == "__main__":
    main()
