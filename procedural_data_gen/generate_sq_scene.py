#!/usr/bin/env python3
"""
Generate ONE basic indoor-like scene and write:
  - coord.npy (float32, Nx3, meters)
  - color.npy (uint8, Nx3, 0..255)
  - meta.json (optional)

This file is intentionally structured as a small pipeline of stages so that
future features (walls, scaling, overlap rejection, etc.) can be toggled on/off
without rewriting the script.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from sample_SQs import _make_cloud_once  # returns points4 Nx4, last col = SQ id


# ---------------------- Config / State ----------------------

@dataclass
class SceneConfig:
    # output / reproducibility
    out_root: Path
    seed: int = 42
    write_meta: bool = True

    # room (meters)
    room_margin: float = 0.05
    L_range: Tuple[float, float] = (3.0, 8.0)
    W_range: Tuple[float, float] = (3.0, 8.0)
    H_range: Tuple[float, float] = (2.2, 3.5)

    # objects
    n_obj_range: Tuple[int, int] = (6, 20)
    n_sq_max: int = 9
    points_per_object: int = 20000
    alpha: float = 2.0
    growth: float = 1.3
    max_rounds: int = 6

    # feature toggles (start adding features behind these)
    enable_overlap_rejection: bool = False
    enable_floor_support: bool = False
    enable_wall_placement: bool = False

    # room shell (walls/floor/ceiling) points
    enable_room_shell: bool = True
    room_shell_density: float = 200.0   # points per m^2 of surface
    room_shell_max_points: int = 200_000

    # scaling (object-level)
    enable_scaling: bool = False

    # size classes for target AABB diagonal (meters)
    enable_size_classes: bool = True
    size_class_probs: Tuple[float, float, float] = (0.50, 0.35, 0.15)  # small, medium, large
    diag_small: Tuple[float, float] = (0.25, 0.80)
    diag_medium: Tuple[float, float] = (0.80, 1.60)
    diag_large: Tuple[float, float] = (1.60, 3.00)

    # used only if enable_size_classes=False
    target_diag_range: Tuple[float, float] = (0.40, 2.50)


@dataclass
class SceneState:
    # sampled per scene
    scene_seed: int
    rng: np.random.Generator
    L: float
    W: float
    H: float

    # accumulated outputs
    xyz: np.ndarray  # (N,3) float32
    rgb: np.ndarray  # (N,3) uint8

    # optional stats/debug
    pts_before_clip: int = 0


Stage = Callable[[SceneState, SceneConfig], SceneState]


# ---------------------- Small helpers ----------------------

def _sample_room(rng: np.random.Generator, cfg: SceneConfig) -> Tuple[float, float, float]:
    L = float(rng.uniform(*cfg.L_range))
    W = float(rng.uniform(*cfg.W_range))
    H = float(rng.uniform(*cfg.H_range))
    return L, W, H

def _aabb_diag(xyz: np.ndarray) -> float:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    d = float(np.linalg.norm(maxs - mins))
    return d

def _place_object_randomly(
    xyz: np.ndarray,
    rng: np.random.Generator,
    L: float,
    W: float,
    H: float,
    margin: float,
) -> np.ndarray:
    """Shift object so its centroid is uniformly sampled inside room interior."""
    centroid = xyz.mean(axis=0)
    target = np.array(
        [
            rng.uniform(margin, max(margin, L - margin)),
            rng.uniform(margin, max(margin, W - margin)),
            rng.uniform(margin, max(margin, H - margin)),
        ],
        dtype=xyz.dtype,
    )
    return xyz + (target - centroid)


def _clip_keep_mask(xyz: np.ndarray, L: float, W: float, H: float, margin: float) -> np.ndarray:
    m = margin
    return (
        (xyz[:, 0] >= m) & (xyz[:, 0] <= (L - m)) &
        (xyz[:, 1] >= m) & (xyz[:, 1] <= (W - m)) &
        (xyz[:, 2] >= m) & (xyz[:, 2] <= (H - m))
    )


def _sample_distinct_hues(rng: np.random.Generator, n: int, min_dist: float = 0.12) -> List[float]:
    """Sample n hues in [0,1) separated by at least min_dist on the hue circle."""
    hues: List[float] = []
    tries = 0
    while len(hues) < n:
        tries += 1
        if tries > 50_000:
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
    Per-object coloring:
      - each SQ gets a solid color (rule 1)
      - SQ colors are small variations around base object color (rule 2)
      - different objects have distinct base hues (rule 3)
    """
    sq_ids_int = sq_ids.astype(np.int64)
    n_sqs = int(sq_ids_int.max()) + 1 if sq_ids_int.size > 0 else 0

    base_rgb01 = (base_rgb_u8.astype(np.float32) / 255.0).tolist()
    base_h, base_s, base_v = colorsys.rgb_to_hsv(*base_rgb01)

    sq_lut = np.zeros((n_sqs, 3), dtype=np.uint8)
    for k in range(n_sqs):
        dh = float(rng.uniform(-0.015, 0.015))
        ds = float(rng.uniform(-0.08, 0.08))
        dv = float(rng.uniform(-0.12, 0.12))
        h = (base_h + dh) % 1.0
        s = float(np.clip(base_s + ds, 0.55, 0.98))
        v = float(np.clip(base_v + dv, 0.45, 0.98))
        sq_lut[k] = _rgb01_to_uint8(colorsys.hsv_to_rgb(h, s, v))

    return sq_lut[sq_ids_int]

def _sample_target_diag(rng: np.random.Generator, cfg: SceneConfig) -> float:
    if not cfg.enable_size_classes:
        return float(rng.uniform(*cfg.target_diag_range))

    p = np.array(cfg.size_class_probs, dtype=np.float64)
    p = p / p.sum()  # normalize (just in case)

    cls = int(rng.choice(3, p=p))  # 0=small, 1=medium, 2=large
    if cls == 0:
        lo, hi = cfg.diag_small
    elif cls == 1:
        lo, hi = cfg.diag_medium
    else:
        lo, hi = cfg.diag_large
    return float(rng.uniform(lo, hi))


# ---------------------- Pipeline stages ----------------------

def stage_add_objects_random(state: SceneState, cfg: SceneConfig) -> SceneState:
    rng = state.rng
    n_obj = int(rng.integers(cfg.n_obj_range[0], cfg.n_obj_range[1] + 1))
    hues = _sample_distinct_hues(rng, n_obj, min_dist=0.12)

    xyz_list = []
    rgb_list = []
    pts_before = 0

    for j in range(n_obj):
        points4, _sq_params = _make_cloud_once(
            cfg.n_sq_max,
            cfg.points_per_object,
            rng=rng,
            alpha=cfg.alpha,
            growth=cfg.growth,
            max_rounds=cfg.max_rounds,
        )

        xyz = points4[:, :3].astype(np.float32, copy=False)
        sq_ids = points4[:, 3]

        if cfg.enable_scaling:
            d0 = _aabb_diag(xyz)
            if d0 > 1e-8:
                d_target = _sample_target_diag(rng, cfg)
                s = d_target / d0
                xyz = xyz * s


        base_rgb_u8 = _rgb01_to_uint8(colorsys.hsv_to_rgb(hues[j], 0.85, 0.90))
        rgb = _make_object_sq_colors(rng, sq_ids, base_rgb_u8)

        xyz = _place_object_randomly(xyz, rng, state.L, state.W, state.H, cfg.room_margin)

        pts_before += xyz.shape[0]
        xyz_list.append(xyz)
        rgb_list.append(rgb)

    if xyz_list:
        state.xyz = np.concatenate([state.xyz] + xyz_list, axis=0)
        state.rgb = np.concatenate([state.rgb] + rgb_list, axis=0)

    state.pts_before_clip += int(pts_before)
    return state


def stage_clip_to_room(state: SceneState, cfg: SceneConfig) -> SceneState:
    keep = _clip_keep_mask(state.xyz, state.L, state.W, state.H, cfg.room_margin)
    state.xyz = state.xyz[keep]
    state.rgb = state.rgb[keep]
    return state


def stage_write_outputs(state: SceneState, cfg: SceneConfig) -> SceneState:
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    np.save(cfg.out_root / "coord.npy", state.xyz.astype(np.float32, copy=False))
    np.save(cfg.out_root / "color.npy", state.rgb.astype(np.uint8, copy=False))

    if cfg.write_meta:
        meta = {
            "seed": state.scene_seed,
            "room": {"L": state.L, "W": state.W, "H": state.H, "margin": cfg.room_margin},
            "n_objects_range": list(cfg.n_obj_range),
            "object_sampler": {
                "n_sq_max": cfg.n_sq_max,
                "points_per_object": cfg.points_per_object,
                "alpha": cfg.alpha,
                "growth": cfg.growth,
                "max_rounds": cfg.max_rounds,
            },
            "object_points_before_clip": int(state.pts_before_clip),
            "points_after_clip": int(state.xyz.shape[0]),
            "toggles": {
                "enable_room_shell": cfg.enable_room_shell,
                "enable_scaling": cfg.enable_scaling,
                "enable_overlap_rejection": cfg.enable_overlap_rejection,
                "enable_floor_support": cfg.enable_floor_support,
                "enable_wall_placement": cfg.enable_wall_placement,
            },
            "room_shell": {
            "enabled": cfg.enable_room_shell,
            "density": cfg.room_shell_density,
            "max_points": cfg.room_shell_max_points,
            },
            "scaling": {
                "enabled": cfg.enable_scaling,
                "enable_size_classes": cfg.enable_size_classes,
                "size_class_probs": list(cfg.size_class_probs),
                "diag_small": list(cfg.diag_small),
                "diag_medium": list(cfg.diag_medium),
                "diag_large": list(cfg.diag_large),
                "fallback_target_diag_range": list(cfg.target_diag_range),
            },
        }
        (cfg.out_root / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote: {cfg.out_root / 'coord.npy'} and {cfg.out_root / 'color.npy'}")
    print(f"Room (L,W,H)=({state.L:.2f},{state.W:.2f},{state.H:.2f})  points={state.xyz.shape[0]}")
    return state

def stage_add_room_shell(state: SceneState, cfg: SceneConfig) -> SceneState:
    """Add points on the *inner* room surfaces so they survive margin-based clipping.

    We place shell faces at:
      x = m and x = L-m
      y = m and y = W-m
      z = m (floor) and z = H-m (ceiling)

    This matches the current clip behavior (which removes points closer than m to the boundary).
    """
    if not cfg.enable_room_shell:
        return state

    rng = state.rng
    m = cfg.room_margin
    L, W, H = state.L, state.W, state.H

    # If the room is too small for the margin, just skip.
    if (L <= 2 * m) or (W <= 2 * m) or (H <= 2 * m):
        return state

    # Areas of the *inner* surfaces
    area_floor = (L - 2 * m) * (W - 2 * m)
    area_ceiling = area_floor
    area_walls_x = 2.0 * (W - 2 * m) * (H - 2 * m)  # x=m and x=L-m
    area_walls_y = 2.0 * (L - 2 * m) * (H - 2 * m)  # y=m and y=W-m
    total_area = area_floor + area_ceiling + area_walls_x + area_walls_y

    n_total = int(round(cfg.room_shell_density * total_area))
    n_total = max(n_total, 0)
    n_total = min(n_total, cfg.room_shell_max_points)

    if n_total <= 0:
        return state

    # Split points proportional to surface area
    def alloc(area: float) -> int:
        return int(round(n_total * (area / max(total_area, 1e-9))))

    n_floor = alloc(area_floor)
    n_ceil = alloc(area_ceiling)
    n_wx_each = alloc(area_walls_x) // 2
    n_wy_each = alloc(area_walls_y) // 2

    # Fix rounding drift so sum matches n_total
    counts = [n_floor, n_ceil, 2 * n_wx_each, 2 * n_wy_each]
    drift = n_total - sum(counts)
    # Add/subtract drift to floor (arbitrary) to preserve exact total
    n_floor = max(0, n_floor + drift)

    xyz_list = []
    rgb_list = []

    # Simple fixed colors for debugging/visual distinction
    floor_rgb = np.array([110, 110, 110], dtype=np.uint8)
    ceil_rgb = np.array([210, 210, 210], dtype=np.uint8)
    wall_rgb = np.array([160, 180, 210], dtype=np.uint8)

    # floor: z = m
    if n_floor > 0:
        x = rng.uniform(m, L - m, size=n_floor).astype(np.float32)
        y = rng.uniform(m, W - m, size=n_floor).astype(np.float32)
        z = np.full(n_floor, m, dtype=np.float32)
        xyz = np.stack([x, y, z], axis=1)
        rgb = np.repeat(floor_rgb[None, :], n_floor, axis=0)
        xyz_list.append(xyz)
        rgb_list.append(rgb)

    # ceiling: z = H-m
    if n_ceil > 0:
        x = rng.uniform(m, L - m, size=n_ceil).astype(np.float32)
        y = rng.uniform(m, W - m, size=n_ceil).astype(np.float32)
        z = np.full(n_ceil, H - m, dtype=np.float32)
        xyz = np.stack([x, y, z], axis=1)
        rgb = np.repeat(ceil_rgb[None, :], n_ceil, axis=0)
        xyz_list.append(xyz)
        rgb_list.append(rgb)

    # walls x=m and x=L-m
    if n_wx_each > 0:
        for x_const in (m, L - m):
            x = np.full(n_wx_each, x_const, dtype=np.float32)
            y = rng.uniform(m, W - m, size=n_wx_each).astype(np.float32)
            z = rng.uniform(m, H - m, size=n_wx_each).astype(np.float32)
            xyz = np.stack([x, y, z], axis=1)
            rgb = np.repeat(wall_rgb[None, :], n_wx_each, axis=0)
            xyz_list.append(xyz)
            rgb_list.append(rgb)

    # walls y=m and y=W-m
    if n_wy_each > 0:
        for y_const in (m, W - m):
            x = rng.uniform(m, L - m, size=n_wy_each).astype(np.float32)
            y = np.full(n_wy_each, y_const, dtype=np.float32)
            z = rng.uniform(m, H - m, size=n_wy_each).astype(np.float32)
            xyz = np.stack([x, y, z], axis=1)
            rgb = np.repeat(wall_rgb[None, :], n_wy_each, axis=0)
            xyz_list.append(xyz)
            rgb_list.append(rgb)

    if xyz_list:
        xyz_shell = np.concatenate(xyz_list, axis=0)
        rgb_shell = np.concatenate(rgb_list, axis=0)
        state.xyz = np.concatenate([state.xyz, xyz_shell], axis=0)
        state.rgb = np.concatenate([state.rgb, rgb_shell], axis=0)

    return state



# Order matters: later you’ll insert new stages here (e.g., room shell before clip).
PIPELINE: List[Stage] = [
    stage_add_room_shell,
    stage_add_objects_random,
    stage_clip_to_room,
    stage_write_outputs,
]


# ---------------------- CLI / main ----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", required=True, help="Directory to write coord.npy + color.npy (+ meta.json).")
    p.add_argument("--seed", type=int, default=42, help="Seed for reproducibility. Use -1 for OS entropy.")
    p.add_argument("--write-meta", action="store_true", default=True, help="Write meta.json next to outputs.")
    p.add_argument("--no-write-meta", dest="write_meta", action="store_false")
    p.add_argument("--walls", action="store_true", default=True, help="Enable room shell points")
    p.add_argument("--no-walls", dest="walls", action="store_false")
    p.add_argument("--wall-density", type=float, default=200.0,
               help="Room shell density in points per m^2 (only used if --walls).")
    p.add_argument("--scale", action="store_true", default=False, help="Enable object scaling")
    p.add_argument("--size-classes", action="store_true", default=True,
                   help="Use small/medium/large diagonal ranges for scaling")
    p.add_argument("--no-size-classes", dest="size_classes", action="store_false")


    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = SceneConfig(
        out_root=Path(args.out_root),
        seed=args.seed,
        write_meta=args.write_meta,
        enable_room_shell=args.walls,
        room_shell_density=args.wall_density,
        enable_scaling=args.scale,
        enable_size_classes=args.size_classes,
    )

    rng_master = np.random.default_rng(None if cfg.seed == -1 else cfg.seed)
    scene_seed = int(rng_master.integers(0, 2**32 - 1, dtype=np.uint32))
    rng = np.random.default_rng(scene_seed)

    L, W, H = _sample_room(rng, cfg)

    state = SceneState(
        scene_seed=scene_seed,
        rng=rng,
        L=L,
        W=W,
        H=H,
        xyz=np.zeros((0, 3), dtype=np.float32),
        rgb=np.zeros((0, 3), dtype=np.uint8),
        pts_before_clip=0,
    )

    for stage in PIPELINE:
        state = stage(state, cfg)


if __name__ == "__main__":
    main()
