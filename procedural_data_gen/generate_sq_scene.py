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

from sample_SQs import _make_cloud_once_with_normals
from dataclasses import asdict, replace


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

    # primitives / SQ mix at object level
    # If enable_primitives=False, this reduces to the old "pure SQ" behavior.
    enable_primitives: bool = False
    # Per-component probability of sampling a primitive instead of a superquadric.
    p_primitive_component: float = 0.0
    # Relative weights for primitive types: (cube, sphere, cylinder, cone, torus)
    primitive_type_probs: Tuple[float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )

    # room shell (walls/floor/ceiling) points
    enable_room_shell: bool = True
    room_shell_density: float = 200.0   # points per m^2 of surface
    room_shell_max_points: int = 200_000

    # --- wall cutouts / occlusions ---
    enable_wall_cutouts: bool = False
    # Number of cutouts per scene (inclusive range)
    wall_cutouts_n_range: Tuple[int, int] = (0, 0)
    # Approx window/door size ranges in meters (in-plane extents on wall)
    wall_cutout_size_xy_range: Tuple[float, float] = (0.5, 2.0)
    wall_cutout_size_z_range: Tuple[float, float] = (0.5, 2.0)
    # Thickness of the cutout along the wall normal (meters)
    wall_cutout_thickness: float = 0.2

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

    # overlap prevention (AABB-based)
    enable_overlap_rejection: bool = False
    overlap_max_ratio: float = 0.05    # reject if inter_vol / min(volA, volB) exceeds this
    overlap_max_tries: int = 50        # placement attempts per object
    overlap_fallback_place_best: bool = True

    # placement priors
    enable_attachments: bool = True

    # z-mode probabilities (floor, float, ceiling) — will be normalized
    z_mode_probs: Tuple[float, float, float] = (0.75, 0.20, 0.05)

    # independent probability to snap to a wall (affects x/y only)
    p_wall: float = 0.45

    # gaps (meters)
    floor_gap_range: Tuple[float, float] = (0.0, 0.02)
    ceiling_gap_range: Tuple[float, float] = (0.0, 0.02)
    wall_gap_range: Tuple[float, float] = (0.0, 0.15)

    # --- sensor-like noise ---
    enable_noise: bool = False
    noise_std: float = 0.0  # meters, std dev of Gaussian noise on xyz


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
    nrm: np.ndarray  # (N,3) float32
    rgb: np.ndarray  # (N,3) uint8

    # placed effective AABBs (after intersecting with room interior)
    placed_aabb_mins: List[np.ndarray]
    placed_aabb_maxs: List[np.ndarray]


    # optional stats/debug
    pts_before_clip: int = 0


Stage = Callable[[SceneState, SceneConfig], SceneState]


# ---------------------- Small helpers ----------------------

def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml") from e

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _get(d: dict, *keys, default=None):
    """Nested dict getter: _get(cfg, 'room','margin', default=0.05)."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _merge_config_into_sceneconfig(cfg: SceneConfig, y: dict) -> SceneConfig:
    """
    Merge YAML dict into SceneConfig. YAML values override cfg defaults.
    CLI should override *after* this.
    """
    # output
    out_root = _get(y, "output", "out_root", default=None)
    seed = _get(y, "output", "seed", default=None)
    write_meta = _get(y, "output", "write_meta", default=None)

    # room
    room_margin = _get(y, "room", "margin", default=None)
    L_range = _get(y, "room", "L_range", default=None)
    W_range = _get(y, "room", "W_range", default=None)
    H_range = _get(y, "room", "H_range", default=None)

    # objects
    n_obj_range = _get(y, "objects", "n_obj_range", default=None)
    n_sq_max = _get(y, "objects", "n_sq_max", default=None)
    points_per_object = _get(y, "objects", "points_per_object", default=None)
    alpha = _get(y, "objects", "alpha", default=None)
    growth = _get(y, "objects", "growth", default=None)
    max_rounds = _get(y, "objects", "max_rounds", default=None)

    # primitives (inside "objects" section)
    obj_enable_primitives = _get(y, "objects", "enable_primitives", default=None)
    obj_p_primitive = _get(y, "objects", "p_primitive_component", default=None)
    obj_primitive_type_probs = _get(y, "objects", "primitive_type_probs", default=None)

    # walls
    walls_enable = _get(y, "walls", "enable", default=None)
    walls_density = _get(y, "walls", "density", default=None)
    walls_max_points = _get(y, "walls", "max_points", default=None)

    # wall cutouts
    enable_wall_cutouts = _get(y, "room", "enable_wall_cutouts", default=None)
    wall_cutouts_n_range = _get(y, "room", "wall_cutouts_n_range", default=None)
    wall_cutout_size_xy_range = _get(y, "room", "wall_cutout_size_xy_range", default=None)
    wall_cutout_size_z_range = _get(y, "room", "wall_cutout_size_z_range", default=None)
    wall_cutout_thickness = _get(y, "room", "wall_cutout_thickness", default=None)

    # scaling
    sc_enable = _get(y, "scaling", "enable", default=None)
    sc_enable_classes = _get(y, "scaling", "enable_size_classes", default=None)
    sc_probs = _get(y, "scaling", "size_class_probs", default=None)
    sc_small = _get(y, "scaling", "diag_small", default=None)
    sc_med = _get(y, "scaling", "diag_medium", default=None)
    sc_large = _get(y, "scaling", "diag_large", default=None)
    sc_fallback = _get(y, "scaling", "target_diag_range", default=None)

    # overlap
    ov_enable = _get(y, "overlap", "enable", default=None)
    ov_ratio = _get(y, "overlap", "max_ratio", default=None)
    ov_tries = _get(y, "overlap", "max_tries", default=None)
    ov_fallback = _get(y, "overlap", "fallback_place_best", default=None)

    # extra toggles (future)
    t_floor = _get(y, "toggles", "enable_floor_support", default=None)
    t_wall = _get(y, "toggles", "enable_wall_placement", default=None)

    # placement priors
    pl_enable = _get(y, "placement", "enable_attachments", default=None)
    pl_z_probs = _get(y, "placement", "z_mode_probs", default=None)
    pl_p_wall = _get(y, "placement", "p_wall", default=None)
    pl_floor_gap = _get(y, "placement", "floor_gap_range", default=None)
    pl_ceil_gap = _get(y, "placement", "ceiling_gap_range", default=None)
    pl_wall_gap = _get(y, "placement", "wall_gap_range", default=None)

    # noise
    enable_noise = _get(y, "room", "enable_noise", default=None)
    noise_std = _get(y, "room", "noise_std", default=None)

    # Apply overrides (convert lists->tuples where needed)
    new_cfg = replace(
        cfg,
        out_root=Path(out_root) if out_root is not None else cfg.out_root,
        seed=int(seed) if seed is not None else cfg.seed,
        write_meta=bool(write_meta) if write_meta is not None else cfg.write_meta,

        room_margin=float(room_margin) if room_margin is not None else cfg.room_margin,
        L_range=tuple(L_range) if L_range is not None else cfg.L_range,
        W_range=tuple(W_range) if W_range is not None else cfg.W_range,
        H_range=tuple(H_range) if H_range is not None else cfg.H_range,

        n_obj_range=tuple(n_obj_range) if n_obj_range is not None else cfg.n_obj_range,
        n_sq_max=int(n_sq_max) if n_sq_max is not None else cfg.n_sq_max,
        points_per_object=int(points_per_object) if points_per_object is not None else cfg.points_per_object,
        alpha=float(alpha) if alpha is not None else cfg.alpha,
        growth=float(growth) if growth is not None else cfg.growth,
        max_rounds=int(max_rounds) if max_rounds is not None else cfg.max_rounds,

        enable_primitives=bool(obj_enable_primitives)
        if obj_enable_primitives is not None
        else cfg.enable_primitives,
        p_primitive_component=float(obj_p_primitive)
        if obj_p_primitive is not None
        else cfg.p_primitive_component,
        primitive_type_probs=tuple(obj_primitive_type_probs)
        if obj_primitive_type_probs is not None
        else cfg.primitive_type_probs,

        enable_room_shell=bool(walls_enable) if walls_enable is not None else cfg.enable_room_shell,
        room_shell_density=float(walls_density) if walls_density is not None else cfg.room_shell_density,
        room_shell_max_points=int(walls_max_points) if walls_max_points is not None else cfg.room_shell_max_points,

        enable_wall_cutouts=bool(enable_wall_cutouts)
        if enable_wall_cutouts is not None
        else cfg.enable_wall_cutouts,
        wall_cutouts_n_range=tuple(wall_cutouts_n_range)
        if wall_cutouts_n_range is not None
        else cfg.wall_cutouts_n_range,
        wall_cutout_size_xy_range=tuple(wall_cutout_size_xy_range)
        if wall_cutout_size_xy_range is not None
        else cfg.wall_cutout_size_xy_range,
        wall_cutout_size_z_range=tuple(wall_cutout_size_z_range)
        if wall_cutout_size_z_range is not None
        else cfg.wall_cutout_size_z_range,
        wall_cutout_thickness=float(wall_cutout_thickness)
        if wall_cutout_thickness is not None
        else cfg.wall_cutout_thickness,

        enable_scaling=bool(sc_enable) if sc_enable is not None else cfg.enable_scaling,
        enable_size_classes=bool(sc_enable_classes) if sc_enable_classes is not None else cfg.enable_size_classes,
        size_class_probs=tuple(sc_probs) if sc_probs is not None else cfg.size_class_probs,
        diag_small=tuple(sc_small) if sc_small is not None else cfg.diag_small,
        diag_medium=tuple(sc_med) if sc_med is not None else cfg.diag_medium,
        diag_large=tuple(sc_large) if sc_large is not None else cfg.diag_large,
        target_diag_range=tuple(sc_fallback) if sc_fallback is not None else cfg.target_diag_range,

        enable_overlap_rejection=bool(ov_enable) if ov_enable is not None else cfg.enable_overlap_rejection,
        overlap_max_ratio=float(ov_ratio) if ov_ratio is not None else cfg.overlap_max_ratio,
        overlap_max_tries=int(ov_tries) if ov_tries is not None else cfg.overlap_max_tries,
        overlap_fallback_place_best=bool(ov_fallback) if ov_fallback is not None else cfg.overlap_fallback_place_best,

        enable_attachments=bool(pl_enable) if pl_enable is not None else cfg.enable_attachments,
        z_mode_probs=tuple(pl_z_probs) if pl_z_probs is not None else cfg.z_mode_probs,
        p_wall=float(pl_p_wall) if pl_p_wall is not None else cfg.p_wall,
        floor_gap_range=tuple(pl_floor_gap) if pl_floor_gap is not None else cfg.floor_gap_range,
        ceiling_gap_range=tuple(pl_ceil_gap) if pl_ceil_gap is not None else cfg.ceiling_gap_range,
        wall_gap_range=tuple(pl_wall_gap) if pl_wall_gap is not None else cfg.wall_gap_range,

        enable_noise=bool(enable_noise)
        if enable_noise is not None
        else cfg.enable_noise,
        noise_std=float(noise_std)
        if noise_std is not None
        else cfg.noise_std,

    )
    return new_cfg

def _sceneconfig_to_jsonable(cfg: SceneConfig) -> dict:
    d = asdict(cfg)
    d["out_root"] = str(d["out_root"])
    return d

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

def _aabb_min_max(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    return mins.astype(np.float32), maxs.astype(np.float32)


def _aabb_volume(mins: np.ndarray, maxs: np.ndarray) -> float:
    d = np.maximum(0.0, (maxs - mins).astype(np.float32))
    return float(d[0] * d[1] * d[2])


def _aabb_intersection(min_a: np.ndarray, max_a: np.ndarray,
                       min_b: np.ndarray, max_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_i = np.maximum(min_a, min_b)
    max_i = np.minimum(max_a, max_b)
    return min_i, max_i


def _aabb_intersection_volume(min_a: np.ndarray, max_a: np.ndarray,
                              min_b: np.ndarray, max_b: np.ndarray) -> float:
    min_i, max_i = _aabb_intersection(min_a, max_a, min_b, max_b)
    return _aabb_volume(min_i, max_i)


def _room_inner_aabb(cfg: SceneConfig, L: float, W: float, H: float) -> Tuple[np.ndarray, np.ndarray]:
    m = cfg.room_margin
    room_min = np.array([m, m, m], dtype=np.float32)
    room_max = np.array([L - m, W - m, H - m], dtype=np.float32)
    return room_min, room_max


def _effective_aabb(mins_w: np.ndarray, maxs_w: np.ndarray,
                    room_min: np.ndarray, room_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Intersect object AABB with room interior AABB and return (min,max,vol_effective)."""
    eff_min, eff_max = _aabb_intersection(mins_w, maxs_w, room_min, room_max)
    vol = _aabb_volume(eff_min, eff_max)
    return eff_min, eff_max, vol


def _overlap_ratio_with_placed(
    cand_min: np.ndarray,
    cand_max: np.ndarray,
    cand_vol: float,
    placed_mins: List[np.ndarray],
    placed_maxs: List[np.ndarray],
) -> float:
    """Return max overlap ratio vs placed boxes: inter_vol / min(vol_cand, vol_other)."""
    if cand_vol <= 0.0 or len(placed_mins) == 0:
        return 0.0

    max_ratio = 0.0
    for mn, mx in zip(placed_mins, placed_maxs):
        vol_other = _aabb_volume(mn, mx)
        if vol_other <= 0.0:
            continue
        inter = _aabb_intersection_volume(cand_min, cand_max, mn, mx)
        if inter <= 0.0:
            continue
        ratio = inter / min(cand_vol, vol_other)
        if ratio > max_ratio:
            max_ratio = ratio
    return float(max_ratio)


def _sample_translation_free(
    rng: np.random.Generator,
    L: float,
    W: float,
    H: float,
    margin: float,
) -> np.ndarray:
    """Sample a translation target (centroid) anywhere in the room interior."""
    return np.array(
        [
            rng.uniform(margin, max(margin, L - margin)),
            rng.uniform(margin, max(margin, W - margin)),
            rng.uniform(margin, max(margin, H - margin)),
        ],
        dtype=np.float32,
    )

def _sample_z_mode(rng: np.random.Generator, cfg: SceneConfig) -> int:
    """
    Returns: 0=floor, 1=float, 2=ceiling
    """
    p = np.array(cfg.z_mode_probs, dtype=np.float64)
    s = float(p.sum())
    if s <= 0:
        p = np.array([1/3, 1/3, 1/3], dtype=np.float64)
    else:
        p = p / s
    return int(rng.choice(3, p=p))

# ---------------------- Pipeline stages ----------------------

def stage_add_objects_random(state: SceneState, cfg: SceneConfig) -> SceneState:
    rng = state.rng
    room_min, room_max = _room_inner_aabb(cfg, state.L, state.W, state.H)
    n_obj = int(rng.integers(cfg.n_obj_range[0], cfg.n_obj_range[1] + 1))
    hues = _sample_distinct_hues(rng, n_obj, min_dist=0.12)

    xyz_list = []
    nrm_list = []
    rgb_list = []
    pts_before = 0

    for j in range(n_obj):
        points4, normals, _components = _make_cloud_once_with_normals(
            cfg.n_sq_max,
            cfg.points_per_object,
            rng=rng,
            alpha=cfg.alpha,
            growth=cfg.growth,
            max_rounds=cfg.max_rounds,
            use_primitives=cfg.enable_primitives,
            p_primitive=cfg.p_primitive_component,
            primitive_type_probs=cfg.primitive_type_probs,
        )

        xyz = points4[:, :3].astype(np.float32, copy=False)
        nrm = normals.astype(np.float32, copy=False)
        sq_ids = points4[:, 3]

        if cfg.enable_scaling:
            d0 = _aabb_diag(xyz)
            if d0 > 1e-8:
                d_target = _sample_target_diag(rng, cfg)
                s = d_target / d0
                xyz = xyz * s


        base_rgb_u8 = _rgb01_to_uint8(colorsys.hsv_to_rgb(hues[j], 0.85, 0.90))
        rgb = _make_object_sq_colors(rng, sq_ids, base_rgb_u8)

        # --- overlap-aware placement (optional) ---
        centroid = xyz.mean(axis=0).astype(np.float32)
        obj_mins, obj_maxs = _aabb_min_max(xyz)

        best_xyz = None
        best_nrm = None
        best_eff_min = None
        best_eff_max = None
        best_score = float("inf")  # lower is better (max overlap ratio)
        best_eff_vol = 0.0

        # --- attachment intent for this object (sample once; stable across tries) ---
        if cfg.enable_attachments:
            z_mode = _sample_z_mode(rng, cfg)            # 0=floor,1=float,2=ceiling
            on_wall = (rng.random() < cfg.p_wall)

            # decide wall details if on_wall
            wall_axis = int(rng.integers(0, 2))          # 0=x, 1=y
            wall_side = int(rng.integers(0, 2))          # 0=near min, 1=near max
            wall_gap = float(rng.uniform(*cfg.wall_gap_range))

            floor_gap = float(rng.uniform(*cfg.floor_gap_range))
            ceil_gap = float(rng.uniform(*cfg.ceiling_gap_range))
        else:
            z_mode = 1
            on_wall = False
            wall_axis = 0
            wall_side = 0
            wall_gap = 0.0
            floor_gap = 0.0
            ceil_gap = 0.0

        tries = cfg.overlap_max_tries if cfg.enable_overlap_rejection else 1
        for _ in range(max(1, tries)):
            target = _sample_translation_free(rng, state.L, state.W, state.H, cfg.room_margin)
            t = (target - centroid).astype(np.float32)

            # --- apply wall snap (x/y) ---
            if cfg.enable_attachments and on_wall:
                if wall_axis == 0:
                    # snap in x
                    if wall_side == 0:
                        # near x = margin + wall_gap (use mins)
                        desired = cfg.room_margin + wall_gap
                        t[0] += desired - (obj_mins[0] + t[0])
                    else:
                        # near x = (L - margin - wall_gap) (use maxs)
                        desired = (state.L - cfg.room_margin - wall_gap)
                        t[0] += desired - (obj_maxs[0] + t[0])
                else:
                    # snap in y
                    if wall_side == 0:
                        desired = cfg.room_margin + wall_gap
                        t[1] += desired - (obj_mins[1] + t[1])
                    else:
                        desired = (state.W - cfg.room_margin - wall_gap)
                        t[1] += desired - (obj_maxs[1] + t[1])

            # --- apply z-mode snap (floor / ceiling / float) ---
            if cfg.enable_attachments:
                if z_mode == 0:
                    # floor: mins_w[z] -> margin + floor_gap
                    desired = cfg.room_margin + floor_gap
                    t[2] += desired - (obj_mins[2] + t[2])
                elif z_mode == 2:
                    # ceiling: maxs_w[z] -> (H - margin - ceil_gap)
                    desired = (state.H - cfg.room_margin - ceil_gap)
                    t[2] += desired - (obj_maxs[2] + t[2])
                # z_mode == 1 => float: do nothing

            xyz_cand = xyz + t
            nrm_cand = nrm
            mins_w = obj_mins + t
            maxs_w = obj_maxs + t

            eff_min, eff_max, eff_vol = _effective_aabb(mins_w, maxs_w, room_min, room_max)

            # If object contributes nothing inside the room after clipping, skip this try.
            if eff_vol <= 0.0:
                score = float("inf")
            else:
                score = _overlap_ratio_with_placed(
                    eff_min, eff_max, eff_vol,
                    state.placed_aabb_mins, state.placed_aabb_maxs
                )

            if not cfg.enable_overlap_rejection:
                # old behavior: accept immediately
                best_xyz, best_nrm, best_eff_min, best_eff_max, best_eff_vol = xyz_cand, nrm_cand, eff_min, eff_max, eff_vol
                best_score = score
                break

            # accept if passes threshold
            if score <= cfg.overlap_max_ratio:
                best_xyz, best_nrm, best_eff_min, best_eff_max, best_eff_vol = xyz_cand, nrm_cand, eff_min, eff_max, eff_vol
                best_score = score
                break

            # otherwise track best candidate
            if score < best_score:
                best_xyz, best_nrm, best_eff_min, best_eff_max, best_eff_vol = xyz_cand, nrm_cand, eff_min, eff_max, eff_vol
                best_score = score

        # Decide what to do if we never found a "good enough" placement
        if best_xyz is None:
            # extremely unlikely, but safe
            continue

        if cfg.enable_overlap_rejection and (best_score > cfg.overlap_max_ratio) and (not cfg.overlap_fallback_place_best):
            # Skip this object entirely
            continue

        xyz = best_xyz
        nrm = best_nrm

        # Register effective AABB for future overlap tests (only if it has volume)
        if best_eff_vol > 0.0:
            state.placed_aabb_mins.append(best_eff_min)
            state.placed_aabb_maxs.append(best_eff_max)


        pts_before += xyz.shape[0]
        xyz_list.append(xyz)
        rgb_list.append(rgb)
        nrm_list.append(nrm)

    if xyz_list:
        state.xyz = np.concatenate([state.xyz] + xyz_list, axis=0)
        state.rgb = np.concatenate([state.rgb] + rgb_list, axis=0)
        state.nrm = np.concatenate([state.nrm] + nrm_list, axis=0)

    state.pts_before_clip += int(pts_before)
    return state


def stage_clip_to_room(state: SceneState, cfg: SceneConfig) -> SceneState:
    keep = _clip_keep_mask(state.xyz, state.L, state.W, state.H, cfg.room_margin)
    state.xyz = state.xyz[keep]
    state.rgb = state.rgb[keep]
    state.nrm = state.nrm[keep]
    return state



def stage_write_outputs(state: SceneState, cfg: SceneConfig) -> SceneState:
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    np.save(cfg.out_root / "coord.npy", state.xyz.astype(np.float32, copy=False))
    np.save(cfg.out_root / "normal.npy", state.nrm.astype(np.float32, copy=False))
    np.save(cfg.out_root / "color.npy", state.rgb.astype(np.uint8, copy=False))

    if cfg.write_meta:
        meta = build_meta(state, cfg)
        (cfg.out_root / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote: {cfg.out_root / 'coord.npy'} and {cfg.out_root / 'color.npy'}")
    print(f"Room (L,W,H)=({state.L:.2f},{state.W:.2f},{state.H:.2f})  points={state.xyz.shape[0]}")
    return state

def build_meta(state: SceneState, cfg: SceneConfig) -> dict:
    return {
        "scene_seed": int(state.scene_seed),
        "sampled_room": {"L": float(state.L), "W": float(state.W), "H": float(state.H)},
        "points": {
            "object_points_before_clip": int(state.pts_before_clip),
            "points_after_clip": int(state.xyz.shape[0]),
        },
        "scene_config": _sceneconfig_to_jsonable(cfg),
    }

def stage_apply_wall_cutouts(state: SceneState, cfg: SceneConfig) -> SceneState:
    """
    Remove points inside a few randomly placed boxes that touch the room walls.

    This simulates windows, doors, or LiDAR "missing wall" regions.
    The cutouts affect BOTH wall points and nearby object points, which is
    actually nice if you want occlusion-like artifacts.
    """
    import numpy as np

    if not cfg.enable_wall_cutouts:
        return state

    xyz = state.xyz
    if xyz.shape[0] == 0:
        return state

    # We assume state has L, W, H and a rng like the other stages.
    L, W, H = state.L, state.W, state.H
    m = cfg.room_margin

    # How many cutouts this scene gets
    n_min, n_max = cfg.wall_cutouts_n_range
    if n_max <= 0:
        return state

    if hasattr(state, "rng"):
        rng = state.rng
    else:
        rng = np.random.default_rng()

    n_cutouts = int(rng.integers(n_min, n_max + 1))
    if n_cutouts <= 0:
        return state

    keep = np.ones(xyz.shape[0], dtype=bool)

    for _ in range(n_cutouts):
        # pick a wall: 0=x=m, 1=x=L-m, 2=y=m, 3=y=W-m
        wall = int(rng.integers(0, 4))

        # sample sizes (in-plane extents) and thickness
        size_xy = rng.uniform(*cfg.wall_cutout_size_xy_range)
        size_z = rng.uniform(*cfg.wall_cutout_size_z_range)
        half_xy = 0.5 * size_xy
        half_z = 0.5 * size_z
        half_t = 0.5 * cfg.wall_cutout_thickness

        if wall in (0, 1):
            # x-walls: x ~ m or L-m, in-plane coords are (y, z)
            x_plane = m if wall == 0 else (L - m)

            y_center = rng.uniform(m, W - m)
            z_center = rng.uniform(m, H - m)

            mask = (
                (xyz[:, 0] >= x_plane - half_t) & (xyz[:, 0] <= x_plane + half_t) &
                (xyz[:, 1] >= y_center - half_xy) & (xyz[:, 1] <= y_center + half_xy) &
                (xyz[:, 2] >= z_center - half_z) & (xyz[:, 2] <= z_center + half_z)
            )
        else:
            # y-walls: y ~ m or W-m, in-plane coords are (x, z)
            y_plane = m if wall == 2 else (W - m)

            x_center = rng.uniform(m, L - m)
            z_center = rng.uniform(m, H - m)

            mask = (
                (xyz[:, 1] >= y_plane - half_t) & (xyz[:, 1] <= y_plane + half_t) &
                (xyz[:, 0] >= x_center - half_xy) & (xyz[:, 0] <= x_center + half_xy) &
                (xyz[:, 2] >= z_center - half_z) & (xyz[:, 2] <= z_center + half_z)
            )

        keep &= ~mask

    state.xyz = state.xyz[keep]
    state.nrm = state.nrm[keep]
    state.rgb = state.rgb[keep]

    # Optional: track how many were removed, to store in meta if you want
    if hasattr(state, "pts_removed_cutouts"):
        state.pts_removed_cutouts += int((~keep).sum())
    else:
        state.pts_removed_cutouts = int((~keep).sum())

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
    nrm_list = []
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
        nrm = np.repeat(np.array([[0, 0, 1]], dtype=np.float32), n_floor, axis=0)
        xyz_list.append(xyz)
        rgb_list.append(rgb)
        nrm_list.append(nrm)

    # ceiling: z = H-m
    if n_ceil > 0:
        x = rng.uniform(m, L - m, size=n_ceil).astype(np.float32)
        y = rng.uniform(m, W - m, size=n_ceil).astype(np.float32)
        z = np.full(n_ceil, H - m, dtype=np.float32)
        xyz = np.stack([x, y, z], axis=1)
        rgb = np.repeat(ceil_rgb[None, :], n_ceil, axis=0)
        nrm = np.repeat(np.array([[0, 0, -1]], dtype=np.float32), n_ceil, axis=0)
        xyz_list.append(xyz)
        rgb_list.append(rgb)
        nrm_list.append(nrm)


    # walls x=m and x=L-m
    if n_wx_each > 0:
        for x_const, nx in ((m, 1.0), (L - m, -1.0)):
            x = np.full(n_wx_each, x_const, dtype=np.float32)
            y = rng.uniform(m, W - m, size=n_wx_each).astype(np.float32)
            z = rng.uniform(m, H - m, size=n_wx_each).astype(np.float32)
            xyz = np.stack([x, y, z], axis=1)
            rgb = np.repeat(wall_rgb[None, :], n_wx_each, axis=0)
            nrm = np.repeat(np.array([[nx, 0, 0]], dtype=np.float32), n_wx_each, axis=0)

            xyz_list.append(xyz); rgb_list.append(rgb); nrm_list.append(nrm)

    # walls y=m and y=W-m
    if n_wy_each > 0:
        for y_const, ny in ((m, 1.0), (W - m, -1.0)):
            x = rng.uniform(m, L - m, size=n_wy_each).astype(np.float32)
            y = np.full(n_wy_each, y_const, dtype=np.float32)
            z = rng.uniform(m, H - m, size=n_wy_each).astype(np.float32)
            xyz = np.stack([x, y, z], axis=1)
            rgb = np.repeat(wall_rgb[None, :], n_wy_each, axis=0)
            nrm = np.repeat(np.array([[0, ny, 0]], dtype=np.float32), n_wy_each, axis=0)

            xyz_list.append(xyz); rgb_list.append(rgb); nrm_list.append(nrm)

    if xyz_list:
        xyz_shell = np.concatenate(xyz_list, axis=0)
        rgb_shell = np.concatenate(rgb_list, axis=0)
        nrm_shell = np.concatenate(nrm_list, axis=0)

        state.xyz = np.concatenate([state.xyz, xyz_shell], axis=0)
        state.rgb = np.concatenate([state.rgb, rgb_shell], axis=0)
        state.nrm = np.concatenate([state.nrm, nrm_shell], axis=0)

    return state

def stage_add_noise(state: SceneState, cfg: SceneConfig) -> SceneState:
    """
    Add simple Gaussian jitter to point coordinates to simulate sensor noise.
    Normals are left unchanged (they remain "true" normals of the underlying geometry).
    """
    import numpy as np

    if (not cfg.enable_noise) or cfg.noise_std <= 0.0:
        return state

    if state.xyz.shape[0] == 0:
        return state

    if hasattr(state, "rng"):
        rng = state.rng
    else:
        rng = np.random.default_rng()

    std = float(cfg.noise_std)
    noise = rng.normal(loc=0.0, scale=std, size=state.xyz.shape).astype(np.float32)

    state.xyz = (state.xyz.astype(np.float32) + noise).astype(np.float32)

    # Optionally track how much we perturbed for meta
    state.noise_std_applied = std

    return state


PIPELINE = [
    stage_add_room_shell,
    stage_add_objects_random,
    stage_apply_wall_cutouts,
    stage_add_noise,
    stage_clip_to_room,
    stage_write_outputs,
]

def generate_one_scene_arrays(cfg: SceneConfig, *, scene_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(int(scene_seed))
    L, W, H = _sample_room(rng, cfg)

    state = SceneState(
        scene_seed=int(scene_seed),
        rng=rng,
        L=L, W=W, H=H,
        xyz=np.zeros((0, 3), dtype=np.float32),
        rgb=np.zeros((0, 3), dtype=np.uint8),
        nrm=np.zeros((0, 3), dtype=np.float32),
        pts_before_clip=0,
        placed_aabb_mins=[],
        placed_aabb_maxs=[],
    )

    for stage in PIPELINE_CORE:
        state = stage(state, cfg)

    meta = build_meta(state, cfg)
    return state.xyz, state.rgb, state.nrm, meta


# ---------------------- CLI / main ----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str, help="Path to YAML config file.")
    return p.parse_args()



def main() -> None:
    args = parse_args()

    y = _load_yaml(args.config)

    # Start from a dummy cfg (out_root will be overwritten by YAML anyway)
    cfg = SceneConfig(out_root=Path("/tmp/unused"))

    cfg = _merge_config_into_sceneconfig(cfg, y)

    # (Optional) sanity check: ensure YAML provided out_root
    if str(cfg.out_root) in ("/tmp/unused", "", "."):
        raise ValueError("Config must set output.out_root to a real path.")

    # RNG setup
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
        nrm=np.zeros((0, 3), dtype=np.float32),
        rgb=np.zeros((0, 3), dtype=np.uint8),
        pts_before_clip=0,
        placed_aabb_mins=[],
        placed_aabb_maxs=[],
    )

    for stage in PIPELINE:
        state = stage(state, cfg)

if __name__ == "__main__":
    main()
