#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Sequence, Dict

import numpy as np
from sample_SQs import sample_SQ_naive_with_normals


# =========================
# Dataclasses for parameters
# =========================

@dataclass
class SuperquadricParams:
    """
    Parameters for a single superquadric in its own local frame.

    a:    axis lengths (a_x, a_y, a_z)
    eps1: shape exponent 1 (controls "squareness" in elevation)
    eps2: shape exponent 2 (controls "squareness" in azimuth)
    R:    3x3 rotation matrix (world ← local)
    t:    translation (world coordinates)
    """
    a: np.ndarray         # (3,)
    eps1: float
    eps2: float
    R: np.ndarray         # (3, 3)
    t: np.ndarray         # (3,)


PrimitiveType = Literal["cube", "sphere", "cylinder", "cone", "torus"]


@dataclass
class PrimitiveParams:
    """
    Parameters for a simple analytic primitive.

    primitive_type: one of {"cube","sphere","cylinder","cone","torus"}

    scale:  per-axis scale applied to the canonical primitive in its own frame.
            For example:
              - sphere: scale=(r, r, r)  → radius ~ r
              - cube:   scale=(sx, sy, sz) on a base cube [-0.5,0.5]^3
              - cylinder/cone: scale applied to a canonical shape with radius=0.5, height=1
              - torus: scale reshapes the embedded coordinates after canonical sampling

    R, t:  pose (world ← local)
    """
    primitive_type: PrimitiveType
    scale: np.ndarray     # (3,)
    R: np.ndarray         # (3, 3)
    t: np.ndarray         # (3,)


@dataclass
class ShapeComponent:
    """
    Unified wrapper for either a superquadric or a primitive.

    kind: "sq" or "primitive"

    For kind == "sq":
        sq_params must be not None, prim_params must be None.
    For kind == "primitive":
        prim_params must be not None, sq_params must be None.

    component_id is just an integer label (e.g. 0..K-1 within an object),
    useful if you want to store per-component ids in the point cloud.
    """
    kind: Literal["sq", "primitive"]
    sq_params: Optional[SuperquadricParams]
    prim_params: Optional[PrimitiveParams]
    component_id: int


# =========================
# Utilities for random rotation
# =========================

def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """
    Sample a random 3x3 rotation matrix using a random unit quaternion.
    """
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()

    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)

    x, y, z, w = q1, q2, q3, q4

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


# =========================
# Parameter samplers
# =========================

def sample_superquadric_params(
    rng: np.random.Generator,
    *,
    a_min: Sequence[float] = (0.1, 0.1, 0.1),
    a_max: Sequence[float] = (1.0, 1.0, 3.0),
    eps1_range: Tuple[float, float] = (0.3, 3.0),
    eps2_range: Tuple[float, float] = (0.3, 3.0),
    t_range: Tuple[float, float] = (-1.0, 1.0),
) -> SuperquadricParams:
    """
    Roughly matches your current get_random_SQ_pars() behaviour, but returns
    a structured SuperquadricParams instead of a raw tuple.
    """
    a = np.array(
        [
            rng.uniform(a_min[0], a_max[0]),
            rng.uniform(a_min[1], a_max[1]),
            rng.uniform(a_min[2], a_max[2]),
        ],
        dtype=np.float32,
    )
    eps1 = float(rng.uniform(*eps1_range))
    eps2 = float(rng.uniform(*eps2_range))
    R = random_rotation_matrix(rng)
    t = rng.uniform(t_range[0], t_range[1], size=(3,)).astype(np.float32)

    return SuperquadricParams(a=a, eps1=eps1, eps2=eps2, R=R, t=t)


def sample_primitive_type(
    rng: np.random.Generator,
    type_probs: Dict[PrimitiveType, float],
) -> PrimitiveType:
    """
    Sample a primitive type according to a probability dictionary.
    """
    types = list(type_probs.keys())
    probs = np.array(list(type_probs.values()), dtype=np.float64)
    probs = probs / probs.sum()
    idx = rng.choice(len(types), p=probs)
    return types[idx]


def sample_primitive_params(
    rng: np.random.Generator,
    primitive_type: PrimitiveType,
    *,
    scale_range: Tuple[float, float] = (0.3, 1.0),
    t_range: Tuple[float, float] = (-1.0, 1.0),
) -> PrimitiveParams:
    """
    Sample PrimitiveParams for a given primitive_type.

    For now we use a simple per-axis scale in [scale_min, scale_max].
    Later we can refine this per type if needed (e.g. different ranges
    for torus major/minor radius, or cylinders vs spheres).
    """
    smin, smax = scale_range
    scale = rng.uniform(smin, smax, size=(3,)).astype(np.float32)
    R = random_rotation_matrix(rng)
    t = rng.uniform(t_range[0], t_range[1], size=(3,)).astype(np.float32)
    return PrimitiveParams(primitive_type=primitive_type, scale=scale, R=R, t=t)


def sample_shape_component_sq(
    rng: np.random.Generator,
    component_id: int = 0,
    **sq_kwargs,
) -> ShapeComponent:
    """
    Create a ShapeComponent of kind="sq" with randomly sampled SQ params.
    """
    sq = sample_superquadric_params(rng, **sq_kwargs)
    return ShapeComponent(kind="sq", sq_params=sq, prim_params=None, component_id=component_id)


def sample_shape_component_primitive(
    rng: np.random.Generator,
    type_probs: Dict[PrimitiveType, float],
    component_id: int = 0,
    **prim_kwargs,
) -> ShapeComponent:
    """
    Create a ShapeComponent of kind="primitive" with randomly sampled primitive params.
    """
    ptype = sample_primitive_type(rng, type_probs=type_probs)
    prim = sample_primitive_params(rng, primitive_type=ptype, **prim_kwargs)
    return ShapeComponent(kind="primitive", sq_params=None, prim_params=prim, component_id=component_id)


# =========================
# Canonical primitive surface samplers (NumPy)
# =========================

def _sample_sphere_canonical(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unit sphere centered at origin, radius 1.
    Returns:
        pts:  (n, 3)
        nrm:  (n, 3) (same as pts, unit vectors)
    """
    pts = rng.normal(size=(n, 3)).astype(np.float32)
    norms = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    pts = pts / norms
    nrm = pts.copy()
    return pts, nrm


def _sample_cube_canonical(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Axis-aligned cube with corners at [-0.5, 0.5]^3.
    Uniformly samples faces, with correct face normals.
    """
    faces = rng.integers(0, 6, size=(n,))
    pts = np.empty((n, 3), dtype=np.float32)
    nrm = np.empty((n, 3), dtype=np.float32)

    for face in range(6):
        mask = faces == face
        m = int(mask.sum())
        if m == 0:
            continue

        u = rng.random(size=(m,)).astype(np.float32) - 0.5
        v = rng.random(size=(m,)).astype(np.float32) - 0.5

        if face == 0:  # +x
            pts[mask] = np.stack([np.full(m, 0.5, dtype=np.float32), u, v], axis=-1)
            nrm[mask] = np.array([1, 0, 0], dtype=np.float32)
        elif face == 1:  # -x
            pts[mask] = np.stack([np.full(m, -0.5, dtype=np.float32), u, v], axis=-1)
            nrm[mask] = np.array([-1, 0, 0], dtype=np.float32)
        elif face == 2:  # +y
            pts[mask] = np.stack([u, np.full(m, 0.5, dtype=np.float32), v], axis=-1)
            nrm[mask] = np.array([0, 1, 0], dtype=np.float32)
        elif face == 3:  # -y
            pts[mask] = np.stack([u, np.full(m, -0.5, dtype=np.float32), v], axis=-1)
            nrm[mask] = np.array([0, -1, 0], dtype=np.float32)
        elif face == 4:  # +z
            pts[mask] = np.stack([u, v, np.full(m, 0.5, dtype=np.float32)], axis=-1)
            nrm[mask] = np.array([0, 0, 1], dtype=np.float32)
        else:  # -z
            pts[mask] = np.stack([u, v, np.full(m, -0.5, dtype=np.float32)], axis=-1)
            nrm[mask] = np.array([0, 0, -1], dtype=np.float32)

    return pts, nrm


def _sample_cylinder_canonical(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed cylinder, radius 0.5, height 1.0, centered at z=0:
        - side surface
        - top cap at z=+0.5
        - bottom cap at z=-0.5
    Sampling is area-weighted between side and caps.
    """
    Rv = 0.5
    h = 1.0

    A_side = 2.0 * np.pi * Rv * h
    A_cap = np.pi * Rv * Rv
    p_side = A_side / (A_side + 2.0 * A_cap)

    n_side = int(round(n * p_side))
    n_caps = n - n_side
    n_cap_top = n_caps // 2
    n_cap_bottom = n_caps - n_cap_top

    pts_list = []
    nrm_list = []

    # Side
    if n_side > 0:
        z = rng.random(size=(n_side,)).astype(np.float32) - 0.5
        theta = (2.0 * np.pi * rng.random(size=(n_side,))).astype(np.float32)
        x = Rv * np.cos(theta)
        y = Rv * np.sin(theta)
        pts_side = np.stack([x, y, z], axis=-1)
        nrm_side = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)
        pts_list.append(pts_side)
        nrm_list.append(nrm_side)

    def sample_disk(m: int, z_val: float, normal: np.ndarray):
        if m <= 0:
            return None, None
        theta = (2.0 * np.pi * rng.random(size=(m,))).astype(np.float32)
        rho = np.sqrt(rng.random(size=(m,))).astype(np.float32) * Rv
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        z = np.full(m, float(z_val), dtype=np.float32)
        pts = np.stack([x, y, z], axis=-1)
        nrm = np.broadcast_to(normal.astype(np.float32), (m, 3))
        return pts, nrm

    if n_cap_top > 0:
        pts_top, nrm_top = sample_disk(n_cap_top, 0.5, np.array([0, 0, 1], dtype=np.float32))
        pts_list.append(pts_top)
        nrm_list.append(nrm_top)

    if n_cap_bottom > 0:
        pts_bot, nrm_bot = sample_disk(n_cap_bottom, -0.5, np.array([0, 0, -1], dtype=np.float32))
        pts_list.append(pts_bot)
        nrm_list.append(nrm_bot)

    pts = np.concatenate(pts_list, axis=0)
    nrm = np.concatenate(nrm_list, axis=0)
    return pts, nrm


def _sample_cone_canonical(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Right circular cone:
        - base disk radius 0.5 in z=0 plane
        - apex at (0, 0, 1)
        - lateral side + base, area-weighted
    """
    Rv = 0.5
    h = 1.0

    s = np.sqrt(Rv * Rv + h * h)    # slant height
    A_side = np.pi * Rv * s
    A_base = np.pi * Rv * Rv
    p_side = A_side / (A_side + A_base)

    n_side = int(round(n * p_side))
    n_base = n - n_side

    pts_list = []
    nrm_list = []

    # Lateral surface
    if n_side > 0:
        U = rng.random(size=(n_side,)).astype(np.float32)
        u = np.sqrt(U)
        theta = (2.0 * np.pi * rng.random(size=(n_side,))).astype(np.float32)

        r = (1.0 - u) * Rv
        z = u * h
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts_side = np.stack([x, y, z], axis=-1)

        dr_du = -Rv
        dz_du = h
        du_vec = np.stack(
            [
                dr_du * np.cos(theta),
                dr_du * np.sin(theta),
                np.full_like(theta, dz_du),
            ],
            axis=-1,
        )
        dtheta_vec = np.stack(
            [
                -r * np.sin(theta),
                r * np.cos(theta),
                np.zeros_like(theta),
            ],
            axis=-1,
        )
        n_side_vec = np.cross(dtheta_vec, du_vec, axis=-1)
        norms = np.linalg.norm(n_side_vec, axis=1, keepdims=True) + 1e-12
        n_side_vec = n_side_vec / norms

        pts_list.append(pts_side)
        nrm_list.append(n_side_vec)

    # Base disk at z=0, outward normal = -z
    if n_base > 0:
        theta = (2.0 * np.pi * rng.random(size=(n_base,))).astype(np.float32)
        rho = np.sqrt(rng.random(size=(n_base,))).astype(np.float32) * Rv
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        z = np.zeros_like(x, dtype=np.float32)
        pts_base = np.stack([x, y, z], axis=-1)
        n_base_vec = np.broadcast_to(np.array([0, 0, -1], dtype=np.float32), (n_base, 3))

        pts_list.append(pts_base)
        nrm_list.append(n_base_vec)

    pts = np.concatenate(pts_list, axis=0)
    nrm = np.concatenate(nrm_list, axis=0)
    return pts, nrm


def _sample_torus_canonical(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Torus with:
        - major radius Rv (distance from center to tube center)
        - minor radius r (tube radius)

    Uses simple parametric sampling (theta, phi).
    Surface is not perfectly area-uniform, but good enough for now.
    """
    Rv = 0.75
    r = 0.25

    theta = (2.0 * np.pi * rng.random(size=(n,))).astype(np.float32)
    phi = (2.0 * np.pi * rng.random(size=(n,))).astype(np.float32)

    x = (Rv + r * np.cos(phi)) * np.cos(theta)
    y = (Rv + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    pts = np.stack([x, y, z], axis=-1)

    # Derivatives for normals
    x_theta = -(Rv + r * np.cos(phi)) * np.sin(theta)
    y_theta = (Rv + r * np.cos(phi)) * np.cos(theta)
    z_theta = np.zeros_like(theta)
    dtheta_vec = np.stack([x_theta, y_theta, z_theta], axis=-1)

    x_phi = -r * np.sin(phi) * np.cos(theta)
    y_phi = -r * np.sin(phi) * np.sin(theta)
    z_phi = r * np.cos(phi)
    dphi_vec = np.stack([x_phi, y_phi, z_phi], axis=-1)

    n_vec = np.cross(dtheta_vec, dphi_vec, axis=-1)
    norms = np.linalg.norm(n_vec, axis=1, keepdims=True) + 1e-12
    n_vec = n_vec / norms

    return pts, n_vec


def _apply_affine_np(
    pts: np.ndarray,
    nrm: np.ndarray,
    scale: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply anisotropic scale, rotation, translation to points and normals.

    pts:   (N, 3) canonical
    nrm:   (N, 3) canonical unit normals
    scale: (3,)
    R:     (3, 3)
    t:     (3,)
    """
    # points
    pts_scaled = pts * scale[None, :]
    pts_world = pts_scaled @ R.T + t[None, :]

    # normals: n_w ∝ (R * S^{-1}) * n_c
    inv_scale = 1.0 / (scale + 1e-12)
    nrm_scaled = nrm * inv_scale[None, :]
    nrm_world = nrm_scaled @ R.T
    norms = np.linalg.norm(nrm_world, axis=1, keepdims=True) + 1e-12
    nrm_world = nrm_world / norms

    return pts_world.astype(np.float32), nrm_world.astype(np.float32)


def sample_surface_points_primitive(
    prim_params: PrimitiveParams,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample surface points + normals for a PrimitiveParams instance, in WORLD coordinates.
    """
    ptype = prim_params.primitive_type

    if ptype == "sphere":
        pts_c, nrm_c = _sample_sphere_canonical(n_points, rng)
    elif ptype == "cube":
        pts_c, nrm_c = _sample_cube_canonical(n_points, rng)
    elif ptype == "cylinder":
        pts_c, nrm_c = _sample_cylinder_canonical(n_points, rng)
    elif ptype == "cone":
        pts_c, nrm_c = _sample_cone_canonical(n_points, rng)
    elif ptype == "torus":
        pts_c, nrm_c = _sample_torus_canonical(n_points, rng)
    else:
        raise ValueError(f"Unknown primitive_type: {ptype}")

    pts_w, nrm_w = _apply_affine_np(
        pts_c,
        nrm_c,
        scale=prim_params.scale,
        R=prim_params.R,
        t=prim_params.t,
    )
    return pts_w, nrm_w


def sample_surface_points(
    comp: ShapeComponent,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample surface points and normals for a ShapeComponent.

    For now:
      - kind == "primitive" → analytic primitive samplers.
      - kind == "sq"        → uses sample_SQ_naive_with_normals via
                              sample_surface_points_sq_from_params().
    """
    if comp.kind == "primitive":
        assert comp.prim_params is not None
        return sample_surface_points_primitive(comp.prim_params, n_points, rng)

    if comp.kind == "sq":
        assert comp.sq_params is not None
        return sample_surface_points_sq_from_params(comp.sq_params, n_points, rng)

    raise ValueError(f"Unknown ShapeComponent kind: {comp.kind}")

def _to_canonical_primitive_coords(
    prim_params: PrimitiveParams,
    pts_world: np.ndarray,
) -> np.ndarray:
    """
    Map world-space points back to canonical primitive coordinates.

    Our forward mapping was:
        pts_world = (pts_canonical * scale) @ R.T + t

    So the inverse is:
        pts_canonical = ((pts_world - t) @ R) / scale
    """
    if pts_world.size == 0:
        return pts_world.reshape(-1, 3).astype(np.float32)

    R = prim_params.R.astype(np.float32)     # (3,3)
    t = prim_params.t.astype(np.float32)     # (3,)
    s = prim_params.scale.astype(np.float32) # (3,)

    pts_local = (pts_world.astype(np.float32) - t[None, :]) @ R
    pts_canonical = pts_local / (s[None, :] + 1e-12)
    return pts_canonical


def _is_inside_primitive(
    prim_params: PrimitiveParams,
    pts_world: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Check whether world-space points lie inside the volume of the primitive.
    """
    if pts_world.size == 0:
        return np.zeros((0,), dtype=bool)

    ptype = prim_params.primitive_type
    pc = _to_canonical_primitive_coords(prim_params, pts_world)  # (N,3)
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    inside = np.zeros(pc.shape[0], dtype=bool)

    if ptype == "sphere":
        # unit sphere
        r2 = x * x + y * y + z * z
        inside = r2 <= (1.0 + tol) ** 2

    elif ptype == "cube":
        # cube [-0.5,0.5]^3
        inside = np.all(np.abs(pc) <= (0.5 + tol), axis=1)

    elif ptype == "cylinder":
        # radius 0.5 in xy, height 1 along z in [-0.5, 0.5]
        r_xy2 = x * x + y * y
        inside = (r_xy2 <= (0.5 + tol) ** 2) & (np.abs(z) <= (0.5 + tol))

    elif ptype == "cone":
        # base disk radius 0.5 at z=0, apex at z=1
        # inside if 0 <= z <= 1 and radius <= 0.5 * (1 - z)
        r_xy = np.sqrt(x * x + y * y)
        inside = (z >= -tol) & (z <= 1.0 + tol)
        # avoid negative radius at top by clipping (1 - z) >= 0
        max_r = 0.5 * np.maximum(0.0, 1.0 - z)
        inside &= r_xy <= (max_r + tol)

    elif ptype == "torus":
        # torus: (sqrt(x^2 + y^2) - Rv)^2 + z^2 <= r^2
        Rv = 0.75
        r_minor = 0.25
        rho = np.sqrt(x * x + y * y)
        val = (rho - Rv) ** 2 + z * z
        inside = val <= (r_minor + tol) ** 2

    else:
        raise ValueError(f"Unknown primitive_type for inside test: {ptype}")

    return inside


def _is_inside_superquadric(
    sq_params: SuperquadricParams,
    pts_world: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Check whether world-space points lie inside the superquadric volume defined by sq_params.

    Returns:
        inside: (N,) bool
    """
    if pts_world.size == 0:
        return np.zeros((0,), dtype=bool)

    R = sq_params.R.astype(np.float32)  # (3,3)
    t = sq_params.t.astype(np.float32)  # (3,)
    a = sq_params.a.astype(np.float32)  # (3,)
    eps1 = float(sq_params.eps1)
    eps2 = float(sq_params.eps2)

    # world = local @ R.T + t  =>  local = (world - t) @ R
    pts_local = (pts_world.astype(np.float32) - t[None, :]) @ R

    x = pts_local[:, 0] / (a[0] + 1e-12)
    y = pts_local[:, 1] / (a[1] + 1e-12)
    z = pts_local[:, 2] / (a[2] + 1e-12)

    # implicit SQ expression
    term_xy = (np.abs(x) ** (2.0 / eps2) + np.abs(y) ** (2.0 / eps2)) ** (eps2 / eps1)
    term_z = np.abs(z) ** (2.0 / eps1)
    val = term_xy + term_z  # inside if <= 1

    inside = val <= (1.0 + tol)
    return inside

def is_inside_component(
    comp: ShapeComponent,
    pts_world: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Check whether world-space points lie inside the volume of this ShapeComponent.
    """
    if comp.kind == "sq":
        assert comp.sq_params is not None
        return _is_inside_superquadric(comp.sq_params, pts_world, tol=tol)
    elif comp.kind == "primitive":
        assert comp.prim_params is not None
        return _is_inside_primitive(comp.prim_params, pts_world, tol=tol)
    else:
        raise ValueError(f"Unknown ShapeComponent kind: {comp.kind}")



def sample_surface_points_sq_from_params(
    sq_params: SuperquadricParams,
    n_points: int,
    rng: np.random.Generator,
    normal_eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample ~n_points surface samples for a SuperquadricParams using
    your existing sample_SQ_naive_with_normals().

    Strategy:
      - Build the 5-parameter vector [a_x, a_y, a_z, eps1, eps2].
        (We ignore rotation/translation inside sample_SQ and apply
         the SuperquadricParams.pose afterwards, to stay consistent
         with how primitives are handled.)

      - Choose (n_theta, n_phi) such that n_theta * n_phi >= n_points,
        roughly square grid, then randomly subsample exactly n_points.

      - Apply world transform: x_w = x_local @ R.T + t, n_w = n_local @ R.T.
    """
    a_x, a_y, a_z = sq_params.a.astype(float)
    eps1 = float(sq_params.eps1)
    eps2 = float(sq_params.eps2)

    # 5-parameter form, no internal rot/trans
    sq_pars = [a_x, a_y, a_z, eps1, eps2]

    # Choose grid resolution
    # Aim for slightly more than n_points, then subsample
    n_theta = int(np.sqrt(n_points))
    if n_theta < 1:
        n_theta = 1
    n_phi = int(np.ceil(n_points / n_theta))
    n_total = n_theta * n_phi

    # Sample in object frame (no rot/trans inside sq_pars)
    P_local, N_local = sample_SQ_naive_with_normals(
        sq_pars,
        n_theta,
        n_phi,
        normal_eps=normal_eps,
    )  # P_local, N_local: (n_total, 3), float

    # Subsample to exactly n_points if we overshot
    if n_total > n_points:
        idx = rng.choice(n_total, size=n_points, replace=False)
        P_local = P_local[idx]
        N_local = N_local[idx]

    # Apply SuperquadricParams pose (world <- local)
    R = sq_params.R.astype(np.float32)  # (3, 3)
    t = sq_params.t.astype(np.float32)  # (3,)

    P_world = (P_local.astype(np.float32) @ R.T) + t[None, :]
    N_world = (N_local.astype(np.float32) @ R.T)

    # Renormalize normals just in case of numerical drift
    norms = np.linalg.norm(N_world, axis=1, keepdims=True) + 1e-12
    N_world = N_world / norms

    return P_world.astype(np.float32), N_world.astype(np.float32)


def sample_N_components_exactN_with_normals(
    components: Sequence[ShapeComponent],
    n_points: int,
    rng: np.random.Generator,
    *,
    alpha: float = 2.0,
    growth: float = 1.3,
    max_rounds: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic multi-component surface sampler (SQ + primitives).

    Symmetric overlap removal:
      For each component i, we remove any point that lies inside the volume
      of *any other* component j != i. This means that if two shapes overlap,
      the overlapping volume is empty (no points from either shape remain).

    Args:
        components: list of ShapeComponent (length K >= 1)
        n_points:   target number of points in the final object cloud
        rng:        np.random.Generator
        alpha:      oversampling factor (total ~ alpha * n_points per round)
        growth:     multiplicative factor if we don't reach n_points
        max_rounds: max oversampling rounds before giving up/relaxing

    Returns:
        points4: (n_points, 4) float32, where last column is component_id
        normals: (n_points, 3) float32
        comp_ids: (n_points,) int32 (component_id per point)
    """
    K = len(components)
    if K == 0:
        raise ValueError("sample_N_components_exactN_with_normals() called with no components.")
    if n_points <= 0:
        raise ValueError("n_points must be > 0")

    oversample_factor = alpha
    last_pts = None
    last_nrm = None
    last_ids = None

    for round_idx in range(max_rounds):
        pts_per_comp: list[np.ndarray] = []
        nrm_per_comp: list[np.ndarray] = []
        ids_per_comp: list[np.ndarray] = []

        # total oversample per component (roughly)
        n_per = int(np.ceil(oversample_factor * n_points / K))
        if n_per < 1:
            n_per = 1

        # 1) sample candidates for each component independently
        for idx, comp in enumerate(components):
            pts_cand, nrm_cand = sample_surface_points(comp, n_per, rng)
            if pts_cand.shape[0] == 0:
                # keep an empty entry so indexing stays aligned
                pts_per_comp.append(pts_cand)
                nrm_per_comp.append(nrm_cand)
                ids_per_comp.append(np.empty((0,), dtype=np.int32))
                continue

            comp_id_vec = np.full(pts_cand.shape[0], comp.component_id, dtype=np.int32)
            pts_per_comp.append(pts_cand)
            nrm_per_comp.append(nrm_cand)
            ids_per_comp.append(comp_id_vec)

        if all(p.shape[0] == 0 for p in pts_per_comp):
            # no points at all this round
            oversample_factor *= growth
            continue

        # 2) symmetric inside-removal: for each component i, drop points that
        #    lie inside the volume of ANY other component j != i
        for i in range(K):
            pts_i = pts_per_comp[i]
            if pts_i.shape[0] == 0:
                continue

            keep_i = np.ones(pts_i.shape[0], dtype=bool)
            for j in range(K):
                if j == i:
                    continue
                if pts_per_comp[j].shape[0] == 0:
                    # no points for component j, but we only need its *volume*,
                    # so we can still call is_inside_component on pts_i
                    pass
                inside_ij = is_inside_component(components[j], pts_i)
                keep_i &= ~inside_ij
                if not keep_i.any():
                    break

            pts_per_comp[i] = pts_i[keep_i]
            nrm_per_comp[i] = nrm_per_comp[i][keep_i]
            ids_per_comp[i] = ids_per_comp[i][keep_i]

        # 3) concatenate survivors
        pts_list = [p for p in pts_per_comp if p.shape[0] > 0]
        nrm_list = [n for n in nrm_per_comp if n.shape[0] > 0]
        ids_list = [i for i in ids_per_comp if i.shape[0] > 0]

        if not pts_list:
            # all points got removed due to overlaps; increase oversampling
            oversample_factor *= growth
            continue

        pts_concat = np.concatenate(pts_list, axis=0)
        nrm_concat = np.concatenate(nrm_list, axis=0)
        ids_concat = np.concatenate(ids_list, axis=0)

        last_pts, last_nrm, last_ids = pts_concat, nrm_concat, ids_concat

        M = pts_concat.shape[0]
        if M >= n_points:
            # randomly subsample exactly n_points without replacement
            idx_sel = rng.choice(M, size=n_points, replace=False)
            pts_sel = pts_concat[idx_sel]
            nrm_sel = nrm_concat[idx_sel]
            ids_sel = ids_concat[idx_sel]
            points4 = np.concatenate(
                [pts_sel.astype(np.float32), ids_sel[:, None].astype(np.float32)],
                axis=1,
            )
            return points4, nrm_sel.astype(np.float32), ids_sel

        # not enough points, increase oversample factor and try again
        oversample_factor *= growth

    # Fallback after max_rounds: sample with replacement from last round if needed
    if last_pts is None or last_pts.shape[0] == 0:
        raise RuntimeError("sample_N_components_exactN_with_normals(): no points generated after max_rounds.")

    M = last_pts.shape[0]
    if M >= n_points:
        idx_sel = rng.choice(M, size=n_points, replace=False)
    else:
        idx_sel = rng.choice(M, size=n_points, replace=True)

    pts_sel = last_pts[idx_sel]
    nrm_sel = last_nrm[idx_sel]
    ids_sel = last_ids[idx_sel]

    points4 = np.concatenate(
        [pts_sel.astype(np.float32), ids_sel[:, None].astype(np.float32)],
        axis=1,
    )
    return points4, nrm_sel.astype(np.float32), ids_sel