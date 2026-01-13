#import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from numpy.random import Generator
import os, io, json, math, time, pathlib
from typing import Dict, Tuple, List
import lmdb

# --- cache for trig grids so we don't re-mesh every call with same (n_theta, n_phi) ---
_TRIG_CACHE = {}  # key: (n_theta, n_phi) -> dict with theta, phi, ct, st, cp, sp

def _get_trig_grid(n_theta: int, n_phi: int):
    key = (n_theta, n_phi)
    c = _TRIG_CACHE.get(key)
    if c is not None:
        return c["ct"], c["st"], c["cp"], c["sp"]

    # theta starts at -pi/2 + pi/(2*n_theta), step d_theta = pi/n_theta, i=0..n_theta-1
    # phi   starts at -pi   + pi/(  n_phi), step d_phi   = 2pi/n_phi, j=0..n_phi-1
    d_theta = np.pi / n_theta
    d_phi   = 2.0 * np.pi / n_phi
    theta0  = -np.pi/2.0 + np.pi/(2.0 * n_theta)
    phi0    = -np.pi     + np.pi/   (1.0 * n_phi)

    theta = theta0 + d_theta * np.arange(n_theta)[:, None]   # (n_theta, 1)
    phi   = phi0   + d_phi   * np.arange(n_phi)[None, :]     # (1, n_phi)

    ct, st = np.cos(theta), np.sin(theta)  # (n_theta,1)
    cp, sp = np.cos(phi),   np.sin(phi)    # (1,n_phi)

    _TRIG_CACHE[key] = {"ct": ct, "st": st, "cp": cp, "sp": sp}
    return ct, st, cp, sp

def mirror_octants(pts):
    # all 8 combinations of ±1 for x,y,z
    signs = np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
        [-1, -1, -1],
    ], dtype=pts.dtype)  # shape (8,3)

    # broadcast-multiply:
    # (8,3)   * (N,3)[None,:,:] -> (8,N,3)
    mirrored = signs[None, :, :] * pts[:, None, :]

    # reshape to (8N, 3)
    out = mirrored.reshape(-1, 3)

    return out

import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _pilu_delta_theta_central(theta: float, a: float, b: float, eps: float, D: float) -> float:
    """
    Central-region chord/arc-length approximation:
      Δθ ≈ D / ||d r(θ)/dθ||
    for superellipse: x=a cos^eps θ, y=b sin^eps θ, θ in (0, π/2).

    This form *does* include negative powers when eps<1, but we only use it
    away from the endpoints (handled by the singularity branches).
    """
    c = math.cos(theta)
    s = math.sin(theta)
    # avoid 0 in the central region; endpoints are handled elsewhere
    c_abs = abs(c)
    s_abs = abs(s)

    # speed = eps * sqrt( a^2 cos^(2eps-2) θ sin^2 θ + b^2 sin^(2eps-2) θ cos^2 θ )
    # (derived from dx/dθ, dy/dθ)
    speed_sq = (
        (a * eps) ** 2 * (c_abs ** (2.0 * eps - 2.0)) * (s ** 2) +
        (b * eps) ** 2 * (s_abs ** (2.0 * eps - 2.0)) * (c ** 2)
    )
    if speed_sq <= 0.0 or not math.isfinite(speed_sq):
        return 0.0
    return D / math.sqrt(speed_sq)


def _pilu_delta_theta_near_zero(theta: float, b: float, eps: float, D: float) -> float:
    """
    Near θ≈0: y(θ) ≈ b θ^eps, enforce y(θ+Δ)-y(θ) ≈ D:

      b((θ+Δ)^eps - θ^eps) = D
      (θ+Δ)^eps = θ^eps + D/b
      Δ = (θ^eps + D/b)^(1/eps) - θ
    """
    if b <= 0.0:
        return 0.0
    base = (theta ** eps) + (D / b)
    if base <= 0.0:
        return 0.0
    return (base ** (1.0 / eps)) - theta


def _pilu_delta_theta_near_halfpi(theta: float, a: float, eps: float, D: float) -> float:
    """
    Near θ≈π/2: let u = π/2 - θ, x(θ) ≈ a u^eps decreases as θ increases.
    Enforce x(θ)-x(θ+Δ) ≈ D:

      a(u^eps - (u-Δ)^eps) = D
      (u-Δ)^eps = u^eps - D/a
      Δ = u - (u^eps - D/a)^(1/eps)

    If u^eps - D/a <= 0, we clamp to jump to the endpoint.
    """
    half_pi = 0.5 * math.pi
    u = half_pi - theta
    if u <= 0.0 or a <= 0.0:
        return 0.0
    base = (u ** eps) - (D / a)
    if base <= 0.0:
        # can't take a real root; just move to endpoint
        return u
    u_next = base ** (1.0 / eps)
    return u - u_next


def pilu_angles_superellipse(a: float, b: float, eps: float, D: float, theta_eps: float = 1e-2) -> np.ndarray:
    """
    Return a monotone increasing list of θ in [0, π/2] using Pilu-style stepping.
    """
    half_pi = 0.5 * math.pi
    theta = 0.0
    out = [0.0]

    # defensive hard cap to prevent infinite loops if D is too small
    for _ in range(300000):
        if theta >= half_pi:
            break

        if theta <= theta_eps:
            dth = _pilu_delta_theta_near_zero(theta, b, eps, D)
        elif (half_pi - theta) <= theta_eps:
            dth = _pilu_delta_theta_near_halfpi(theta, a, eps, D)
        else:
            dth = _pilu_delta_theta_central(theta, a, b, eps, D)

        # guardrails
        if not np.isfinite(dth) or dth <= 0.0:
            dth = max(1e-4, 0.5 * theta_eps)

        theta_next = theta + dth
        if theta_next <= theta:
            theta_next = theta + 1e-4

        theta = min(theta_next, half_pi)
        out.append(theta)

        if theta >= half_pi - 1e-12:
            break

    if out[-1] < half_pi:
        out.append(half_pi)

    return np.array(out, dtype=float)


def sample_SQ_pilu_dense(sq_pars, D: float, theta_eps: float = 1e-2) -> np.ndarray:
    """
    Pilu-style dense sampling of ONE SQ.
    Returns (M,3) points in WORLD coordinates (rotation+translation applied),
    matching the convention of sample_SQ_naive:
        P_world = P_obj @ R.T + t
    """
    assert len(sq_pars) in (5, 11)

    if len(sq_pars) == 5:
        ax, ay, az, eps1, eps2 = sq_pars
        euler = None
        t = None
    else:
        ax, ay, az, eps1, eps2 = sq_pars[:5]
        euler = sq_pars[5:8]
        t = np.asarray(sq_pars[8:11], dtype=float)

    # angles for vertical (η) and horizontal (ω) generating superellipses
    H = pilu_angles_superellipse(1.0, az, eps1, D, theta_eps=theta_eps)      # η in [0, π/2]
    O = pilu_angles_superellipse(ax, ay, eps2, D, theta_eps=theta_eps)       # ω in [0, π/2]

    cosH_e1 = (np.cos(H) ** eps1)
    sinH_e1 = (np.sin(H) ** eps1)
    cosO_e2 = (np.cos(O) ** eps2)
    sinO_e2 = (np.sin(O) ** eps2)

    # first-octant grid (|x|,|y|,|z|)
    X0 = ax * np.outer(cosH_e1, cosO_e2)
    Y0 = ay * np.outer(cosH_e1, sinO_e2)
    Z0 = az * (sinH_e1[:, None] * np.ones((1, len(O))))

    # mirror to all 8 octants
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append(np.stack([ (sx * X0).ravel(),
                                      (sy * Y0).ravel(),
                                      (sz * Z0).ravel() ], axis=1))
    P = np.vstack(pts)

    # apply optional rotation/translation (same convention as sample_SQ_naive)
    if euler is not None:
        R = Rot.from_euler("xyz", euler).as_matrix()
        P = P @ R.T
    if t is not None:
        P = P + t

    return P

def sample_N_SQs_pilu_exactN(
    sq_pars_N,
    n_points: int,
    *,
    alpha: float = 2.0,
    D0: float = 0.03,
    shrink: float = 0.8,
    max_rounds: int = 8,
    theta_eps: float = 1e-2,
    max_pts_per_sq: int | None = None,
    rng: Generator,
):
    """
    Global oversample (Pilu) → remove overlaps → global thin to exactly n_points.

    Returns: (n_points, 4) float array, last col = SQ id.

    Notes:
      - Oversampling density is controlled by D (smaller D => more points).
      - We retry with D *= shrink each round if we don't get enough survivors.
      - Optional max_pts_per_sq can cap per-SQ points for speed (randomly thinned before overlap removal).
    """
    rng = _require_gen(rng)
    n_SQs = len(sq_pars_N)
    if n_points <= 0 or n_SQs == 0:
        return np.empty((0, 4), dtype=float)

    # rough per-component target for oversampling (used only for a default cap)
    pts_target_per_sq = int(np.ceil(alpha * n_points / max(n_SQs, 1)))
    if max_pts_per_sq is None:
        # conservative: allow some headroom so overlap removal still leaves enough
        max_pts_per_sq = max(5000, 4 * pts_target_per_sq)

    M = 0
    D = float(D0)

    for _round in range(max_rounds):
        all_pts = []

        for i in range(n_SQs):
            pts = sample_SQ_pilu_dense(sq_pars_N[i], D=D, theta_eps=theta_eps)  # (Mi, 3)

            # Optional early cap for speed/memory (keeps overlap removal tractable)
            if pts.shape[0] > max_pts_per_sq:
                idx = rng.choice(pts.shape[0], size=max_pts_per_sq, replace=False)
                pts = pts[idx]

            # Remove points that lie inside other SQs (same logic as naive path)
            for j in range(n_SQs):
                if j != i:
                    pts = remove_points_inside_SQ(pts, sq_pars_N[j])
                    if pts.size == 0:
                        break

            if pts.size:
                ids = np.full((pts.shape[0], 1), i, dtype=float)
                all_pts.append(np.concatenate([pts, ids], axis=1))

        if not all_pts:
            # nothing survived; densify
            D *= shrink
            continue

        survivors = np.concatenate(all_pts, axis=0)
        M = survivors.shape[0]

        if M >= n_points:
            idx = rng.choice(M, n_points, replace=False)
            return survivors[idx]

        # Not enough points survived; densify and retry
        D *= shrink

    raise ValueError(
        f"Could not reach {n_points} points after {max_rounds} rounds; "
        f"last count={M}, last D={D}"
    )


def sample_SQ_Pilu(sq_pars, n_theta_quad, n_phi_quad):
    assert (len(sq_pars) in (5, 11))
    assert n_theta_quad > 0 and n_phi_quad > 0

    if len(sq_pars) == 5:
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars
        euler = None
        t = None
    else:
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars[:5]
        euler = sq_pars[5:8]
        t     = sq_pars[8:11]
    # optional rotation
    if euler is not None:
        R = Rot.from_euler('xyz', euler).as_matrix()
        P = P @ R.T

    points = np.zeros((n_theta_quad * n_phi_quad, 3), dtype=float)

    # constant helper vars
    a_x_sq = a_x ** 2.0
    a_y_sq = a_y ** 2.0
    a_z_sq = a_z ** 2.0

    theta_min = 0.1
    theta_max = (np.pi / 2.0) - theta_min
    theta = theta_min

    phi_min = 0.1
    phi_max = (np.pi / 2.0) - phi_min
    phi = phi_min

    # Approximate "Radius" of shape at z = 0. ~Avg. distance to z-axis of shape in z=0 plane
    # Needed for getting reasonable value for D (arclength)
    R_approx = (a_x + a_y) / 2.0

    idx = 0

    while theta < theta_max:
        #D = np.pi/2.0 * R_approx * np.cos(theta) / n_theta_quad
        D = np.pi/2.0 * R_approx * 1.0 / n_theta_quad

        while phi < phi_max:
            
            # helper variables
            cos_theta_eps_1 = np.cos(theta)**eps_1
            sin_theta_eps_1 = np.sin(theta)**eps_1
            cos_phi_eps_2 = np.cos(phi)**eps_2
            sin_phi_eps_2 = np.sin(phi)**eps_2

            cos_phi_sq = np.cos(phi)**2.0
            cos_phi_sq_eps_2 = cos_phi_sq ** eps_2

            sin_phi_sq = np.sin(phi)**2.0
            sin_phi_sq_eps_2 = sin_phi_sq ** eps_2

            # x, y, z for previous phi theta pair
            x = a_x * cos_theta_eps_1 * cos_phi_eps_2
            y = a_y * cos_theta_eps_1 * sin_phi_eps_2
            z = a_z * sin_theta_eps_1

            points[idx] = np.array([x,y,z], dtype=float) # Save to array

            # helper variables to calc phi increase
            sin_phi_pow_4 = sin_phi_sq ** 2.0
            cos_phi_pow_4 = cos_phi_sq ** 2.0

            frac_1 = D / (eps_2 * cos_theta_eps_1)
            frac_2_num = sin_phi_sq * cos_phi_sq
            frac_2_den_p1 = a_x_sq * cos_phi_sq_eps_2 * sin_phi_pow_4
            frac_2_den_p2 = a_y_sq * sin_phi_sq_eps_2 * cos_phi_pow_4
            frac_2 = frac_2_num / (frac_2_den_p1 + frac_2_den_p2)

            # Final delta phi
            delta_phi = frac_1 * np.sqrt(frac_2)

            phi = phi + delta_phi
            idx += 1

        # Calculate delta theta for same starting phi
        phi = phi_min

        # helper vars
        cos_theta_sq = np.cos(theta) ** 2.0
        cos_theta_pow_4 = cos_theta_sq ** 2.0
        cos_theta_sq_pow_eps_1 = cos_theta_sq ** eps_1

        sin_theta_sq = np.sin(theta) ** 2.0
        sin_theta_pow_4 = sin_theta_sq ** 2.0
        sin_theta_sq_pow_eps_1 = sin_theta_sq ** eps_1

        # (2nd time of calculation -> optimize ?)
        cos_phi_sq = np.cos(phi)**2.0
        cos_phi_sq_eps_2 = cos_phi_sq ** eps_2
        sin_phi_sq = np.sin(phi)**2.0
        sin_phi_sq_eps_2 = sin_phi_sq ** eps_2

        frac_1 = D / eps_1
        frac_2_num = cos_theta_sq * sin_theta_sq

        # Duplicate calc cos_theta_sq_pow_eps_1 * sin_theta_pow_4
        frac_2_den_p1 = a_x_sq * cos_phi_sq_eps_2 * cos_theta_sq_pow_eps_1 * sin_theta_pow_4
        frac_2_den_p2 = a_y_sq * sin_phi_sq_eps_2 * cos_theta_sq_pow_eps_1 * sin_theta_pow_4

        frac_2_den_p3 = a_z_sq * sin_theta_sq_pow_eps_1 * cos_theta_pow_4

        frac_2 = frac_2_num / (frac_2_den_p1 + frac_2_den_p2 + frac_2_den_p3)

        delta_theta = frac_1 * np.sqrt(frac_2)

        theta += delta_theta

    points = mirror_octants(points)

    return points

def sample_SQ_naive(sq_pars, n_theta, n_phi):
    """
    Vectorized (parallel) sampling.
    Returns (n_theta*n_phi, 3) float array.
    """
    assert (len(sq_pars) in (5, 11))
    assert n_theta > 0 and n_phi > 0

    if len(sq_pars) == 5:
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars
        euler = None
        t = None
    else:
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars[:5]
        euler = sq_pars[5:8]
        t     = sq_pars[8:11]

    # trig grids (broadcastable to (n_theta, n_phi))
    ct, st, cp, sp = _get_trig_grid(n_theta, n_phi)

    # sign * |.|^eps, done fully vectorized
    # shapes: ct/st => (n_theta,1), cp/sp => (1,n_phi), broadcasts to (n_theta,n_phi)
    cx = np.sign(ct) * np.abs(ct) ** eps_1
    sx = np.sign(st) * np.abs(st) ** eps_1
    c2 = np.sign(cp) * np.abs(cp) ** eps_2
    s2 = np.sign(sp) * np.abs(sp) ** eps_2

    X = a_x * (cx * c2)   # (n_theta, n_phi)
    Y = a_y * (cx * s2)
    Z = a_z * np.broadcast_to(sx, (n_theta, n_phi))

    # flatten in the SAME order as the nested loops: i over theta outer, j over phi inner
    P = np.empty((n_theta * n_phi, 3), dtype=float)
    P[:, 0] = X.reshape(-1)
    P[:, 1] = Y.reshape(-1)
    P[:, 2] = Z.reshape(-1)

    # optional rotation
    if euler is not None:
        R = Rot.from_euler('xyz', euler).as_matrix()
        P = P @ R.T

    # optional translation
    if t is not None:
        P = P + np.asarray(t, dtype=float)

    return P

import numpy as np
from scipy.spatial.transform import Rotation as Rot

def sample_SQ_naive_with_normals(sq_pars, n_theta, n_phi, *, normal_eps=1e-12):
    """
    Vectorized sampling + analytic normals for the SAME SQ parametrization as sample_SQ_naive().

    Returns:
      P: (n_theta*n_phi, 3) float
      N: (n_theta*n_phi, 3) float  (unit normals, outward)

    SQ pars:
      len==5  : [a_x, a_y, a_z, eps_1, eps_2]  (no rot, no trans)
      len==11 : first 5 + euler(xyz) + t(xyz)
    """
    assert len(sq_pars) in (5, 11)
    assert n_theta > 0 and n_phi > 0

    if len(sq_pars) == 5:
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars
        euler = None
        t = None
    else:
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars[:5]
        euler = sq_pars[5:8]
        t     = sq_pars[8:11]

    # trig grids (broadcastable to (n_theta, n_phi))
    ct, st, cp, sp = _get_trig_grid(n_theta, n_phi)  # ct/st: (n_theta,1), cp/sp: (1,n_phi)

    # sign * |.|^eps exactly like your existing code
    cx = np.sign(ct) * np.abs(ct) ** eps_1   # C_{eps1}(theta)
    sx = np.sign(st) * np.abs(st) ** eps_1   # S_{eps1}(theta)
    c2 = np.sign(cp) * np.abs(cp) ** eps_2   # C_{eps2}(phi)
    s2 = np.sign(sp) * np.abs(sp) ** eps_2   # S_{eps2}(phi)

    # positions (n_theta, n_phi)
    X = a_x * (cx * c2)
    Y = a_y * (cx * s2)
    Z = a_z * np.broadcast_to(sx, (n_theta, n_phi))

    # ---- analytic derivatives wrt theta/phi (object frame) ----
    # d/du [sign(cos u)|cos u|^e] = -e * sin(u) * |cos u|^(e-1)
    # d/du [sign(sin u)|sin u|^e] =  e * cos(u) * |sin u|^(e-1)
    abs_ct = np.maximum(np.abs(ct), normal_eps)
    abs_st = np.maximum(np.abs(st), normal_eps)
    abs_cp = np.maximum(np.abs(cp), normal_eps)
    abs_sp = np.maximum(np.abs(sp), normal_eps)

    dC1 = -eps_1 * st * (abs_ct ** (eps_1 - 1.0))   # shape (n_theta,1)
    dS1 =  eps_1 * ct * (abs_st ** (eps_1 - 1.0))   # shape (n_theta,1)

    dC2 = -eps_2 * sp * (abs_cp ** (eps_2 - 1.0))   # shape (1,n_phi)
    dS2 =  eps_2 * cp * (abs_sp ** (eps_2 - 1.0))   # shape (1,n_phi)

    # dP/dtheta = [a_x*dC1*c2, a_y*dC1*s2, a_z*dS1]
    Ax = a_x * (dC1 * c2)
    Ay = a_y * (dC1 * s2)
    Az = a_z * np.broadcast_to(dS1, (n_theta, n_phi))

    # dP/dphi = [a_x*cx*dC2, a_y*cx*dS2, 0]
    Bx = a_x * (cx * dC2)
    By = a_y * (cx * dS2)

    # normal = dP/dtheta x dP/dphi
    # cross([Ax,Ay,Az],[Bx,By,0]) = [-Az*By, Az*Bx, Ax*By - Ay*Bx]
    Nx = -Az * By
    Ny =  Az * Bx
    Nz =  Ax * By - Ay * Bx

    # flatten in SAME order as sample_SQ_naive (theta outer, phi inner)
    P = np.empty((n_theta * n_phi, 3), dtype=float)
    P[:, 0] = X.reshape(-1)
    P[:, 1] = Y.reshape(-1)
    P[:, 2] = Z.reshape(-1)

    N = np.empty((n_theta * n_phi, 3), dtype=float)
    N[:, 0] = Nx.reshape(-1)
    N[:, 1] = Ny.reshape(-1)
    N[:, 2] = Nz.reshape(-1)

    # orient outward: ensure n · p >= 0 (SQ centered at origin in object frame)
    dots = (N * P).sum(axis=1)
    flip = dots < 0.0
    N[flip] *= -1.0

    # normalize
    nrm = np.linalg.norm(N, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-20)
    N = N / nrm

    # optional rotation (same convention as sample_SQ_naive: P = P @ R.T)
    if euler is not None:
        R = Rot.from_euler('xyz', euler).as_matrix()
        P = P @ R.T
        N = N @ R.T

    # optional translation
    if t is not None:
        P = P + np.asarray(t, dtype=float)

    return P, N

def sq_inside_value(P_world: np.ndarray, sq_pars, *, eps=1e-12) -> np.ndarray:
    """
    Compute inside-outside value f(P) for the superellipsoid that matches your sampler.
    Inside: f <= 1
    """
    assert len(sq_pars) in (5, 11)
    P = P_world

    if len(sq_pars) == 5:
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars
        euler = None
        t = None
    else:
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars[:5]
        euler = sq_pars[5:8]
        t     = sq_pars[8:11]

    # transform world -> object (inverse of: P_obj @ R.T + t)
    if t is not None:
        P = P - np.asarray(t, dtype=float)
    if euler is not None:
        R = Rot.from_euler('xyz', euler).as_matrix()
        P = P @ R  # because row-vectors and P_world = P_obj @ R.T => P_obj = P_world @ R

    x = P[:, 0] / max(float(a_x), eps)
    y = P[:, 1] / max(float(a_y), eps)
    z = P[:, 2] / max(float(a_z), eps)

    # f = ((|x|^(2/eps2) + |y|^(2/eps2))^(eps2/eps1) + |z|^(2/eps1))
    p = 2.0 / max(float(eps_2), eps)
    q = 2.0 / max(float(eps_1), eps)

    s = (np.abs(x) ** p + np.abs(y) ** p)
    f = (s ** (float(eps_2) / max(float(eps_1), eps))) + (np.abs(z) ** q)
    return f

def remove_points_inside_SQ_mask(P_world: np.ndarray, sq_pars, *, eps=1e-12) -> np.ndarray:
    """
    Return keep-mask for points that are NOT inside the SQ (i.e., keep outside points).
    """
    f = sq_inside_value(P_world, sq_pars, eps=eps)
    return f > 1.0


def set_equal_axes_quadrant_aware(ax, points):
    P = np.asarray(points)[:, :3]
    mins, maxs = P.min(0), P.max(0)
    lo, hi = mins.copy(), maxs.copy()

    # Clamp to zero if data is one-sided on an axis
    for k in range(3):
        if mins[k] >= 0: lo[k] = 0       # all ≥ 0 -> start at 0
        if maxs[k] <= 0: hi[k] = 0       # all ≤ 0 -> end at 0

    spans = hi - lo
    R = spans.max()  # target common span

    # Grow each axis to length R with minimal empty space
    for k in range(3):
        if spans[k] == R: 
            continue
        if lo[k] == 0 and hi[k] > 0:        # positive-only axis
            hi[k] = R
        elif hi[k] == 0 and lo[k] < 0:      # negative-only axis
            lo[k] = -R
        else:                               # mixed-sign: expand both sides evenly
            d = (R - spans[k]) / 2.0
            lo[k] -= d; hi[k] += d

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1,1,1))  # equal visual aspect


def show_points(points, point_size=5):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    if points.shape[1] > 3:
        for i in np.unique(points[:, 3]):
            p = points[points[:, 3] == i]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=point_size, label=int(i))
        ax.legend()
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size)

    set_equal_axes_quadrant_aware(ax, points)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

# Note: Does not save labels
def save_pc(path, points):
    pts = np.asarray(points)
    xyz = pts[:, :3].astype(np.float64, copy=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)

def read_pc(path):
    if not os.path.isfile(path): raise FileNotFoundError(path)
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float32)

# Note: Saves labels
def save_npy(path, points, keep_labels=True):
    pts = np.asarray(points)
    xyz = pts[:, :3].astype(np.float32, copy=False)
    if keep_labels and pts.shape[1] >= 4:
        np.save(path, {'points': xyz, 'labels': pts[:, 3].astype(np.int64, copy=False)})
    else:
        np.save(path, xyz)

def read_npy(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, dict):
        pts = np.asarray(obj['points'], dtype=np.float32)
        lbl = obj.get('labels'); lbl = None if lbl is None else np.asarray(lbl, dtype=np.int64)
        return pts, lbl
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0 and obj.dtype == object:
            d = obj.item(); return np.asarray(d['points'], dtype=np.float32), np.asarray(d.get('labels'), dtype=np.int64) if d.get('labels') is not None else None
        if obj.ndim == 2 and obj.shape[1] == 3:
            return obj.astype(np.float32, copy=False), None
    raise ValueError("Unsupported .npy content (expected dict or (N,3) array).")

def remove_points_inside_SQ(points, sq_pars):
    """
    Removes all points that are within SQ defined by sq_pars
    """
    pts = np.asarray(points, dtype=float)  # preserve original dtype/shape at return
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"`points` must be (N,3); got {pts.shape}")

    # --- unpack params & optional inverse transform (once) ---
    if len(sq_pars) == 5:
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars
        P = pts  # no transform
    elif len(sq_pars) == 11:
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars[:5]
        euler = sq_pars[5:8]
        t = np.asarray(sq_pars[8:11], dtype=float)
        R_inv = Rot.from_euler('xyz', euler).inv().as_matrix()
        # inverse of (R, t): x' = R^{-1}(x - t)
        P = (pts - t) @ R_inv.T
    else:
        raise ValueError("sq_pars must have length 5 or 11")

    X = np.abs(P[:, 0])
    Y = np.abs(P[:, 1])
    Z = np.abs(P[:, 2])

    # --- implicit function (vectorized) ---
    # f = ( (|x|/a_x)^(2/eps2) + (|y|/a_y)^(2/eps2) )^(eps2/eps1) + (|z|/a_z)^(2/eps1)
    rx = (X / a_x) ** (2.0 / eps_2)
    ry = (Y / a_y) ** (2.0 / eps_2)
    rxy = (rx + ry) ** (eps_2 / eps_1)
    rz = (Z / a_z) ** (2.0 / eps_1)
    f = rxy + rz

    # filter if inside
    inside = f < 1.0
    return pts[~inside]

def remove_points_inside_SQ_v2(P_world: np.ndarray, sq_pars, *, eps=1e-12) -> np.ndarray:
    keep = remove_points_inside_SQ_mask(P_world, sq_pars, eps=eps)
    return P_world[keep]


def sample_two_SQ_naive(sq_pars_1st, sq_pars_2nd, n_theta, n_phi):
    points_1st = sample_SQ_naive(sq_pars_1st, n_theta, n_phi)
    points_2nd = sample_SQ_naive(sq_pars_2nd, n_theta, n_phi)

    points_1st = remove_points_inside_SQ(points_1st, sq_pars_2nd)
    points_2nd = remove_points_inside_SQ(points_2nd, sq_pars_1st)

    # add ids to distinguish sq
    points_1st = np.concatenate((points_1st, np.full((points_1st.shape[0], 1), 0)), axis=1)
    points_2nd = np.concatenate((points_2nd, np.full((points_2nd.shape[0], 1), 1)), axis=1)

    return np.concatenate((points_1st, points_2nd), axis=0)

def sample_N_SQs_naive(sq_pars_N, n_theta, n_phi):
    n_SQs = len(sq_pars_N) # number of SQs

    all_points_list = []

    for i in range(n_SQs):
        current_points = sample_SQ_naive(sq_pars_N[i], n_theta, n_phi)

        # Remove points within other SQs
        for j in range(n_SQs):
            if j != i:
                current_points = remove_points_inside_SQ(current_points, sq_pars_N[j])


        # add ids to distinguish sq
        current_points = np.concatenate((current_points, np.full((current_points.shape[0], 1), i)), axis=1)
        all_points_list.append(current_points)

    return np.concatenate(all_points_list, axis=0)

def get_random_SQ_pars_v_2(seed=None, centered=False):
    """
    Sample a realistic, numerically stable set of superquadric parameters.
    Returns a list:
        [a_x, a_y, a_z, eps_1, eps_2, euler_x, euler_y, euler_z, t_x, t_y, t_z]
    """

    if seed is not None:
        np.random.seed(seed)

    # --- 1. Scale parameters (avoid degeneracies) ---
    # Log-uniform for better coverage of small and large scales
    a_x = 10 ** np.random.uniform(-0.3, 0.3)  # ~[0.5, 2.0]
    a_y = 10 ** np.random.uniform(-0.3, 0.3)
    a_z = 10 ** np.random.uniform(-0.3, 0.6)  # allow taller shapes

    # Optionally normalize to roughly constant volume
    volume_norm = (a_x * a_y * a_z) ** (1/3)
    a_x, a_y, a_z = a_x / volume_norm, a_y / volume_norm, a_z / volume_norm

    # --- 2. Exponents (roundness) ---
    eps_1 = np.random.uniform(0.3, 3.0)
    eps_2 = np.random.uniform(0.3, 3.0)

    # --- 3. Orientation (uniform over SO(3)) ---
    rot = Rot.random()
    euler_x, euler_y, euler_z = rot.as_euler('xyz', degrees=False)

    # --- 4. Translation ---
    if centered:
        t_x, t_y, t_z = 0.0, 0.0, 0.0
    else:
        t_x = np.random.uniform(-1.0, 1.0)
        t_y = np.random.uniform(-1.0, 1.0)
        t_z = np.random.uniform(-1.0, 1.0)

    return [a_x, a_y, a_z, eps_1, eps_2, euler_x, euler_y, euler_z, t_x, t_y, t_z]

def _require_gen(rng):
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator (e.g., np.random.default_rng(42))")
    return rng

def get_random_SQ_pars(rng: Generator, centered: bool=False):
    rng = _require_gen(rng)
    U = rng.uniform

    a_x = U(0.1, 1.0); a_y = U(0.1, 1.0); a_z = U(0.1, 3.0)
    eps_1 = U(0.3, 3.0); eps_2 = U(0.3, 3.0)
    euler_x = U(0.0, 2*np.pi); euler_y = U(0.0, 2*np.pi); euler_z = U(0.0, 2*np.pi)

    if centered:
        t_x = t_y = t_z = 0.0
    else:
        t_x = U(-1.0, 1.0); t_y = U(-1.0, 1.0); t_z = U(-1.0, 1.0)

    return [a_x, a_y, a_z, eps_1, eps_2, euler_x, euler_y, euler_z, t_x, t_y, t_z]

def sample_SQ_naive_exactN(sq_pars, n_points: int, rng: Generator):
    """Uses your existing sample_SQ_naive; oversample to k×k then thin to exactly n_points."""
    rng = _require_gen(rng)
    k = int(np.ceil(np.sqrt(n_points)))
    dense = sample_SQ_naive(sq_pars, k, k)
    if dense.shape[0] == n_points:
        return dense
    idx = rng.choice(dense.shape[0], size=n_points, replace=False)
    return dense[idx]

def sample_N_SQs_naive_exactN(sq_pars_N, n_points: int, *, alpha=2.0, growth=1.3, max_rounds=6, rng: Generator):
    """Global oversample → remove overlaps → global thin to exactly n_points."""
    rng = _require_gen(rng)
    n_SQs = len(sq_pars_N)
    if n_points <= 0 or n_SQs == 0:
        return np.empty((0,4), dtype=float)

    k = int(np.ceil(np.sqrt(max(1.0, alpha * n_points / n_SQs))))
    for _ in range(max_rounds):
        all_pts = []
        for i in range(n_SQs):
            pts = sample_SQ_naive(sq_pars_N[i], k, k)  # (k*k, 3)
            for j in range(n_SQs):
                if j != i:
                    pts = remove_points_inside_SQ(pts, sq_pars_N[j])
            if pts.size:
                ids = np.full((pts.shape[0], 1), i)
                all_pts.append(np.concatenate([pts, ids], axis=1))
        if not all_pts:
            k = int(np.ceil(k * growth)); continue

        survivors = np.concatenate(all_pts, axis=0)
        M = survivors.shape[0]
        if M >= n_points:
            idx = rng.choice(M, n_points, replace=False)
            return survivors[idx]
        k = int(np.ceil(k * growth))
    raise ValueError(f"Could not reach {n_points} points after {max_rounds} rounds; last count={M}, k={k}")

def sample_N_SQs_naive_exactN_with_normals(
    sq_pars_N, n_points: int, *, alpha=2.0, growth=1.3, max_rounds=6, rng=None, normal_eps=1e-12
):
    """
    Global oversample → remove overlaps → global thin to exactly n_points.
    Returns:
      survivors: (n_points, 4)  with last col = SQ id
      normals:   (n_points, 3)  unit normals corresponding to survivors
    """
    rng = _require_gen(rng)
    n_SQs = len(sq_pars_N)
    if n_points <= 0 or n_SQs == 0:
        return np.empty((0, 4), dtype=float), np.empty((0, 3), dtype=float)

    k = int(np.ceil(np.sqrt(max(1.0, alpha * n_points / n_SQs))))
    M = 0

    for _ in range(max_rounds):
        all_pts = []
        all_nrm = []

        for i in range(n_SQs):
            P, N = sample_SQ_naive_with_normals(sq_pars_N[i], k, k, normal_eps=normal_eps)  # (k*k,3), (k*k,3)

            for j in range(n_SQs):
                if j == i:
                    continue
                keep = remove_points_inside_SQ_mask(P, sq_pars_N[j])
                P = P[keep]
                N = N[keep]
                if P.size == 0:
                    break

            if P.size:
                ids = np.full((P.shape[0], 1), i, dtype=float)
                all_pts.append(np.concatenate([P, ids], axis=1))
                all_nrm.append(N)

        if not all_pts:
            k = int(np.ceil(k * growth))
            continue

        survivors = np.concatenate(all_pts, axis=0)
        normals = np.concatenate(all_nrm, axis=0)
        M = survivors.shape[0]

        if M >= n_points:
            idx = rng.choice(M, n_points, replace=False)
            return survivors[idx], normals[idx]

        k = int(np.ceil(k * growth))

    raise ValueError(f"Could not reach {n_points} points after {max_rounds} rounds; last count={M}, k={k}")

def gen_random_SQs_points(n_sqs: int, n_points: int, *, rng: Generator, alpha=2.0, growth=1.3, max_rounds=6):
    """Create n_sqs random SQs with get_random_SQ_pars(rng=child_rng) and sample exactly n_points total."""
    rng = _require_gen(rng)
    # derive independent child generators deterministically from the parent
    child_seeds = rng.integers(0, 2**32 - 1, size=n_sqs, dtype=np.uint32)
    sq_pars_list = [get_random_SQ_pars(np.random.default_rng(int(s))) for s in child_seeds]
    points = sample_N_SQs_naive_exactN(
        sq_pars_list, n_points, alpha=alpha, growth=growth, max_rounds=max_rounds, rng=rng
    )
    return points, sq_pars_list

def _ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def _np_serialize_to_bytes(obj: object) -> bytes:
    buf = io.BytesIO()
    np.save(buf, obj, allow_pickle=True)
    return buf.getvalue()

def _estimate_item_bytes(n_points: int, mode: str, dtype_points: np.dtype) -> int:
    # very conservative + overhead padding
    dt = np.dtype(dtype_points)
    if mode == "xyz_only":
        core = n_points * 3 * dt.itemsize
        return int(core * 1.35)  # ~35% padding for key/val overhead in LMDB
    elif mode == "enriched":
        # points + labels(int64) + small params list + header/np.save overhead
        core = (n_points * 3 * dt.itemsize) + (n_points * np.dtype(np.int64).itemsize) + 512
        return int(core * 1.35)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _items_per_shard(max_shard_bytes: int, n_points: int, mode: str, dtype_points: np.dtype, safety: float=0.9) -> int:
    usable = int(max_shard_bytes * safety)
    per = _estimate_item_bytes(n_points, mode, dtype_points)
    k = max(1, usable // max(per, 1))
    # keep it round-ish for nicer shard sizes
    if k > 10000:
        k = (k // 1000) * 1000
    elif k > 1000:
        k = (k // 100) * 100
    return max(1, int(k))

def _child_generators(parent: Generator, n: int) -> List[Generator]:
    seeds = parent.integers(0, 2**32 - 1, size=n, dtype=np.uint32)
    return [np.random.default_rng(int(s)) for s in seeds]

def _make_cloud_once(
    n_SQ_max: int,
    n_points: int,
    *,
    rng: Generator,
    alpha: float = 2.0,
    growth: float = 1.3,
    max_rounds: int = 6,
    sampling: str = "naive",          # "naive" or "pilu"
    pilu_D0: float = 0.03,
    pilu_shrink: float = 0.8,
    pilu_theta_eps: float = 1e-2,
    pilu_max_rounds: int | None = None,
    pilu_max_pts_per_sq: int | None = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    """One attempt: returns (points_with_ids Nx4 float64, sq_pars_list) or raises ValueError if cannot reach N."""
    rng = _require_gen(rng)

    n_sqs = int(rng.integers(1, n_SQ_max + 1))
    gens = _child_generators(rng, n_sqs)
    sq_pars_list = [get_random_SQ_pars(g) for g in gens]

    if sampling == "naive":
        pts = sample_N_SQs_naive_exactN(
            sq_pars_list,
            n_points,
            alpha=alpha,
            growth=growth,
            max_rounds=max_rounds,
            rng=rng,
        )
    elif sampling == "pilu":
        pts = sample_N_SQs_pilu_exactN(
            sq_pars_list,
            n_points,
            alpha=alpha,
            D0=pilu_D0,
            shrink=pilu_shrink,
            max_rounds=(max_rounds if pilu_max_rounds is None else pilu_max_rounds),
            theta_eps=pilu_theta_eps,
            max_pts_per_sq=pilu_max_pts_per_sq,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown sampling='{sampling}' (expected 'naive' or 'pilu')")

    return pts, sq_pars_list


def _make_cloud_once_with_normals(
    n_SQ_max: int,
    n_points: int,
    *,
    rng: Generator,
    alpha: float = 2.0,
    growth: float = 1.3,
    max_rounds: int = 6,
    normal_eps: float = 1e-12,
    use_primitives: bool = False,
    p_primitive: float = 0.0,
    primitive_type_probs=None,
):
    """
    One attempt: returns (points_with_ids Nx4 float64, normals Nx3 float64, components)

    This is now a generic multi-component sampler:
      - We sample n_components in [1, n_SQ_max].
      - For each component:
          with probability p_primitive (if use_primitives=True):
            sample a primitive (cube/sphere/cylinder/cone/torus),
          otherwise:
            sample a superquadric.
      - Then we call shapes.sample_N_components_exactN_with_normals()
        to get a union surface with symmetric overlap removal.

    compatibility notes:
      - If use_primitives=False or p_primitive <= 0, this reduces to a
        "pure SQ" object, but using the new generic sampler instead of
        sample_N_SQs_naive_exactN_with_normals().
      - The third return value is now the list of ShapeComponent objects
        instead of sq_pars_list, but the scene generator does not use it.
    """
    from shapes import (
        sample_shape_component_sq,
        sample_shape_component_primitive,
        sample_N_components_exactN_with_normals,
    )

    rng = _require_gen(rng)

    # Number of components (SQs + primitives) in this object
    n_components = int(rng.integers(1, n_SQ_max + 1))

    # Build primitive type prob dict if provided (cube, sphere, cylinder, cone, torus)
    if primitive_type_probs is not None:
        names = ["cube", "sphere", "cylinder", "cone", "torus"]
        # Handle len mismatch defensively
        n_names = min(len(names), len(primitive_type_probs))
        prim_type_probs_dict = {
            names[i]: float(primitive_type_probs[i]) for i in range(n_names)
        }
        # If the user passed fewer entries, fill the rest with 0.0
        for name in names[n_names:]:
            prim_type_probs_dict[name] = 0.0
    else:
        prim_type_probs_dict = {
            "cube": 1.0,
            "sphere": 1.0,
            "cylinder": 1.0,
            "cone": 1.0,
            "torus": 1.0,
        }

    components = []
    for cid in range(n_components):
        if use_primitives and (p_primitive > 0.0) and (rng.random() < p_primitive):
            comp = sample_shape_component_primitive(
                rng,
                type_probs=prim_type_probs_dict,
                component_id=cid,
            )
        else:
            comp = sample_shape_component_sq(
                rng,
                component_id=cid,
            )
        components.append(comp)

    # Use the generic multi-component sampler (SQ + primitives)
    points4, nrm, comp_ids = sample_N_components_exactN_with_normals(
        components,
        n_points=n_points,
        rng=rng,
        alpha=alpha,
        growth=growth,
        max_rounds=max_rounds,
    )

    # points4 already has shape (N, 4): [x, y, z, component_id]
    # normals is (N, 3)
    # We return components instead of sq_pars_list; the scene generator only
    # cares about points4/normals, so this is safe.
    return points4, nrm, components


# ---- NPY writer (batched) ----
def _write_npy_batch(batch: List[Tuple[np.ndarray, List[List[float]]]],
                     out_dir: pathlib.Path, start_idx: int, mode: str, dtype_points: np.dtype):
    _ensure_dir(out_dir)
    idx = start_idx
    for points4, params in batch:
        if mode == "xyz_only":
            arr = points4[:, :3].astype(dtype_points, copy=False)
            np.save(out_dir / f"sample_{idx:08d}.npy", arr)
        else:
            xyz = points4[:, :3].astype(dtype_points, copy=False)
            labels = points4[:, 3].astype(np.int64, copy=False)
            obj = {'points': xyz, 'labels': labels, 'sq_params': params}
            np.save(out_dir / f"sample_{idx:08d}.npy", obj, allow_pickle=True)
        idx += 1


# ---- LMDB shard writer ----
class _ShardWriter:
    def __init__(self, shard_dir: pathlib.Path, map_size: int):
        if lmdb is None:
            raise RuntimeError("lmdb package not available; install it or use storage='npy'")
        _ensure_dir(shard_dir)
        self.shard_dir = shard_dir
        self.env = lmdb.open(
            str(shard_dir),
            map_size=map_size,
            subdir=True,
            max_dbs=1,
            lock=True,
            writemap=True,
            map_async=False,
            metasync=False,
            sync=False,
        )
        self.txn = self.env.begin(write=True)
        self.count = 0
        self.bytes = 0
        self.shapes: Dict[str, int] = {}
        self.dtypes: Dict[str, int] = {}
        self.t0 = time.time()

    def put(self, key: str, value_bytes: bytes, *, pts_shape: Tuple[int, int], dtype_points: str):
        self.txn.put(key.encode('utf-8'), value_bytes)
        self.count += 1
        self.bytes += len(value_bytes) + len(key)
        self.shapes[str(pts_shape)] = self.shapes.get(str(pts_shape), 0) + 1
        self.dtypes[dtype_points] = self.dtypes.get(dtype_points, 0) + 1

    def commit(self):
        self.txn.commit()
        self.txn = self.env.begin(write=True)

    def close_with_metadata(self):
        duration = time.time() - self.t0
        meta = {
            "manifest": str(self.shard_dir / "data.mdb"),
            "items": self.count,
            "written": self.count,
            "bytes": self.bytes,
            "shapes": self.shapes,
            "dtypes": self.dtypes,
            "duration_sec": round(duration, 2)
        }
        with open(self.shard_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        self.txn.commit()
        self.env.sync()
        self.env.close()


def _bytes_from_json(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def _write_split_files_always(out_root: pathlib.Path, total: int, *,
                              train_ratio: float = 1.0,
                              shuffle: bool = True) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Always writes train.txt and test.txt (filenames fixed).
    Lines contain LMDB keys of the form "train:{idx:08d}" (the *name* inside is arbitrary).
    """
    rng_local = np.random.default_rng(0)
    indices = np.arange(total)
    if shuffle:
        rng_local.shuffle(indices)

    n_train = int(round(train_ratio * total))
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    # Keys must match what we actually wrote into LMDB (see main loop below)
    train_keys = [f"train:{i:08d}" for i in train_idx]
    test_keys  = [f"train:{i:08d}" for i in test_idx]   # same shards are fine; we split by keys

    (out_root / "train.txt").write_text("\n".join(train_keys) + ("\n" if train_keys else ""))
    (out_root / "test.txt").write_text("\n".join(test_keys)   + ("\n" if test_keys else ""))

    return (out_root / "train.txt", out_root / "test.txt")

def generate_pointzero_like_dataset(
    out_root: str,
    n_clouds: int,
    n_SQ_max: int,
    *,
    dataset_name: str = "train",
    n_points_per_cloud: int = 8192,
    storage: str = "lmdb",                 # "lmdb" or "npy"
    mode: str = "enriched",                # "enriched" or "xyz_only"
    dtype_points: np.dtype = np.float32,
    lmdb_shard_bytes: int = (8 << 30),     # 8 GB
    lmdb_safety: float = 0.90,
    lmdb_txn_batch: int = 100,
    npy_batch_size: int = 200,
    alpha: float = 2.0, growth: float = 1.3, max_rounds: int = 6,
    sampling: str = "naive",
    pilu_D0: float = 0.03,
    pilu_shrink: float = 0.8,
    pilu_theta_eps: float = 1e-2,
    pilu_max_rounds: int | None = None,
    pilu_max_pts_per_sq: int | None = None,
    rng: np.random.Generator = None,
    train_ratio: float = 0.95,
    shuffle_split: bool = True,
) -> Dict[str, object]:
    """
    Generate exactly `n_clouds` clouds with `n_points_per_cloud` points each.

    - storage="lmdb": write shards under {out_root}/{dataset_name}/shard_xxxxx/
      IMPORTANT: We ALWAYS write split files named {out_root}/train.txt and {out_root}/test.txt
                 that list keys of the form "train:{idx:08d}" so the repo's dataloader
                 can use subset='train' / 'test' in YAML without any code changes.

    - storage="npy": write files under {out_root}/npy/{dataset_name}/  (same semantics)

    - mode="xyz_only": main LMDB/Npy value contains ONLY (N,3) float32 points.
    - mode="enriched": main value is STILL (N,3) float32 points (loader-compatible),
                       and we ALSO write sidecar keys:
                         "<key>.labels"    -> (N,) int32 superquadric index per point
                         "<key>.sq_params" -> (S,P) float32 per-cloud superquadric params
                         "<key>.meta"      -> JSON bytes with shapes/dtypes

    Returns a summary dict and writes train.txt / test.txt split files to `out_root`.
    """
    rng = _require_gen(rng)

    out_root = pathlib.Path(out_root)
    if storage == "lmdb":
        base_dir = out_root / dataset_name
    elif storage == "npy":
        base_dir = out_root / dataset_name
    else:
        raise ValueError("storage must be 'lmdb' or 'npy'")
    _ensure_dir(base_dir)

    per_item = _estimate_item_bytes(n_points_per_cloud, mode, np.dtype(dtype_points))
    items_per = _items_per_shard(lmdb_shard_bytes, n_points_per_cloud, mode, np.dtype(dtype_points), lmdb_safety)

    summary = {
        "out_dir": str(base_dir),
        "storage": storage,
        "mode": mode,
        "dtype_points": str(np.dtype(dtype_points)),
        "n_points_per_cloud": n_points_per_cloud,
        "target_clouds": n_clouds,
        "saved_clouds": 0,
        "skipped_attempts": 0,
        "est_bytes_per_item": per_item,
        "items_per_shard": items_per,
        "shards_written": 0,
        "train_split": None,
        "test_split": None,
        "sampling": sampling,
        "pilu_D0": pilu_D0 if sampling == "pilu" else None,
        "pilu_shrink": pilu_shrink if sampling == "pilu" else None,
        "pilu_theta_eps": pilu_theta_eps if sampling == "pilu" else None,
    }

    # --- LMDB writer init ---
    if storage == "lmdb":
        shard_idx = 0
        written_in_shard = 0
        shard_dir = base_dir / f"shard_{shard_idx:05d}"
        writer = _ShardWriter(shard_dir, map_size=lmdb_shard_bytes)
        summary["shards_written"] = 1

    batch_mem: List[Tuple[np.ndarray, List[List[float]]]] = []
    next_idx = 0

    while summary["saved_clouds"] < n_clouds:
        try:
            # points4: (N,4) where last column are int labels (SQ index)
            # params:  list-of-lists with per-SQ parameter vectors for this cloud
            points4, params = _make_cloud_once(
                n_SQ_max,
                n_points_per_cloud,
                rng=rng,
                alpha=alpha,
                growth=growth,
                max_rounds=max_rounds,
                sampling=sampling,
                pilu_D0=pilu_D0,
                pilu_shrink=pilu_shrink,
                pilu_theta_eps=pilu_theta_eps,
                pilu_max_rounds=pilu_max_rounds,
                pilu_max_pts_per_sq=pilu_max_pts_per_sq,
            )
        except ValueError:
            summary["skipped_attempts"] += 1
            continue

        if storage == "lmdb":
            # MAIN KEY (loader-compatible): (N,3) float32
            key = f"train:{next_idx:08d}"  # fixed "train:" so split files can be train.txt/test.txt
            xyz = points4[:, :3].astype(dtype_points, copy=False)
            writer.put(key, _np_serialize_to_bytes(xyz),
                       pts_shape=(n_points_per_cloud, 3),
                       dtype_points=str(np.dtype(dtype_points)))

            if mode == "enriched":
                # SIDECARS
                labels = points4[:, 3].astype(np.int32, copy=False)
                sq_params = np.asarray(params, dtype=np.float32)  # (S,P)

                writer.put(f"{key}.labels", _np_serialize_to_bytes(labels),
                           pts_shape=(n_points_per_cloud,), dtype_points="int32")
                writer.put(f"{key}.sq_params", _np_serialize_to_bytes(sq_params),
                           pts_shape=tuple(sq_params.shape), dtype_points="float32")

            written_in_shard += 1
            summary["saved_clouds"] += 1
            next_idx += 1

            # commit batched
            if (written_in_shard % lmdb_txn_batch) == 0:
                writer.commit()

            # rotate shard if full
            if written_in_shard >= items_per:
                writer.close_with_metadata()
                shard_idx += 1
                shard_dir = base_dir / f"shard_{shard_idx:05d}"
                writer = _ShardWriter(shard_dir, map_size=lmdb_shard_bytes)
                summary["shards_written"] += 1
                written_in_shard = 0

        else:  # storage == "npy"
            batch_mem.append((points4, params))
            summary["saved_clouds"] += 1
            next_idx += 1

            if len(batch_mem) >= npy_batch_size:
                _write_npy_batch(batch_mem, base_dir, start_idx=next_idx - len(batch_mem),
                                 mode=mode, dtype_points=np.dtype(dtype_points))
                batch_mem.clear()

    # flush tails
    if storage == "lmdb":
        writer.commit()
        writer.close_with_metadata()
    else:
        if batch_mem:
            _write_npy_batch(batch_mem, base_dir, start_idx=next_idx - len(batch_mem),
                             mode=mode, dtype_points=np.dtype(dtype_points))
            batch_mem.clear()

    # Write split files train.txt / test.txt
    train_split, test_split = _write_split_files_always(
        base_dir, total=summary["saved_clouds"],
        train_ratio=train_ratio, shuffle=shuffle_split
    )
    summary["train_split"] = str(train_split)
    summary["test_split"]  = str(test_split)
    return summary

from time import perf_counter
from contextlib import contextmanager
from collections import defaultdict

class Prof:
    def __init__(self):
        self.t = defaultdict(float)   # total time per section
        self.n = defaultdict(int)     # calls per section

    @contextmanager
    def section(self, name: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            self.t[name] += perf_counter() - t0
            self.n[name] += 1

    def report(self, top=None):
        items = sorted(self.t.items(), key=lambda kv: kv[1], reverse=True)
        if top is not None: items = items[:top]
        for k, v in items:
            calls = self.n[k]
            avg = (v / calls) if calls else 0.0
            print(f"{k:24s} total={v*1000:8.1f} ms  calls={calls:6d}  avg={avg*1000:7.2f} ms")

def make_cloud_profiled(n_SQ_max, n_points, *, rng, alpha=2.0, growth=1.3, max_rounds=6):
    prof = Prof()
    with prof.section("total_make_cloud"):
        with prof.section("draw_n_sqs"):
            n_sqs = int(rng.integers(1, n_SQ_max + 1))
        with prof.section("draw_params"):
            from numpy.random import default_rng
            child_seeds = rng.integers(0, 2**32 - 1, size=n_sqs, dtype=np.uint32)
            sq_pars = [get_random_SQ_pars(np.random.default_rng(int(s))) for s in child_seeds]
        # mirror your exact-N routine but time its parts:
        k = int(np.ceil(np.sqrt(max(1.0, alpha * n_points / n_sqs))))
        for round_id in range(max_rounds):
            all_pts = []
            with prof.section("round_loop"):
                for i in range(n_sqs):
                    with prof.section("sample_one_SQ"):
                        pts = sample_SQ_naive(sq_pars[i], k, k)  # (k*k,3)
                    with prof.section("overlap_removal"):
                        for j in range(n_sqs):
                            if j != i:
                                pts = remove_points_inside_SQ(pts, sq_pars[j])
                    with prof.section("label_concat"):
                        if pts.size:
                            ids = np.full((pts.shape[0], 1), i)
                            all_pts.append(np.concatenate([pts, ids], axis=1))
            if not all_pts:
                k = int(np.ceil(k * growth)); continue
            with prof.section("concat_all"):
                survivors = np.concatenate(all_pts, axis=0)
            M = survivors.shape[0]
            if M >= n_points:
                with prof.section("global_subsample"):
                    idx = rng.choice(M, n_points, replace=False)
                    out = survivors[idx]
                prof.t["rounds"] = prof.t.get("rounds", 0) + (round_id + 1)
                return out, sq_pars, prof
            k = int(np.ceil(k * growth))
        raise ValueError("could not reach target points")
    
import numpy as np

# ----------------------------
# Geometry: superellipsoid
# ----------------------------

def r_theta(ax, ay, az, eps1, eps2, theta, phi):
    """
    ∂r/∂θ for
      x = ax * cos^eps1(theta) * cos^eps2(phi)
      y = ay * cos^eps1(theta) * sin^eps2(phi)
      z = az * sin^eps1(theta)
    """
    cth  = np.cos(theta)
    sth  = np.sin(theta)
    cph  = np.cos(phi)
    sph  = np.sin(phi)

    # powers (first octant -> all nonnegative; no abs needed)
    cth_e1    = cth**eps1
    sth_e1    = sth**eps1
    cph_e2    = cph**eps2
    sph_e2    = sph**eps2

    # derivatives wrt theta
    # d/dθ cos^e1 = -e1 * cos^(e1-1) * sin
    # d/dθ sin^e1 =  e1 * sin^(e1-1) * cos
    d_cth_e1  = -eps1 * (cth**(eps1 - 1.0)) * sth
    d_sth_e1  =  eps1 * (sth**(eps1 - 1.0)) * cth

    dx = ax * d_cth_e1 * cph_e2
    dy = ay * d_cth_e1 * sph_e2
    dz = az * d_sth_e1

    return np.array([dx, dy, dz])


def r_phi(ax, ay, az, eps1, eps2, theta, phi):
    """
    ∂r/∂φ for
      x = ax * cos^eps1(theta) * cos^eps2(phi)
      y = ay * cos^eps1(theta) * sin^eps2(phi)
      z = az * sin^eps1(theta)
    """
    cth  = np.cos(theta)
    sth  = np.sin(theta)
    cph  = np.cos(phi)
    sph  = np.sin(phi)

    cth_e1 = cth**eps1

    # d/dφ cos^e2 = -e2 * cos^(e2-1) * sin
    # d/dφ sin^e2 =  e2 * sin^(e2-1) * cos
    d_cph_e2 = -eps2 * (cph**(eps2 - 1.0)) * sph
    d_sph_e2 =  eps2 * (sph**(eps2 - 1.0)) * cph

    dx = ax * cth_e1 * d_cph_e2
    dy = ay * cth_e1 * d_sph_e2
    dz = 0.0

    return np.array([dx, dy, dz])


def J_area_element(ax, ay, az, eps1, eps2, theta, phi):
    """
    Area element J(θ, φ) = || ∂r/∂θ × ∂r/∂φ || (first octant)
    """
    vt = r_theta(ax, ay, az, eps1, eps2, theta, phi)
    vp = r_phi  (ax, ay, az, eps1, eps2, theta, phi)
    return np.linalg.norm(np.cross(vt, vp))


def r_point(ax, ay, az, eps1, eps2, theta, phi):
    """
    Position on the surface at (θ, φ), first octant.
    """
    cth = np.cos(theta)
    sth = np.sin(theta)
    cph = np.cos(phi)
    sph = np.sin(phi)

    x = ax * (cth**eps1) * (cph**eps2)
    y = ay * (cth**eps1) * (sph**eps2)
    z = az * (sth**eps1)
    return np.array([x, y, z])


# ----------------------------
# Simple numerical helpers
# ----------------------------

def trapz(y, x):
    """Trapezoidal rule with nonuniform x allowed (1D)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))

def cumtrapz_pdf_to_cdf(pdf, x):
    """
    Cum. trapezoid to build CDF from a pdf sampled on x.
    Returns CDF normalized to [0,1].
    """
    pdf = np.maximum(pdf, 0.0)  # guard tiny negatives from roundoff
    area = trapz(pdf, x)
    if area <= 0:
        # fallback: uniform if pdf degenerate
        return np.linspace(0, 1, len(pdf))
    pdfn = pdf / area
    cdf = np.zeros_like(pdfn)
    dx  = np.diff(x)
    avg = 0.5 * (pdfn[1:] + pdfn[:-1])
    cdf[1:] = np.cumsum(avg * dx)
    # ensure exactly 1 at end (numerical nicety)
    cdf[-1] = 1.0
    return cdf

def inverse_cdf(cdf, x, u):
    """
    Invert monotone CDF sampled on x at probability u in [0,1].
    Linear interpolation within bracketing bin.
    """
    cdf = np.asarray(cdf)
    x   = np.asarray(x)
    u   = np.asarray(u)

    # clip u to [0,1]
    u = np.clip(u, 0.0, 1.0)

    # indices where cdf[i] <= u < cdf[i+1]
    idx = np.searchsorted(cdf, u, side='right') - 1
    idx = np.clip(idx, 0, len(cdf) - 2)

    c0 = cdf[idx]
    c1 = cdf[idx + 1]
    x0 = x[idx]
    x1 = x[idx + 1]

    # avoid division by zero if flat
    w = np.where(c1 > c0, (u - c0) / (c1 - c0), 0.0)
    return x0 + w * (x1 - x0)


# ----------------------------
# Equal-area sampler (first octant)
# ----------------------------

def sample_superellipsoid_equal_area_first_octant(ax, ay, az, eps1, eps2,
                                                  N,
                                                  G_theta=256, G_phi=256,
                                                  eps_angle=1e-6,
                                                  K=None,
                                                  rng=None):
    """
    Equal-area sampling on the first octant patch (θ∈(0,π/2), φ∈(0,π/2)).

    Parameters
    ----------
    ax, ay, az : float
        Semi-axes.
    eps1, eps2 : float
        Superellipsoid exponents.
    N : int
        Desired number of points on the first octant.
    G_theta, G_phi : int
        Grid sizes for building θ-marginal and φ-conditionals via quadrature.
    eps_angle : float
        Small margin to avoid exact endpoints (singular behavior).
    K : int or None
        Number of rings. If None, use round(sqrt(N)).
    rng : np.random.Generator or None
        If provided, use randomized stratification; otherwise deterministic mid-quantiles.

    Returns
    -------
    pts : (N, 3) array
        Points on the first octant patch, approximately uniform w.r.t. surface area.
    """
    # 1) Quadrature grids
    theta_grid = np.linspace(eps_angle, 0.5*np.pi - eps_angle, G_theta)
    phi_grid   = np.linspace(eps_angle, 0.5*np.pi - eps_angle, G_phi)

    # 2) Precompute J(θ_i, φ_j)
    J_table = np.empty((G_theta, G_phi), dtype=float)
    for i, th in enumerate(theta_grid):
        for j, ph in enumerate(phi_grid):
            J_table[i, j] = J_area_element(ax, ay, az, eps1, eps2, th, ph)

    # 3) θ-marginal M(θ) = ∫ J(θ, φ) dφ  and its CDF
    #    Use trapezoidal rule across φ for each θ_i.
    M_theta = np.array([trapz(J_table[i, :], phi_grid) for i in range(G_theta)])
    cdf_theta = cumtrapz_pdf_to_cdf(M_theta, theta_grid)

    # 4) Choose number of rings K and place ring angles via θ-CDF inversion
    if K is None:
        K = max(1, int(round(np.sqrt(N))))
    # targets for θ (deterministic mid-quantiles or random)
    if rng is None:
        t_targets = (np.arange(K) + 0.5) / K
    else:
        t_targets = (np.arange(K) + rng.random(K)) / K
    theta_rings = inverse_cdf(cdf_theta, theta_grid, t_targets)

    # 5) Allocate points per ring proportional to M(θ_k)
    w = np.interp(theta_rings, theta_grid, M_theta)
    w_sum = np.sum(w)
    if w_sum <= 0:
        # degenerate fallback: spread evenly
        Nk = np.full(K, max(1, N // K), dtype=int)
    else:
        Nk_float = N * (w / w_sum)
        Nk = np.maximum(1, np.floor(Nk_float + 0.5).astype(int))
        # fix rounding to match exactly N
        diff = int(np.sum(Nk) - N)
        if diff != 0:
            # adjust by +/-1 on rings with largest rounding residuals
            resid = Nk_float - Nk
            order = np.argsort(resid)  # ascending
            if diff > 0:   # remove diff points
                for idx in order[:diff]:
                    if Nk[idx] > 1:
                        Nk[idx] -= 1
            else:          # add -diff points
                for idx in order[::-1][:(-diff)]:
                    Nk[idx] += 1

    # 6) For each ring θ_k, build conditional CDF over φ and sample Nk points
    pts = []
    for k, th in enumerate(theta_rings):
        # ring J(φ) at fixed θ=th
        J_ring = np.array([J_area_element(ax, ay, az, eps1, eps2, th, ph) for ph in phi_grid])
        cdf_phi = cumtrapz_pdf_to_cdf(J_ring, phi_grid)

        m = Nk[k]
        if rng is None:
            u = (np.arange(m) + 0.5) / m  # deterministic stratified
        else:
            u = (np.arange(m) + rng.random(m)) / m
        phi_samp = inverse_cdf(cdf_phi, phi_grid, u)

        # to 3D points
        for ph in phi_samp:
            pts.append(r_point(ax, ay, az, eps1, eps2, th, ph))

    pts = np.array(pts, dtype=float)
    # Trim or pad in rare cases due to rounding
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # duplicate a few last points to match N (rare; you can also resample)
        need = N - len(pts)
        pts = np.vstack([pts, pts[-need:]])

    return pts


# ----------------------------
# Mirroring to full surface
# ----------------------------

def mirror_octants(points_first_octant):
    """
    Mirror +x,+y,+z patch to all 8 octants by sign flips.
    Avoid duplicates at planes by assuming input excludes exact zeros.
    """
    signs = np.array([[ 1,  1,  1],
                      [-1,  1,  1],
                      [ 1, -1,  1],
                      [ 1,  1, -1],
                      [-1, -1,  1],
                      [-1,  1, -1],
                      [ 1, -1, -1],
                      [-1, -1, -1]], dtype=float)
    out = (points_first_octant[:, None, :] * signs[None, :, :]).reshape(-1, 3)
    return out


def sample_superellipsoid_equal_area_pointwise(ax, ay, az, eps1, eps2,
                                               N,
                                               G_theta=256, G_phi=256,
                                               eps_angle=1e-6,
                                               rng=None):
    """
    Equal-area sampling on the first octant patch via pointwise 2-stage CDF:
      1) sample theta from its marginal p(θ) ∝ ∫ J(θ, φ) dφ
      2) sample phi  from conditional p(φ|θ) ∝ J(θ, φ)

    Requires helper functions defined earlier:
      J_area_element, trapz, cumtrapz_pdf_to_cdf, inverse_cdf, r_point

    Parameters
    ----------
    ax, ay, az : floats       # semi-axes
    eps1, eps2 : floats       # exponents
    N          : int          # number of points (first octant only)
    G_theta    : int          # θ grid for marginal build (quadrature)
    G_phi      : int          # φ grid for marginal/conditional (quadrature)
    eps_angle  : float        # avoid exact endpoints (singular behavior)
    rng        : np.random.Generator or None
                               If None: deterministic stratification in θ,
                               and a simple deterministic spread for φ.

    Returns
    -------
    pts : (N,3) float array, first-octant points, ~uniform w.r.t. surface area
    """
    # ---- Build θ, φ grids for quadrature (avoid endpoints) ----
    theta_grid = np.linspace(eps_angle, 0.5*np.pi - eps_angle, G_theta)
    phi_grid   = np.linspace(eps_angle, 0.5*np.pi - eps_angle, G_phi)

    # ---- Precompute J(θ_i, φ_j) on a tensor grid (for θ-marginal) ----
    J_table = np.empty((G_theta, G_phi), dtype=float)
    for i, th in enumerate(theta_grid):
        for j, ph in enumerate(phi_grid):
            J_table[i, j] = J_area_element(ax, ay, az, eps1, eps2, th, ph)

    # ---- θ marginal M(θ) = ∫ J(θ, φ) dφ and CDF C_θ ----
    M_theta   = np.array([trapz(J_table[i, :], phi_grid) for i in range(G_theta)])
    C_theta   = cumtrapz_pdf_to_cdf(M_theta, theta_grid)

    # ---- Draw N points: θ_i from C_θ^{-1}, then φ_i from C(·|θ_i) ----
    pts = np.empty((N, 3), dtype=float)

    # Stratification in θ: mid-quantiles if rng is None, else randomized within bins
    if rng is None:
        u_theta = (np.arange(N) + 0.5) / N          # stratified θ
        # decorrelated low-discrepancy φ sequence (Kronecker)
        alpha = (np.sqrt(5.0) - 1.0) / 2.0          # ≈ 0.6180339887
        u_phi_seed = (0.5 + alpha * np.arange(N)) % 1.0
    else:
        u_theta = (np.arange(N) + rng.random(N)) / N
        u_phi_seed = rng.random(N) 

    # Invert θ-CDF
    thetas = inverse_cdf(C_theta, theta_grid, u_theta)

    # For each θ_i, build conditional CDF over φ and sample one φ_i
    for i in range(N):
        th = thetas[i]

        # Conditional along φ at fixed θ_i
        J_ring = np.array([J_area_element(ax, ay, az, eps1, eps2, th, ph) for ph in phi_grid])
        C_phi  = cumtrapz_pdf_to_cdf(J_ring, phi_grid)

        # Choose φ_i by inverting conditional CDF
        # Deterministic but well-spread choice if rng is None; otherwise random
        ph = inverse_cdf(C_phi, phi_grid, u_phi_seed[i])

        # Emit point
        pts[i] = r_point(ax, ay, az, eps1, eps2, th, ph)

    return pts

def sample_superellipsoid_equal_area_pointwise_v2(ax, ay, az, eps1, eps2,
                                                  N,
                                                  G_theta=128, G_phi=128,
                                                  eps_angle=1e-6,
                                                  gamma_theta=0.5, gamma_phi=0.5,
                                                  rng=None):
    """
    Equal-area, pointwise 2-stage sampler (first octant), with boundary-clustered
    θ/φ grids via arcsin(power) warp to handle ε<1 without huge grid sizes.

    Depends on your existing helpers:
      J_area_element, trapz, cumtrapz_pdf_to_cdf, inverse_cdf, r_point
    """

    # ---- helper: warped grid on (θ_min, θ_max) using θ = arcsin( u^γ ) ----
    def warped_theta_grid(G, theta_min, theta_max, gamma):
        # work in u = sin θ, which lives in [sin θ_min, sin θ_max]
        u_min = np.sin(theta_min)
        u_max = np.sin(theta_max)
        t = np.linspace(0.0, 1.0, G)
        u = u_min + (u_max - u_min) * (t ** gamma)   # power warp clusters toward u_min
        # arcsin itself clusters toward u=1 (θ≈π/2), so both edges get extra resolution
        return np.arcsin(u)

    def warped_phi_grid(G, phi_min, phi_max, gamma):
        v_min = np.sin(phi_min)
        v_max = np.sin(phi_max)
        t = np.linspace(0.0, 1.0, G)
        v = v_min + (v_max - v_min) * (t ** gamma)
        return np.arcsin(v)

    # ---- Build θ, φ grids (avoid exact endpoints) ----
    th_min = eps_angle
    th_max = 0.5 * np.pi - eps_angle
    ph_min = eps_angle
    ph_max = 0.5 * np.pi - eps_angle

    theta_grid = warped_theta_grid(G_theta, th_min, th_max, gamma_theta)
    phi_grid   = warped_phi_grid  (G_phi,   ph_min, ph_max, gamma_phi)

    # ---- Precompute J(θ_i, φ_j) on the warped tensor grid ----
    J_table = np.empty((G_theta, G_phi), dtype=float)
    for i, th in enumerate(theta_grid):
        for j, ph in enumerate(phi_grid):
            J_table[i, j] = J_area_element(ax, ay, az, eps1, eps2, th, ph)

    # ---- θ marginal M(θ) = ∫ J(θ, φ) dφ and CDF C_θ (trapezoid on warped φ-grid) ----
    M_theta = np.array([trapz(J_table[i, :], phi_grid) for i in range(G_theta)])
    C_theta = cumtrapz_pdf_to_cdf(M_theta, theta_grid)

    # ---- Draw N points: θ_i from C_θ^{-1}, then φ_i from conditional CDF ----
    pts = np.empty((N, 3), dtype=float)

    # θ targets: stratified (or randomized within bins)
    if rng is None:
        u_theta = (np.arange(N) + 0.5) / N
        # φ targets: decorrelated low-discrepancy (Kronecker sequence)
        alpha = (np.sqrt(5.0) - 1.0) / 2.0
        u_phi_seed = (0.5 + alpha * np.arange(N)) % 1.0
    else:
        u_theta = (np.arange(N) + rng.random(N)) / N
        u_phi_seed = rng.random(N)

    # Invert θ-CDF
    thetas = inverse_cdf(C_theta, theta_grid, u_theta)

    # For each θ_i, build conditional over φ on the warped φ-grid and sample one φ_i
    for i in range(N):
        th = thetas[i]
        # J along the ring at fixed θ_i
        J_ring = np.array([J_area_element(ax, ay, az, eps1, eps2, th, ph) for ph in phi_grid])
        C_phi  = cumtrapz_pdf_to_cdf(J_ring, phi_grid)
        ph     = inverse_cdf(C_phi, phi_grid, u_phi_seed[i])
        pts[i] = r_point(ax, ay, az, eps1, eps2, th, ph)

    return pts

def sample_superellipsoid_equal_area_pointwise_v3(ax, ay, az, eps1, eps2,
                                                  N,
                                                  G_theta=128, G_phi=128,
                                                  eps_angle=1e-6,
                                                  rng=None):
    """
    Equal-area pointwise sampler (first octant) with change-of-variables
    that cancels boundary singularities for eps<1.

    Integrals:
      θ-marginal over u = sin^eps1(θ):  pdf_u(u) = Mθ(θ(u)) * dθ/du
      φ-conditional over w = sin^eps2(φ): pdf_w(w) = J(θ, φ(w)) * dφ/dw
    """

    # ---------------------------
    # 0) Maps and Jacobians
    # ---------------------------
    # θ <-> u
    def theta_from_u(u):
        # u = sin^eps1 θ  =>  θ = arcsin(u^(1/eps1))
        return np.arcsin(np.clip(u, 0.0, 1.0) ** (1.0 / eps1))

    def dtheta_du(u):
        # dθ/du = (1/eps1) * u^(1/eps1 - 1) / sqrt(1 - u^(2/eps1))
        pow1 = (1.0 / eps1) - 1.0
        num  = (1.0 / eps1) * np.maximum(u, 1e-300) ** pow1
        den  = np.sqrt(np.maximum(1.0 - np.clip(u, 0.0, 1.0) ** (2.0 / eps1), 1e-300))
        return num / den

    # φ <-> w
    def phi_from_w(w):
        # w = sin^eps2 φ  =>  φ = arcsin(w^(1/eps2))
        return np.arcsin(np.clip(w, 0.0, 1.0) ** (1.0 / eps2))

    def dphi_dw(w):
        # dφ/dw = (1/eps2) * w^(1/eps2 - 1) / sqrt(1 - w^(2/eps2))
        pow1 = (1.0 / eps2) - 1.0
        num  = (1.0 / eps2) * np.maximum(w, 1e-300) ** pow1
        den  = np.sqrt(np.maximum(1.0 - np.clip(w, 0.0, 1.0) ** (2.0 / eps2), 1e-300))
        return num / den

    # ---------------------------
    # 1) Grids in transformed vars
    # ---------------------------
    # Avoid exact endpoints (singular arithmetic) via eps_angle
    th_min, th_max = eps_angle, 0.5 * np.pi - eps_angle
    ph_min, ph_max = eps_angle, 0.5 * np.pi - eps_angle

    # Corresponding u,w intervals
    u_min = np.sin(th_min) ** eps1
    u_max = np.sin(th_max) ** eps1
    w_min = np.sin(ph_min) ** eps2
    w_max = np.sin(ph_max) ** eps2

    # Uniform grids in u and w (the whole point of v3)
    u_grid = np.linspace(u_min, u_max, G_theta)
    w_grid = np.linspace(w_min, w_max, G_phi)

    # Physical angle grids (for evaluating J/Mθ on nodes)
    theta_grid = theta_from_u(u_grid)
    phi_grid   = phi_from_w(w_grid)

    # ---------------------------
    # 2) θ-marginal in transformed variable
    # ---------------------------
    # For each θ_i, integrate over φ via w with Jacobian dφ/dw
    M_theta = np.empty(G_theta, dtype=float)
    for i, th in enumerate(theta_grid):
        # J(θ_i, φ(w_j)) * dφ/dw evaluated on w-grid
        J_times_jac = np.array([
            J_area_element(ax, ay, az, eps1, eps2, th, phi_grid[j]) * dphi_dw(w_grid[j])
            for j in range(G_phi)
        ])
        # ∫ J dφ  ==  ∫ [J(θ, φ(w)) * dφ/dw] dw
        M_theta[i] = trapz(J_times_jac, w_grid)

    # Build CDF over u: pdf_u(u) = Mθ(θ(u)) * dθ/du
    pdf_u = M_theta * dtheta_du(u_grid)
    C_u   = cumtrapz_pdf_to_cdf(pdf_u, u_grid)

    # ---------------------------
    # 3) Draw N thetas (via u) and then φ conditionals (via w)
    # ---------------------------
    pts = np.empty((N, 3), dtype=float)

    if rng is None:
        # θ targets: stratified in [0,1]
        u_theta = (np.arange(N) + 0.5) / N
        # φ targets: decorrelated low-discrepancy (Kronecker sequence)
        alpha = (np.sqrt(5.0) - 1.0) / 2.0
        u_phi_seed = (0.5 + alpha * np.arange(N)) % 1.0
    else:
        u_theta = (np.arange(N) + rng.random(N)) / N
        u_phi_seed = rng.random(N)

    # Invert C_u on u_grid, map to θ
    u_samples = inverse_cdf(C_u, u_grid, u_theta)
    theta_samples = theta_from_u(u_samples)

    # For each θ_i, build conditional CDF over w and sample one w_i, then φ_i
    for i in range(N):
        th = theta_samples[i]

        # Conditional pdf over w: J(θ, φ(w)) * dφ/dw
        J_times_jac = np.array([
            J_area_element(ax, ay, az, eps1, eps2, th, phi_grid[j]) * dphi_dw(w_grid[j])
            for j in range(G_phi)
        ])
        C_w = cumtrapz_pdf_to_cdf(J_times_jac, w_grid)

        w_i  = inverse_cdf(C_w, w_grid, u_phi_seed[i])
        phi  = phi_from_w(w_i)

        pts[i] = r_point(ax, ay, az, eps1, eps2, th, phi)

    return pts


def sample_superellipsoid_equal_area_rejection_v1(
    ax, ay, az, eps1, eps2,
    N,
    theta_min=1e-6, theta_max=np.pi/2 - 1e-6,
    phi_min=1e-6,   phi_max=np.pi/2 - 1e-6,
    grid_u=64, grid_w=64,
    safety=1.10,
    use_local_envelope=False,
    rng=None
):
    """
    Equal-area Monte Carlo sampler (rejection in transformed variables)
    for the first-octant superellipsoid surface.

    Target density over (u,w):
        u = sin^eps1(theta),  w = sin^eps2(phi)
        \tilde p(u,w) ∝ J(theta(u), phi(w)) * dtheta/du(u) * dphi/dw(w)

    Steps:
      1) Build a coarse grid in (u,w), evaluate f(u,w) = J(...) * dθ/du * dφ/dw
      2) Take a global max (or per-cell local max) as the envelope bound
      3) Propose (u,w) uniformly and accept with prob f(u,w)/M
      4) Map to (θ,φ) and emit r_point(θ,φ)

    Parameters
    ----------
    ax, ay, az : float
        Semi-axes.
    eps1, eps2 : float
        Superellipsoid exponents (>0).
    N : int
        Number of points to generate (first octant only).
    theta_min, theta_max, phi_min, phi_max : float
        Angular limits for the octant (avoid exact endpoints).
    grid_u, grid_w : int
        Coarse grid resolution for precomputing the envelope.
    safety : float
        Safety factor > 1 on the envelope bound (e.g., 1.10).
    use_local_envelope : bool
        If False: use a single global bound M (simplest).
        If True : use a per-cell local bound M_ij (higher acceptance).
    rng : np.random.Generator or None
        Random generator; if None, uses a default.

    Returns
    -------
    pts : (N,3) array
        Equal-area samples on the first-octant surface.
    accept_rate : float
        Acceptance rate of the rejection sampler.
    """

    if rng is None:
        rng = np.random.default_rng()

    # ---------- transforms & Jacobians ----------
    def theta_from_u(u):
        # θ = arcsin(u^{1/eps1})
        return np.arcsin(np.clip(u, 0.0, 1.0) ** (1.0 / eps1))

    def dtheta_du(u):
        # dθ/du = (1/eps1) * u^{1/eps1 - 1} / sqrt(1 - u^{2/eps1})
        u = np.clip(u, 0.0, 1.0)
        num = (1.0 / eps1) * np.maximum(u, 1e-300) ** (1.0 / eps1 - 1.0)
        den = np.sqrt(np.maximum(1.0 - u ** (2.0 / eps1), 1e-300))
        return num / den

    def phi_from_w(w):
        # φ = arcsin(w^{1/eps2})
        return np.arcsin(np.clip(w, 0.0, 1.0) ** (1.0 / eps2))

    def dphi_dw(w):
        # dφ/dw = (1/eps2) * w^{1/eps2 - 1} / sqrt(1 - w^{2/eps2})
        w = np.clip(w, 0.0, 1.0)
        num = (1.0 / eps2) * np.maximum(w, 1e-300) ** (1.0 / eps2 - 1.0)
        den = np.sqrt(np.maximum(1.0 - w ** (2.0 / eps2), 1e-300))
        return num / den

    # ---------- transformed-domain bounds ----------
    u_min = np.sin(theta_min) ** eps1
    u_max = np.sin(theta_max) ** eps1
    w_min = np.sin(phi_min)   ** eps2
    w_max = np.sin(phi_max)   ** eps2

    # ---------- coarse grid & envelope ----------
    ug = np.linspace(u_min, u_max, grid_u)
    wg = np.linspace(w_min, w_max, grid_w)

    # Evaluate f(u,w) on coarse grid
    F = np.empty((grid_u, grid_w), dtype=float)
    for i, u in enumerate(ug):
        th = theta_from_u(u)
        jt = dtheta_du(u)
        for j, w in enumerate(wg):
            ph = phi_from_w(w)
            jp = dphi_dw(w)
            F[i, j] = J_area_element(ax, ay, az, eps1, eps2, th, ph) * jt * jp

    # Envelope(s)
    if not use_local_envelope:
        M = safety * np.max(F)
    else:
        # Per-cell local maxima (use 2x2 neighborhood max per cell for safety)
        # Build a padded array and take a local max filter (simple variant)
        M_local = np.zeros_like(F)
        for i in range(grid_u):
            i0 = max(i - 1, 0)
            i1 = min(i + 1, grid_u - 1)
            for j in range(grid_w):
                j0 = max(j - 1, 0)
                j1 = min(j + 1, grid_w - 1)
                M_local[i, j] = np.max(F[i0:i1+1, j0:j1+1])
        M_local *= safety

    # ---------- rejection sampling ----------
    pts = np.empty((N, 3), dtype=float)
    accepted = 0
    tried = 0

    while accepted < N:
        # Propose uniform in (u,w) rectangle
        u = rng.uniform(u_min, u_max)
        w = rng.uniform(w_min, w_max)

        # Map to angles and compute target density * Jacobian
        th = theta_from_u(u)
        ph = phi_from_w(w)
        q  = J_area_element(ax, ay, az, eps1, eps2, th, ph) * dtheta_du(u) * dphi_dw(w)

        # Envelope value at proposal
        if not use_local_envelope:
            bound = M
        else:
            # Find the nearest coarse cell to (u,w)
            i = min(max(int((u - u_min) / max(u_max - u_min, 1e-300) * (grid_u - 1)), 0), grid_u - 1)
            j = min(max(int((w - w_min) / max(w_max - w_min, 1e-300) * (grid_w - 1)), 0), grid_w - 1)
            bound = M_local[i, j]

        # Accept/reject
        r = rng.uniform(0.0, 1.0)
        if r * bound <= q:
            pts[accepted] = r_point(ax, ay, az, eps1, eps2, th, ph)
            accepted += 1

        tried += 1
        # (Optional) you can add a safety break if tried is enormous

    accept_rate = accepted / max(tried, 1)
    return pts, accept_rate


# ----------------------------
# Helpers (Pilu-style angle stepping)
# ----------------------------

def _delta_theta_central(theta, a, b, eps, D):
    """
    Central-region increment (Pilu Eq. (5)/(8) in the paper’s numbering):
        Δθ = (D/eps) * cosθ * sinθ / sqrt(a^2 cos^(2eps)θ sin^4θ + b^2 sin^(2eps)θ cos^4θ)
    """
    c = math.cos(theta)
    s = math.sin(theta)
    num = (D / eps) * c * s
    den = math.sqrt((a*a) * (c ** (2.0*eps)) * (s ** 4.0) +
                    (b*b) * (s ** (2.0*eps)) * (c ** 4.0))
    if den <= 0.0:
        return 0.0
    return num / den


def _delta_theta_near_zero(theta, b, eps, D, theta_eps):
    """
    Near θ≈0 increment (Pilu Eq. (9)-style asymptotic):
        Δθ ≈ ( (D/b) - θ_eps )^(1/eps) - θ
    """
    val = (D / b) - theta_eps
    if val <= 0.0:
        # fallback: small step
        return max(1e-4, 0.5 * theta_eps)
    return (val ** (1.0 / eps)) - theta


def _delta_theta_near_halfpi(theta, a, eps, D):
    """
    Near θ≈π/2 increment (Pilu Eq. (9)-style, mirrored):
        θn = ( (D/a) - (π/2 - θ)^eps )^(1/eps) - (π/2 - θ)
        Δθ = θn
    """
    half_pi = 0.5 * math.pi
    tau = (half_pi - theta)
    val = (D / a) - (tau ** eps)
    if val <= 0.0:
        return max(1e-4, 0.5 * tau)
    return (val ** (1.0 / eps)) - tau


def SampleSE(a, b, eps, D, theta_eps=1e-2):
    """
    Algorithm 3 (SuperEllipseSampler) – returns Θ ∈ [0, π/2]
    using the dual update rule (near 0, near π/2, central).
    """
    # march upward from 0 to π/2
    thetas_up = [0.0]
    theta = 0.0
    half_pi = 0.5 * math.pi
    while theta < half_pi - 1e-12:
        if theta <= theta_eps:
            dth = _delta_theta_near_zero(theta, b, eps, D, theta_eps)
        elif (half_pi - theta) <= theta_eps:
            dth = _delta_theta_near_halfpi(theta, a, eps, D)
        else:
            dth = _delta_theta_central(theta, a, b, eps, D)

        # guardrails
        if not np.isfinite(dth) or dth <= 0.0:
            dth = max(1e-4, 0.5 * theta_eps)

        theta_next = theta + dth
        if theta_next <= theta:
            theta_next = theta + 1e-4
        theta = min(theta_next, half_pi)
        thetas_up.append(theta)

        # very defensive bail-out
        if len(thetas_up) > 200000:
            break

    # ensure exact endpoint
    if thetas_up[-1] < half_pi:
        thetas_up.append(half_pi)

    # The original pseudocode also marches back down with negative steps.
    # For constructing the first-octant angle set on [0, π/2] we don’t need
    # to double back; return the monotone list (duplicates get handled later).
    return np.array(thetas_up, dtype=float)


# ----------------------------
# SuperEllipsoid (Algorithm 2)
# ----------------------------

def SuperEllipsoid(a, b, c, eps1, eps2, D, theta_eps=1e-2):
    """
    Algorithm 2 (SuperEllipsoidSampler) – Pilu-style.
    Returns an (N,3) array of points (full surface via mirroring).

    Parameters
    ----------
    a, b, c : floats
        Semi-axes.
    eps1, eps2 : floats
        Superellipsoid exponents (vertical eps1, horizontal eps2).
    D : float
        Target (approximate) arclength step in parameter lines (Pilu parameter).
    theta_eps : float
        Small threshold for "near 0" and "near π/2" regions.

    Notes
    -----
    - This follows the paper’s “spherical product” idea:
        H = SampleSE(1, c, eps1, D)    # η angles (vertical superellipse)
        Ω = SampleSE(a, b, eps2, D)    # ω angles (horizontal superellipse)
      Then combine via:
        x = ± a * cos^eps2(±Ω) * cos^eps1(±H)
        y = ± b * sin^eps2(±Ω) * cos^eps1(±H)
        z = ± c * sin^eps1(±H)
      covering all 8 symmetric octants.
    """
    # 1) Sample angles
    H = SampleSE(1.0, c, eps1, D, theta_eps=theta_eps)    # η (vertical)
    Omega = SampleSE(a, b, eps2, D, theta_eps=theta_eps)  # ω (horizontal)

    # 2) Precompute powers on the first octant (nonnegative cos/sin)
    cosH_e1 = (np.cos(H)) ** eps1          # (|cos H|)^eps1 (first octant -> positive)
    sinH_e1 = (np.sin(H)) ** eps1
    cosO_e2 = (np.cos(Omega)) ** eps2
    sinO_e2 = (np.sin(Omega)) ** eps2

    # 3) First-octant grid (|x|,|y|,|z|)
    X0 = a * np.outer(cosH_e1,            cosO_e2)        # shape (len(H), len(Omega))
    Y0 = b * np.outer(cosH_e1,            sinO_e2)
    Z0 = c * (sinH_e1[:, None] * np.ones((1, len(Omega))))

    # 4) Mirror to all 8 octants (sign flips across x=0, y=0, z=0)
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                X = sx * X0
                Y = sy * Y0
                Z = sz * Z0
                pts.append(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1))

    pts = np.vstack(pts)
    return pts, H, Omega

def superellipsoid_sampler(a, b, c, eps1, eps2, D):
    """
    Python translation of the 'SuperEllipsoid' algorithm (Alg. 2 in the paper).

    a, b, c   : semi-axes
    eps1      : vertical exponent (their ε1)
    eps2      : horizontal exponent (their ε2)
    D         : desired (approx) chord length on the generating superellipses

    Returns
    -------
    pts : (N, 3) numpy array
        Sampled points on the full, mirrored superellipsoid.
    """
    # 1) sample η like a superellipse with (1, c, eps1)  -- vertical
    H = sample_superellipse(1.0, c, eps1, D)   # η values in [0, π/2] plus mirrored ones inside
    # BUT: the paper mirrors inside SampleSuperEllipse; we only need the [0, π/2] part for the 1/8th
    # so let's take only the part from 0..π/2:
    H_quadrant = H[(H >= 0) & (H <= np.pi/2)]

    # 2) sample ω like a superellipse with (a, b, eps2)  -- horizontal
    Omega = sample_superellipse(a, b, eps2, D)
    Omega_quadrant = Omega[(Omega >= 0) & (Omega <= np.pi/2)]

    n_eta = len(H_quadrant)
    n_omega = len(Omega_quadrant)

    # we will first build the 1/8 surface (η, ω >= 0), then mirror
    X_list = []
    Y_list = []
    Z_list = []

    # nested over signs, exactly like lines 8-20 in the pseudocode
    # i, j, k are the sign flips for x, y, z in the final object
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                # we need cos/sin for all ω,η with signs applied like in the MATLAB
                cosOmega = np.cos(sy * Omega_quadrant)    # line 11 in pseudocode
                sinOmega = np.sin(sy * Omega_quadrant)    # line 12
                cosH     = np.cos(sz * H_quadrant)        # line 13
                sinH     = np.sin(sz * H_quadrant)        # line 14

                # raise to powers (first octant -> abs not needed, but with signs we better abs)
                # in MATLAB they rely on symmetry; we do abs(...)**eps
                cosOmega_eps2 = np.abs(cosOmega) ** eps2
                sinOmega_eps2 = np.abs(sinOmega) ** eps2
                cosH_eps1     = np.abs(cosH) ** eps1
                sinH_eps1     = np.abs(sinH) ** eps1

                # Now form the grid of size (n_eta, n_omega)
                # Xnext ← i ∗ a ∗ cos^eps2(Ω) ∗ cos^eps1(H)
                # they do outer products
                Xnext = sx * a * np.outer(cosH_eps1, cosOmega_eps2)    # (n_eta, n_omega)
                Ynext = sx * b * np.outer(cosH_eps1, sinOmega_eps2)    # (n_eta, n_omega)
                Znext = sz * c * np.outer(sinH_eps1, np.ones_like(Omega_quadrant))

                # flatten and concat
                X_list.append(Xnext.ravel())
                Y_list.append(Ynext.ravel())
                Z_list.append(Znext.ravel())

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    Z = np.concatenate(Z_list)

    pts = np.stack([X, Y, Z], axis=1)
    return pts


def sample_superellipse(a, b, eps, D):
    """
    Python translation of Algorithm 3 (SuperEllipseSampler) and its helper.
    We follow the paper's structure:

      Θ(1) = 0
      while Θ(N) < π/2:
          θ_next = UpdateTheta(...)
          append
      append π/2
      then run backward to 0 with negative updates
      then mirror x/y once more.

    But for our 3D use we actually only need 0..π/2 part, so the caller
    will cut that out.
    """
    # first quarter
    thetas = [0.0]
    while thetas[-1] < np.pi / 2:
        theta_next = update_theta(thetas[-1], a, b, eps, D)
        # safety in case D is too big
        if theta_next <= thetas[-1]:
            theta_next = thetas[-1] + 1e-6
        thetas.append(theta_next)
    # ensure exact π/2 (like line 18)
    thetas[-1] = np.pi / 2

    # now backward (lines 19-22): generate the negative side
    # this is for the full superellipse; we keep it to stay faithful
    # (even if we only need 0..π/2 in the 3D sampler)
    N = len(thetas)
    # start from last (π/2) and step down
    thetas_down = [thetas[-1]]
    while thetas_down[-1] > 0.0:
        theta_next = thetas_down[-1] - update_theta(thetas_down[-1], a, b, eps, D)
        if theta_next >= thetas_down[-1]:
            theta_next = thetas_down[-1] - 1e-6
        thetas_down.append(theta_next)
    thetas_down[-1] = 0.0

    thetas_full = np.array(thetas + thetas_down[1:])  # join, avoid double 0
    return thetas_full


def update_theta(theta, a, b, eps, D):
    """
    This is lines 24-33 from the pseudocode:

    θ_eps ← 0.01
    if θ ≤ θ_eps:
        Δθ(θ) ← ( D / b - θ_eps )^(1/ε) - θ
    else if π/2 − θ ≤ θ_eps:
        Δθ(θ) ← (( D / a ) − ( π/2 − θ )^ε )^(1/ε) − ( π/2 − θ )
    else:
        Δθ(θ) ← D / ( ε * cosθ * sinθ ) *
                 sqrt( sin^2θ*cos^2θ / ( a^2 cos^{2ε}θ sin^4θ + b^2 sin^{2ε}θ cos^4θ ) )

    I’ll translate this as literally as possible.
    """
    theta_eps = 0.01  # boundary width, exactly as in pseudocode

    # region near 0
    if theta <= theta_eps:
        # ( D / b - θ_eps )^(1/ε) - θ
        # guard if D/b < theta_eps (might happen if D is small)
        base = (D / b) - theta_eps
        if base <= 0:
            dtheta = theta_eps - theta  # just jump to theta_eps
        else:
            dtheta = base ** (1.0 / eps) - theta
        return dtheta

    # region near π/2
    if (np.pi / 2) - theta <= theta_eps:
        # θn ← ( D / a − ( π/2 − θ )^ε )^(1/ε) − ( π/2 − θ)
        rem = (np.pi / 2) - theta
        base = (D / a) - (rem ** eps)
        if base <= 0:
            dtheta = rem  # jump to the end
        else:
            dtheta = base ** (1.0 / eps) - rem
        return dtheta

    # central rule (their Eq. 8-style)
    ct = np.cos(theta)
    st = np.sin(theta)

    # numerator: D / (ε * cosθ * sinθ)
    num = D / (eps * ct * st)

    # the big denominator under the sqrt:
    # a^2 cos^{2ε}(θ) sin^4(θ) + b^2 sin^{2ε}(θ) cos^4(θ)
    term1 = (a ** 2) * (ct ** (2.0 * eps)) * (st ** 4)
    term2 = (b ** 2) * (st ** (2.0 * eps)) * (ct ** 4)
    den_inside = term1 + term2

    # the fraction under the sqrt in the pseudocode:
    # (sin^2 θ * cos^2 θ) / den_inside
    frac = (st ** 2) * (ct ** 2) / den_inside

    dtheta = num * np.sqrt(frac)
    return dtheta