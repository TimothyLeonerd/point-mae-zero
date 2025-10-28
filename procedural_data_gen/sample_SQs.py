import open3d as o3d
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
            print("here")
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

def _make_cloud_once(n_SQ_max: int, n_points: int, *, rng: Generator,
                     alpha=2.0, growth=1.3, max_rounds=6) -> Tuple[np.ndarray, List[List[float]]]:
    """One attempt: returns (points_with_ids Nx4 float64, sq_pars_list) or raises ValueError if cannot reach N."""
    rng = _require_gen(rng)
    # number of Sqs: Uniform{1..n_SQ_max}
    n_sqs = int(rng.integers(1, n_SQ_max + 1))
    gens = _child_generators(rng, n_sqs)
    sq_pars_list = [get_random_SQ_pars(g) for g in gens]
    pts = sample_N_SQs_naive_exactN(sq_pars_list, n_points, alpha=alpha, growth=growth, max_rounds=max_rounds, rng=rng)
    return pts, sq_pars_list


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


def generate_pointzero_like_dataset(
    out_root: str,
    n_clouds: int,
    n_SQ_max: int,
    *,
    split: str = "train",
    n_points_per_cloud: int = 8192,
    storage: str = "lmdb",                 # "lmdb" or "npy"
    mode: str = "enriched",                # "enriched" or "xyz_only"
    dtype_points: np.dtype = np.float32,
    lmdb_shard_bytes: int = (8 << 30),     # 8 GB
    lmdb_safety: float = 0.90,             # keep 10% headroom
    lmdb_txn_batch: int = 100,             # commit interval
    npy_batch_size: int = 200,             # RAM batch before writing
    alpha: float = 2.0, growth: float = 1.3, max_rounds: int = 6,
    rng: Generator = None
) -> Dict[str, object]:
    """
    Generate exactly `n_clouds` clouds with `n_points_per_cloud` points each.
    - storage="lmdb": write shards under {out_root}/lmdb_shards/{split}/shard_xxxxx/
    - storage="npy" : write files under {out_root}/npy/{split}/
    - mode="enriched": save points+labels+sq_params
    - mode="xyz_only": save only (N,3) arrays
    Returns a summary dict (counts, skips, paths).
    """
    rng = _require_gen(rng)

    out_root = pathlib.Path(out_root)
    if storage == "lmdb":
        base_dir = out_root / "lmdb_shards" / split
    elif storage == "npy":
        base_dir = out_root / "npy" / split
    else:
        raise ValueError("storage must be 'lmdb' or 'npy'")
    _ensure_dir(base_dir)

    # LMDB: compute items per shard
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
        "shards_written": 0
    }

    if storage == "lmdb":
        shard_idx = 0
        written_in_shard = 0
        shard_dir = base_dir / f"shard_{shard_idx:05d}"
        writer = _ShardWriter(shard_dir, map_size=lmdb_shard_bytes)
        summary["shards_written"] = 1

    batch_mem: List[Tuple[np.ndarray, List[List[float]]]] = []  # for NPY mode
    next_idx = 0

    while summary["saved_clouds"] < n_clouds:
        try:
            points4, params = _make_cloud_once(n_SQ_max, n_points_per_cloud, rng=rng,
                                               alpha=alpha, growth=growth, max_rounds=max_rounds)
        except ValueError:
            summary["skipped_attempts"] += 1
            continue

        if storage == "lmdb":
            # build key and value
            key = f"{split}:{next_idx:08d}"
            if mode == "xyz_only":
                val = _np_serialize_to_bytes(points4[:, :3].astype(dtype_points, copy=False))
                shp = (n_points_per_cloud, 3)
                dt_name = str(np.dtype(dtype_points))
            else:
                xyz = points4[:, :3].astype(dtype_points, copy=False)
                labels = points4[:, 3].astype(np.int64, copy=False)
                obj = {'points': xyz, 'labels': labels, 'sq_params': params}
                val = _np_serialize_to_bytes(obj)
                shp = (n_points_per_cloud, 3)
                dt_name = str(np.dtype(dtype_points))

            writer.put(key, val, pts_shape=shp, dtype_points=dt_name)
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