import os
import io as _io

import numpy as np
import h5py

# Lazy LMDB setup (only used if ZERO_LMDB_DIR is set)
_LMDB_ENVS = None

def _lmdb_envs():
    """Open all LMDB dirs under ZERO_LMDB_DIR once, lazily."""
    global _LMDB_ENVS
    if _LMDB_ENVS is not None:
        return _LMDB_ENVS
    lmdb_dir = os.environ.get("ZERO_LMDB_DIR")
    if not lmdb_dir or not os.path.isdir(lmdb_dir):
        _LMDB_ENVS = []
        return _LMDB_ENVS
    try:
        import lmdb  # only import if needed
    except Exception:
        _LMDB_ENVS = []
        return _LMDB_ENVS
    envs = []
    for name in sorted(os.listdir(lmdb_dir)):
        p = os.path.join(lmdb_dir, name)
        if os.path.isdir(p):
            try:
                envs.append(lmdb.open(p, readonly=True, lock=False, readahead=True, subdir=True))
            except Exception:
                pass
    _LMDB_ENVS = envs
    return _LMDB_ENVS

def _rel_from_abs(file_path: str):
    """Compute dataset-relative key (e.g., 'uuid/object_aug.npy')."""
    pc_root = os.environ.get("ZERO_PC_ROOT")
    if pc_root and file_path.startswith(pc_root.rstrip("/") + "/"):
        return os.path.relpath(file_path, pc_root)
    # fallback: split by known segment
    marker = "/data/results/"
    if marker in file_path:
        return file_path.split(marker, 1)[1]
    return None  # give up → disk fallback

def _try_lmdb_get(file_path: str):
    """Return np array from LMDB if present; else None."""
    rel = _rel_from_abs(file_path)
    if not rel:
        return None
    envs = _lmdb_envs()
    if not envs:
        return None
    key = rel.encode("utf-8")
    for env in envs:
        with env.begin(write=False) as txn:
            raw = txn.get(key)
            if raw is not None:
                return np.load(_io.BytesIO(raw))
    return None

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        # elif file_extension in ['.pcd']:
        #     return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        # Try LMDB first (if enabled), then disk
        arr = _try_lmdb_get(file_path)
        if arr is not None:
            return arr
        return np.load(file_path)

       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]