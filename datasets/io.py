import os
import io as _io

import numpy as np
import h5py
import lmdb


def _lmdb_envs(lmdb_path):
    envs = []
    for name in sorted(os.listdir(lmdb_path)):
        p = os.path.join(lmdb_path, name)
        if os.path.isdir(p):
            try:
                envs.append(lmdb.open(p, readonly=True, lock=False, readahead=True, subdir=True))
            except Exception:
                pass
    return envs

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
        
    @classmethod
    def get_lmdb(cls, relative_path, lmdb_path):
        """Return np array from LMDB if present; else None."""
        if not relative_path:
            return None
        envs = _lmdb_envs(lmdb_path)
        if not envs:
            return None
        key = relative_path.encode("utf-8")
        for env in envs:
            with env.begin(write=False) as txn:
                raw = txn.get(key)
                if raw is not None:
                    return np.load(_io.BytesIO(raw))
        return None

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
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