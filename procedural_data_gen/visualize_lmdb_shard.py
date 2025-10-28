# visualize_lmdb_shard.py
import lmdb, io, numpy as np
from sample_SQs import show_points  # your function

def view_lmdb_shard(shard_dir, limit=None, point_size=3):
    env = lmdb.open(shard_dir, readonly=True, lock=False, readahead=False, meminit=False, subdir=True)
    shown = 0
    with env.begin(buffers=True) as txn:
        cur = txn.cursor()
        for k, v in cur:
            # load numpy payload from bytes
            obj = np.load(io.BytesIO(bytes(v)), allow_pickle=True)
            title = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)

            # enriched dict {'points', 'labels', 'sq_params'}  OR plain (N,3) array
            if isinstance(obj, dict):
                pts = np.asarray(obj["points"], dtype=np.float32)
                lbl = obj.get("labels", None)
                if lbl is not None:
                    arr = np.concatenate([pts, np.asarray(lbl, dtype=np.int64).reshape(-1, 1)], axis=1)
                else:
                    arr = pts
            else:
                # could be object-array wrapping a dict
                if isinstance(obj, np.ndarray) and obj.ndim == 0 and obj.dtype == object:
                    d = obj.item()
                    pts = np.asarray(d["points"], dtype=np.float32)
                    lbl = d.get("labels", None)
                    if lbl is not None:
                        arr = np.concatenate([pts, np.asarray(lbl, dtype=np.int64).reshape(-1, 1)], axis=1)
                    else:
                        arr = pts
                else:
                    # plain (N,3) array
                    arr = np.asarray(obj, dtype=np.float32)

            print(f"Showing {title}  shape={arr.shape}")
            show_points(arr, point_size=point_size)  # close the figure to advance

            shown += 1
            if limit is not None and shown >= limit:
                break
    env.close()

# --- run it ---
view_lmdb_shard(
    "/home/tylerich/work/Master_Arbeit/point-mae-zero/procedural_data_gen/lmdb_shards/train_scalar/shard_00000",
    limit=9,  # show all 9
    point_size=2
)
