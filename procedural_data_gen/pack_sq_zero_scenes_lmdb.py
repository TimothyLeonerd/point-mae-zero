#!/usr/bin/env python3
import os
import io
import json
import argparse
from pathlib import Path

import numpy as np
import lmdb

from generate_sq_scene import (
    SceneConfig,
    _load_yaml,
    _merge_config_into_sceneconfig,
    generate_one_scene_arrays,
)

SPLITS = ("train", "val", "test")


def open_env(dst_root: Path, split: str, shard_idx: int, map_size: int) -> lmdb.Environment:
    shard_dir = dst_root / split / f"shard_{shard_idx:05d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{split}] opening shard {shard_idx:05d} at {shard_dir}")
    return lmdb.open(
        str(shard_dir),
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
    )


def pack_split(
    split: str,
    n_samples: int,
    cfg_scene: SceneConfig,
    dst_root: Path,
    shard_target_bytes: int,
    map_size_bytes: int,
    key_file,
    base_seed: int,
):
    # deterministic per-split seed stream
    split_offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[split]
    rng = np.random.default_rng(base_seed + split_offset)

    shard_idx = 0
    env = open_env(dst_root, split, shard_idx, map_size_bytes)
    shard_bytes = 0

    for i in range(n_samples):
        scene_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))

        coord, color, normal, meta = generate_one_scene_arrays(cfg_scene, scene_seed=scene_seed)

        # pack payload (NO compression)
        buf = io.BytesIO()
        np.savez(
            buf,
            coord=coord.astype(np.float32, copy=False),
            color=color.astype(np.uint8, copy=False),
            normal=normal.astype(np.float32, copy=False),
            meta=np.string_(json.dumps(meta)),
        )
        payload = buf.getvalue()

        # rotate shard by target size
        if shard_bytes + len(payload) > shard_target_bytes and shard_bytes > 0:
            env.close()
            shard_idx += 1
            env = open_env(dst_root, split, shard_idx, map_size_bytes)
            shard_bytes = 0

        key_str = f"{split}:{i:08d}"
        with env.begin(write=True) as txn:
            txn.put(key_str.encode("utf-8"), payload)

        key_file.write(key_str + "\n")
        shard_bytes += len(payload)

        if i % 100 == 0:
            print(f"  [{split}] packed {i}/{n_samples}  (shard={shard_idx:05d}, shard_bytes~{shard_bytes/1e9:.2f}GB)")

    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-config", required=True, help="YAML config used by generate_sq_scene.py")
    p.add_argument("--dst-root", required=True, help="Output root for LMDB dataset")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--shard-size-gb", type=float, default=12.0, help="Target shard payload size (GB)")
    p.add_argument("--map-size-gb", type=float, default=15.0, help="LMDB map_size per shard (GB), must exceed shard size")
    p.add_argument("--seed", type=int, default=123, help="Base seed for dataset generation")
    args = p.parse_args()

    dst_root = Path(args.dst_root)
    for s in SPLITS:
        (dst_root / s).mkdir(parents=True, exist_ok=True)

    # load scene config YAML -> SceneConfig
    y = _load_yaml(args.scene_config)
    cfg_scene = SceneConfig(out_root=Path("/tmp/unused"))  # out_root unused for in-memory generation
    cfg_scene = _merge_config_into_sceneconfig(cfg_scene, y)

    shard_target_bytes = int(args.shard_size_gb * (1024**3))
    map_size_bytes = int(args.map_size_gb * (1024**3))

    if map_size_bytes <= shard_target_bytes:
        raise ValueError("--map-size-gb must be > --shard-size-gb (LMDB cannot grow beyond map_size).")

    # write key lists
    key_files = {
        "train": open(dst_root / "train.txt", "w"),
        "val": open(dst_root / "val.txt", "w"),
        "test": open(dst_root / "test.txt", "w"),
    }

    try:
        pack_split("train", args.n_train, cfg_scene, dst_root, shard_target_bytes, map_size_bytes, key_files["train"], args.seed)
        pack_split("val",   args.n_val,   cfg_scene, dst_root, shard_target_bytes, map_size_bytes, key_files["val"],   args.seed)
        pack_split("test",  args.n_test,  cfg_scene, dst_root, shard_target_bytes, map_size_bytes, key_files["test"],  args.seed)
    finally:
        for f in key_files.values():
            f.close()

    # optional global meta
    (dst_root / "dataset_meta.json").write_text(json.dumps({
        "scene_config_path": str(Path(args.scene_config).resolve()),
        "counts": {"train": args.n_train, "val": args.n_val, "test": args.n_test},
        "shard_size_gb": args.shard_size_gb,
        "map_size_gb": args.map_size_gb,
        "seed": args.seed,
        "format": "np.savez (coord,color,normal,meta)",
    }, indent=2))

    print("Done.")


if __name__ == "__main__":
    main()
