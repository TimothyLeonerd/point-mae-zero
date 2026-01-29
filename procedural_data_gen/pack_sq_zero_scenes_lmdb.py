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

def collect_existing_keys_from_lmdb(dst_root: Path, split: str) -> list[str]:
    """
    Faster version: use LMDB stats to count entries per shard, then
    synthesize keys split:00000000..split:000NNNN.
    Prints per-shard progress.
    """
    split_dir = dst_root / split
    if not split_dir.is_dir():
        return []

    shard_dirs = sorted(split_dir.glob("shard_*"))
    if not shard_dirs:
        return []

    n_shards = len(shard_dirs)
    print(f"[{split}] scanning {n_shards} shard(s) via env.stat()...")

    total = 0
    for idx, shard_dir in enumerate(shard_dirs):
        print(f"[{split}]  shard {idx+1}/{n_shards}: {shard_dir.name} ...", flush=True)
        env = lmdb.open(
            str(shard_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin() as txn:
            st = txn.stat()
            entries = st.get("entries", 0)
            total += entries
        env.close()

    print(f"[{split}] found {total} entries across all shards.")
    keys = [f"{split}:{i:08d}" for i in range(total)]
    return keys

def pack_split(
    split: str,
    n_samples: int,
    cfg_scene: SceneConfig,
    dst_root: Path,
    shard_target_bytes: int,
    map_size_bytes: int,
    key_file,
    base_seed: int,
    start_index: int = 0,
):
    # deterministic per-split seed stream
    split_offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[split]
    rng = np.random.default_rng(base_seed + split_offset)

    # start writing into a new shard if resuming
    if start_index > 0:
        # count existing shards and start at the next index
        existing_shards = sorted((dst_root / split).glob("shard_*"))
        shard_idx = len(existing_shards)
    else:
        shard_idx = 0

    env = open_env(dst_root, split, shard_idx, map_size_bytes)
    shard_bytes = 0

    for i in range(start_index, n_samples):
        scene_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))

        coord, color, normal, meta = generate_one_scene_arrays(cfg_scene, scene_seed=scene_seed)

        # pack payload (NO compression)
        buf = io.BytesIO()
        np.savez(
            buf,
            coord=coord.astype(np.float32, copy=False),
            color=color.astype(np.uint8, copy=False),
            normal=normal.astype(np.float32, copy=False),
            meta=np.bytes_(json.dumps(meta).encode("utf-8")),
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

    # how many samples already exist for each split?
    existing_counts = {}
    key_files = {}

    for split in SPLITS:
        key_path = dst_root / f"{split}.txt"

        existing_keys: list[str] = []

        if key_path.is_file():
            # read existing keys from file
            with open(key_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            existing_keys = lines

        if not existing_keys:
            # Either key file doesn't exist or it's empty: try to rebuild from LMDB
            existing_keys = collect_existing_keys_from_lmdb(dst_root, split)
            if existing_keys:
                print(f"[{split}] rebuilding {key_path.name} with {len(existing_keys)} entries")
                with open(key_path, "w") as f:
                    f.write("\n".join(existing_keys) + "\n")

        n_existing = len(existing_keys)
        existing_counts[split] = n_existing

        # Open for appending new keys (or fresh if none existed)
        key_files[split] = open(key_path, "a")
        print(f"[{split}] existing samples: {n_existing}")


    # load scene config YAML -> SceneConfig
    y = _load_yaml(args.scene_config)
    cfg_scene = SceneConfig(out_root=Path("/tmp/unused"))  # out_root unused for in-memory generation
    cfg_scene = _merge_config_into_sceneconfig(cfg_scene, y)

    shard_target_bytes = int(args.shard_size_gb * (1024**3))
    map_size_bytes = int(args.map_size_gb * (1024**3))

    if map_size_bytes <= shard_target_bytes:
        raise ValueError("--map-size-gb must be > --shard-size-gb (LMDB cannot grow beyond map_size).")

    try:
        # TRAIN
        if existing_counts["train"] < args.n_train:
            pack_split(
                "train",
                args.n_train,
                cfg_scene,
                dst_root,
                shard_target_bytes,
                map_size_bytes,
                key_files["train"],
                args.seed,
                start_index=existing_counts["train"],
            )
        else:
            print("[train] already has required samples, skipping.")

        # VAL
        if args.n_val > 0 and existing_counts["val"] < args.n_val:
            pack_split(
                "val",
                args.n_val,
                cfg_scene,
                dst_root,
                shard_target_bytes,
                map_size_bytes,
                key_files["val"],
                args.seed,
                start_index=existing_counts["val"],
            )
        elif args.n_val > 0:
            print("[val] already has required samples, skipping.")

        # TEST
        if args.n_test > 0 and existing_counts["test"] < args.n_test:
            pack_split(
                "test",
                args.n_test,
                cfg_scene,
                dst_root,
                shard_target_bytes,
                map_size_bytes,
                key_files["test"],
                args.seed,
                start_index=existing_counts["test"],
            )
        elif args.n_test > 0:
            print("[test] already has required samples, skipping.")
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