#!/usr/bin/env python3
import argparse
import io
import os
from pathlib import Path

import lmdb
import numpy as np


def iter_shard_dirs(split_dir: Path):
    # split_dir/shard_00000, shard_00001, ...
    return sorted([p for p in split_dir.glob("shard_*") if p.is_dir()])


def shard_disk_bytes(shard_dir: Path) -> int:
    # LMDB typically stores data in data.mdb (+ lock.mdb)
    total = 0
    for name in ("data.mdb", "lock.mdb"):
        p = shard_dir / name
        if p.exists():
            total += p.stat().st_size
    return total


def analyze_shard(shard_dir: Path, *, every: int = 1000):
    """
    Returns stats dict for one shard:
      n_samples, total_points, min_points, max_points
    """
    env = lmdb.open(
        str(shard_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
        max_readers=512,
    )

    n_samples = 0
    total_points = 0
    min_points = None
    max_points = None

    with env.begin(write=False) as txn:
        with txn.cursor() as cur:
            for k, v in cur:
                n_samples += 1

                # payload is a np.savez() bytes blob (zip). Not compressed in your pipeline.
                buf = io.BytesIO(v)
                with np.load(buf, allow_pickle=False) as data:
                    coord = data["coord"]
                    n = int(coord.shape[0])

                total_points += n
                min_points = n if min_points is None else min(min_points, n)
                max_points = n if max_points is None else max(max_points, n)

                if every > 0 and (n_samples % every == 0):
                    print(f"      {shard_dir.name}: scanned {n_samples} samples...")

    env.close()

    if min_points is None:
        min_points = 0
    if max_points is None:
        max_points = 0

    return {
        "n_samples": n_samples,
        "total_points": total_points,
        "min_points": min_points,
        "max_points": max_points,
    }


def analyze_split(root: Path, split: str, *, every: int = 1000):
    split_dir = root / split
    if not split_dir.exists():
        return None

    shard_dirs = iter_shard_dirs(split_dir)
    if not shard_dirs:
        return {
            "split": split,
            "n_samples": 0,
            "total_points": 0,
            "min_points": 0,
            "max_points": 0,
            "disk_bytes": 0,
            "n_shards": 0,
        }

    n_samples = 0
    total_points = 0
    min_points = None
    max_points = None
    disk_bytes = 0

    print(f"[{split}] {len(shard_dirs)} shard(s) found.")
    for sd in shard_dirs:
        disk_bytes += shard_disk_bytes(sd)
        st = analyze_shard(sd, every=every)
        n_samples += st["n_samples"]
        total_points += st["total_points"]
        min_points = st["min_points"] if min_points is None else min(min_points, st["min_points"])
        max_points = st["max_points"] if max_points is None else max(max_points, st["max_points"])

    if min_points is None:
        min_points = 0
    if max_points is None:
        max_points = 0

    return {
        "split": split,
        "n_samples": n_samples,
        "total_points": total_points,
        "min_points": min_points,
        "max_points": max_points,
        "disk_bytes": disk_bytes,
        "n_shards": len(shard_dirs),
    }


def fmt_gb(x_bytes: int) -> str:
    return f"{x_bytes / (1024**3):.2f}G"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root of SQ_Zero_Scenes_lmdb (contains train/val/test dirs).")
    ap.add_argument("--every", type=int, default=0, help="Progress print every N samples per shard (0 disables).")
    args = ap.parse_args()

    root = Path(args.root)

    print(f"Using dataset root: {root}\n")
    splits = ["train", "val", "test"]

    split_stats = []
    for s in splits:
        st = analyze_split(root, s, every=args.every)
        if st is not None:
            split_stats.append(st)

    print("\n=== Per-split statistics ===")
    grand_samples = 0
    grand_points = 0
    grand_min = None
    grand_max = None
    grand_disk = 0

    for st in split_stats:
        n = st["n_samples"]
        tp = st["total_points"]
        mn = st["min_points"]
        mx = st["max_points"]
        disk = st["disk_bytes"]

        avg = (tp / n) if n > 0 else 0.0

        print(f"{st['split']}:")
        print(f"  num scenes              : {n}")
        print(f"  avg points per scene    : {avg:.2f}")
        print(f"  min points in a scene   : {mn}")
        print(f"  max points in a scene   : {mx}")
        print(f"  total points in split   : {tp}")
        print(f"  num shards              : {st['n_shards']}")
        print(f"  disk usage (split)      : {fmt_gb(disk)}")

        grand_samples += n
        grand_points += tp
        grand_disk += disk
        grand_min = mn if grand_min is None else min(grand_min, mn)
        grand_max = mx if grand_max is None else max(grand_max, mx)

    if grand_min is None:
        grand_min = 0
    if grand_max is None:
        grand_max = 0

    grand_avg = (grand_points / grand_samples) if grand_samples > 0 else 0.0

    print("\n=== Dataset-level statistics ===")
    print(f"Total number of splits           : {len(split_stats)}")
    print(f"Total number of scenes (dataset) : {grand_samples}")
    print(f"Avg points per scene (dataset)   : {grand_avg:.2f}")
    print(f"Min points in a scene (dataset)  : {grand_min}")
    print(f"Max points in a scene (dataset)  : {grand_max}")
    print(f"Total number of points (dataset) : {grand_points}")
    print(f"Disk usage (dataset)             : {fmt_gb(grand_disk)}")


if __name__ == "__main__":
    main()
