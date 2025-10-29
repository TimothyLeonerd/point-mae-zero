#!/usr/bin/env python3
# make_sq_dataset.py
"""
CLI to generate a superquadric LMDB/NPY dataset using sample_SQs.generate_pointzero_like_dataset.

Required:
  --out-dir /path/to/root
  --n-clouds 150000
  --dataset-name 150k

All other options mirror the function defaults and are optional.
"""

import argparse
import json
import numpy as np

from sample_SQs import generate_pointzero_like_dataset  # adjust import if placed elsewhere


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required
    p.add_argument("--out-dir", required=True, help="Root output directory")
    p.add_argument("--n-clouds", required=True, type=int, help="Total number of clouds to generate")
    p.add_argument("--dataset-name", required=True, help="Name used for the subset folder (e.g., '2k', '150k').")

    # Common sampling / dataset options
    p.add_argument("--n-SQ-max", type=int, default=9, help="Max number of superquadrics per cloud")
    p.add_argument("--n-points-per-cloud", type=int, default=8192, help="Points per cloud")
    p.add_argument(
        "--storage", choices=["lmdb", "npy"], default="lmdb",
        help="Storage backend"
    )
    p.add_argument(
        "--mode", choices=["enriched", "xyz_only"], default="enriched",
        help="Payload mode. 'enriched' writes sidecar keys (.labels, .sq_params) for LMDB."
    )
    p.add_argument(
        "--dtype-points", choices=["float32", "float64"], default="float32",
        help="Floating dtype for points"
    )

    # LMDB/Numpy writer parameters
    p.add_argument("--lmdb-shard-bytes", type=int, default=(8 << 30),
                   help="Max LMDB map size per shard in bytes (e.g., 8589934592 for 8 GiB)")
    p.add_argument("--lmdb-safety", type=float, default=0.90,
                   help="Safety factor (usable = shard_bytes * safety)")
    p.add_argument("--lmdb-txn-batch", type=int, default=100,
                   help="LMDB commit interval")
    p.add_argument("--npy-batch-size", type=int, default=200,
                   help="NPY mode: how many samples to buffer before writing")

    # Cloud sampling knobs
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--growth", type=float, default=1.3)
    p.add_argument("--max-rounds", type=int, default=6)

    # Split writing
    p.add_argument("--train-ratio", type=float, default=0.95,
                   help="Fraction of items written into train.txt (rest into test.txt)")
    p.add_argument("--shuffle-split", action="store_true", default=True,
                   help="Shuffle indices before splitting (default: True)")
    p.add_argument("--no-shuffle-split", dest="shuffle_split", action="store_false",
                   help="Disable shuffling before split")

    # RNG control
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for NumPy Generator; set to -1 to use random OS entropy")

    return p.parse_args()


def main():
    args = parse_args()

    rng = None if args.seed == -1 else np.random.default_rng(args.seed)
    dtype = np.float32 if args.dtype_points == "float32" else np.float64

    summary = generate_pointzero_like_dataset(
        out_root=args.out_dir,
        n_clouds=args.n_clouds,
        n_SQ_max=args.n_SQ_max,
        dataset_name=args.dataset_name,
        n_points_per_cloud=args.n_points_per_cloud,
        storage=args.storage,
        mode=args.mode,
        dtype_points=dtype,
        lmdb_shard_bytes=args.lmdb_shard_bytes,
        lmdb_safety=args.lmdb_safety,
        lmdb_txn_batch=args.lmdb_txn_batch,
        npy_batch_size=args.npy_batch_size,
        alpha=args.alpha,
        growth=args.growth,
        max_rounds=args.max_rounds,
        rng=rng,
        train_ratio=args.train_ratio,
        shuffle_split=args.shuffle_split,
    )

    # Pretty print a stable JSON summary
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
