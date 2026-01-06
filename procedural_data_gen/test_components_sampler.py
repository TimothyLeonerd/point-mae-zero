#!/usr/bin/env python3
"""
Quick test driver for the new multi-component sampler.

Generates a few synthetic "objects", each being a mixture of
superquadrics and analytic primitives (cube/sphere/cylinder/cone/torus),
then saves them as .npy with shape (N, 6) = [x,y,z,nx,ny,nz].

You can then visualize them with viz_primitives.py, e.g.:

    # from repo root:
    python viz_primitives.py --data-dir data/test_components --index 0 --color-by-normals

Usage example (from procedural_data_gen/):

    python test_components_sampler.py \
        --num-objects 5 \
        --points-per-object 4096 \
        --n-components-min 2 \
        --n-components-max 5 \
        --p-primitive 0.5 \
        --out-dir ../data/test_components \
        --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shapes import (
    PrimitiveType,
    ShapeComponent,
    sample_shape_component_sq,
    sample_shape_component_primitive,
    sample_N_components_exactN_with_normals,
)


def build_components_for_object(
    rng: np.random.Generator,
    *,
    n_components: int,
    p_primitive: float,
    primitive_type_probs: dict[PrimitiveType, float],
) -> list[ShapeComponent]:
    """
    Build a list of ShapeComponent for a single test object.

    Each component is either:
      - a superquadric (kind="sq"), or
      - a primitive (kind="primitive", type in primitive_type_probs)
    """
    comps: list[ShapeComponent] = []

    for cid in range(n_components):
        if rng.random() < p_primitive:
            comp = sample_shape_component_primitive(
                rng,
                type_probs=primitive_type_probs,
                component_id=cid,
            )
        else:
            comp = sample_shape_component_sq(
                rng,
                component_id=cid,
            )
        comps.append(comp)

    return comps


def main():
    parser = argparse.ArgumentParser(description="Test sampler for SQ + primitive ShapeComponents.")
    parser.add_argument(
        "--num-objects",
        type=int,
        default=3,
        help="Number of test objects to generate.",
    )
    parser.add_argument(
        "--points-per-object",
        type=int,
        default=4096,
        help="Number of surface points per object.",
    )
    parser.add_argument(
        "--n-components-min",
        type=int,
        default=2,
        help="Minimum number of components (SQ or primitive) per object.",
    )
    parser.add_argument(
        "--n-components-max",
        type=int,
        default=5,
        help="Maximum number of components (SQ or primitive) per object.",
    )
    parser.add_argument(
        "--p-primitive",
        type=float,
        default=0.5,
        help="Probability that a component is a primitive (else superquadric).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for object_*.npy files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Uniform over primitive types for now
    primitive_type_probs: dict[PrimitiveType, float] = {
        "cube": 1.0,
        "sphere": 1.0,
        "cylinder": 1.0,
        "cone": 1.0,
        "torus": 1.0,
    }

    print(f"[INFO] Generating {args.num_objects} objects into {out_dir}")
    print(f"[INFO] points_per_object = {args.points_per_object}")
    print(f"[INFO] components per object in [{args.n_components_min}, {args.n_components_max}]")
    print(f"[INFO] p_primitive = {args.p_primitive}")

    for obj_idx in range(args.num_objects):
        n_components = rng.integers(args.n_components_min, args.n_components_max + 1)
        comps = build_components_for_object(
            rng,
            n_components=n_components,
            p_primitive=args.p_primitive,
            primitive_type_probs=primitive_type_probs,
        )

        points4, normals, comp_ids = sample_N_components_exactN_with_normals(
            comps,
            n_points=args.points_per_object,
            rng=rng,
            alpha=2.0,
            growth=1.3,
            max_rounds=6,
        )

        xyz = points4[:, :3]
        data = np.concatenate([xyz, normals], axis=1).astype(np.float32)  # (N, 6)

        out_file = out_dir / f"object_{obj_idx:06d}.npy"
        np.save(out_file, data)

        print(
            f"[INFO] object {obj_idx:06d}: "
            f"n_components={n_components}, "
            f"n_points={xyz.shape[0]}, "
            f"saved to {out_file}"
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
