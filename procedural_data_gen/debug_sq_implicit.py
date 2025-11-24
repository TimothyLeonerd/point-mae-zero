# debug_sq_implicit.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sample_SQs import *


from sq_implicit_torch import (
    superquadric_implicit_field,
    sq_bounding_box,
    sample_grid_in_box,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Example SQ params:
    # a_x, a_y, a_z, eps_1, eps_2, euler_x, euler_y, euler_z, t_x, t_y, t_z
    params = torch.tensor(
        [1.0, 1.0, 2.0,   # scales
         2.0, 2.0,         # eps_1, eps_2
         0.5, 0.0, 0.0,    # small rotation
         0.0, 0.0, 0.0],   # centered at origin
        dtype=dtype,
        device=device,
    )

    # 1) Build grid
    min_xyz, max_xyz = sq_bounding_box(params, margin=1.2)
    points, grid_shape = sample_grid_in_box(min_xyz, max_xyz, resolution=64,
                                            device=device, dtype=dtype)
    nx, ny, nz = grid_shape

    # 2) Evaluate field
    vals = superquadric_implicit_field(points, params, signed=True)  # (N,)
    grid = vals.view(nx, ny, nz).detach().cpu().numpy()

    # 3) Visualize a central Z-slice
    z_idx = nz // 2
    slice_ = grid[:, :, z_idx]


    plt.figure()
    plt.title(f"Implicit field slice at z index {z_idx}")
    plt.imshow(slice_, origin="lower")
    plt.colorbar(label="f(x) - 1")
    plt.xlabel("y index")
    plt.ylabel("x index")
    plt.tight_layout()
    plt.show()


    # 4) Approximate iso-surface via marching cubes (optional)
    try:
        from skimage import measure
    except ImportError:
        print("skimage not installed; skipping marching cubes demo.")
        return

    # Marching cubes on grid (iso-level 0 corresponds to surface)
    verts, faces, normals, values = measure.marching_cubes(grid, level=-0.3)

    # verts are in voxel coordinates (i,j,k); map to world coordinates
    # We map [0, nx-1] -> [min_x, max_x] etc.
    i = verts[:, 0] / (nx - 1)
    j = verts[:, 1] / (ny - 1)
    k = verts[:, 2] / (nz - 1)

    min_xyz_np = min_xyz.cpu().numpy()
    max_xyz_np = max_xyz.cpu().numpy()

    xw = min_xyz_np[0] + i * (max_xyz_np[0] - min_xyz_np[0])
    yw = min_xyz_np[1] + j * (max_xyz_np[1] - min_xyz_np[1])
    zw = min_xyz_np[2] + k * (max_xyz_np[2] - min_xyz_np[2])

    verts_world = np.stack([xw, yw, zw], axis=-1)

    # 5) Plot iso-surface as a point cloud (quick & dirty)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # analytic surface sample
    points_2 = sample_SQ_naive(params.tolist(), 32, 32)  # (N,3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # iso-surface vertices (blue)
    ax.scatter(verts_world[:, 0], verts_world[:, 1], verts_world[:, 2],
            s=1, alpha=0.6, label="iso-surface (grid)")

    # analytic sampler points (red)
    ax.scatter(points_2[:, 0], points_2[:, 1], points_2[:, 2],
            s=4, alpha=0.6, label="sample_SQ_naive")

    ax.set_title("Superquadric: iso-surface vs analytic sampling")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 6) Quick check: points classified as inside/outside
    inside_mask = grid < 0.0
    print(f"Fraction of voxels inside (f-1 < 0): {inside_mask.mean():.4f}")


if __name__ == "__main__":
    main()
