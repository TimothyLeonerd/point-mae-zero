# sq_implicit_torch.py

import torch
from typing import Tuple


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert XYZ Euler angles (radians) to a 3x3 rotation matrix.

    This is intended to match SciPy's Rot.from_euler('xyz', euler).as_matrix()
    convention used in sample_SQs.py.

    euler: (3,) tensor [euler_x, euler_y, euler_z]
    returns: (3, 3) rotation matrix R
    """
    if euler.shape != (3,):
        raise ValueError(f"euler must be shape (3,), got {tuple(euler.shape)}")

    cx = torch.cos(euler[0])
    sx = torch.sin(euler[0])
    cy = torch.cos(euler[1])
    sy = torch.sin(euler[1])
    cz = torch.cos(euler[2])
    sz = torch.sin(euler[2])

    # Rotation about x
    R_x = torch.tensor([[1.0, 0.0, 0.0],
                        [0.0, cx, -sx],
                        [0.0, sx,  cx]], dtype=euler.dtype, device=euler.device)

    # Rotation about y
    R_y = torch.tensor([[ cy, 0.0, sy],
                        [0.0, 1.0, 0.0],
                        [-sy, 0.0, cy]], dtype=euler.dtype, device=euler.device)

    # Rotation about z
    R_z = torch.tensor([[cz, -sz, 0.0],
                        [sz,  cz, 0.0],
                        [0.0, 0.0, 1.0]], dtype=euler.dtype, device=euler.device)

    # Apply rotations in x, then y, then z (XYZ intrinsic)
    # This corresponds to R = Rz * Ry * Rx in matrix form.
    R = R_z @ (R_y @ R_x)
    return R


def superquadric_implicit_field(
    points: torch.Tensor,   # (N, 3)
    params: torch.Tensor,   # (11,)
    signed: bool = True,
) -> torch.Tensor:
    """
    Compute the implicit superquadric field f(x) or (f(x) - 1) for a single SQ.

    points: (N, 3) world-space coordinates
    params: (11,) tensor with:
        [a_x, a_y, a_z, eps_1, eps_2, euler_x, euler_y, euler_z, t_x, t_y, t_z]

    returns:
        vals: (N,) tensor
            if signed=True:  f(x) - 1  (≈ signed field: <0 inside, 0 on surface, >0 outside)
            else:            f(x)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"`points` must be (N, 3); got {tuple(points.shape)}")
    if params.shape != (11,):
        raise ValueError(f"`params` must be (11,); got {tuple(params.shape)}")

    device = points.device
    dtype = points.dtype

    params = params.to(device=device, dtype=dtype)

    # Unpack parameters
    a_x, a_y, a_z = params[0:3]
    eps_1, eps_2   = params[3], params[4]
    euler          = params[5:8]
    t              = params[8:11]   # translation in world coords

    # --- transform world points into SQ's canonical frame ---
    R = euler_xyz_to_matrix(euler)      # (3,3)
    # inverse transform for row-vectors: x' = (x - t) @ R
    P = (points - t) @ R                # (N, 3) in object frame

    X = torch.abs(P[:, 0])
    Y = torch.abs(P[:, 1])
    Z = torch.abs(P[:, 2])

    # Avoid silly corner cases (just in case):
    a_x = torch.clamp(a_x, min=1e-6)
    a_y = torch.clamp(a_y, min=1e-6)
    a_z = torch.clamp(a_z, min=1e-6)
    eps_1 = torch.clamp(eps_1, min=1e-6)
    eps_2 = torch.clamp(eps_2, min=1e-6)

    # --- implicit function (vectorized) ---
    # f = ((|x|/a_x)^(2/eps2) + (|y|/a_y)^(2/eps2))^(eps2/eps1) + (|z|/a_z)^(2/eps1)
    rx  = (X / a_x) ** (2.0 / eps_2)
    ry  = (Y / a_y) ** (2.0 / eps_2)
    rxy = (rx + ry) ** (eps_2 / eps_1)
    rz  = (Z / a_z) ** (2.0 / eps_1)
    f   = rxy + rz

    if signed:
        return f - 1.0
    else:
        return f

def sq_bounding_box(
    params: torch.Tensor,
    margin: float = 1.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an axis-aligned bounding box in world coordinates for a single SQ.

    params: (11,) tensor as above
    margin: multiplier on max(a_x, a_y, a_z) to give some slack

    returns:
        min_xyz: (3,) tensor
        max_xyz: (3,) tensor
    """
    if params.shape != (11,):
        raise ValueError(f"`params` must be (11,); got {tuple(params.shape)}")

    a_x, a_y, a_z = params[0:3]
    t = params[8:11]

    r = margin * torch.max(torch.stack([torch.abs(a_x), torch.abs(a_y), torch.abs(a_z)]))
    # Ensure r > 0
    r = torch.clamp(r, min=1e-6)

    min_xyz = t - r
    max_xyz = t + r
    return min_xyz, max_xyz


def sample_grid_in_box(
    min_xyz: torch.Tensor,
    max_xyz: torch.Tensor,
    resolution: int = 32,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Sample a regular 3D grid of points inside an axis-aligned box.

    min_xyz, max_xyz: (3,) tensors
    resolution: either int (same for x,y,z) or a tuple (nx, ny, nz)

    returns:
        points: (N, 3) tensor with N = nx * ny * nz
        grid_shape: (nx, ny, nz)
    """
    if min_xyz.shape != (3,) or max_xyz.shape != (3,):
        raise ValueError("min_xyz and max_xyz must be shape (3,)")

    if isinstance(resolution, int):
        nx = ny = nz = resolution
    else:
        nx, ny, nz = resolution

    if device is None:
        device = min_xyz.device

    xs = torch.linspace(min_xyz[0], max_xyz[0], steps=nx, device=device, dtype=dtype)
    ys = torch.linspace(min_xyz[1], max_xyz[1], steps=ny, device=device, dtype=dtype)
    zs = torch.linspace(min_xyz[2], max_xyz[2], steps=nz, device=device, dtype=dtype)

    # 'ij' indexing so x -> axis 0, y -> axis 1, z -> axis 2
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    grid_shape = (nx, ny, nz)

    points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (N, 3)
    return points, grid_shape

def global_bounding_box_for_all_SQs(
    sq_params: torch.Tensor,
    margin: float = 1.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a global axis-aligned bounding box covering all S SQs in world space.

    sq_params: (S, 11) tensor, each row:
               [a_x, a_y, a_z, eps_1, eps_2,
                euler_x, euler_y, euler_z,
                t_x, t_y, t_z]
    margin:    passed through to sq_bounding_box (inflates each SQ's box)

    Returns:
        min_xyz: (3,) tensor — global minimum over all per-SQ boxes
        max_xyz: (3,) tensor — global maximum over all per-SQ boxes
    """
    if sq_params.ndim != 2 or sq_params.shape[1] != 11:
        raise ValueError(f"`sq_params` must be (S, 11); got {tuple(sq_params.shape)}")

    S = sq_params.shape[0]
    if S == 0:
        raise ValueError("`sq_params` has zero superquadrics (S == 0).")

    mins = []
    maxs = []

    for s in range(S):
        min_s, max_s = sq_bounding_box(sq_params[s], margin=margin)
        mins.append(min_s.unsqueeze(0))  # (1, 3)
        maxs.append(max_s.unsqueeze(0))  # (1, 3)

    mins_stack = torch.cat(mins, dim=0)  # (S, 3)
    maxs_stack = torch.cat(maxs, dim=0)  # (S, 3)

    min_xyz = mins_stack.min(dim=0).values  # (3,)
    max_xyz = maxs_stack.max(dim=0).values  # (3,)

    return min_xyz, max_xyz

def multi_sq_implicit_union(
    points: torch.Tensor,      # (N, 3)
    sq_params: torch.Tensor,   # (S, 11)
    signed: bool = True,
) -> torch.Tensor:
    """
    Compute the implicit field for a union of S superquadrics at given points.

    points:   (N, 3) world-space coordinates
    sq_params:(S, 11) tensor, each row:
              [a_x, a_y, a_z, eps_1, eps_2,
               euler_x, euler_y, euler_z,
               t_x, t_y, t_z]

    Returns:
        vals: (N,) tensor
            If signed=True:  union_signed(x) = min_i (f_i(x) - 1)
                             -> <0 inside union, 0 on surface, >0 outside
            If signed=False: union_f(x) = union_signed(x) + 1
                             -> 1 on surface, <1 inside, >1 outside
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"`points` must be (N, 3); got {tuple(points.shape)}")
    if sq_params.ndim != 2 or sq_params.shape[1] != 11:
        raise ValueError(f"`sq_params` must be (S, 11); got {tuple(sq_params.shape)}")

    S = sq_params.shape[0]
    if S == 0:
        raise ValueError("`sq_params` has zero superquadrics (S == 0).")

    device = points.device
    dtype = points.dtype
    sq_params = sq_params.to(device=device, dtype=dtype)

    # Compute signed field for each SQ: (S, N)
    vals_list = []
    for s in range(S):
        vals_s = superquadric_implicit_field(
            points, sq_params[s], signed=True
        )  # (N,)
        vals_list.append(vals_s.unsqueeze(0))  # (1, N)

    vals_stack = torch.cat(vals_list, dim=0)  # (S, N)

    # Union of implicit surfaces ≈ min over SQs
    union_signed = vals_stack.min(dim=0).values  # (N,)

    if signed:
        return union_signed
    else:
        return union_signed + 1.0
