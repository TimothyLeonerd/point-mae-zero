import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

def sample_SQ_naive(sq_pars, n_theta, n_phi):
    assert(len(sq_pars) == 5 or len(sq_pars) == 11)

    if(len(sq_pars) == 5):
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars
        euler = None # xyz
        t = None
    elif(len(sq_pars) == 11):
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars[0:5]
        euler = sq_pars[5:8]
        t = sq_pars[8:11]

    points = np.zeros((n_theta * n_phi, 3), dtype=float) # allocate array

    d_theta = np.pi / n_theta # stepsize for theta
    d_phi = 2.0 * np.pi / n_phi # stepsize for phi

    # starting theta. Small offset from -pi/2 avoids poles 
    theta = -np.pi/2.0 + np.pi/(2.0 * n_theta)

    idx = 0 # array index

    for i in range(0, n_theta):

        phi = -np.pi + np.pi/n_phi
        for j in range(0, n_phi):

            # intermediate variables
            ct = np.cos(theta)
            cp = np.cos(phi)

            st = np.sin(theta)
            sp = np.sin(phi)

            # Calc point coord
            x = a_x * np.sign(ct) * np.abs(ct)**eps_1 * np.sign(cp) * np.abs(cp)**eps_2
            y = a_y * np.sign(ct) * np.abs(ct)**eps_1 * np.sign(sp) * np.abs(sp)**eps_2
            z = a_z * np.sign(st) * np.abs(st)**eps_1

            
            points[idx] = np.array([x,y,z], dtype=float) # Save to array
            
            phi += d_phi # increment phi (East-West)
            idx += 1

        theta += d_theta # increment theta (North-South)

    # optional rotation
    if euler is not None:
        R = Rot.from_euler('xyz', euler).as_matrix()
        points = points @ R.T

    # optional translation
    if t is not None:
        points = points + np.asarray(t)

    return points

def sample_SQ_naive_exactN(sq_pars, n_points, rng=None):
    """
    Oversample on a k×k grid with k=ceil(sqrt(n_points)) using sample_SQ_naive,
    then uniformly subsample to exactly n_points. Inherits the same parameterization
    (so it keeps the naive pole bias, as intended for now).
    """
    import numpy as np

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    k = int(np.ceil(np.sqrt(n_points)))
    M = k * k
    dense = sample_SQ_naive(sq_pars, k, k)   # (M, 3)

    if M == n_points:
        return dense

    # rng can be: None (global), an int seed, or a np.random.Generator
    if rng is None:
        idx = np.random.choice(M, size=n_points, replace=False)
    else:
        gen = np.random.default_rng(rng) if isinstance(rng, (int, np.integer)) else rng
        idx = gen.choice(M, size=n_points, replace=False)

    return dense[idx]



def set_equal_axes_quadrant_aware(ax, points):
    P = np.asarray(points)[:, :3]
    mins, maxs = P.min(0), P.max(0)
    lo, hi = mins.copy(), maxs.copy()

    # Clamp to zero if data is one-sided on an axis
    for k in range(3):
        if mins[k] >= 0: lo[k] = 0       # all ≥ 0 -> start at 0
        if maxs[k] <= 0: hi[k] = 0       # all ≤ 0 -> end at 0

    spans = hi - lo
    R = spans.max()  # target common span

    # Grow each axis to length R with minimal empty space
    for k in range(3):
        if spans[k] == R: 
            continue
        if lo[k] == 0 and hi[k] > 0:        # positive-only axis
            hi[k] = R
        elif hi[k] == 0 and lo[k] < 0:      # negative-only axis
            lo[k] = -R
        else:                               # mixed-sign: expand both sides evenly
            d = (R - spans[k]) / 2.0
            lo[k] -= d; hi[k] += d

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1,1,1))  # equal visual aspect


def show_points(points, point_size=5):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    if points.shape[1] > 3:
        for i in np.unique(points[:, 3]):
            p = points[points[:, 3] == i]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=point_size, label=int(i))
        ax.legend()
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size)

    set_equal_axes_quadrant_aware(ax, points)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

# Note: Does not save labels
def save_pc(path, points):
    pts = np.asarray(points)
    xyz = pts[:, :3].astype(np.float64, copy=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)

def read_pc(path):
    if not os.path.isfile(path): raise FileNotFoundError(path)
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float32)

# Note: Saves labels
def save_npy(path, points, keep_labels=True):
    pts = np.asarray(points)
    xyz = pts[:, :3].astype(np.float32, copy=False)
    if keep_labels and pts.shape[1] >= 4:
        np.save(path, {'points': xyz, 'labels': pts[:, 3].astype(np.int64, copy=False)})
    else:
        np.save(path, xyz)

def read_npy(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, dict):
        pts = np.asarray(obj['points'], dtype=np.float32)
        lbl = obj.get('labels'); lbl = None if lbl is None else np.asarray(lbl, dtype=np.int64)
        return pts, lbl
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0 and obj.dtype == object:
            d = obj.item(); return np.asarray(d['points'], dtype=np.float32), np.asarray(d.get('labels'), dtype=np.int64) if d.get('labels') is not None else None
        if obj.ndim == 2 and obj.shape[1] == 3:
            return obj.astype(np.float32, copy=False), None
    raise ValueError("Unsupported .npy content (expected dict or (N,3) array).")



def point_is_inside_SQ(point, sq_pars):
    assert(len(sq_pars) == 5 or len(sq_pars) == 11)

    if(len(sq_pars) == 5):
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars
        euler = None # xyz
        t = None
    elif(len(sq_pars) == 11):
        a_x, a_y, a_z, eps_1, eps_2 = sq_pars[0:5]
        euler = sq_pars[5:8]
        R_inv = Rot.from_euler('xyz', euler).inv().as_matrix()
        t = sq_pars[8:11]

        # Inv. rot. and translation that was applied
        point = R_inv @ (point - t)

    # ToDo: This only works with np.abs (heuristically tested)
    # Without it, only one quadrant correctly detected. Why?
    x = np.abs(point[0])
    y = np.abs(point[1])
    z = np.abs(point[2])

    #a_x, a_y, a_z, eps_1, eps_2 = sq_pars

    # implicit Superellipsoid function
    f = ( (x/a_x)**(2.0/eps_2) + (y/a_y)**(2.0/eps_2) )**(eps_2/eps_1) + (z/a_z)**(2.0/eps_1)

    return f < 1.0 # if f < 1 -> point is inside of SQ
def remove_points_inside_SQ(points, sq_pars):
    indices = []

    n = points.shape[0]

    for i in range(n):
        if(point_is_inside_SQ(points[i], sq_pars)):
            indices.append(i)

    return np.delete(points, indices, axis=0)

    
def sample_two_SQ_naive(sq_pars_1st, sq_pars_2nd, n_theta, n_phi):
    points_1st = sample_SQ_naive(sq_pars_1st, n_theta, n_phi)
    points_2nd = sample_SQ_naive(sq_pars_2nd, n_theta, n_phi)

    points_1st = remove_points_inside_SQ(points_1st, sq_pars_2nd)
    points_2nd = remove_points_inside_SQ(points_2nd, sq_pars_1st)

    # add ids to distinguish sq
    points_1st = np.concatenate((points_1st, np.full((points_1st.shape[0], 1), 0)), axis=1)
    points_2nd = np.concatenate((points_2nd, np.full((points_2nd.shape[0], 1), 1)), axis=1)

    return np.concatenate((points_1st, points_2nd), axis=0)

def sample_N_SQs_naive(sq_pars_N, n_theta, n_phi):
    n_SQs = len(sq_pars_N) # number of SQs

    all_points_list = []

    for i in range(n_SQs):
        current_points = sample_SQ_naive(sq_pars_N[i], n_theta, n_phi)

        # Remove points within other SQs
        for j in range(n_SQs):
            if j is not i:
                current_points = remove_points_inside_SQ(current_points, sq_pars_N[j])


        # add ids to distinguish sq
        current_points = np.concatenate((current_points, np.full((current_points.shape[0], 1), i)), axis=1)
        all_points_list.append(current_points)

    return np.concatenate(all_points_list, axis=0)

def sample_N_SQs_naive_exactN(sq_pars_N, n_points, alpha=2.0, growth=1.3, max_rounds=6, rng=None):
    """
    Oversample each SQ on a k×k grid (same k for all), remove overlaps, then
    uniformly subsample globally to exactly n_points. If not enough survivors,
    increase k multiplicatively and retry (up to max_rounds).
    """

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    n_SQs = len(sq_pars_N)
    if n_SQs == 0:
        return np.empty((0, 4), dtype=float)

    k = int(np.ceil(np.sqrt(max(1.0, alpha * n_points / n_SQs))))
    gen = (np.random.default_rng(rng) if rng is not None and
           not isinstance(rng, np.random.Generator) else rng)

    for _ in range(max_rounds):
        all_pts = []
        for i in range(n_SQs):
            pts = sample_SQ_naive(sq_pars_N[i], k, k)  # (k*k, 3)
            for j in range(n_SQs):
                if j != i:  # (avoid 'is not' for ints)
                    pts = remove_points_inside_SQ(pts, sq_pars_N[j])
            if pts.size:
                ids = np.full((pts.shape[0], 1), i)
                all_pts.append(np.concatenate([pts, ids], axis=1))

        if not all_pts:
            k = int(np.ceil(k * growth))
            continue

        survivors = np.concatenate(all_pts, axis=0)
        M = survivors.shape[0]
        if M >= n_points:
            if gen is None:
                idx = np.random.choice(M, n_points, replace=False)
            else:
                idx = gen.choice(M, n_points, replace=False)
            return survivors[idx]

        # Not enough survivors: increase density and try again
        k = int(np.ceil(k * growth))

    raise ValueError(f"Could not reach {n_points} points after {max_rounds} rounds; last count={M}, k={k}")



def get_random_SQ_pars():

    # Set min and max vals (ToDo: Handle better)
    # a values
    a_default_min = 0.1
    a_x_min = a_default_min
    a_x_max = 1.0
    a_y_min = a_default_min
    a_y_max = 1.0
    a_z_min = a_default_min
    a_z_max = 3.0 # perhaps good due to sampling

    # Exponents
    eps_1_min = 0.0
    eps_1_max = 3.0
    eps_2_min = 0.0
    eps_2_max = 3.0

    # Euler-angles
    euler_x_min = 0.0
    euler_x_max = 2* np.pi
    euler_y_min = 0.0
    euler_y_max = 2* np.pi
    euler_z_min = 0.0
    euler_z_max = 2* np.pi

    # Translations
    t_default_min = -0.5
    t_default_max = -t_default_min
    t_x_min = t_default_min
    t_x_max = t_default_max
    t_y_min = t_default_min
    t_y_max = t_default_max
    t_z_min = t_default_min
    t_z_max = t_default_max

    # Sample values
    a_x = np.random.uniform(a_x_min, a_x_max)
    a_y = np.random.uniform(a_y_min, a_y_max)
    a_z = np.random.uniform(a_z_min, a_z_max)

    eps_1 = np.random.uniform(eps_1_min, eps_1_max)
    eps_2 = np.random.uniform(eps_2_min, eps_2_max)

    euler_x = np.random.uniform(euler_x_min, euler_x_max)
    euler_y = np.random.uniform(euler_y_min, euler_y_max)
    euler_z = np.random.uniform(euler_z_min, euler_z_max)

    t_x = np.random.uniform(t_x_min, t_x_max)
    t_y = np.random.uniform(t_y_min, t_y_max)
    t_z = np.random.uniform(t_z_min, t_z_max)

    sq_pars = [a_x, a_y, a_z, eps_1, eps_2, euler_x, euler_y, euler_z, t_x, t_y, t_z]

    return sq_pars


def get_random_SQ_pars_v_2(seed=None, centered=False):
    """
    Sample a realistic, numerically stable set of superquadric parameters.
    Returns a list:
        [a_x, a_y, a_z, eps_1, eps_2, euler_x, euler_y, euler_z, t_x, t_y, t_z]
    """

    if seed is not None:
        np.random.seed(seed)

    # --- 1. Scale parameters (avoid degeneracies) ---
    # Log-uniform for better coverage of small and large scales
    a_x = 10 ** np.random.uniform(-0.3, 0.3)  # ~[0.5, 2.0]
    a_y = 10 ** np.random.uniform(-0.3, 0.3)
    a_z = 10 ** np.random.uniform(-0.3, 0.6)  # allow taller shapes

    # Optionally normalize to roughly constant volume
    volume_norm = (a_x * a_y * a_z) ** (1/3)
    a_x, a_y, a_z = a_x / volume_norm, a_y / volume_norm, a_z / volume_norm

    # --- 2. Exponents (roundness) ---
    eps_1 = np.random.uniform(0.3, 3.0)
    eps_2 = np.random.uniform(0.3, 3.0)

    # --- 3. Orientation (uniform over SO(3)) ---
    rot = Rot.random()
    euler_x, euler_y, euler_z = rot.as_euler('xyz', degrees=False)

    # --- 4. Translation ---
    if centered:
        t_x, t_y, t_z = 0.0, 0.0, 0.0
    else:
        t_x = np.random.uniform(-1.0, 1.0)
        t_y = np.random.uniform(-1.0, 1.0)
        t_z = np.random.uniform(-1.0, 1.0)

    return [a_x, a_y, a_z, eps_1, eps_2, euler_x, euler_y, euler_z, t_x, t_y, t_z]
