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

def save_pc(file, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(file, pcd, write_ascii=True)

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

    print(len(indices))

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