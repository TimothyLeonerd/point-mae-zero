import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def sample_SQ_naive(sq_pars, n_theta, n_phi):
    assert(len(sq_pars)== 5)

    a_x, a_y, a_z, eps_1, eps_2 = sq_pars

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

    return points

def show_points(points):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()

def save_pc(file, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(file, pcd, write_ascii=True)
    