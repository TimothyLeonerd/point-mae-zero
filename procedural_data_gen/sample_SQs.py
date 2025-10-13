import numpy as np

def sample_SQ_naive(sq_pars, n_theta, n_phi):
    assert(len(sq_pars)== 5)

    a_x, a_y, a_z, eps_1, eps_2 = sq_pars

    points = np.zeros((n_theta * n_phi, 3), dtype=float)

    d_theta = np.pi / n_theta
    d_phi = 2.0 * np.pi / n_phi

    theta = -np.pi/2.0 + np.pi/(2.0 * n_theta)

    idx = 0

    for i in range(0, n_theta):

        phi = -np.pi + np.pi/n_phi
        for j in range(0, n_phi):

            x = a_x * np.cos(theta)**eps_1 * np.cos(phi)**eps_2
            y = a_y * np.cos(theta)**eps_1 * np.sin(phi)**eps_2
            z = a_z * np.sin(theta)**eps_1

            points[idx] = np.array([x,y,z], dtype=float)
            
            phi += d_phi
            idx += 1
            
        theta += d_theta

    return points


    