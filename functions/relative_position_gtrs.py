import numpy as np
from scipy.optimize import bisect

class RelativePositionGTRS():
    def __init__(self):
        pass

    def get_landmarks_srls_gtrs(self, dr_positions, dcm, d, range_measurements, time, velocity, sigma_range, sigma_theta, Q_dvl, max_iterations=100, tol=1e-1):
        """
        This function gets an estimate for the positions of landmarks relative to the robot using the 
        SR-LS problem posed as a GTRS.  A covariance associated with these relative position estimates
        is also computed.
        
        Parameters
        ----------
        dr_positions : N,2 array
            Dead-reckoned robot positions
        dcm : list of 2,2 arrays
            Direction cosine matrices for robot orientations
        d : float
            Nominal z distance from robot to landmark
        range_measurements : N array
            Range measurements from robot to landmark
        time : N array
            Time vector
        velocity : N,2 array
            Robot body frame velocity measurements
        sigma_range : float
            Standard deviation of range measurements
        sigma_theta : float
            Standard deviation of heading measurements
        Q_dvl : 2 array
            Covariance matrix of the DVL velocity measurements
        """
        A = np.zeros((len(dr_positions), 3))
        for i in range(len(dr_positions)):
            A[i,:] = np.array([-2*dr_positions[i,0], -2*dr_positions[i,1], 1])

        b = np.zeros((len(dr_positions), 1))
        for i in range(len(dr_positions)):
            b[i] = range_measurements[i]**2 - d**2 - np.linalg.norm(dr_positions[i])**2

        D = np.eye(3)
        D[2,2] = 0

        f = np.array([[0],[0],[-0.5]])

        # solve for lambda with bisection method
        def g(lam):
            y = np.linalg.inv(A.T @ A + lam * D) @ (A.T @ b - lam * f)
            return y.T @ D @ y + 2 * f.T @ y

        lam = bisect(g, -1e10, 1e10, xtol=tol, maxiter=max_iterations)

        y = np.linalg.inv(A.T @ A + lam * D) @ (A.T @ b - lam * f)
        landmark_est = y[0:2]
        
        # get covariance of landmark estimate
        H = np.zeros((len(dr_positions), 2))
        for i in range(len(dr_positions)):
            H[i,:] = 2*(landmark_est.reshape(2,) - dr_positions[i][0:2].reshape(2,))

        # dead reckoning covariance
        w_dr_i = np.zeros((2,1))
        w_dr = [w_dr_i]
        gamma = np.array([[0, -1], [1, 0]])
        for i in range(1, len(dr_positions)):
            dt = time[i] - time[i-1]
            w_dr_i += -dt * sigma_theta * (dcm[i] @ gamma @ velocity[i-1].reshape(2,1)) + dt * dcm[i] @ np.diag(Q_dvl).reshape(2,1)
            w_dr.append(w_dr_i)

        W = np.zeros((len(dr_positions), len(dr_positions)))
        for i in range(len(dr_positions)):
            M_i = np.array([0, 0, -2*range_measurements[i]])
            M_i[0:2] = 2*(landmark_est.reshape(2,) - dr_positions[i][0:2].reshape(2,))
            W[i,i] = M_i @ np.diag([(w_dr[i])[0,0]**2, (w_dr[i])[1,0]**2, sigma_range**2]) @ M_i.T

        cov_l = np.linalg.inv(H.T @ np.linalg.inv(W) @ H)

        # get relative position estimates and covariances
        rel_pos = []
        rel_pos_cov = []
        for i in range(len(dr_positions)):
            rel_pos.append((dcm[i].T @ (landmark_est - dr_positions[i].reshape(2,1))).flatten())
            M_i = np.zeros((2,5))
            M_i[0:2, 0:2], M_i[0:2, 2:4], M_i[0:2, 4:5] = dcm[i].T, -dcm[i].T, -dcm[i].T @ gamma @ (landmark_est - dr_positions[i].reshape(2,1))
            if i == 0:
                R = np.diag([cov_l[0,0], cov_l[1,1], (1e-6)**2, (1e-6)**2, (sigma_theta**2)])
            else:
                R = np.diag([cov_l[0,0], cov_l[1,1], ((w_dr[i-1])[0,0])**2, ((w_dr[i-1])[1,0])**2, (sigma_theta**2)])
            rel_pos_cov.append(M_i @ R @ M_i.T)

        return rel_pos, landmark_est, rel_pos_cov