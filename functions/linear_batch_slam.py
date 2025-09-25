import numpy as np
from scipy import linalg
#from navlie.types import (State, StateWithCovariance, Input, Measurement, ProcessModel, MeasurementModel)
#from navlie.batch.problem import Problem
#from navlie.lib import SingleIntegrator, RangePointToAnchor, VectorState

class LinearSLAM():
    def __init__(self):
        pass

    def linear_batch_slam(self, dcm, rel_posn_measurements, rel_posn_cov, velocity_measurements, Q_dvl, sigma_theta, x0, P0):
        """
        This function solve the linear batch SLAM problem, returning an estimate for robot and landmark positions.
        
        Parameters
        ----------
        dcm : list of 2,2 arrays
            Direction cosine matrices for robot orientations
        rel_posn_measurements : list of N,3 array for each landmark
            Relative position measurements for each landmark
            time, rel_posn_x, rel_posn_y
        rel_posn_cov : list of N,2,2 array for each landmark
            Covariances associated with relative position measurements for each landmark
        velocity_measurements : N,3 array
            Robot body frame velocity measurements
            time, v_x, v_y
        Q_dvl : 2,2 array
            Covariance matrix of the DVL velocity measurements
        sigma_theta : float
            Standard deviation of heading measurements
        x0 : 2, array
            Initial robot position
        P0 : 2,2 array
            Covariance matrix of the initial robot position
        """
        num_measurements = sum(len(i) for i in rel_posn_measurements)
        num_landmarks = len(rel_posn_measurements)

        u = velocity_measurements[:, 1:3]

        N = len(velocity_measurements) # number of estimated robot positions

        H = np.zeros(((2*(1 + (N-1) + num_measurements)), (2*(N + num_landmarks))))

        dt_dvl = velocity_measurements[1,0] - velocity_measurements[0,0]

        Q_d_list, B_d_list = self.get_Qd_Bd(dcm[0:N-1], Q_dvl, dt_dvl)
        z = []

        # adding prior x0
        z.append(x0.reshape(1,2)[0])
        H[0:2, 0:2] = np.eye(2,2)

        # adding velocity measurement residuals
        for i in range(N-1):
            z.append(B_d_list[i] @ u[i].reshape(1,2)[0])
            H[(2+2*i):(2+2*(i+1)), 2*i:2*(i+1)] = -1 * np.eye(2,2)
            H[(2+2*i):(2+2*(i+1)), 2*(i+1):2*(i+2)] = np.eye(2,2)

        Q_ = self.Q_(sigma_theta, (1/dt_dvl)*Q_dvl)
        L_ = self.L_((-dt_dvl)*dcm[0], u[0].reshape(1,2)[0])

        weight_blocks = []
        weight_blocks.append(self.inverse(P0))
        weight_blocks.append(self.inverse(L_ @ Q_ @ L_.T))
        for i in range(len(Q_d_list)-1):
            Q_ = self.Q_(sigma_theta, (1/dt_dvl)*Q_dvl)
            L_ = self.L_((-dt_dvl)*dcm[i+1], u[i+1].reshape(1,2)[0])
            weight_blocks.append(self.inverse(L_ @ Q_ @ L_.T))

        # adding measurement residuals for each landmark
        measurements_added = 0
        for l in range(num_landmarks):
            indices = np.where(np.isin(velocity_measurements[:,0], rel_posn_measurements[l][:,0]))[0]
            i2 = 0
            for i in indices:
                meas = (rel_posn_measurements[l][i2, 1:3]).flatten()
                z.append(meas)
                M_k = self.M_(meas)
                R_ = self.R_(sigma_theta, rel_posn_cov[l][i2])
                weight_blocks.append(self.inverse(M_k @ np.diag(np.diag(R_)) @ M_k.T))

                H[(2*N+ 2*measurements_added + 2*i2):(2*N + 2*measurements_added + 2*(i2+1)), 2*i:2*(i+1)] = -1 * dcm[i].T
                H[(2*N + 2*measurements_added + 2*i2):(2*N + 2*measurements_added + 2*(i2+1)), (2*l + 2*N):(2*(l+1) + 2*N)] = dcm[i].T

                i2 += 1
            
            measurements_added += len(indices)
        
        weight = linalg.block_diag(*weight_blocks)

        # solve the problem
        x_est = linalg.solve(H.T @ weight @ H, H.T @ weight @ np.array(z).flatten())

        # format robot estimates nicely
        x_est = np.array(x_est).reshape(-1, 2)

        robot_est = x_est[0:N]
        landmarks_est = x_est[N:N+num_landmarks]

        # get covariances
        P = self.covariance(H, weight)
        sigma3 = (3 * np.sqrt(np.diag(P))).reshape(-1,2)

        sigma3_robot = sigma3[0:N]
        sigma3_landmarks = sigma3[N:N+num_landmarks]

        return robot_est, sigma3_robot, landmarks_est, sigma3_landmarks
    
    def B_cont(self, Cab):
        return Cab
    
    def L_cont(self, Cab):
        return -Cab
    
    def L_(self, L_k, u_k):
        # new L_k with heading uncertainty characterization
        gamma = np.array([[0, -1], [1, 0]])
        return np.block([L_k @ gamma @ u_k.reshape(2,1), L_k])
    
    def M_(self, r_l_z):
        # new M_k with heading uncertainty characterization
        gamma = np.array([[0, -1], [1, 0]])
        return np.block([gamma @ r_l_z.reshape(2,1), np.eye(2,2)])
    
    def R_(self, sigma_angle, R_k):
        # new R_k with heading uncertainty characterization
        return linalg.block_diag([sigma_angle**2], R_k)
    
    def Q_(self, sigma_angle, Q_k):
        # new Q_k with heading uncertainty characterization
        return linalg.block_diag([sigma_angle**2], Q_k)

    def discretize(self, A, B, L, Q, T):
        """
        This method discretizes a state space system using Steven Dahdah's method.
        """
        m = np.shape(A)[0]
        n = np.shape(B)[1]

        # building large matrix
        xi = np.block(
            [
                [A, L @ Q @ L.T, np.zeros((m,m)), np.zeros((m, n))],
                [np.zeros((m, m)), -(A.T), np.zeros((m, m)), np.zeros((m, n))],
                [np.zeros((m, m)), np.zeros((m, m)), A, B],
                [
                    np.zeros((n, m)),
                    np.zeros((n, m)),
                    np.zeros((n, m)),
                    np.zeros((n, n)),
                ],
            ]
        )
        upsilon = linalg.expm(xi * T)

        # extract A_d, B_d, Q_d
        upsilon_11 = upsilon[0:m, 0:m]
        upsilon_12 = upsilon[0:m, m:2*m]
        upsilon_34 = upsilon[2*m:3*m, 3*m:3*m + n]
        A_d = upsilon_11
        B_d = upsilon_34
        Q_d = upsilon_12 @ (upsilon_11.T)

        # initialize parameters
        #self.A_d = A_d
        #self.B_d = B_d
        #self.Q_d = Q_d
        return A_d, B_d, Q_d
    
    def H_u(self, u, C_ab, landmarks, x1, robot_positions, Q, dt):
        H_u = np.eye(2, 2*(len(robot_positions)-1 + len(landmarks)))
        z = x1
        x_true = []
        Q_d = []
        for j in range(len(robot_positions)-1):
            x_true = np.concatenate((x_true, robot_positions[j+1]))
            # not including x0 because assuming known

        x_true = np.concatenate((x_true, landmarks.flatten()))

        # following block rows are B_k u_k = x_k - x_{k-1}
        for i in range(len(u)-2):
            H_u_i = np.zeros((2, 2*(len(robot_positions) - 1 + len(landmarks))))
            B = C_ab[i+1]
            L = -C_ab[i+1]
            Ad, Bd, Qd = self.discretize(np.zeros((2,2)), B, L, Q, dt)
            Q_d.append(Qd)
            H_u_i[0:2, 2*i:2*(i+1)] = -np.eye(2,2)
            H_u_i[0:2, 2*(i+1):2*(i+2)] = np.eye(2,2)

            H_u = np.row_stack((H_u, H_u_i))
            z = np.concatenate((z, Bd @ u[i+1]))

        return H_u, z, x_true, Q_d
    
    def reshape_H_c(self, H_c, H_u, landmarks, frequ_y, frequ_u):
        f = int(frequ_u/frequ_y) 
        rows = []
        for i in range(np.shape(H_c)[1]):
            rows.append(H_c[:,i])
        
        H_c_new = np.zeros((np.shape(H_c)[0], np.shape(H_u)[1]))
        for i in range(int((np.shape(H_u)[1] - len(landmarks))/f)):
            H_c_new[0: np.shape(H_c)[0], f*i:f*i+1] = rows[i].reshape(len(rows[0]), 1)

        for i in range(len(landmarks)):
            H_c_new[:, (np.shape(H_u)[1] - len(landmarks) + i)] = H_c[:, (np.shape(H_c)[1] - len(landmarks) + i)]

        return H_c_new
    
    def covariance(self, H, W):
        info_matrix = H.T @ W @ H
        return np.linalg.inv(info_matrix)
    
    def true_x(self, u, x0, C_ab, Q, dt):
        x_true = []
        x_true.append((x0.reshape(1,2))[0])
        x_k_1 = x0
        for i in range(len(u)):
            ad, bd, qd = self.discretize(np.zeros((2,2)), C_ab[i], -C_ab[i], Q, dt)
            x_k = ad @ x_k_1.reshape(2,1) + bd @ u[i].reshape(2,1)
            x_true.append((x_k.reshape(1,2))[0])
            x_k_1 = x_k

        return x_true
    
    def get_Qd_Bd(self, dcm, Q, T):
        Q_d = []
        B_d = []
        for i in range(len(dcm)):
            ad, bd, qd = self.discretize(np.zeros((2,2)), dcm[i], np.eye(2,2), Q, T) 
            Q_d.append(qd)
            B_d.append(bd)
        return Q_d, B_d
    
    def inverse(self, A):
        """
        compute matrix inverse with Cholesky
        """
        c, lower = linalg.cho_factor(A)
        inv = linalg.cho_solve((c, lower), np.eye(A.shape[0]))
        return inv