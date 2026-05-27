import numpy as np
from scipy.optimize import bisect

class RelativePositionGTRS():
    def __init__(self):
        pass

    def get_landmarks_srls_gtrs(self, dr_positions, dcm, d, range_measurements, time, velocity, sigma_range, sigma_theta, Q_dvl, max_iterations=100, tol=1e-1, use_pdop_weights=False, use_greedy_keypoints=False, num_keypoints=None, greedy_gamma=1e-2):
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

        Q_dvl : 2,2 array
            Covariance matrix of the DVL velocity measurements

        max_iterations : int, optional
            Maximum number of iterations used in the GTRS bisection solve

        tol : float, optional
            Tolerance used in the GTRS bisection solve

        use_pdop_weights : bool, optional
            If True, solve the GTRS once, compute PDOP values from the
            resulting relative positions, and solve again using PDOP-weighted
            equations

        use_greedy_keypoints : bool, optional
            If True, solve the GTRS once, compute PDOP values from the
            resulting relative positions, then select a subset of range
            measurements using greedy PDOP minimization

        num_keypoints : int, optional
            Number of range measurements to retain when
            use_greedy_keypoints=True

        greedy_gamma : float, optional
            Weight on temporal spread penalty in greedy keypoint selection

        Returns
        -------
        rel_pos : list of 2 arrays
            Landmark positions relative to the robot body frame

        landmark_est : 2,1 array
            Estimated landmark position in the navigation frame

        rel_pos_cov : list of 2,2 arrays
            Covariance matrices associated with the relative position estimates
        """
        A = np.zeros((len(dr_positions), 3))
        for i in range(len(dr_positions)):
            A[i,:] = np.array([-2*dr_positions[i,0], -2*dr_positions[i,1], 1])

        b = np.zeros((len(dr_positions), 1))
        for i in range(len(dr_positions)):
            b[i] = range_measurements[i]**2 - d**2 - np.linalg.norm(dr_positions[i])**2

        # first solve GTRS with no PDOP information
        landmark_est = self.solve_gtrs(A, b, max_iterations=max_iterations, tol=tol)

        # get PDOP values if necessary
        if use_pdop_weights or use_greedy_keypoints:
            rel_pos_initial = []
            for i in range(len(dr_positions)):
                rel_pos_initial.append(
                    (dcm[i].T @ (landmark_est - dr_positions[i].reshape(2,1))).flatten()
                )

            pdop = self.compute_pdop_from_rel_pos(rel_pos_initial)

        # optional second solve with subset of ranges
        if use_greedy_keypoints:
            if num_keypoints is None:
                raise ValueError("num_keypoints must be provided when use_greedy_keypoints=True")

            num_keypoints = min(num_keypoints, len(dr_positions))

            selected_indices = self.greedy_keypoints(
                pdop,
                0,
                len(pdop) - 1,
                num_keypoints,
                greedy_gamma
            )

            A_solve = A[selected_indices, :]
            b_solve = b[selected_indices, :]

            dr_positions_solve = dr_positions[selected_indices]
            dcm_solve = np.asarray(dcm)[selected_indices]
            range_measurements_solve = range_measurements[selected_indices]
            time_solve = time[selected_indices]
            velocity_solve = velocity[selected_indices]

            pdop_solve = pdop[selected_indices]

        else:
            selected_indices = np.arange(len(dr_positions))

            A_solve = A
            b_solve = b

            dr_positions_solve = dr_positions
            dcm_solve = dcm
            range_measurements_solve = range_measurements
            time_solve = time
            velocity_solve = velocity

            if use_pdop_weights:
                pdop_solve = pdop

        # optionally weight the equations by PDOP
        if use_pdop_weights:
            A_solve, b_solve = self.apply_pdop_weights(A_solve, b_solve, pdop_solve)

        # final GTRS solve
        landmark_est = self.solve_gtrs(
            A_solve,
            b_solve,
            max_iterations=max_iterations,
            tol=tol,
        )
        
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
    
    def compute_pdop_from_rel_pos(self, rel_pos):
        """
        Compute individual PDOP values for a list of relative landmark positions.

        Parameters
        ----------
        rel_pos : list of 2 arrays
            Landmark positions relative to the robot body frame

        Returns
        -------
        pdop : N array
            PDOP value associated with each relative position
        """
        pdop = np.zeros(len(rel_pos))

        for i, p in enumerate(rel_pos):
            p = np.asarray(p).reshape(2)
            r = np.linalg.norm(p)

            if r < 1e-9:
                pdop[i] = np.inf
                continue

            H = np.zeros((2, 3))
            H[:, 0] = p / r
            H[:, 1:3] = np.eye(2)

            try:
                P = np.linalg.inv(H.T @ H)
                pdop[i] = np.sqrt(np.abs(np.trace(P)))
            except np.linalg.LinAlgError:
                pdop[i] = np.inf

        return pdop
    
    def compute_combined_pdop(self, rel_pos):
        """
        Compute the combined PDOP for a set of relative landmark positions.

        Parameters
        ----------
        rel_pos : list of 2 arrays
            Landmark positions relative to the robot body frame

        Returns
        -------
        pdop : float
            Combined PDOP value for the set of relative positions
        """
        m = len(rel_pos)
        dim = 2

        H = np.zeros((m * dim, 1 + dim))

        for i, p in enumerate(rel_pos):
            p = np.asarray(p).reshape(dim)
            r = np.linalg.norm(p)

            if r < 1e-9:
                return np.inf

            H[i*dim:(i+1)*dim, 0] = p / r
            H[i*dim:(i+1)*dim, 1:] = np.eye(dim)

        try:
            P = np.linalg.inv(H.T @ H)
            return np.sqrt(np.abs(np.trace(P)))
        except np.linalg.LinAlgError:
            return np.inf
    
    def apply_pdop_weights(self, A, b, pdop, eps=1e-6):
        """
        Apply PDOP-based weights to the GTRS system.

        Parameters
        ----------
        A : N,3 array
            GTRS system matrix

        b : N,1 array
            GTRS right-hand side vector

        pdop : N array
            PDOP values associated with each GTRS equation

        eps : float
            Minimum normalized PDOP value used to avoid division by zero

        Returns
        -------
        A_weighted : N,3 array
            Weighted GTRS system matrix

        b_weighted : N,1 array
            Weighted GTRS right-hand side vector
        """
        pdop = np.asarray(pdop, dtype=float)

        finite = np.isfinite(pdop)
        if not np.any(finite):
            return A, b

        max_pdop = np.max(pdop[finite])
        pdop[~finite] = max_pdop

        # lower PDOP = higher confidence
        sigma = pdop / max_pdop
        sigma = np.maximum(sigma, eps)

        W_sqrt = np.diag(1.0 / np.sqrt(sigma))

        return W_sqrt @ A, W_sqrt @ b
    
    def solve_gtrs(self, A, b, max_iterations=100, tol=1e-1):
        """
        Solve the SR-LS GTRS problem.

        Parameters
        ----------
        A : N,3 array
            GTRS system matrix

        b : N,1 array
            GTRS right-hand side vector

        max_iterations : int
            Maximum number of bisection iterations

        tol : float
            Bisection tolerance

        Returns
        -------
        landmark_est : 2,1 array
            Estimated landmark position in the navigation frame
        """
        D = np.eye(3)
        D[2, 2] = 0

        f = np.array([[0], [0], [-0.5]])

        def g(lam):
            y = np.linalg.solve(A.T @ A + lam * D, A.T @ b - lam * f)
            return (y.T @ D @ y + 2 * f.T @ y).item()

        lam = bisect(g, -1e10, 1e10, xtol=tol, maxiter=max_iterations)

        y = np.linalg.solve(A.T @ A + lam * D, A.T @ b - lam * f)
        return y[0:2]
    
    def greedy_keypoints(self, PDOP_values, i, N, num_equations, gamma):
        """
        Select range-measurement keypoints using greedy combined-PDOP minimization.

        Parameters
        ----------
        rel_pos : list of 2 arrays
            Landmark positions relative to the robot body frame

        i : int
            Starting index of the candidate window

        N : int
            Ending index of the candidate window

        num_equations : int
            Number of keypoints to select

        gamma : float
            Weight on temporal spread penalty

        Returns
        -------
        selected : list of int
            Indices of selected keypoints
        """
        candidates = list(range(i, N + 1))
        selected = []

        delta_star = N / (num_equations - 1) if num_equations > 1 else 0
        num_to_select = min(num_equations, len(candidates))

        for step in range(num_to_select):
            best_idx = None
            best_cost = float("inf")

            for idx in candidates:

                trial = sorted(selected + [idx])

                if len(trial) > 1:
                    spread = sum(
                        (trial[k+1] - trial[k] - delta_star)**2
                        for k in range(len(trial)-1)
                    )
                else:
                    spread = 0

                pdop_cost = sum(PDOP_values[t]**2 for t in trial)
                total_cost = pdop_cost + gamma * spread

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_idx = idx

            selected.append(best_idx)
            candidates.remove(best_idx)

        return sorted(selected)