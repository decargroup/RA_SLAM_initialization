import matplotlib.pyplot as plt
import numpy as np

#plotting parameters
plt.rc("figure", figsize=(11.5, 8.5))
plt.rc("font", family="Times New Roman", size=20)
plt.rc("axes", grid=True, labelsize=20)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("grid", linestyle="--")
plt.rcParams["lines.markersize"] = 2

class PlottingTools():
    def __init__(self):
        pass

    def plot_trajectory(self, dr_positions, landmark_est, robot_est, landmarks_true, robot_true, landmark_ids):
        """
        This function plots the true vs. estimated robot trajectory and true vs. estimated landmark positions.

        Parameters
        ----------
        dr_positions : N,2 array
            Dead-reckoned robot positions
        landmark_est : N_l,2 array
            Estimated landmark positions
        robot_est : N,2 array
            Estimated robot positions
        landmarks_true : N_l,2 array
            True landmark positions
        robot_true : N,2 array
            True robot positions
        landmark_ids : list of int
            List of landmark IDs
        """

        fig, ax = plt.subplots(1,1)
        ax.set_xlabel(r"$x(t)$ (m)")
        ax.set_ylabel(r"$y(t)$ (m)")
        ax.plot(robot_true[:,0], robot_true[:,1], label = 'Groundtruth')
        ax.plot(robot_est[:,0], robot_est[:,1], label = 'Proposed')
        #ax.plot(dr_positions[:,0], dr_positions[:,1], label = 'Dead-reckoned')

        ax.scatter(landmarks_true[:,0], landmarks_true[:,1], marker="o", s=25)
        ax.scatter(landmark_est[:,0], landmark_est[:,1], marker="o", s=25)


        # add landmark ID labels to the plot
        for (x, y), lm_id in zip(landmarks_true, landmark_ids):
            ax.text(x+2, y-2, str(lm_id),
                    fontsize=20,
                    ha="left", va="top")
            
        ax.legend()
        plt.show()

    def plot_landmark_error(self, landmark_est, landmarks_true, sigma3, landmark_ids):
        """
        This function plots the error in the landmarks' estimated positions.

        Parameters
        ----------
        landmark_est : N_l,2 array
            Estimated landmark positions
        landmarks_true : N_l,2 array
            True landmark positions
        cov : list of 2,2 arrays
            Covariance matrices of the robot position estimates
        landmark_ids : list
            List of landmark IDs
        """

        error_l = landmark_est - landmarks_true

        marker_type = "."
        error_color = "tab:blue"
        bound_color = "tab:blue"
        capsize = 8

        fig, ax = plt.subplots(2,1)
        ax[0].set_ylabel(r"$e_x$ (m)")
        ax[1].set_ylabel(r"$e_y$ (m)")

        ax[0].set_xlabel(r"landmark id")
        ax[1].set_xlabel(r"landmark id")
        
        # plotting 3 sigma bounds
        ax[0].errorbar(
            landmark_ids,
            np.zeros_like(error_l[:,0]),   # center 3 sigma bound at 0
            yerr=sigma3[:,0],
            linestyle="", color=bound_color,
            marker=marker_type, markeredgecolor=error_color, markerfacecolor=error_color,
            capsize=capsize, markersize=0
        )

        # plotting error
        ax[0].plot(landmark_ids, error_l[:,0],
                color=error_color, marker=marker_type,
                linestyle="", label="Proposed", markersize=20)
        
        # plotting 3 sigma bounds
        ax[1].errorbar(
            landmark_ids,
            np.zeros_like(error_l[:,1]),   # center 3 sigma bound at 0
            yerr=sigma3[:,1],
            linestyle="", color=bound_color,
            marker=marker_type, markeredgecolor=error_color, markerfacecolor=error_color,
            capsize=capsize, markersize=0
        )

        # plotting error
        ax[1].plot(landmark_ids, error_l[:,1],
                color=error_color, marker=marker_type,
                linestyle="", label="Proposed", markersize=20)

        ax[0].set_xticks(landmark_ids)
        ax[1].set_xticks(landmark_ids)
        #ax[1].legend(loc = "lower right")
        plt.tight_layout()
        plt.show()

    def plot_robot_error(self, robot_est, robot_true, sigma3, time):
        """
        This function plots the error in the robot's estimated position, and compares
        it to the error in the dead-reckoned position estimate.

        Parameters
        ----------
        robot_est : N,2 array
            Estimated robot positions
        robot_true : N,2 array
            True robot positions
        dr_est : N,2 array
            Dead-reckoned robot positions
        cov : list of 2,2 arrays
            Covariance matrices of the robot position estimates
        time : N array
            Time vector
        """

        #error_dr = dr_est - robot_true
        error_robot_est = robot_est - robot_true

        fig, ax = plt.subplots(2,1)
        ax[0].set_xlabel(r"$t$ (s)")
        ax[1].set_xlabel(r"$t$ (s)")
        ax[0].set_ylabel(r"$e_x$ (m)")
        ax[1].set_ylabel(r"$e_y$ (m)")
        #ax[0].plot(time, error_dr[:,0], label = "Dead-reckoned")
        ax[0].plot(time, error_robot_est[:, 0], label = "Proposed")
        ax[0].fill_between(time[0:len(sigma3)], sigma3[:,0], -sigma3[:,0], color = "lightblue", label=r"$\pm 3 \sigma$")
        #ax[1].plot(time, error_dr[:,1], label = "Dead-reckoned")
        ax[1].plot(time, error_robot_est[:, 1], label = "Proposed")
        ax[1].fill_between(time[0:len(sigma3)], sigma3[:,1], -sigma3[:,1], color = "lightblue", label=r"$\pm 3 \sigma_2$")

        #ax[0].legend(loc = "lower right")
        ax[1].legend(loc = "lower right")
        plt.tight_layout()