from functions.plotting_tools import PlottingTools
from functions.linear_batch_slam import LinearSLAM
from functions.relative_position_gtrs import RelativePositionGTRS
import numpy as np

# Load data
gt = np.loadtxt('data/groundtruth.csv', delimiter=',')
dvl = np.loadtxt('data/dvl_measurements.csv', delimiter=',')
ranges = np.loadtxt('data/range_measurements.csv', delimiter=',')
orientation = np.loadtxt('data/orientation_measurements.csv', delimiter=',')

landmarks_true = np.array([[10, -30],
                           [0, 100],
                           [-60, 150]])

sigma_r = np.sqrt(0.4)
sigma_theta = np.sqrt(0.7615e-6)
sigma_v = np.sqrt(0.02)
Q_dvl = sigma_v**2 * np.eye(2)
num_landmarks = 3

# create dcm from orientation measurements
dcm = []
for i in range(len(orientation)):
    theta = orientation[i, 1]
    c, s = np.cos(theta), np.sin(theta)
    dcm.append(np.array([[c, -s], [s, c]]))

# get dead-reckoned positions
dead_reckoned = []
x = gt[0, 1:3]
dead_reckoned.append(x)
dt = 0.1
for i in range(len(dvl)):
    x = x + (dt * dcm[i] @ (dvl[i, 1:3].reshape(2,1))).flatten()
    dead_reckoned.append(x)

dead_reckoned = np.array(dead_reckoned)

# Get relative position estimates using SR-LS GTRS method
rel_posns = RelativePositionGTRS()

rel_posns_all = []
rel_posns_all_cov = []

for i in range(num_landmarks):
    # get range measurements for landmark i
    ranges_i = ranges[ranges[:, 1] == i+1]
    dcm_subset = np.array(dcm)[np.isin(orientation[:,0], ranges_i[:,0])]
    dvl_subset = dvl[np.isin(dvl[:,0], ranges_i[:,0])]
    dead_reckoned_subset = dead_reckoned[:-1][np.isin(dvl[:,0], ranges_i[:,0])]

    rel_pos, landmark_est, rel_pos_cov = rel_posns.get_landmarks_srls_gtrs(dead_reckoned_subset, dcm_subset, 0, ranges_i[:,2], ranges_i[:,0], dvl_subset[:,1:3], sigma_r, sigma_theta, Q_dvl)

    rel_posns_all.append(np.column_stack((ranges_i[:,0], rel_pos)))
    rel_posns_all_cov.append(rel_pos_cov)

# Get robot and landmark position estimated using linear batch SLAM
slam = LinearSLAM()
robot_est, sigma3_robot, landmarks_est, sigma3_landmarks = slam.linear_batch_slam(dcm, rel_posns_all, rel_posns_all_cov, dvl, Q_dvl, sigma_theta, np.array([0,0]), 1e-4 * np.eye(2,2))

# Plot results
plotting = PlottingTools()

plotting.plot_trajectory(dead_reckoned, landmarks_est, robot_est, landmarks_true, gt[:,2:4], [1,2,3])
plotting.plot_landmark_error(landmarks_est, landmarks_true, sigma3_landmarks, [1,2,3])
plotting.plot_robot_error(robot_est, gt[:,2:4], sigma3_robot, gt[:,0])
