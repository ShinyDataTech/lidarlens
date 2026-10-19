"""
Chapter 5: Point Cloud Registration & LiDAR Odometry
Code Example 5.2: IMU Trajectory SLERP Motion Deskewing (Motion Compensation).

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第5章 5.2节 & 第9章 9.1.1节
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import lidarlens as ll
from lidarlens.registration import de_skew_pointcloud


def generate_distorted_sweep(num_points: int = 15_000, duration: float = 0.1) -> tuple:
    """Generate a 100ms LiDAR sweep distorted by vehicle forward drive and yaw turning."""
    # Point timestamps spanning [0, duration]
    timestamps = np.linspace(0, duration, num_points, dtype=np.float64)
    
    # Static ground truth scene (pillars at known positions)
    angles = np.linspace(0, 2 * np.pi, num_points)
    radius = 12.0
    true_x = radius * np.cos(angles)
    true_y = radius * np.sin(angles)
    true_z = np.sin(4 * angles)
    true_points = np.column_stack([true_x, true_y, true_z]).astype(np.float32)
    
    # Vehicle trajectory during the sweep:
    # Forward velocity v = 15 m/s (~54 km/h), Angular yaw rate w = 0.8 rad/s (~45 deg/s)
    v_x = 15.0
    w_z = 0.8
    
    distorted_points = np.zeros_like(true_points)
    for i in range(num_points):
        dt = timestamps[i]
        # Robot pose at time dt
        yaw = w_z * dt
        c, s = np.cos(yaw), np.sin(yaw)
        R_ego = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        t_ego = np.array([v_x * dt, 0.0, 0.0], dtype=np.float32)
        
        # Point in sensor frame at time dt
        distorted_points[i] = R_ego.T @ (true_points[i] - t_ego)
        
    # High frequency IMU pose data sampled at 100 Hz
    imu_times = np.linspace(0, duration, 11)
    imu_rots = []
    imu_trans = []
    for t in imu_times:
        yaw = w_z * t
        r = R.from_euler("z", yaw)
        imu_rots.append(r)
        imu_trans.append([v_x * t, 0.0, 0.0])
        
    pose_data = {
        "timestamps": imu_times,
        "rotations": R.concatenate(imu_rots),
        "translations": np.array(imu_trans, dtype=np.float32)
    }
    
    return distorted_points, timestamps, pose_data, true_points


def main():
    print("=== Chapter 5: IMU SLERP Point Cloud Motion Deskewing ===")
    
    duration = 0.1 # 100 ms sweep (10 Hz spin)
    distorted_pts, timestamps, pose_data, true_pts = generate_distorted_sweep(num_points=15_000, duration=duration)
    t_ref = duration # Deskew to the end-of-sweep frame
    
    # Ground truth points in the sensor frame at t_ref
    yaw_ref = 0.8 * t_ref
    c_ref, s_ref = np.cos(yaw_ref), np.sin(yaw_ref)
    R_ref = np.array([[c_ref, -s_ref, 0], [s_ref, c_ref, 0], [0, 0, 1]], dtype=np.float32)
    p_ref = np.array([15.0 * t_ref, 0.0, 0.0], dtype=np.float32)
    true_pts_ref = (R_ref.T @ (true_pts - p_ref).T).T.astype(np.float32)
    
    # Distortion error before compensation relative to t_ref frame
    init_err = np.mean(np.linalg.norm(distorted_pts - true_pts_ref, axis=1))
    print(f"Mean spatial distortion before deskewing: {init_err:.4f} m (caused by 15 m/s velocity + 45 deg/s turn)")
    
    # Run IMU SLERP Deskewing
    t0 = time.perf_counter()
    deskewed_pts = de_skew_pointcloud(
        points=distorted_pts,
        timestamps=timestamps,
        pose_data=pose_data,
        t_ref=t_ref
    )
    deskew_ms = (time.perf_counter() - t0) * 1000.0
    
    residual_err = np.mean(np.linalg.norm(deskewed_pts - true_pts_ref, axis=1))
    print(f"Motion deskewing completed in {deskew_ms:.2f} ms.")
    print(f"Residual error after SLERP compensation: {residual_err:.4f} m (reduced by {(init_err - residual_err)/init_err*100:.1f}%)")
    
    assert residual_err < 0.05, f"Residual error {residual_err:.4f} too high after compensation!"
    print("Chapter 5.2 verification completed successfully.\n")


if __name__ == "__main__":
    main()
