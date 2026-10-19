"""
Chapter 5: Point Cloud Registration & LiDAR Odometry
Code Example 5.3: Scan-to-Submap LowLatencyGICP Real-Time Odometry.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第5章 5.3.2节
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.registration import LowLatencyGICP


def generate_simulated_submap(num_points: int = 10_000) -> np.ndarray:
    """Generate 3D geometric point cloud of an indoor corridor with walls and pillars."""
    pts = []
    # Left & right walls
    n_wall = num_points // 3
    x_wall = np.random.uniform(-10, 10, n_wall)
    z_wall = np.random.uniform(-1, 3, n_wall)
    pts.append(np.column_stack([x_wall, np.full(n_wall, -3.5), z_wall]))
    pts.append(np.column_stack([x_wall, np.full(n_wall, 3.5), z_wall]))
    
    # Ground floor
    pts.append(np.column_stack([x_wall, np.random.uniform(-3.5, 3.5, n_wall), np.full(n_wall, -1.0)]))
    
    cloud = np.vstack(pts).astype(np.float32)
    return cloud


def main():
    print("=== Chapter 5: LowLatencyGICP Scan-to-Submap Odometry ===")
    
    # 1. Initialize Submap and Current Scan with known motion displacement
    target_submap = generate_simulated_submap(num_points=12_000)
    
    # Ground truth relative ego motion: forward 0.25m, lateral 0.05m, yaw 2.0 degrees
    yaw_gt = np.deg2rad(2.0)
    R_gt = np.array([
        [np.cos(yaw_gt), -np.sin(yaw_gt), 0],
        [np.sin(yaw_gt),  np.cos(yaw_gt), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    t_gt = np.array([0.25, 0.05, 0.0], dtype=np.float32)
    
    # Transform submap to create incoming scan in sensor frame
    source_scan = (R_gt.T @ (target_submap - t_gt).T).T.astype(np.float32)
    # Add random sensor measurement noise
    source_scan += np.random.normal(0, 0.015, size=source_scan.shape).astype(np.float32)
    
    # 2. Configure LowLatencyGICP Odometry (6 iterations for high-frequency scan matching)
    gicp = LowLatencyGICP(leaf_size=0.5, max_iterations=6)
    
    # 3. Set Target and Align Source Scan
    t0 = time.perf_counter()
    gicp.set_target(target_submap)
    align_res = gicp.align(source_scan)
    gicp_ms = (time.perf_counter() - t0) * 1000.0
    
    est_transform = align_res["transform"]
    hessian = align_res["hessian"]
    iterations = align_res["iterations"]
    
    trans_err = np.linalg.norm(est_transform[:3, 3] - t_gt)
    
    print(f"GICP alignment completed in {gicp_ms:.2f} ms ({iterations} iterations).")
    print(f"Ground Truth Motion:  dx={t_gt[0]:.3f}m, dy={t_gt[1]:.3f}m, yaw={np.rad2deg(yaw_gt):.2f}°")
    print(f"Estimated Motion:     dx={est_transform[0, 3]:.3f}m, dy={est_transform[1, 3]:.3f}m")
    print(f"Translation Error:    {trans_err:.4f} m")
    print(f"Hessian shape:        {hessian.shape}")
    
    assert trans_err < 0.08, f"Odometry error {trans_err:.4f} exceeds threshold!"
    assert gicp_ms < 150.0, f"GICP latency {gicp_ms:.2f} ms exceeds real-time budget!"
    print("Chapter 5.3 verification completed successfully.\n")


if __name__ == "__main__":
    main()
