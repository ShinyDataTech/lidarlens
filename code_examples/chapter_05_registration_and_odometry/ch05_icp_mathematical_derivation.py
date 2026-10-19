"""
Chapter 5: Point Cloud Registration & LiDAR Odometry
Code Example 5.1: Mathematical Derivation & SVD Implementation of Point-to-Point ICP.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第5章 5.1.2节
"""

import time
import numpy as np
from typing import Tuple
from scipy.spatial import cKDTree


def icp_match(
    source_cloud: np.ndarray,
    target_points: np.ndarray,
    initial_pose: np.ndarray,
    max_iters: int = 50,
    epsilon: float = 1e-5,
    max_corr_dist: float = 2.0
) -> Tuple[np.ndarray, float, int]:
    """Point-to-point ICP alignment via SVD decomposition of cross-covariance matrix.
    
    Args:
        source_cloud: (N, 3) source point cloud coordinates.
        target_points: (M, 3) reference target point cloud.
        initial_pose: (4, 4) initial guess transformation matrix.
        max_iters: Maximum convergence iterations.
        epsilon: Transformation update stopping tolerance.
        max_corr_dist: Maximum nearest-neighbor correspondence distance threshold.
        
    Returns:
        Tuple of (estimated_pose, final_rmse, num_iterations).
    """
    target_tree = cKDTree(target_points)
    current_pose = initial_pose.copy()
    
    prev_error = float("inf")
    converged_iter = max_iters

    for iteration in range(max_iters):
        # 1. Transform source points by current estimated pose
        src_homo = np.hstack([source_cloud[:, :3], np.ones((len(source_cloud), 1), dtype=np.float32)])
        transformed_pts = (current_pose @ src_homo.T).T[:, :3]

        # 2. Query nearest neighbor correspondences
        dists, indices = target_tree.query(transformed_pts, k=1)
        valid = dists < max_corr_dist
        if np.sum(valid) < 6:
            break

        p_matched = transformed_pts[valid]
        q_matched = target_points[indices[valid]]
        current_error = float(np.mean(dists[valid] ** 2))

        # 3. Compute centroids and centered coordinates
        mu_p = np.mean(p_matched, axis=0)
        mu_q = np.mean(q_matched, axis=0)
        p_centered = p_matched - mu_p
        q_centered = q_matched - mu_q

        # 4. Cross-covariance matrix H = sum(p_centered * q_centered^T)
        H = p_centered.T @ q_centered

        # 5. SVD decomposition: H = U * S * V^T
        U, S, Vt = np.linalg.svd(H)
        R_delta = Vt.T @ U.T

        # Reflection correction: det(R) must equal +1
        if np.linalg.det(R_delta) < 0:
            Vt[-1, :] *= -1
            R_delta = Vt.T @ U.T

        t_delta = mu_q - R_delta @ mu_p

        # 6. Compose incremental transformation
        delta_T = np.eye(4, dtype=np.float32)
        delta_T[:3, :3] = R_delta
        delta_T[:3, 3] = t_delta
        current_pose = delta_T @ current_pose

        # 7. Check convergence tolerance
        if abs(prev_error - current_error) < epsilon:
            converged_iter = iteration + 1
            break
        prev_error = current_error

    rmse = np.sqrt(max(current_error, 0.0))
    return current_pose, rmse, converged_iter


def main():
    print("=== Chapter 5: SVD Point-to-Point ICP Registration ===")
    
    # 1. Create target point cloud (bunny / sphere shape)
    n_pts = 3000
    theta = np.random.uniform(0, 2*np.pi, n_pts)
    phi = np.random.uniform(-np.pi/2, np.pi/2, n_pts)
    r = 5.0 + 0.5 * np.sin(3*theta)
    
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    target = np.column_stack([x, y, z]).astype(np.float32)
    
    # 2. Create ground-truth perturbed source cloud
    # Rotation: ~15 deg around Z, Translation: [0.5, -0.3, 0.2]
    yaw_gt = np.deg2rad(15.0)
    R_gt = np.array([
        [np.cos(yaw_gt), -np.sin(yaw_gt), 0],
        [np.sin(yaw_gt),  np.cos(yaw_gt), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    t_gt = np.array([0.5, -0.3, 0.2], dtype=np.float32)
    
    T_gt = np.eye(4, dtype=np.float32)
    T_gt[:3, :3] = R_gt
    T_gt[:3, 3] = t_gt
    
    # Invert to transform target into source sensor frame
    T_gt_inv = np.linalg.inv(T_gt)
    source = (T_gt_inv[:3, :3] @ target.T).T + T_gt_inv[:3, 3]
    # Add sensor noise
    source += np.random.normal(0, 0.01, size=source.shape).astype(np.float32)
    
    # 3. Perform ICP Matching
    t0 = time.perf_counter()
    est_pose, rmse, iters = icp_match(
        source_cloud=source,
        target_points=target,
        initial_pose=np.eye(4, dtype=np.float32),
        max_iters=40
    )
    icp_ms = (time.perf_counter() - t0) * 1000.0
    
    trans_err = np.linalg.norm(est_pose[:3, 3] - T_gt[:3, 3])
    rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(est_pose[:3, :3].T @ T_gt[:3, :3]) - 1) / 2, -1.0, 1.0)))
    
    print(f"ICP completed in {icp_ms:.2f} ms ({iters} iterations).")
    print(f"Final Residual RMSE: {rmse:.4f} m")
    print(f"Translation Error:  {trans_err:.4f} m")
    print(f"Rotation Error:     {rot_err_deg:.4f} deg")
    
    assert trans_err < 0.05, f"Translation error {trans_err} too high"
    assert rot_err_deg < 0.5, f"Rotation error {rot_err_deg} too high"
    print("Chapter 5.1 verification completed successfully.\n")


if __name__ == "__main__":
    main()
