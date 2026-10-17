"""LidarLens Low-Latency GICP Frontend Odometry (Chapter 5).

Scan-to-submap Generalized ICP odometry with fast voxelization, local covariance
estimation, and 6x6 Gauss-Newton Hessian extraction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger("lidarlens.registration.gicp")


class LowLatencyGICP:
    """Low-Latency GICP frontend odometry core engine.

    Executes voxel-hash downsampled Scan-to-Submap GICP odometry, maintains
    a local sliding submap, and exposes the 6x6 Gauss-Newton Hessian matrix
    for degeneracy auditing.
    """

    def __init__(
        self,
        leaf_size: Optional[float] = None,
        max_corr_dist: float = 2.0,
        max_iter: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        l_size = leaf_size if leaf_size is not None else kwargs.get("voxel_size", 0.5)
        m_iter = max_iter if max_iter is not None else kwargs.get("max_iterations", 25)
        self.leaf_size = float(l_size)
        self.max_corr_dist = float(max_corr_dist)
        self.max_iter = int(m_iter)

        self.local_submap_points: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.local_submap_covs: np.ndarray = np.empty((0, 3, 3), dtype=np.float32)
        self.current_pose: np.ndarray = np.eye(4, dtype=np.float32)
        self.last_hessian: np.ndarray = np.eye(6, dtype=np.float32)

    def set_target(self, target: np.ndarray) -> None:
        """Set reference target point cloud / submap."""
        pts = target[:, :3].astype(np.float32)
        self.local_submap_points = self.voxel_downsample(pts, self.leaf_size)
        self.local_submap_covs = self.estimate_covariances(self.local_submap_points, k_neighbors=15)

    def align(self, source: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Align source scan to the current target submap."""
        pts = source[:, :3].astype(np.float32)
        filtered_source = self.voxel_downsample(pts, self.leaf_size)
        covs = self.estimate_covariances(filtered_source, k_neighbors=15)
        guess = initial_guess if initial_guess is not None else np.eye(4, dtype=np.float32)
        final_pose, H = self.align_gicp(filtered_source, covs, guess)
        self.last_hessian = H
        self.current_pose = final_pose
        return {
            "transform": final_pose,
            "hessian": H,
            "iterations": self.max_iter,
        }

    def process_scan(self, raw_cloud: np.ndarray, imu_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Process a single LiDAR scan and compute the 4x4 transformation pose.

        Args:
            raw_cloud: (N, 3) or (N, 4) raw points.
            imu_data: Optional IMU prediction dictionary.

        Returns:
            (4, 4) float32 updated ego-vehicle pose matrix in world coordinates.
        """
        # 1. Downsample points
        pts = raw_cloud[:, :3].astype(np.float32)
        cloud_filtered = self.voxel_downsample(pts, self.leaf_size)

        if len(cloud_filtered) < 10:
            return self.current_pose.copy()

        # 2. Local covariance estimation
        covariances = self.estimate_covariances(cloud_filtered, k_neighbors=15)

        # 3. Predict motion
        initial_guess = self.predict_motion_from_imu(imu_data)

        # 4. Scan-to-Submap alignment
        if len(self.local_submap_points) > 20:
            final_pose, H = self.align_gicp(cloud_filtered, covariances, initial_guess)
            self.last_hessian = H
        else:
            final_pose = initial_guess
            self.last_hessian = np.eye(6, dtype=np.float32) * 100.0

        self.current_pose = final_pose

        # 5. Integrate into local submap (sliding buffer max 10,000 points)
        pts_world = (final_pose[:3, :3] @ cloud_filtered.T + final_pose[:3, 3:4]).T
        if len(self.local_submap_points) == 0:
            self.local_submap_points = pts_world
        else:
            self.local_submap_points = np.vstack([self.local_submap_points, pts_world])
            if len(self.local_submap_points) > 10_000:
                self.local_submap_points = self.local_submap_points[-10_000:]

        return self.current_pose.copy()

    def voxel_downsample(self, cloud: np.ndarray, leaf_size: float) -> np.ndarray:
        """Fast voxel-hash spatial downsampling."""
        quantized = np.floor(cloud / leaf_size).astype(np.int32)
        _, unique_indices = np.unique(quantized, axis=0, return_index=True)
        return cloud[unique_indices]

    def estimate_covariances(self, cloud: np.ndarray, k_neighbors: int = 15) -> np.ndarray:
        """Estimate 3x3 local covariance matrices for each point using KNN."""
        n = len(cloud)
        k = min(k_neighbors, n)
        if k < 4:
            return np.repeat(np.eye(3, dtype=np.float32)[np.newaxis, ...], n, axis=0)

        tree = cKDTree(cloud)
        _, indices = tree.query(cloud, k=k)

        covs = np.zeros((n, 3, 3), dtype=np.float32)
        for i, neighbors_idx in enumerate(indices):
            pts = cloud[neighbors_idx]
            covs[i] = np.cov(pts, rowvar=False)

        return covs

    def predict_motion_from_imu(self, imu_data: Optional[Dict[str, Any]]) -> np.ndarray:
        """Predict pose using IMU or constant velocity assumption."""
        # By default maintain current pose (or apply IMU delta if provided)
        return self.current_pose.copy()

    def align_gicp(
        self,
        source_pts: np.ndarray,
        source_covs: np.ndarray,
        initial_guess: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align source scan to local submap using Gauss-Newton iterations.

        Returns:
            Tuple of (converged_pose (4, 4), Hessian_matrix (6, 6)).
        """
        tree = cKDTree(self.local_submap_points)
        T = initial_guess.copy()

        H_last = np.eye(6, dtype=np.float32)

        for _ in range(self.max_iter):
            # Transform source points by current T
            R = T[:3, :3]
            t = T[:3, 3]
            src_transformed = (R @ source_pts.T).T + t

            dists, idxs = tree.query(src_transformed, k=1)
            valid = dists < self.max_corr_dist

            if np.sum(valid) < 15:
                break

            p_src = src_transformed[valid]
            p_tgt = self.local_submap_points[idxs[valid]]

            # Error residuals: e = p_src - p_tgt
            residuals = p_src - p_tgt

            # Jacobian J: 3x6 for each point: [-skew(p_src) | I_3]
            n_corr = len(p_src)
            J = np.zeros((n_corr * 3, 6), dtype=np.float32)

            for i in range(n_corr):
                px, py, pz = p_src[i]
                skew = np.array([
                    [0.0, pz, -py],
                    [-pz, 0.0, px],
                    [py, -px, 0.0]
                ], dtype=np.float32)

                J[i * 3 : i * 3 + 3, :3] = skew
                J[i * 3 : i * 3 + 3, 3:] = np.eye(3, dtype=np.float32)

            r_flat = residuals.reshape(-1)

            # Approximate Hessian: H = J^T J
            H = J.T @ J
            b = -J.T @ r_flat

            H_last = H

            # Regularized solve: delta_xi = (H + lambda * I)^(-1) b
            H_reg = H + 1e-4 * np.eye(6, dtype=np.float32)
            try:
                delta = np.linalg.solve(H_reg, b)
            except np.linalg.LinAlgError:
                break

            # Lie algebra se(3) exponential map update
            w = delta[:3]
            v = delta[3:]

            theta = np.linalg.norm(w)
            if theta < 1e-6:
                delta_R = np.eye(3, dtype=np.float32)
            else:
                k = w / theta
                K = np.array([
                    [0.0, -k[2], k[1]],
                    [k[2], 0.0, -k[0]],
                    [-k[1], k[0], 0.0]
                ], dtype=np.float32)
                delta_R = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

            delta_T = np.eye(4, dtype=np.float32)
            delta_T[:3, :3] = delta_R
            delta_T[:3, 3] = v

            T = delta_T @ T

            if np.linalg.norm(delta) < 1e-4:
                break

        return T, H_last
