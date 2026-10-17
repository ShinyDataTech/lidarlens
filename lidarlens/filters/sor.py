"""LidarLens Statistical Outlier Removal (SOR) & Dynamic Denoising (Chapters 3 & 9).

Provides distance-adaptive DynamicTemporalSOR with sliding-window temporal voting,
classic SOR, and multi-stage cascaded rain/fog point cloud denoising.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
from scipy.spatial import cKDTree

from lidarlens.core import LidarFrame
from lidarlens.pipeline import RingBuffer

logger = logging.getLogger("lidarlens.filters.sor")


class DynamicTemporalSOR:
    """Distance-adaptive Statistical Outlier Removal with temporal voting.

    Addresses the LiDAR beam divergence problem: distant points have naturally
    lower density. Using static SOR thresholds aggressively prunes distant valid
    obstacles (e.g. oncoming vehicles at 50 m). DynamicTemporalSOR automatically
    scales tolerance by radial range and verifies noise across sliding temporal votes.
    """

    def __init__(
        self,
        base_neighbors: Optional[int] = None,
        base_std_ratio: float = 2.0,
        distance_scale: float = 0.05,
        temporal_votes: int = 2,
        **kwargs: Any,
    ) -> None:
        neighbors = base_neighbors if base_neighbors is not None else kwargs.get("k_neighbors", 30)
        self.base_neighbors = int(neighbors)
        self.base_std_ratio = float(base_std_ratio)
        self.distance_scale = float(distance_scale)
        self.temporal_votes = int(temporal_votes)

    def __call__(self, frame_or_points: Union[LidarFrame, np.ndarray]) -> Union[LidarFrame, np.ndarray]:
        return self.process(frame_or_points)

    def process(self, frame_or_points: Union[LidarFrame, np.ndarray]) -> Union[LidarFrame, np.ndarray]:
        return self.process_temporal(frame_or_points, ring_buffer=None)

    def process_temporal(
        self,
        frame_or_points: Union[LidarFrame, np.ndarray],
        ring_buffer: Optional[RingBuffer] = None,
    ) -> Union[LidarFrame, np.ndarray]:
        is_frame = isinstance(frame_or_points, LidarFrame)
        current_frame = frame_or_points if is_frame else LidarFrame(points=frame_or_points)
        pts = current_frame.points

        if len(pts) < max(self.base_neighbors, 5):
            return current_frame if is_frame else pts

        xyz = pts[:, :3]
        tree = cKDTree(xyz)

        k = min(self.base_neighbors + 1, len(xyz))
        dists, _ = tree.query(xyz, k=k)

        # Exclude self-distance (column 0)
        mean_knn = dists[:, 1:].mean(axis=1)
        global_mean = float(np.mean(mean_knn))
        global_std = float(np.std(mean_knn)) + 1e-9

        # Radial distance calculation
        ranges = np.linalg.norm(xyz, axis=1)

        # Distance-adaptive threshold: relaxes threshold for distant points
        adaptive_std_ratio = self.base_std_ratio + self.distance_scale * ranges
        thresholds = global_mean + adaptive_std_ratio * global_std
        inlier_mask = mean_knn <= thresholds

        # Temporal Voting: if ring_buffer is available, evaluate temporal stability
        if ring_buffer is not None and len(ring_buffer) >= self.temporal_votes:
            # Query candidate noise points against historical frames
            noise_indices = np.where(~inlier_mask)[0]
            if len(noise_indices) > 0:
                noise_xyz = xyz[noise_indices]
                # Count in how many historical frames this point is considered isolated
                historical_votes = np.zeros(len(noise_indices), dtype=int)
                for past_frame in ring_buffer.frames[:-1]:
                    if past_frame.num_points > 5:
                        past_tree = cKDTree(past_frame.xyz)
                        past_dists, _ = past_tree.query(noise_xyz, k=1)
                        # If a point was completely absent or isolated in previous frame
                        historical_votes += (past_dists > 0.5).astype(int)

                # Keep points that do not meet the minimum temporal noise votes
                confirmed_noise = historical_votes >= (self.temporal_votes - 1)
                for i, is_noise in enumerate(confirmed_noise):
                    if not is_noise:
                        inlier_mask[noise_indices[i]] = True

        filtered_pts = pts[inlier_mask]

        if is_frame:
            return LidarFrame(
                points=filtered_pts,
                timestamp=current_frame.timestamp,
                frame_id=current_frame.frame_id,
                metadata={**current_frame.metadata, "sor_filtered": True},
            )
        return filtered_pts


class StatisticalOutlierRemoval:
    """Standard Statistical Outlier Removal (SOR)."""

    def __init__(self, nb_neighbors: int = 20, std_ratio: float = 2.0) -> None:
        self.nb_neighbors = int(nb_neighbors)
        self.std_ratio = float(std_ratio)

    def process(self, points: np.ndarray) -> np.ndarray:
        if len(points) < self.nb_neighbors + 1:
            return points
        tree = cKDTree(points[:, :3])
        dists, _ = tree.query(points[:, :3], k=self.nb_neighbors + 1)
        mean_d = dists[:, 1:].mean(axis=1)
        mu, sigma = float(np.mean(mean_d)), float(np.std(mean_d)) + 1e-9
        mask = mean_d <= (mu + self.std_ratio * sigma)
        return points[mask]


def rain_pointcloud_denoise(
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    intensity_thr: float = 0.12,
    ror_radius: float = 0.45,
    ror_min_neighbors: int = 6,
    sor_k: int = 16,
    sor_z_thr: float = 2.2,
) -> np.ndarray:
    """Cascade denoiser for rain/fog backscatter (Chapter 9).

    Executes a 3-stage hardware-matched cascade:
    1. Reflection Intensity Thresholding (rain backscatter energy is minimal).
    2. Radius Outlier Removal (ROR) for isolated splash/droplet clusters.
    3. Statistical Outlier Removal (SOR) with z-score gating.

    Args:
        points_xyz: (N, 3) 3D coordinate array.
        intensity: (N,) or (N, 1) scalar reflection intensity array.
        intensity_thr: Minimum intensity threshold.
        ror_radius: Search radius for ROR in meters.
        ror_min_neighbors: Minimum neighboring points within ROR radius.
        sor_k: Number of nearest neighbors for SOR.
        sor_z_thr: Standard deviation threshold multiplier.

    Returns:
        (N,) boolean inlier mask.
    """
    intensity_flat = intensity.flatten()
    n_points = points_xyz.shape[0]

    # Stage 1: Intensity Thresholding
    mask_int = intensity_flat >= intensity_thr
    idx_p1 = np.where(mask_int)[0]
    p1 = points_xyz[idx_p1]

    if len(p1) < max(ror_min_neighbors, 4):
        return np.zeros(n_points, dtype=bool)

    # Stage 2: Radius Outlier Removal (ROR)
    kdt1 = cKDTree(p1)
    counts = np.array([len(kdt1.query_ball_point(p, r=ror_radius)) - 1 for p in p1])
    mask_ror = counts >= ror_min_neighbors
    idx_p2 = idx_p1[mask_ror]
    p2 = points_xyz[idx_p2]

    if len(p2) < max(sor_k + 1, 4):
        return np.zeros(n_points, dtype=bool)

    # Stage 3: Statistical Outlier Removal (SOR)
    kdt2 = cKDTree(p2)
    distances, _ = kdt2.query(p2, k=min(sor_k + 1, len(p2)))
    mean_knn = distances[:, 1:].mean(axis=1)
    mu = float(np.mean(mean_knn))
    sigma = float(np.std(mean_knn)) + 1e-9
    z_score = (mean_knn - mu) / sigma
    mask_sor = z_score <= sor_z_thr
    idx_p3 = idx_p2[mask_sor]

    final_mask = np.zeros(n_points, dtype=bool)
    final_mask[idx_p3] = True
    return final_mask
