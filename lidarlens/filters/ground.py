"""LidarLens Ground Segmentation Algorithms (Chapter 4).

Provides Concentric Zone Model (Patchwork++) and planar RANSAC ground extraction.
"""

from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np

from lidarlens.core import LidarFrame

logger = logging.getLogger("lidarlens.filters.ground")


class RANSACGround:
    """Planar RANSAC ground plane segmentation.

    Estimates the dominant ground plane equation ax + by + cz + d = 0 and
    separates ground points from non-ground obstacles.
    """

    def __init__(self, distance_threshold: float = 0.2, max_iterations: int = 100, seed: int = 42) -> None:
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations
        self.seed = seed

    def segment(self, points: Union[np.ndarray, LidarFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Segment ground plane from 3D points.

        Args:
            points: (N, 3) or (N, 4) point cloud array or LidarFrame.

        Returns:
            Tuple of (ground_mask, plane_model [a, b, c, d]).
        """
        pts_arr = points.points if isinstance(points, LidarFrame) else points
        xyz = pts_arr[:, :3]
        n = xyz.shape[0]
        if n < 3:
            return np.zeros(n, dtype=bool), np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        rng = np.random.default_rng(self.seed)
        best_inliers = np.zeros(n, dtype=bool)
        best_plane = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        max_inlier_count = -1

        for _ in range(self.max_iterations):
            idx = rng.choice(n, size=3, replace=False)
            p1, p2, p3 = xyz[idx[0]], xyz[idx[1]], xyz[idx[2]]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal /= norm

            # Ground plane normal is typically predominantly vertical (|nz| > 0.6)
            if abs(normal[2]) < 0.6:
                continue

            d = -np.dot(normal, p1)
            plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

            # Point-to-plane distance: |ax + by + cz + d|
            dists = np.abs(xyz @ normal + d)
            inliers = dists <= self.distance_threshold
            inlier_count = int(np.sum(inliers))

            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_inliers = inliers
                best_plane = plane

        return best_inliers, best_plane


class PatchworkPlusPlus:
    """Concentric Zone Model (CZM) based ground segmentation (Patchwork++).

    Partitions polar BEV space into concentric zones with increasing radial bin
    sizes and angular azimuth sectors. Locally extracts the lowest point elevation
    and fits piecewise smooth ground surfaces.
    """

    def __init__(
        self,
        czm_rings: int = 4,
        czm_sectors: int = 16,
        max_range: float = 60.0,
        elevation_thresh: float = 0.25,
    ) -> None:
        self.czm_rings = czm_rings
        self.czm_sectors = czm_sectors
        self.max_range = max_range
        self.elevation_thresh = elevation_thresh

    def segment(self, points: Union[np.ndarray, LidarFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Segment points into ground and non-ground.

        Args:
            points: (N, 3) or (N, 4) point cloud or LidarFrame.

        Returns:
            Tuple of (ground_mask, non_ground_mask).
        """
        pts_arr = points.points if isinstance(points, LidarFrame) else points
        xyz = pts_arr[:, :3]
        n = xyz.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

        r = np.linalg.norm(xyz[:, :2], axis=1)
        theta = np.arctan2(xyz[:, 1], xyz[:, 0])  # [-pi, pi]

        # Filter out points beyond max_range
        valid_range = r <= self.max_range
        ground_mask = np.zeros(n, dtype=bool)

        ring_edges = np.linspace(0.5, self.max_range, self.czm_rings + 1)
        sector_edges = np.linspace(-np.pi, np.pi, self.czm_sectors + 1)

        for i in range(self.czm_rings):
            r_low, r_high = ring_edges[i], ring_edges[i + 1]
            for j in range(self.czm_sectors):
                th_low, th_high = sector_edges[j], sector_edges[j + 1]

                bin_mask = valid_range & (r >= r_low) & (r < r_high) & (theta >= th_low) & (theta < th_high)
                bin_indices = np.where(bin_mask)[0]

                if len(bin_indices) < 3:
                    continue

                bin_pts = xyz[bin_indices]
                # Find lowest elevation seeds (bottom 20% elevation)
                z_vals = bin_pts[:, 2]
                z_seed_thresh = np.percentile(z_vals, 20)
                lowest_pts = bin_pts[z_vals <= z_seed_thresh]

                # Estimate local ground plane height
                z_ground = float(np.mean(lowest_pts[:, 2]))

                # Points within elevation_thresh of local ground are classified as ground
                is_ground = np.abs(bin_pts[:, 2] - z_ground) <= self.elevation_thresh
                ground_mask[bin_indices[is_ground]] = True

        return ground_mask, ~ground_mask
