"""LidarLens Density-Based Spatial Clustering (Chapter 4).

Provides distance-adaptive AdaptiveDBSCAN with range-varying neighborhood radii
and fast Euclidean point cloud clustering.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger("lidarlens.cluster.dbscan")


class AdaptiveDBSCAN:
    """Distance-adaptive DBSCAN clustering for 3D LiDAR point clouds.

    In real LiDAR sensors, point density decays quadratically with radial distance d.
    A static epsilon causes distant vehicles to fracture into noise or near objects
    to fuse together. AdaptiveDBSCAN dynamically scales the search radius:
        eps(d) = eps_min + alpha * d
    """

    def __init__(
        self,
        eps_min: float = 0.3,
        alpha: float = 0.015,
        min_samples: int = 8,
    ) -> None:
        self.eps_min = float(eps_min)
        self.alpha = float(alpha)
        self.min_samples = int(min_samples)

    def fit_predict(self, points: np.ndarray) -> np.ndarray:
        """Compute cluster labels for input 3D points.

        Args:
            points: (N, 3) or (N, 4) point cloud array.

        Returns:
            (N,) int32 array of cluster labels (-1 for noise, 0, 1, ... for clusters).
        """
        n = points.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int32)

        xyz = points[:, :3].astype(np.float32)
        ranges = np.linalg.norm(xyz, axis=1)
        tree = cKDTree(xyz)

        labels = np.full(n, -1, dtype=np.int32)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            eps_i = self.eps_min + self.alpha * ranges[i]
            neighbors = tree.query_ball_point(xyz[i], r=eps_i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Mark as noise
                continue

            # Expand new cluster
            labels[i] = cluster_id
            queue = list(neighbors)

            idx = 0
            while idx < len(queue):
                neighbor_idx = queue[idx]
                idx += 1

                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    eps_n = self.eps_min + self.alpha * ranges[neighbor_idx]
                    n_neighbors = tree.query_ball_point(xyz[neighbor_idx], r=eps_n)
                    if len(n_neighbors) >= self.min_samples:
                        queue.extend([p for p in n_neighbors if p not in queue])

                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id

            cluster_id += 1

        return labels

    def extract_clusters(self, points: np.ndarray) -> List[np.ndarray]:
        """Extract individual cluster point clouds excluding noise.

        Args:
            points: (N, 3) or (N, 4) point cloud.

        Returns:
            List of cluster arrays [ (N_k, 3), ... ].
        """
        labels = self.fit_predict(points)
        unique_labels = set(labels) - {-1}
        clusters = [points[labels == cid] for cid in sorted(unique_labels)]
        return clusters
