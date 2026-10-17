"""LidarLens 3D Voxelization & Pillarization Operations (Chapter 8).

Transforms raw 3D point clouds into regular 2D pseudo-image Pillar representations.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger("lidarlens.models.ops")


def pillarize_points(
    points: np.ndarray,
    voxel_size: Tuple[float, float, float] = (0.16, 0.16, 4.0),
    point_cloud_range: Tuple[float, float, float, float, float, float] = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
    max_points_per_pillar: int = 32,
    max_pillars: int = 16000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition 3D points into vertical pillar columns (Chapter 8).

    Generates augmented 9D point features:
    [x, y, z, r, x_c, y_c, z_c, x_p, y_p] where (x_c, y_c, z_c) is the pillar
    centroid and (x_p, y_p) is the displacement from the geometric pillar center.

    Args:
        points: (N, 4) raw point cloud (x, y, z, intensity).
        voxel_size: (vx, vy, vz) physical dimensions of each pillar column.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        max_points_per_pillar: Maximum points kept in a single pillar.
        max_pillars: Maximum active pillars retained.

    Returns:
        Tuple of (pillar_features, pillar_coords, num_points_per_pillar):
            - pillar_features: (max_pillars, max_points_per_pillar, 9) float32 array.
            - pillar_coords: (max_pillars, 2) int32 2D grid coordinates (y_idx, x_idx).
            - num_points_per_pillar: (max_pillars,) int32 point count per pillar.
    """
    xyz = points[:, :3]
    intensity = points[:, 3:4] if points.shape[1] >= 4 else np.zeros((len(points), 1), dtype=np.float32)

    # Filter points within point cloud bounds
    mask = (
        (xyz[:, 0] >= point_cloud_range[0]) & (xyz[:, 0] < point_cloud_range[3]) &
        (xyz[:, 1] >= point_cloud_range[1]) & (xyz[:, 1] < point_cloud_range[4]) &
        (xyz[:, 2] >= point_cloud_range[2]) & (xyz[:, 2] < point_cloud_range[5])
    )
    valid_pts = xyz[mask]
    valid_intensity = intensity[mask]

    if len(valid_pts) == 0:
        return (
            np.zeros((max_pillars, max_points_per_pillar, 9), dtype=np.float32),
            np.zeros((max_pillars, 2), dtype=np.int32),
            np.zeros((max_pillars,), dtype=np.int32),
        )

    # 2D Grid cell coordinates
    ix = np.floor((valid_pts[:, 0] - point_cloud_range[0]) / voxel_size[0]).astype(np.int32)
    iy = np.floor((valid_pts[:, 1] - point_cloud_range[1]) / voxel_size[1]).astype(np.int32)

    # Group points by pillar hash
    grid_w = int(np.ceil((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]))
    cell_keys = iy * grid_w + ix

    unique_keys, inverse_indices, counts = np.unique(cell_keys, return_inverse=True, return_counts=True)
    n_pillars = min(len(unique_keys), max_pillars)

    pillar_features = np.zeros((max_pillars, max_points_per_pillar, 9), dtype=np.float32)
    pillar_coords = np.zeros((max_pillars, 2), dtype=np.int32)
    num_points_arr = np.zeros((max_pillars,), dtype=np.int32)

    # Populate top pillars
    sort_order = np.argsort(-counts)[:n_pillars]

    for p_idx, orig_p_idx in enumerate(sort_order):
        key = unique_keys[orig_p_idx]
        pt_indices = np.where(cell_keys == key)[0][:max_points_per_pillar]
        k_pts = len(pt_indices)
        num_points_arr[p_idx] = k_pts

        pts_p = valid_pts[pt_indices]
        int_p = valid_intensity[pt_indices]

        # Geometric center of pillar cell
        cx = (key % grid_w) * voxel_size[0] + point_cloud_range[0] + 0.5 * voxel_size[0]
        cy = (key // grid_w) * voxel_size[1] + point_cloud_range[1] + 0.5 * voxel_size[1]

        # Centroid of points in pillar
        mean_xyz = np.mean(pts_p, axis=0)

        # 9D features: x, y, z, r, x - mean_x, y - mean_y, z - mean_z, x - cx, y - cy
        delta_centroid = pts_p - mean_xyz
        delta_center_x = pts_p[:, 0:1] - cx
        delta_center_y = pts_p[:, 1:2] - cy

        feat_9d = np.hstack([pts_p, int_p, delta_centroid, delta_center_x, delta_center_y])
        pillar_features[p_idx, :k_pts] = feat_9d
        pillar_coords[p_idx] = [key // grid_w, key % grid_w]

    return pillar_features, pillar_coords, num_points_arr
