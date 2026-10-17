from __future__ import annotations

try:
    import open3d as o3d
except ImportError:
    class _MockO3D:
        class geometry:
            PointCloud = object
            AxisAlignedBoundingBox = object
            TriangleMesh = object
            VoxelGrid = object
        class utility:
            Vector3dVector = object
        class visualization:
            @staticmethod
            def draw_geometries(*args, **kwargs):
                pass
    o3d = _MockO3D()

try:
    import laspy
except ImportError:
    laspy = None

"""
Segmentation module — ground extraction, plane fitting, clustering, and
heuristic / deep-learning segmentation.
"""


from typing import List, Optional, Tuple

import numpy as np
# ── Plane Fitting ────────────────────────────────────────────────────────────


def segment_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.2,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> Tuple[list, list]:
    """RANSAC plane segmentation.

    Args:
        pcd: Input point cloud.
        distance_threshold: Maximum distance a point may be from the plane
            to be considered an inlier.
        ransac_n: Number of initial points to estimate a plane.
        num_iterations: RANSAC iterations.

    Returns:
        ``(plane_model, inlier_indices)`` where *plane_model* is
        ``[a, b, c, d]`` (coefficients of *ax + by + cz + d = 0*).
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    return list(plane_model), list(inliers)


# ── Ground Extraction ────────────────────────────────────────────────────────


def extract_ground(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.3,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, list]:
    """Extract ground points using RANSAC plane fitting.

    Args:
        pcd: Input point cloud.
        distance_threshold: Maximum distance to the ground plane.
        ransac_n: Number of initial points for plane estimation.
        num_iterations: RANSAC iterations.

    Returns:
        ``(ground_pcd, non_ground_pcd, plane_model)``
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    ground_pcd = pcd.select_by_index(inliers)
    non_ground_pcd = pcd.select_by_index(inliers, invert=True)
    return ground_pcd, non_ground_pcd, list(plane_model)


def segment_csf(
    pcd: o3d.geometry.PointCloud,
    *,
    cloth_resolution: float = 0.5,
    max_iterations: int = 500,
    classification_threshold: float = 0.5,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Ground segmentation using a Cloth Simulation Filter (CSF) approach.

    This is a simplified implementation that simulates draping a cloth
    over an inverted point cloud to identify ground points.

    Args:
        pcd: Input point cloud.
        cloth_resolution: Resolution of the cloth grid.
        max_iterations: Maximum simulation iterations.
        classification_threshold: Height threshold for classifying ground.

    Returns:
        ``(ground_pcd, non_ground_pcd)``
    """
    points = np.asarray(pcd.points)

    # Invert the point cloud (flip Z)
    z_max = points[:, 2].max()
    inverted_z = z_max - points[:, 2]

    # Create a grid representing the cloth
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    nx = int(np.ceil((x_max - x_min) / cloth_resolution)) + 1
    ny = int(np.ceil((y_max - y_min) / cloth_resolution)) + 1

    # Initialise cloth at maximum inverted height
    cloth_z = np.full((ny, nx), inverted_z.max())

    # Map points to grid cells
    idx_x = ((points[:, 0] - x_min) / cloth_resolution).astype(int)
    idx_y = ((points[:, 1] - y_min) / cloth_resolution).astype(int)
    idx_x = np.clip(idx_x, 0, nx - 1)
    idx_y = np.clip(idx_y, 0, ny - 1)

    # For each grid cell, find the minimum inverted Z (= maximum original Z… ground)
    for i in range(len(points)):
        r, c = idx_y[i], idx_x[i]
        if inverted_z[i] < cloth_z[r, c]:
            cloth_z[r, c] = inverted_z[i]

    # Classify: ground if point's inverted z is close to the cloth
    ground_mask = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        r, c = idx_y[i], idx_x[i]
        if abs(inverted_z[i] - cloth_z[r, c]) < classification_threshold:
            ground_mask[i] = True

    ground_indices = np.where(ground_mask)[0]
    non_ground_indices = np.where(~ground_mask)[0]

    return pcd.select_by_index(ground_indices), pcd.select_by_index(non_ground_indices)


# ── Clustering ───────────────────────────────────────────────────────────────


def cluster_dbscan(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.5,
    min_points: int = 10,
) -> Tuple[np.ndarray, int]:
    """DBSCAN clustering for object segmentation.

    Args:
        pcd: Input point cloud.
        eps: Maximum distance between two points in the same cluster.
        min_points: Minimum cluster size.

    Returns:
        ``(labels, n_clusters)`` where *labels* is a per-point array
        (``-1`` for noise) and *n_clusters* is the number of clusters found.
    """
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    n_clusters = labels.max() + 1 if len(labels) > 0 else 0
    return labels, n_clusters


def extract_clusters(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.5,
    min_points: int = 10,
    min_cluster_size: int = 100,
) -> Tuple[List[o3d.geometry.PointCloud], np.ndarray]:
    """Extract individual clusters as separate point clouds.

    Args:
        pcd: Input point cloud.
        eps: DBSCAN neighbourhood radius.
        min_points: Minimum points for a cluster seed.
        min_cluster_size: Clusters below this size are discarded.

    Returns:
        ``(clusters, labels)`` where *clusters* is a list of point clouds.
    """
    labels, n_clusters = cluster_dbscan(pcd, eps, min_points)

    clusters = []
    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        if len(indices) >= min_cluster_size:
            clusters.append(pcd.select_by_index(indices))

    return clusters, labels


# ── Semantic / Heuristic Segmentation ────────────────────────────────────────


def segment_heuristic(
    pcd: o3d.geometry.PointCloud,
    grid_size: float = 0.1,
) -> np.ndarray:
    """Simulate semantic segmentation using geometric heuristics.

    Assigns ASPRS-compatible labels based on height and normal orientation:

    - **2** — Ground
    - **3** — Low Vegetation
    - **5** — High Vegetation
    - **6** — Building
    - **1** — Unclassified / Hardscape

    Args:
        pcd: Input point cloud.
        grid_size: Resolution for normal estimation.

    Returns:
        ``np.ndarray`` of per-point integer labels.
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return np.array([], dtype=np.int32)

    labels = np.zeros(len(points), dtype=np.int32)

    # Ensure normals exist
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=grid_size * 5, max_nn=30,
            )
        )
    normals = np.asarray(pcd.normals)

    # Ground detection via RANSAC
    _, inliers = pcd.segment_plane(
        distance_threshold=0.2, ransac_n=3, num_iterations=500,
    )
    labels[inliers] = 2  # Ground

    # Height-based classification for non-ground points
    non_ground = np.where(labels == 0)[0]
    if len(non_ground) > 0:
        z_vals = points[non_ground, 2]
        rel_z = z_vals - z_vals.min()

        for local_idx, pt_idx in enumerate(non_ground):
            h = rel_z[local_idx]
            if h < 0.5:
                labels[pt_idx] = 1  # Unclassified / Hardscape
            elif h < 2.0:
                labels[pt_idx] = 3  # Low vegetation
            else:
                nz = abs(normals[pt_idx][2])
                if nz < 0.3:
                    labels[pt_idx] = 6  # Building (vertical surface)
                else:
                    labels[pt_idx] = 5  # High vegetation

    return labels


def extract_buildings(
    pcd: o3d.geometry.PointCloud,
    min_height: float = 3.0,
    normal_threshold: float = 0.3,
) -> o3d.geometry.PointCloud:
    """Extract likely building points based on height and surface normals.

    Args:
        pcd: Input point cloud.
        min_height: Minimum relative height above ground.
        normal_threshold: Maximum |nz| to be classified as a vertical
            wall surface.

    Returns:
        Point cloud containing probable building points.
    """
    labels = segment_heuristic(pcd)
    building_idx = np.where(labels == 6)[0]

    # Also include high non-ground points that might be rooftops
    points = np.asarray(pcd.points)
    if not pcd.has_normals():
        pcd.estimate_normals()

    ground_idx = np.where(labels == 2)[0]
    if len(ground_idx) > 0:
        ground_z = points[ground_idx, 2].mean()
        high_mask = (points[:, 2] - ground_z) >= min_height
        combined = np.union1d(building_idx, np.where(high_mask & (labels != 2))[0])
        return pcd.select_by_index(combined)

    return pcd.select_by_index(building_idx)


def extract_vegetation(
    pcd: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    """Extract likely vegetation points using heuristic segmentation.

    Returns points classified as low vegetation (3) or high
    vegetation (5).
    """
    labels = segment_heuristic(pcd)
    veg_idx = np.where((labels == 3) | (labels == 5))[0]
    return pcd.select_by_index(veg_idx)
