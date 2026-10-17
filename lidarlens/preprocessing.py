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
Preprocessing module — downsampling and noise removal.
"""


from typing import Tuple

import numpy as np
# ── Downsampling ─────────────────────────────────────────────────────────────


def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.1,
) -> o3d.geometry.PointCloud:
    """Voxel-grid downsampling — replaces points within each voxel with the
    centroid.

    Args:
        pcd: Input point cloud.
        voxel_size: Edge length of each voxel cube.

    Returns:
        Down-sampled point cloud.
    """
    return pcd.voxel_down_sample(voxel_size)


def uniform_downsample(
    pcd: o3d.geometry.PointCloud,
    every_k_points: int = 5,
) -> o3d.geometry.PointCloud:
    """Uniform downsampling — keeps every *k*-th point.

    Args:
        pcd: Input point cloud.
        every_k_points: Sampling interval.

    Returns:
        Down-sampled point cloud.
    """
    return pcd.uniform_down_sample(every_k_points)


def random_downsample(
    pcd: o3d.geometry.PointCloud,
    target_points: int = 100_000,
) -> o3d.geometry.PointCloud:
    """Random downsampling to a target number of points.

    If the cloud already has fewer points than *target_points* it is
    returned unchanged.

    Args:
        pcd: Input point cloud.
        target_points: Desired number of output points.

    Returns:
        Down-sampled point cloud.
    """
    n = len(np.asarray(pcd.points))
    if n <= target_points:
        return pcd
    indices = np.random.choice(n, target_points, replace=False)
    return pcd.select_by_index(indices)


# ── Noise Removal ────────────────────────────────────────────────────────────


def remove_statistical_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> Tuple[o3d.geometry.PointCloud, list]:
    """Statistical outlier removal — removes points that are far from their
    neighbours.

    Args:
        pcd: Input point cloud.
        nb_neighbors: Number of nearest neighbours to evaluate.
        std_ratio: Standard-deviation multiplier threshold.

    Returns:
        ``(clean_pcd, inlier_indices)``.
    """
    return pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )


def remove_radius_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_points: int = 16,
    radius: float = 0.5,
) -> Tuple[o3d.geometry.PointCloud, list]:
    """Radius outlier removal — removes points with few neighbours within a
    given radius.

    Args:
        pcd: Input point cloud.
        nb_points: Minimum number of neighbours required.
        radius: Search radius.

    Returns:
        ``(clean_pcd, inlier_indices)``.
    """
    return pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)


def denoise(
    pcd: o3d.geometry.PointCloud,
    method: str = "statistical",
    **kwargs,
) -> o3d.geometry.PointCloud:
    """Unified denoising interface.

    Args:
        pcd: Input point cloud.
        method: ``"statistical"`` or ``"radius"``.
        **kwargs: Forwarded to the underlying removal function.

    Returns:
        Cleaned point cloud (indices discarded for convenience).
    """
    if method == "statistical":
        clean, _ = remove_statistical_outliers(pcd, **kwargs)
    elif method == "radius":
        clean, _ = remove_radius_outliers(pcd, **kwargs)
    else:
        raise ValueError(f"Unknown denoise method: {method!r}. Use 'statistical' or 'radius'.")
    return clean
