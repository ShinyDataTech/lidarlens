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
Filtering module — spatial, elevation, classification, and return filters.
"""


from typing import List, Optional

import numpy as np
try:
    from matplotlib.path import Path as mplPath
except ImportError:
    mplPath = None


def crop_by_bounds(
    pcd: o3d.geometry.PointCloud,
    min_bound: List[float],
    max_bound: List[float],
) -> o3d.geometry.PointCloud:
    """Crop a point cloud by an axis-aligned bounding box.

    Args:
        pcd: Input point cloud.
        min_bound: ``[min_x, min_y, min_z]``.
        max_bound: ``[max_x, max_y, max_z]``.

    Returns:
        Cropped point cloud.
    """
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array(min_bound),
        max_bound=np.array(max_bound),
    )
    return pcd.crop(bbox)


def filter_by_elevation(
    pcd: o3d.geometry.PointCloud,
    min_z: Optional[float] = None,
    max_z: Optional[float] = None,
) -> o3d.geometry.PointCloud:
    """Filter points by elevation (z-value) range.

    Args:
        pcd: Input point cloud.
        min_z: Minimum elevation (inclusive). ``None`` for no lower bound.
        max_z: Maximum elevation (inclusive). ``None`` for no upper bound.

    Returns:
        Filtered point cloud.
    """
    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)

    if min_z is not None:
        mask &= points[:, 2] >= min_z
    if max_z is not None:
        mask &= points[:, 2] <= max_z

    return pcd.select_by_index(np.where(mask)[0])


def filter_by_classification(
    pcd: o3d.geometry.PointCloud,
    classes: List[int],
    las_path: str,
    *,
    invert: bool = False,
) -> o3d.geometry.PointCloud:
    """Keep or remove points by ASPRS classification code.

    This reads classification data directly from the LAS file (since
    Open3D point clouds don't carry classification).

    Args:
        pcd: The point cloud (must match the LAS file point count).
        classes: List of ASPRS class codes to keep (or remove if
            *invert* is True).
        las_path: Path to the source LAS/LAZ file.
        invert: If True, *remove* the listed classes instead.

    Returns:
        Filtered point cloud.
    """
    with laspy.open(las_path) as f:
        las = f.read()

    class_array = np.array(las.classification)
    mask = np.isin(class_array, classes)

    if invert:
        mask = ~mask

    return pcd.select_by_index(np.where(mask)[0])


def filter_by_return_number(
    pcd: o3d.geometry.PointCloud,
    returns: List[int],
    las_path: str,
) -> o3d.geometry.PointCloud:
    """Filter points by return number.

    Args:
        pcd: The point cloud (must match the LAS file point count).
        returns: List of return numbers to keep (e.g. ``[1]`` for first
            returns only).
        las_path: Path to the source LAS/LAZ file.

    Returns:
        Filtered point cloud.
    """
    with laspy.open(las_path) as f:
        las = f.read()

    return_num = np.array(las.return_number)
    mask = np.isin(return_num, returns)
    return pcd.select_by_index(np.where(mask)[0])


def filter_by_intensity(
    pcd: o3d.geometry.PointCloud,
    las_path: str,
    min_intensity: Optional[float] = None,
    max_intensity: Optional[float] = None,
) -> o3d.geometry.PointCloud:
    """Filter points by intensity value.

    Args:
        pcd: The point cloud (must match the LAS file point count).
        las_path: Path to the source LAS/LAZ file.
        min_intensity: Minimum intensity (inclusive).
        max_intensity: Maximum intensity (inclusive).

    Returns:
        Filtered point cloud.
    """
    with laspy.open(las_path) as f:
        las = f.read()

    intensity = np.array(las.intensity, dtype=np.float64)
    mask = np.ones(len(intensity), dtype=bool)

    if min_intensity is not None:
        mask &= intensity >= min_intensity
    if max_intensity is not None:
        mask &= intensity <= max_intensity

    return pcd.select_by_index(np.where(mask)[0])


def crop_by_polygon(
    pcd: o3d.geometry.PointCloud,
    polygon: List[List[float]],
) -> o3d.geometry.PointCloud:
    """Crop a point cloud using a 2D polygon (X/Y plane).

    Args:
        pcd: Input point cloud.
        polygon: List of ``[x, y]`` vertices defining the polygon
            boundary. The polygon is automatically closed.

    Returns:
        Points that lie inside the polygon.
    """
    points = np.asarray(pcd.points)
    path = mplPath(np.array(polygon))

    # Fast bbox pre-filter
    poly_arr = np.array(polygon)
    min_xy = poly_arr.min(axis=0)
    max_xy = poly_arr.max(axis=0)
    bbox_mask = (
        (points[:, 0] >= min_xy[0])
        & (points[:, 0] <= max_xy[0])
        & (points[:, 1] >= min_xy[1])
        & (points[:, 1] <= max_xy[1])
    )

    candidates = np.where(bbox_mask)[0]
    if len(candidates) == 0:
        return pcd.select_by_index([])

    xy = points[candidates, :2]
    inside = path.contains_points(xy)
    final = candidates[inside]

    return pcd.select_by_index(final)
