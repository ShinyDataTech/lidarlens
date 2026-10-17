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
Analysis module — statistics, histograms, cloud-to-cloud comparison,
density analysis, roughness, cross-section, and volume computation.
"""


from typing import Dict, List, Optional, Tuple

import numpy as np
# ── Statistics ───────────────────────────────────────────────────────────────


def compute_statistics(pcd: o3d.geometry.PointCloud) -> Dict:
    """Compute comprehensive statistics for a point cloud (EDA).

    Returns a dictionary with:

    - ``point_count``, ``bounds``, ``centroid``, ``std_dev``
    - ``elevation`` stats (mean, median, std, range)
    - ``density`` info (width, height, area, points_per_unit)
    - ``has_colors``, ``has_normals``
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return {"error": "Empty point cloud"}

    width = float(points[:, 0].max() - points[:, 0].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    area = width * height

    stats = {
        "point_count": len(points),
        "bounds": {
            "min_x": float(points[:, 0].min()),
            "max_x": float(points[:, 0].max()),
            "min_y": float(points[:, 1].min()),
            "max_y": float(points[:, 1].max()),
            "min_z": float(points[:, 2].min()),
            "max_z": float(points[:, 2].max()),
        },
        "centroid": {
            "x": float(points[:, 0].mean()),
            "y": float(points[:, 1].mean()),
            "z": float(points[:, 2].mean()),
        },
        "std_dev": {
            "x": float(points[:, 0].std()),
            "y": float(points[:, 1].std()),
            "z": float(points[:, 2].std()),
        },
        "elevation": {
            "mean": float(points[:, 2].mean()),
            "median": float(np.median(points[:, 2])),
            "std": float(points[:, 2].std()),
            "range": float(points[:, 2].max() - points[:, 2].min()),
        },
        "density": {
            "width": width,
            "height": height,
            "area": area,
            "points_per_unit": float(len(points) / area) if area > 0 else 0.0,
        },
        "has_colors": pcd.has_colors(),
        "has_normals": pcd.has_normals(),
    }
    return stats


def elevation_histogram(
    pcd: o3d.geometry.PointCloud,
    bins: int = 50,
) -> Dict:
    """Compute elevation histogram for visualisation.

    Returns:
        Dict with ``counts``, ``bin_edges``, ``min_z``, ``max_z``.
    """
    z = np.asarray(pcd.points)[:, 2]
    counts, edges = np.histogram(z, bins=bins)
    return {
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
        "min_z": float(z.min()),
        "max_z": float(z.max()),
    }


# ── Cloud Comparison ─────────────────────────────────────────────────────────


def compare_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, Dict]:
    """Compare two point clouds by computing per-point nearest distances.

    For each point in *source*, the nearest neighbour in *target* is
    found and the Euclidean distance recorded.

    Returns:
        ``(result_pcd, distances, stats)``

        - *result_pcd* — source cloud coloured by distance heatmap.
        - *distances* — per-point distance array.
        - *stats* — min, max, mean, std, median, rmse, point counts.
    """
    distances = np.asarray(source.compute_point_cloud_distance(target))

    stats = {
        "min_distance": float(distances.min()),
        "max_distance": float(distances.max()),
        "mean_distance": float(distances.mean()),
        "std_distance": float(distances.std()),
        "median_distance": float(np.median(distances)),
        "rmse": float(np.sqrt(np.mean(distances ** 2))),
        "source_points": len(np.asarray(source.points)),
        "target_points": len(np.asarray(target.points)),
    }

    # Heatmap: blue (close) → red (far)
    d_min, d_max = distances.min(), distances.max()
    d_range = d_max - d_min if d_max > d_min else 1.0
    norm = (distances - d_min) / d_range

    colors = np.zeros((len(distances), 3))
    colors[:, 0] = norm
    colors[:, 1] = 1.0 - norm
    colors[:, 2] = 1.0 - norm

    result = o3d.geometry.PointCloud(source)
    result.colors = o3d.utility.Vector3dVector(colors)

    return result, distances, stats


# ── Density Analysis ─────────────────────────────────────────────────────────


def point_density(
    pcd: o3d.geometry.PointCloud,
    resolution: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """Compute a 2-D point density heat-map.

    Args:
        pcd: Input point cloud.
        resolution: Grid cell size.

    Returns:
        ``(density_grid, metadata)`` — 2-D array of point counts per cell
        and a metadata dict.
    """
    points = np.asarray(pcd.points)
    x, y = points[:, 0], points[:, 1]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    width = max(1, int(np.ceil((x_max - x_min) / resolution)))
    height = max(1, int(np.ceil((y_max - y_min) / resolution)))

    grid = np.zeros((height, width), dtype=np.int32)

    idx_x = np.clip(((x - x_min) / resolution).astype(int), 0, width - 1)
    idx_y = np.clip(((y - y_min) / resolution).astype(int), 0, height - 1)

    for i in range(len(x)):
        grid[idx_y[i], idx_x[i]] += 1

    metadata = {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "resolution": resolution,
        "width": width, "height": height,
        "max_density": int(grid.max()),
        "mean_density": float(grid.mean()),
    }
    return grid, metadata


# ── Roughness ────────────────────────────────────────────────────────────────


def roughness(
    pcd: o3d.geometry.PointCloud,
    radius: float = 0.5,
) -> np.ndarray:
    """Compute per-point surface roughness.

    Roughness is defined as the distance from each point to the best-fit
    plane of its local neighbourhood.

    Args:
        pcd: Input point cloud.
        radius: Neighbourhood search radius.

    Returns:
        1-D array of roughness values (one per point).
    """
    points = np.asarray(pcd.points)
    n = len(points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    rough = np.zeros(n)

    for i in range(n):
        _, idx, _ = tree.search_radius_vector_3d(pcd.points[i], radius)
        if len(idx) < 3:
            rough[i] = 0.0
            continue

        neighbours = points[idx]
        centroid = neighbours.mean(axis=0)
        centred = neighbours - centroid

        # PCA — smallest eigenvalue direction is the plane normal
        cov = centred.T @ centred
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # smallest eigenvalue

        # Distance from point to the fitted plane
        rough[i] = abs(np.dot(points[i] - centroid, normal))

    return rough


# ── Cross-Section ────────────────────────────────────────────────────────────


def cross_section(
    pcd: o3d.geometry.PointCloud,
    line_start: List[float],
    line_end: List[float],
    width: float = 1.0,
) -> Dict:
    """Extract a 2-D elevation profile along a line.

    Args:
        pcd: Input point cloud.
        line_start: ``[x, y]`` start of the profile line.
        line_end: ``[x, y]`` end of the profile line.
        width: Swath half-width perpendicular to the line.

    Returns:
        Dictionary with ``distance`` and ``elevation`` arrays, and line
        metadata.
    """
    points = np.asarray(pcd.points)
    start = np.array(line_start[:2])
    end = np.array(line_end[:2])

    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        return {"distance": [], "elevation": [], "length": 0.0}

    unit = direction / length
    normal = np.array([-unit[1], unit[0]])

    # Project all points onto the line
    relative = points[:, :2] - start
    along = relative @ unit
    across = relative @ normal

    # Filter by swath width and line extent
    mask = (np.abs(across) <= width) & (along >= 0) & (along <= length)
    indices = np.where(mask)[0]

    distances = along[indices]
    elevations = points[indices, 2]

    # Sort by distance
    order = np.argsort(distances)
    return {
        "distance": distances[order].tolist(),
        "elevation": elevations[order].tolist(),
        "length": float(length),
        "n_points": len(indices),
    }


# ── Volume Computation ───────────────────────────────────────────────────────


def compute_volume(
    pcd: o3d.geometry.PointCloud,
    reference_z: Optional[float] = None,
    resolution: float = 1.0,
) -> Dict:
    """Estimate the volume of a point cloud above a reference surface.

    Uses a simple grid-based approach: for each cell, the volume is
    ``(max_z − reference_z) × cell_area``.

    Args:
        pcd: Input point cloud.
        reference_z: Reference elevation. Defaults to the minimum Z of
            the cloud.
        resolution: Grid cell size.

    Returns:
        Dictionary with ``volume``, ``area``, ``reference_z``,
        ``mean_height``.
    """
    points = np.asarray(pcd.points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    if reference_z is None:
        reference_z = float(z.min())

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    width = max(1, int(np.ceil((x_max - x_min) / resolution)))
    height_cells = max(1, int(np.ceil((y_max - y_min) / resolution)))

    grid = np.full((height_cells, width), reference_z)

    idx_x = np.clip(((x - x_min) / resolution).astype(int), 0, width - 1)
    idx_y = np.clip(((y - y_min) / resolution).astype(int), 0, height_cells - 1)

    for i in range(len(z)):
        r, c = idx_y[i], idx_x[i]
        if z[i] > grid[r, c]:
            grid[r, c] = z[i]

    cell_area = resolution * resolution
    heights = np.maximum(grid - reference_z, 0.0)
    total_volume = float(heights.sum() * cell_area)
    filled_cells = np.count_nonzero(heights > 0)

    return {
        "volume": total_volume,
        "area": float(filled_cells * cell_area),
        "reference_z": reference_z,
        "mean_height": float(heights[heights > 0].mean()) if filled_cells > 0 else 0.0,
    }
