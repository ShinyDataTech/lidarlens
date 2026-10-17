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
Geometry module — normal estimation, mesh generation, transformations,
registration, and convex hull.
"""


from typing import List, Optional, Tuple

import numpy as np
# ── Normals ──────────────────────────────────────────────────────────────────


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float = 0.5,
    max_nn: int = 30,
) -> o3d.geometry.PointCloud:
    """Estimate point normals using local neighbourhood.

    Args:
        pcd: Input point cloud (modified in-place and returned).
        radius: Search radius for the hybrid KD-tree.
        max_nn: Maximum number of nearest neighbours.

    Returns:
        The same point cloud with normals attached.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn,
        )
    )
    return pcd


def orient_normals(
    pcd: o3d.geometry.PointCloud,
    k: int = 15,
) -> o3d.geometry.PointCloud:
    """Orient normals to be consistent (pointing outward).

    Args:
        pcd: Point cloud with normals.
        k: Number of neighbours for tangent-plane estimation.

    Returns:
        The same point cloud with oriented normals.
    """
    pcd.orient_normals_consistent_tangent_plane(k=k)
    return pcd


# ── Surface Reconstruction ───────────────────────────────────────────────────


def generate_mesh(
    pcd: o3d.geometry.PointCloud,
    strategy: str = "poisson",
    depth: int = 9,
) -> Optional[o3d.geometry.TriangleMesh]:
    """Generate a triangle mesh from a point cloud.

    Args:
        pcd: Input point cloud.
        strategy: ``"poisson"`` or ``"ball_pivoting"``.
        depth: Octree depth (Poisson only).

    Returns:
        An ``open3d.geometry.TriangleMesh``, or ``None`` on failure.
    """
    if not pcd.has_normals():
        pcd.estimate_normals()

    if strategy == "poisson":
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth,
        )
        return mesh

    elif strategy == "ball_pivoting":
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * 0.5, avg_dist, avg_dist * 2]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii),
        )
        return mesh

    return None


# ── Transformations ──────────────────────────────────────────────────────────


def translate(
    pcd: o3d.geometry.PointCloud,
    translation: List[float],
) -> o3d.geometry.PointCloud:
    """Translate a point cloud by ``[tx, ty, tz]``."""
    return pcd.translate(np.array(translation))


def scale(
    pcd: o3d.geometry.PointCloud,
    scale_factor: float,
    center: Optional[List[float]] = None,
) -> o3d.geometry.PointCloud:
    """Scale a point cloud by a factor around a centre point.

    Args:
        pcd: Input point cloud.
        scale_factor: Uniform scale factor.
        center: Centre of scaling. Defaults to cloud centroid.
    """
    c = np.array(center) if center is not None else pcd.get_center()
    return pcd.scale(scale_factor, c)


def rotate(
    pcd: o3d.geometry.PointCloud,
    rotation_matrix: List[List[float]],
    center: Optional[List[float]] = None,
) -> o3d.geometry.PointCloud:
    """Rotate a point cloud using a 3×3 rotation matrix.

    Args:
        pcd: Input point cloud.
        rotation_matrix: 3×3 rotation matrix.
        center: Centre of rotation. Defaults to cloud centroid.
    """
    c = np.array(center) if center is not None else pcd.get_center()
    return pcd.rotate(np.array(rotation_matrix), c)


# ── Registration ─────────────────────────────────────────────────────────────


def register_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 1.0,
    init_transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Iterative Closest Point (ICP) registration.

    Args:
        source: Source point cloud to align.
        target: Target (reference) point cloud.
        max_correspondence_distance: Maximum pairing distance.
        init_transform: 4×4 initial transformation matrix.
            Defaults to identity.

    Returns:
        ``(transformation, fitness)`` — the 4×4 transformation matrix and
        the fitness score (fraction of inlier correspondences).
    """
    if init_transform is None:
        init_transform = np.eye(4)

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return result.transformation, result.fitness


# ── Hull & Voxel ─────────────────────────────────────────────────────────────


def compute_convex_hull(
    pcd: o3d.geometry.PointCloud,
) -> Tuple[o3d.geometry.TriangleMesh, list]:
    """Compute the 3-D convex hull of a point cloud.

    Returns:
        ``(hull_mesh, hull_point_indices)``
    """
    mesh, indices = pcd.compute_convex_hull()
    return mesh, list(indices)


def voxelize(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.5,
) -> o3d.geometry.VoxelGrid:
    """Create a voxel-grid representation of the point cloud.

    Args:
        pcd: Input point cloud.
        voxel_size: Size of each voxel.

    Returns:
        An ``open3d.geometry.VoxelGrid``.
    """
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
