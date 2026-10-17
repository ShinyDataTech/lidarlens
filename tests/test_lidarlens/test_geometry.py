"""Tests for lidarlens.geometry — normals, mesh, transforms, ICP, hull."""

import numpy as np
import pytest


def test_estimate_normals(simple_pcd):
    """Normal estimation should attach normals."""
    from lidarlens.geometry import estimate_normals

    result = estimate_normals(simple_pcd)
    assert result.has_normals()


def test_generate_mesh_poisson(simple_pcd):
    """Poisson mesh should produce a valid mesh."""
    from lidarlens.geometry import generate_mesh

    mesh = generate_mesh(simple_pcd, strategy="poisson", depth=5)
    assert mesh is not None
    assert len(mesh.triangles) > 0


def test_translate(simple_pcd):
    """Translation should shift centroid."""
    from lidarlens.geometry import translate

    center_before = np.asarray(simple_pcd.points).mean(axis=0)
    result = translate(simple_pcd, [10, 20, 30])
    center_after = np.asarray(result.points).mean(axis=0)

    np.testing.assert_allclose(
        center_after - center_before, [10, 20, 30], atol=0.01,
    )


def test_scale(simple_pcd):
    """Scaling by 2 should double the extent."""
    from lidarlens.geometry import scale

    pts_before = np.asarray(simple_pcd.points)
    extent_before = pts_before.max(axis=0) - pts_before.min(axis=0)

    center = simple_pcd.get_center().tolist()
    result = scale(simple_pcd, 2.0, center=center)
    pts_after = np.asarray(result.points)
    extent_after = pts_after.max(axis=0) - pts_after.min(axis=0)

    np.testing.assert_allclose(extent_after, extent_before * 2, atol=0.01)


def test_register_icp():
    """ICP should align a translated cloud to the original."""
    import pytest
    o3d = pytest.importorskip('open3d')

    np.random.seed(42)
    pts = np.random.rand(200, 3)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pts + [1, 0, 0])
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pts)

    transform, fitness = register_icp(source, target, max_correspondence_distance=2.0)
    assert fitness > 0.5
    assert transform.shape == (4, 4)


def test_convex_hull(simple_pcd):
    """Convex hull should return a mesh and indices."""
    from lidarlens.geometry import compute_convex_hull

    mesh, indices = compute_convex_hull(simple_pcd)
    assert len(mesh.triangles) > 0
    assert len(indices) > 0


def test_voxelize(simple_pcd):
    """Voxelization should return a VoxelGrid."""
    from lidarlens.geometry import voxelize
    import pytest
    o3d = pytest.importorskip('open3d')

    vg = voxelize(simple_pcd, voxel_size=0.2)
    assert isinstance(vg, o3d.geometry.VoxelGrid)
