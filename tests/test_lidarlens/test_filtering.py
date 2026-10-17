"""Tests for lidarlens.filtering — spatial and attribute filters."""

import numpy as np
import pytest
o3d = pytest.importorskip('open3d')
import pytest


def test_crop_by_bounds(simple_pcd):
    """Cropping should return points within the bounding box."""
    from lidarlens.filtering import crop_by_bounds

    result = crop_by_bounds(simple_pcd, [0.2, 0.2, 0.2], [0.8, 0.8, 0.8])
    pts = np.asarray(result.points)
    assert len(pts) > 0
    assert len(pts) < len(simple_pcd.points)
    assert np.all(pts >= 0.2 - 1e-6)
    assert np.all(pts <= 0.8 + 1e-6)


def test_filter_by_elevation(simple_pcd):
    """Elevation filter should limit Z range."""
    from lidarlens.filtering import filter_by_elevation

    result = filter_by_elevation(simple_pcd, min_z=0.3, max_z=0.7)
    pts = np.asarray(result.points)
    assert len(pts) > 0
    assert np.all(pts[:, 2] >= 0.3 - 1e-6)
    assert np.all(pts[:, 2] <= 0.7 + 1e-6)


def test_crop_by_polygon(simple_pcd):
    """Polygon crop should keep only inside points."""
    from lidarlens.filtering import crop_by_polygon

    polygon = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    result = crop_by_polygon(simple_pcd, polygon)
    pts = np.asarray(result.points)
    assert len(pts) > 0
    assert len(pts) < len(simple_pcd.points)


def test_filter_by_classification(tmp_classified_las):
    """Classification filter should only keep specified classes."""
    from lidarlens.io import read
    from lidarlens.filtering import filter_by_classification

    pcd, _, _, _ = read(tmp_classified_las)
    result = filter_by_classification(pcd, [2], tmp_classified_las)
    assert len(result.points) > 0
    assert len(result.points) < len(pcd.points)
