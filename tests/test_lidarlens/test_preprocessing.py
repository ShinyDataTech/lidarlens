"""Tests for lidarlens.preprocessing — downsampling and noise removal."""

import numpy as np
import pytest
o3d = pytest.importorskip('open3d')
import pytest


def test_voxel_downsample(simple_pcd):
    """Voxel downsampling should reduce point count."""
    from lidarlens.preprocessing import voxel_downsample

    result = voxel_downsample(simple_pcd, voxel_size=0.2)
    assert len(result.points) < len(simple_pcd.points)
    assert len(result.points) > 0


def test_uniform_downsample(simple_pcd):
    """Uniform downsampling keeps every k-th point."""
    from lidarlens.preprocessing import uniform_downsample

    result = uniform_downsample(simple_pcd, every_k_points=5)
    assert len(result.points) == len(simple_pcd.points) // 5


def test_random_downsample(simple_pcd):
    """Random downsampling should hit target count exactly."""
    from lidarlens.preprocessing import random_downsample

    target = 200
    result = random_downsample(simple_pcd, target_points=target)
    assert len(result.points) == target


def test_random_downsample_noop(simple_pcd):
    """If target >= n_points, nothing changes."""
    from lidarlens.preprocessing import random_downsample

    result = random_downsample(simple_pcd, target_points=99999)
    assert len(result.points) == len(simple_pcd.points)


def test_remove_statistical_outliers(simple_pcd):
    """Should return a clean cloud + indices."""
    from lidarlens.preprocessing import remove_statistical_outliers

    clean, indices = remove_statistical_outliers(simple_pcd, nb_neighbors=20, std_ratio=2.0)
    assert len(clean.points) <= len(simple_pcd.points)
    assert len(clean.points) > 0


def test_denoise_statistical(simple_pcd):
    """Unified denoise interface with statistical method."""
    from lidarlens.preprocessing import denoise

    result = denoise(simple_pcd, method="statistical")
    assert len(result.points) > 0


def test_denoise_invalid_method(simple_pcd):
    """Should raise ValueError for unknown method."""
    from lidarlens.preprocessing import denoise

    with pytest.raises(ValueError):
        denoise(simple_pcd, method="magic")
