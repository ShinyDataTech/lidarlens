"""Tests for lidarlens.io — read/write LAS files."""

import numpy as np
import pytest
o3d = pytest.importorskip('open3d')
import pytest


def test_read_write_roundtrip(simple_pcd, tmp_path):
    """Write a point cloud then read it back — points should be close."""
    from lidarlens.io import read, save

    out_path = str(tmp_path / "roundtrip.laz")
    save(simple_pcd, out_path)

    pcd_out, header, vlrs, wkt = read(out_path)
    pts_in = np.asarray(simple_pcd.points)
    pts_out = np.asarray(pcd_out.points)

    assert len(pts_out) == len(pts_in)
    # Allow small rounding due to LAS quantisation
    np.testing.assert_allclose(pts_out, pts_in, atol=0.01)


def test_read_header(tmp_las_file):
    """read_header should return metadata without loading points."""
    from lidarlens.io import read_header

    h = read_header(tmp_las_file)
    assert h["point_count"] == 1000
    assert "version" in h
    assert "scales" in h


def test_read_bounds(tmp_las_file):
    """read_bounds should return spatial extents."""
    from lidarlens.io import read_bounds

    b = read_bounds(tmp_las_file)
    assert b["min_x"] <= b["max_x"]
    assert b["min_y"] <= b["max_y"]
    assert b["min_z"] <= b["max_z"]


def test_save_with_colors(colored_pcd, tmp_path):
    """Colours should survive save/read roundtrip."""
    from lidarlens.io import read, save

    out = str(tmp_path / "colored.laz")
    save(colored_pcd, out)
    pcd_out, _, _, _ = read(out)
    assert pcd_out.has_colors()


def test_save_with_extra_dims(simple_pcd, tmp_path):
    """Extra dimensions (ClusterID) should be written without error."""
    from lidarlens.io import save

    out = str(tmp_path / "extra.laz")
    n = len(simple_pcd.points)
    extra = {"ClusterID": np.arange(n, dtype=np.int32)}
    save(simple_pcd, out, extra_dims=extra)

    import laspy
    with laspy.open(out) as f:
        las = f.read()
    assert hasattr(las, "ClusterID")
