"""Tests for lidarlens.crs — CRS utilities."""

import numpy as np
import pytest


def test_create_wkt_vlr():
    """WKT VLR should have correct user_id, record_id, and data."""
    pytest.importorskip("laspy")
    from lidarlens.crs import create_wkt_vlr

    wkt = 'GEOGCS["GCS_WGS_1984"]'
    vlr = create_wkt_vlr(wkt)
    assert vlr.user_id == "LASF_Projection"
    assert vlr.record_id == 2112
    assert wkt.encode("utf-8") in vlr.record_data


def test_transform_bounds():
    """Bounding box transform should return 4 corners."""
    pytest.importorskip("pyproj")
    from lidarlens.crs import transform_bounds

    bounds = {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100}
    result = transform_bounds(bounds, "EPSG:32601", "EPSG:4326")
    assert len(result) == 4
    for corner in result:
        assert len(corner) == 2


def test_transform_coordinates():
    """Coordinate transform should change values."""
    pytest.importorskip("pyproj")
    from lidarlens.crs import transform_coordinates

    coords = np.array([[500000, 0, 100]])
    result = transform_coordinates(coords, "EPSG:32601", "EPSG:4326")
    assert result.shape == coords.shape
    # Should be in geographic coordinates now
    assert abs(result[0, 0]) < 360


def test_detect_crs(tmp_las_file):
    """detect_crs should return None for a file without CRS VLR."""
    from lidarlens.crs import detect_crs

    # tmp_las_file has no WKT VLR
    result = detect_crs(tmp_las_file)
    assert result is None
