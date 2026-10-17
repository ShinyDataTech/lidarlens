"""
CRS utilities — coordinate reference system detection, transformation, and
WKT VLR creation for LAS files.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import laspy
    from laspy import VLR
except ImportError:
    laspy = None
    VLR = object

try:
    import pyproj
except ImportError:
    pyproj = None


# Default WKT for files without CRS information (WGS 84)
DEFAULT_WKT = (
    'GEOGCS["GCS_WGS_1984",'
    'DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)


def create_wkt_vlr(wkt_string: str) -> VLR:
    """Create a WKT VLR with the correct user_id and record_id for
    copc.js compatibility.

    Args:
        wkt_string: Well-Known Text string describing the CRS.

    Returns:
        A ``laspy.VLR`` instance ready to be appended to a LAS header.
    """
    wkt_bytes = wkt_string.encode("utf-8") + b"\x00"
    vlr = VLR(
        user_id="LASF_Projection",
        record_id=2112,
        description="OGC Coordinate System WKT",
        record_data=wkt_bytes,
    )
    return vlr


def transform_bounds(
    bounds: dict,
    src_crs: str,
    dst_crs: str = "EPSG:4326",
) -> List[List[float]]:
    """Transform bounding box corners between CRS.

    Args:
        bounds: Dict with keys ``x_min``, ``x_max``, ``y_min``, ``y_max``.
        src_crs: Source CRS (EPSG code, WKT, or proj4 string).
        dst_crs: Destination CRS (default WGS 84).

    Returns:
        List of ``[x, y]`` corner pairs in the destination CRS
        (order: TL, TR, BR, BL).

    Raises:
        ImportError: If *pyproj* is not installed.
    """
    if pyproj is None:
        raise ImportError("pyproj is required for CRS transformations.")

    crs_src = pyproj.CRS.from_user_input(src_crs)
    crs_dst = pyproj.CRS.from_user_input(dst_crs)
    transformer = pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    corners = [
        (bounds["x_min"], bounds["y_max"]),  # TL
        (bounds["x_max"], bounds["y_max"]),  # TR
        (bounds["x_max"], bounds["y_min"]),  # BR
        (bounds["x_min"], bounds["y_min"]),  # BL
    ]

    result = []
    for x, y in corners:
        tx, ty = transformer.transform(x, y)
        result.append([float(tx), float(ty)])
    return result


def transform_coordinates(
    coords: np.ndarray,
    src_crs: str,
    dst_crs: str = "EPSG:4326",
) -> np.ndarray:
    """Reproject an array of coordinates.

    Args:
        coords: ``(N, 2)`` or ``(N, 3)`` array of coordinates.
        src_crs: Source CRS.
        dst_crs: Destination CRS.

    Returns:
        Transformed coordinate array with the same shape.
    """
    if pyproj is None:
        raise ImportError("pyproj is required for CRS transformations.")

    crs_src = pyproj.CRS.from_user_input(src_crs)
    crs_dst = pyproj.CRS.from_user_input(dst_crs)
    transformer = pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    coords = np.asarray(coords)
    tx, ty = transformer.transform(coords[:, 0], coords[:, 1])
    result = np.column_stack([tx, ty])

    if coords.shape[1] > 2:
        result = np.column_stack([result, coords[:, 2:]])

    return result


def detect_crs(path: str) -> Optional[str]:
    """Auto-detect the CRS from a LAS/LAZ file by inspecting VLRs.

    Args:
        path: Path to the LAS/LAZ file.

    Returns:
        WKT string if found, ``None`` otherwise.
    """
    with laspy.open(path) as f:
        for vlr in f.header.vlrs:
            uid = getattr(vlr, "user_id", "")
            rid = getattr(vlr, "record_id", 0)
            if uid == "LASF_Projection" and rid == 2112:
                try:
                    return vlr.record_data.decode("utf-8").strip("\x00")
                except Exception:
                    pass
            desc = getattr(vlr, "description", "")
            if "WKT" in desc:
                try:
                    return vlr.record_data.decode("utf-8").strip("\x00")
                except Exception:
                    pass
    return None
