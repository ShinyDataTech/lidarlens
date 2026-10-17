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
Classification module — ASPRS class definitions, polygon-based
reclassification, and class distribution analysis.
"""

import os
import uuid
from typing import Dict, List, Optional, Tuple

try:
    import laspy
except ImportError:
    laspy = None

try:
    from matplotlib.path import Path as mplPath
except ImportError:
    mplPath = None

try:
    import pyproj
except ImportError:
    pyproj = None


# ── ASPRS Standard Classes ───────────────────────────────────────────────────

ASPRS_CLASSES: Dict[int, str] = {
    0: "Created, never classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (Noise)",
    8: "Model Key-point",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Overlap Points",
    13: "Wire – Guard",
    14: "Wire – Conductor",
    15: "Transmission Tower",
    16: "Wire-structure Connector",
    17: "Bridge Deck",
    18: "High Noise",
}


# ── Reclassification ────────────────────────────────────────────────────────


def reclassify_in_polygon(
    file_path: str,
    polygon_wgs84: List[List[float]],
    target_class: int,
    *,
    source_class: Optional[int] = None,
    wkt: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Reclassify points within a 2-D polygon to a target ASPRS class.

    Args:
        file_path: Path to the source LAS/LAZ file.
        polygon_wgs84: List of ``[lon, lat]`` vertices (WGS 84).
        target_class: Target ASPRS class code.
        source_class: If given, only reclassify points currently in this
            class.
        wkt: CRS of the file (for polygon reprojection). If ``None`` the
            function attempts to read it from the file VLRs.
        output_path: Destination path for the modified file. If ``None``
            a new file is created alongside the original.

    Returns:
        Path to the saved (reclassified) file.
    """
    with laspy.open(file_path) as f:
        header = f.header
        vlrs = list(f.header.vlrs)
        las = f.read()

    # Detect CRS from file VLRs
    file_wkt = None
    for vlr in vlrs:
        rid = getattr(vlr, "record_id", 0)
        desc = getattr(vlr, "description", "")
        if rid == 2112 or "WKT" in desc:
            try:
                file_wkt = vlr.record_data.decode("utf-8").strip("\x00")
            except Exception:
                pass
    wkt_to_use = wkt if wkt else file_wkt

    # Transform polygon to file CRS
    poly_coords = np.array(polygon_wgs84)
    if wkt_to_use and pyproj is not None:
        try:
            crs_dst = pyproj.CRS.from_user_input(wkt_to_use)
            if not crs_dst.is_geographic:
                crs_src = pyproj.CRS.from_epsg(4326)
                transformer = pyproj.Transformer.from_crs(
                    crs_src, crs_dst, always_xy=True,
                )
                tx, ty = transformer.transform(poly_coords[:, 0], poly_coords[:, 1])
                poly_coords = np.column_stack((tx, ty))
        except Exception:
            pass

    path = mplPath(poly_coords)

    # BBox pre-filter
    min_xy = poly_coords.min(axis=0)
    max_xy = poly_coords.max(axis=0)
    x, y = np.array(las.x), np.array(las.y)
    bbox_mask = (x >= min_xy[0]) & (x <= max_xy[0]) & (y >= min_xy[1]) & (y <= max_xy[1])

    candidates = np.where(bbox_mask)[0]
    if len(candidates) > 0:
        xy = np.column_stack((x[candidates], y[candidates]))
        inside = path.contains_points(xy)
        final_indices = candidates[inside]

        classes = np.array(las.classification)
        if source_class is not None:
            final_indices = final_indices[classes[final_indices] == source_class]

        if len(final_indices) > 0:
            classes[final_indices] = target_class
            las.classification = classes

    if output_path is None:
        base = os.path.dirname(file_path)
        name = f"reclassified_{uuid.uuid4().hex[:8]}_{os.path.basename(file_path)}"
        output_path = os.path.join(base, name)

    las.write(output_path)
    return output_path


def reclassify_by_elevation(
    file_path: str,
    ranges: List[Tuple[float, float, int]],
    *,
    output_path: Optional[str] = None,
) -> str:
    """Reclassify points based on elevation ranges.

    Args:
        file_path: Path to the LAS/LAZ file.
        ranges: List of ``(min_z, max_z, target_class)`` tuples.
        output_path: Destination path. Auto-generated if ``None``.

    Returns:
        Path to the saved file.
    """
    with laspy.open(file_path) as f:
        las = f.read()

    z = np.array(las.z)
    classes = np.array(las.classification)

    for zmin, zmax, cls in ranges:
        mask = (z >= zmin) & (z <= zmax)
        classes[mask] = cls

    las.classification = classes

    if output_path is None:
        base = os.path.dirname(file_path)
        name = f"reclass_elev_{uuid.uuid4().hex[:8]}_{os.path.basename(file_path)}"
        output_path = os.path.join(base, name)

    las.write(output_path)
    return output_path


# ── Class Analysis ───────────────────────────────────────────────────────────


def get_class_distribution(file_path: str) -> Dict[int, Dict]:
    """Get the distribution of ASPRS classes in a LAS/LAZ file.

    Returns:
        Dict mapping class code → ``{name, count, percentage}``.
    """
    with laspy.open(file_path) as f:
        las = f.read()

    classes = np.array(las.classification)
    total = len(classes)
    unique, counts = np.unique(classes, return_counts=True)

    result = {}
    for cls, cnt in zip(unique, counts):
        result[int(cls)] = {
            "name": ASPRS_CLASSES.get(int(cls), f"User Defined ({cls})"),
            "count": int(cnt),
            "percentage": round(float(cnt) / total * 100, 2) if total > 0 else 0.0,
        }
    return result


def merge_classes(
    file_path: str,
    mapping: Dict[int, int],
    *,
    output_path: Optional[str] = None,
) -> str:
    """Merge classification codes according to a mapping.

    Args:
        file_path: Path to the LAS/LAZ file.
        mapping: ``{old_class: new_class}`` mapping.
        output_path: Destination path. Auto-generated if ``None``.

    Returns:
        Path to the saved file.
    """
    with laspy.open(file_path) as f:
        las = f.read()

    classes = np.array(las.classification)
    for old, new in mapping.items():
        classes[classes == old] = new

    las.classification = classes

    if output_path is None:
        base = os.path.dirname(file_path)
        name = f"merged_{uuid.uuid4().hex[:8]}_{os.path.basename(file_path)}"
        output_path = os.path.join(base, name)

    las.write(output_path)
    return output_path
