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

"""I/O module — read and write LAS/LAZ point cloud files with backward compatibility."""

from typing import Dict, List, Optional, Tuple
import numpy as np

_HAS_O3D = False
try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    o3d = None

_HAS_LASPY = False
try:
    import laspy
    _HAS_LASPY = True
except ImportError:
    laspy = None

from lidarlens.crs import DEFAULT_WKT, create_wkt_vlr


def read(
    path: str,
) -> Tuple[Any, Any, List, Optional[str]]:
    """Load a LAS/LAZ file and return an Open3D PointCloud plus metadata."""
    if not _HAS_LASPY or laspy is None:
        raise ImportError("laspy is required to read LAS/LAZ files. Install via pip install laspy[lazrs].")

    with laspy.open(path) as f:
        header = f.header
        vlrs = list(f.header.vlrs)
        las = f.read()

    # Extract WKT from VLRs
    wkt: Optional[str] = None
    for vlr in vlrs:
        uid = getattr(vlr, "user_id", "")
        rid = getattr(vlr, "record_id", 0)
        desc = getattr(vlr, "description", "")
        if (uid == "LASF_Projection" and rid == 2112) or "WKT" in desc:
            try:
                wkt = vlr.record_data.decode("utf-8").strip("\x00")
                break
            except Exception:
                pass

    points = np.vstack([las.x, las.y, las.z]).T
    if _HAS_O3D and o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        try:
            if hasattr(las, "red"):
                red = np.array(las.red, dtype=np.float64)
                green = np.array(las.green, dtype=np.float64)
                blue = np.array(las.blue, dtype=np.float64)

                max_val = max(red.max(), green.max(), blue.max(), 1)
                scale = 65535.0 if max_val > 255 else 255.0

                colors = np.vstack([red / scale, green / scale, blue / scale]).T
                pcd.colors = o3d.utility.Vector3dVector(colors)
        except Exception:
            pass
    else:
        pcd = points

    return pcd, header, vlrs, wkt


def save(
    pcd: Any,
    output_path: str,
    *,
    header: Optional[Any] = None,
    vlrs: Optional[List] = None,
    wkt: Optional[str] = None,
    extra_dims: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Save a PointCloud to a LAS/LAZ file."""
    if not _HAS_LASPY or laspy is None:
        raise ImportError("laspy is required to save LAS/LAZ files.")

    if hasattr(pcd, "points"):
        points = np.asarray(pcd.points)
    else:
        points = np.asarray(pcd)

    new_header = laspy.LasHeader(point_format=3, version="1.2")
    new_header.scales = np.array([0.001, 0.001, 0.001])
    new_header.offsets = np.min(points, axis=0)

    if header is not None:
        new_header.offsets = header.offsets
        new_header.scales = header.scales

    filtered_vlrs = []
    if vlrs:
        for vlr in vlrs:
            if getattr(vlr, "user_id", "") == "copc":
                continue
            filtered_vlrs.append(vlr)

    existing_wkt = any(
        getattr(v, "user_id", "") == "LASF_Projection"
        and getattr(v, "record_id", 0) == 2112
        for v in filtered_vlrs
    )

    for vlr in filtered_vlrs:
        new_header.vlrs.append(vlr)

    if not existing_wkt:
        target_wkt = wkt if wkt else DEFAULT_WKT
        new_header.vlrs.append(create_wkt_vlr(target_wkt))

    if extra_dims:
        for name, data in extra_dims.items():
            dtype = np.int32 if np.issubdtype(data.dtype, np.integer) else np.float64
            new_header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))

    las = laspy.LasData(new_header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if hasattr(pcd, "has_colors") and pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 65535).clip(0, 65535).astype(np.uint16)
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

    if extra_dims:
        for name, data in extra_dims.items():
            setattr(las, name, data)

    las.write(output_path)


def read_header(path: str) -> Dict:
    """Read only header metadata from a LAS/LAZ file."""
    if not _HAS_LASPY or laspy is None:
        raise ImportError("laspy is required to read LAS/LAZ header.")
    with laspy.open(path) as f:
        h = f.header
        return {
            "point_count": h.point_count,
            "point_format": int(h.point_format.id),
            "version": f"{h.version.major}.{h.version.minor}",
            "scales": h.scales.tolist(),
            "offsets": h.offsets.tolist(),
            "mins": h.mins.tolist(),
            "maxs": h.maxs.tolist(),
        }


def read_bounds(path: str) -> Dict:
    """Read spatial bounding box from LAS/LAZ file."""
    if not _HAS_LASPY or laspy is None:
        raise ImportError("laspy is required to read bounds.")
    with laspy.open(path) as f:
        h = f.header
        return {
            "min_x": float(h.mins[0]),
            "max_x": float(h.maxs[0]),
            "min_y": float(h.mins[1]),
            "max_y": float(h.maxs[1]),
            "min_z": float(h.mins[2]),
            "max_z": float(h.maxs[2]),
        }
