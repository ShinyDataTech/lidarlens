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

"""Terrain module — DTM/DSM/CHM generation, contour extraction, and terrain
analysis (hillshade, slope, aspect) with backward compatibility.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

_HAS_O3D = False
try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    o3d = None

try:
    import scipy.ndimage
except ImportError:
    scipy = None  # type: ignore[assignment]

try:
    import pyproj
except ImportError:
    pyproj = None

from lidarlens.crs import transform_bounds


def _build_grid(
    pcd: Any,
    resolution: float,
    aggregation: str = "min",
    smooth_sigma: float = 1.0,
    wkt: Optional[str] = None,
    dem_type: str = "dtm",
) -> Tuple[np.ndarray, Dict]:
    """Internal helper to rasterise a point cloud onto a regular grid."""
    if hasattr(pcd, "points"):
        points = np.asarray(pcd.points)
    else:
        points = np.asarray(pcd)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    z_min, z_max = float(z.min()), float(z.max())

    width = max(1, int(np.ceil((x_max - x_min) / resolution)))
    height = max(1, int(np.ceil((y_max - y_min) / resolution)))

    init_val = np.inf if aggregation == "min" else -np.inf
    grid = np.full((height, width), init_val)

    idx_x = np.clip(((x - x_min) / resolution).astype(int), 0, width - 1)
    idx_y = np.clip(((y - y_min) / resolution).astype(int), 0, height - 1)

    if aggregation == "min":
        for i in range(len(z)):
            r, c = idx_y[i], idx_x[i]
            if z[i] < grid[r, c]:
                grid[r, c] = z[i]
        grid[grid == np.inf] = z_min
    else:  # max
        for i in range(len(z)):
            r, c = idx_y[i], idx_x[i]
            if z[i] > grid[r, c]:
                grid[r, c] = z[i]
        grid[grid == -np.inf] = z_min

    if smooth_sigma > 0 and scipy is not None:
        grid = scipy.ndimage.gaussian_filter(grid, sigma=smooth_sigma)

    metadata: Dict = {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "resolution": resolution,
        "width": width, "height": height,
        "type": dem_type,
    }

    if wkt:
        try:
            bounds = {
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max,
            }
            metadata["bounds_wgs84"] = transform_bounds(bounds, wkt)
        except Exception:
            pass

    return grid, metadata


def generate_dtm(
    pcd: Any,
    resolution: float = 1.0,
    smooth_sigma: float = 1.0,
    wkt: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    return _build_grid(pcd, resolution, "min", smooth_sigma, wkt, "dtm")


def generate_dsm(
    pcd: Any,
    resolution: float = 1.0,
    smooth_sigma: float = 1.0,
    wkt: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    return _build_grid(pcd, resolution, "max", smooth_sigma, wkt, "dsm")


def generate_chm(
    dtm: np.ndarray,
    dsm: np.ndarray,
    metadata: Dict,
) -> Tuple[np.ndarray, Dict]:
    chm = np.maximum(dsm - dtm, 0.0)
    meta = dict(metadata)
    meta["type"] = "chm"
    meta["z_min"] = float(chm.min())
    meta["z_max"] = float(chm.max())
    return chm, meta


def _simplify_points(points: List, tolerance: float) -> List:
    if len(points) < 3:
        return points
    simplified = [points[0]]
    last_x, last_y = points[0]
    tol_sq = tolerance * tolerance
    for i in range(1, len(points)):
        x, y = points[i]
        dx, dy = x - last_x, y - last_y
        if dx * dx + dy * dy > tol_sq:
            simplified.append(points[i])
            last_x, last_y = x, y
    if simplified[-1] != points[-1]:
        simplified.append(points[-1])
    return simplified


def generate_contours(
    dem_array: np.ndarray,
    metadata: Dict,
    interval: float = 5.0,
    smooth: bool = True,
    wkt: Optional[str] = None,
) -> Dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_min = metadata["z_min"]
    z_max = metadata["z_max"]
    x_min = metadata["x_min"]
    y_min = metadata["y_min"]
    resolution = metadata["resolution"]

    levels = np.arange(
        np.floor(z_min / interval) * interval,
        np.ceil(z_max / interval) * interval + interval,
        interval,
    )

    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    cs = ax.contour(dem_array, levels=levels)
    plt.close(fig)

    transformer = None
    if wkt and pyproj is not None:
        try:
            crs_src = pyproj.CRS.from_user_input(wkt)
            if not crs_src.is_geographic:
                crs_dst = pyproj.CRS.from_epsg(4326)
                transformer = pyproj.Transformer.from_crs(
                    crs_src, crs_dst, always_xy=True,
                )
        except Exception:
            pass

    features = []
    for level_idx, level_val in enumerate(cs.levels):
        segs = cs.allsegs[level_idx] if level_idx < len(cs.allsegs) else []
        for seg in segs:
            if len(seg) < 2:
                continue

            orig_coords = []
            for pt in seg:
                wx = x_min + pt[0] * resolution
                wy = y_min + pt[1] * resolution
                if math.isfinite(wx) and math.isfinite(wy):
                    orig_coords.append([wx, wy])
            if len(orig_coords) < 2:
                continue

            tol = max(0.1, resolution * 0.2)
            simplified = _simplify_points(orig_coords, tol)

            final_coords = []
            for x, y in simplified:
                tx, ty = x, y
                if transformer:
                    try:
                        tx, ty = transformer.transform(x, y)
                    except Exception:
                        pass
                if math.isfinite(tx) and math.isfinite(ty):
                    final_coords.append([round(float(tx), 7), round(float(ty), 7)])

            if len(final_coords) >= 2:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": final_coords,
                    },
                    "properties": {"elevation": float(level_val)},
                })

    return {"type": "FeatureCollection", "features": features}


def hillshade(
    dem_array: np.ndarray,
    metadata: Dict,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    res = metadata["resolution"]
    az_rad = np.radians(360.0 - azimuth + 90.0)
    alt_rad = np.radians(altitude)

    dy, dx = np.gradient(dem_array, res)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(-dy, dx)

    shade = (
        np.sin(alt_rad) * np.cos(slope)
        + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )
    shade = np.clip(shade, 0, 1) * 255
    return shade.astype(np.uint8)


def slope_map(
    dem_array: np.ndarray,
    metadata: Dict,
) -> np.ndarray:
    res = metadata["resolution"]
    dy, dx = np.gradient(dem_array, res)
    return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))


def aspect_map(
    dem_array: np.ndarray,
    metadata: Dict,
) -> np.ndarray:
    res = metadata["resolution"]
    dy, dx = np.gradient(dem_array, res)
    aspect = np.degrees(np.arctan2(-dy, dx))
    aspect = (90.0 - aspect) % 360.0
    return aspect
