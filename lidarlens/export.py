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
Export module — convert point clouds and DEMs to various file formats.
"""


import json
import os
from typing import Dict, Optional

import numpy as np
# ── Point Cloud Export ───────────────────────────────────────────────────────


def to_csv(
    pcd: o3d.geometry.PointCloud,
    output_path: str,
) -> str:
    """Export a point cloud to CSV (x, y, z [, r, g, b] [, nx, ny, nz]).

    Returns:
        The actual output path.
    """
    base, _ = os.path.splitext(output_path)
    actual = base + ".csv"

    points = np.asarray(pcd.points)
    header_parts = ["x", "y", "z"]
    data_parts = [points]

    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        header_parts.extend(["r", "g", "b"])
        data_parts.append(colors)

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        header_parts.extend(["nx", "ny", "nz"])
        data_parts.append(normals)

    combined = np.hstack(data_parts)
    np.savetxt(
        actual, combined, delimiter=",",
        header=",".join(header_parts), comments="", fmt="%.6f",
    )
    return actual


def to_ply(
    pcd: o3d.geometry.PointCloud,
    output_path: str,
) -> str:
    """Export a point cloud to PLY format.

    Returns:
        The actual output path.
    """
    base, _ = os.path.splitext(output_path)
    actual = base + ".ply"
    o3d.io.write_point_cloud(actual, pcd)
    return actual


def to_pcd(
    pcd: o3d.geometry.PointCloud,
    output_path: str,
) -> str:
    """Export a point cloud to PCD (Point Cloud Data) format.

    Returns:
        The actual output path.
    """
    base, _ = os.path.splitext(output_path)
    actual = base + ".pcd"
    o3d.io.write_point_cloud(actual, pcd)
    return actual


def to_numpy(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Convert a point cloud to a NumPy array.

    Returns:
        ``(N, 3)`` array if no colours, ``(N, 6)`` if colours are
        present (x, y, z, r, g, b normalised to 0-1).
    """
    points = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        return np.hstack([points, colors])
    return points


def to_dataframe(pcd: o3d.geometry.PointCloud):
    """Convert a point cloud to a Pandas DataFrame.

    Columns: ``x``, ``y``, ``z`` and optionally ``r``, ``g``, ``b``.

    Returns:
        ``pandas.DataFrame``

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install with: pip install lidarlens[all]"
        )

    points = np.asarray(pcd.points)
    data = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        data["r"] = colors[:, 0]
        data["g"] = colors[:, 1]
        data["b"] = colors[:, 2]

    return pd.DataFrame(data)


# ── DEM Export ───────────────────────────────────────────────────────────────


def dem_to_png(
    dem_array: np.ndarray,
    metadata: Dict,
    output_path: str,
    colormap: str = "terrain",
) -> str:
    """Render and save a DEM as a colourised PNG image.

    Args:
        dem_array: 2-D elevation array.
        metadata: Metadata dict from DTM/DSM generation.
        output_path: Destination ``.png`` path.
        colormap: Matplotlib colourmap name.

    Returns:
        The actual output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_min, z_max = metadata["z_min"], metadata["z_max"]
    z_range = z_max - z_min if z_max > z_min else 1.0
    normalised = np.clip((dem_array - z_min) / z_range, 0, 1)

    cmap = plt.get_cmap(colormap)
    colored = (cmap(normalised) * 255).astype(np.uint8)

    nan_mask = np.isnan(dem_array)
    colored[nan_mask] = [0, 0, 0, 0]

    base, _ = os.path.splitext(output_path)
    actual = base + ".png"

    fig, ax = plt.subplots(
        1, 1,
        figsize=(metadata["width"] / 100, metadata["height"] / 100),
        dpi=100,
    )
    ax.imshow(colored, origin="lower", aspect="auto")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(actual, dpi=100, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return actual


def dem_to_geotiff(
    dem_array: np.ndarray,
    metadata: Dict,
    output_path: str,
) -> str:
    """Export a DEM as a GeoTIFF file.

    Requires the ``rasterio`` package.

    Returns:
        The actual output path.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        raise ImportError(
            "rasterio is required for GeoTIFF export. "
            "Install with: pip install lidarlens[all]"
        )

    base, _ = os.path.splitext(output_path)
    actual = base + ".tif"

    transform = from_bounds(
        metadata["x_min"], metadata["y_min"],
        metadata["x_max"], metadata["y_max"],
        metadata["width"], metadata["height"],
    )

    with rasterio.open(
        actual, "w", driver="GTiff",
        height=metadata["height"], width=metadata["width"],
        count=1, dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(dem_array.astype(np.float32), 1)

    return actual


def dem_to_csv(
    dem_array: np.ndarray,
    metadata: Dict,
    output_path: str,
) -> str:
    """Export a DEM as a CSV file with metadata.

    Returns:
        The actual output path.
    """
    base, _ = os.path.splitext(output_path)
    actual = base + "_dem.csv"

    header_text = (
        f"DEM Type: {metadata.get('type', 'unknown')}, "
        f"Resolution: {metadata['resolution']}, "
        f"Bounds: [{metadata['x_min']:.2f}, {metadata['y_min']:.2f}] to "
        f"[{metadata['x_max']:.2f}, {metadata['y_max']:.2f}]"
    )
    np.savetxt(actual, dem_array, delimiter=",", header=header_text, fmt="%.4f")

    # Side-car metadata JSON
    meta_path = base + "_dem_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return actual
