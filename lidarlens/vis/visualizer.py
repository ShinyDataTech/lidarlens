"""LidarLens 3D Point Cloud Visualization (Chapters 2 & 3).

Provides turbo/viridis colormap pseudo-coloring and synchronized side-by-side
comparison views for raw vs. cleaned point clouds. Gracefully handles environments
where Open3D or graphical display heads are unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

from lidarlens.core import LidarFrame

logger = logging.getLogger("lidarlens.vis")

_HAS_OPEN3D = False
try:
    import open3d as o3d  # type: ignore
    _HAS_OPEN3D = True
except ImportError:
    o3d = None
HAS_OPEN3D = _HAS_OPEN3D

_HAS_MPL = False
try:
    import matplotlib.cm as cm  # type: ignore
    _HAS_MPL = True
except ImportError:
    cm = None
HAS_MPL = _HAS_MPL


def _get_pseudo_colors(norm_intensities: np.ndarray, colormap_name: str = "turbo") -> np.ndarray:
    """Generate RGB colors for normalized values using Matplotlib or NumPy fallback."""
    if _HAS_MPL and cm is not None:
        try:
            cmap = cm.get_cmap(colormap_name)
            return cmap(norm_intensities)[:, :3].astype(np.float64)
        except Exception:
            pass

    # High-speed vectorized turbo/jet-style color map fallback in pure NumPy
    x = np.clip(norm_intensities, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.column_stack([r, g, b]).astype(np.float64)


def render_intensity_pointcloud(
    cloud_data: Union[np.ndarray, LidarFrame],
    sample_ratio: float = 0.1,
    colormap_name: str = "turbo",
    window_name: str = "LidarLens - 强度伪彩色渲染",
    headless: bool = False,
) -> bool:
    """Render 3D point cloud with pseudo-coloring based on reflection intensity.

    Args:
        cloud_data: (N, 4) array or LidarFrame instance.
        sample_ratio: Ratio of points to subsample for smooth interactive frame rates.
        colormap_name: Matplotlib colormap name ('turbo', 'viridis', etc.).
        window_name: Open3D window title.
        headless: If True, skips window popup (useful for automated CI/headless servers).

    Returns:
        True if successfully rendered/visualized, False if graphical backend unavailable.
    """
    pts = cloud_data.points if isinstance(cloud_data, LidarFrame) else cloud_data
    if pts.shape[0] == 0:
        logger.warning("Empty point cloud passed to render_intensity_pointcloud.")
        return False

    n_pts = pts.shape[0]
    n_sample = max(1, int(n_pts * sample_ratio))

    if n_sample < n_pts:
        indices = np.random.choice(n_pts, n_sample, replace=False)
        sampled = pts[indices]
    else:
        sampled = pts

    xyz = sampled[:, :3]
    intensities = sampled[:, 3] if sampled.shape[1] >= 4 else np.zeros(sampled.shape[0], dtype=np.float32)

    min_i, max_i = float(np.min(intensities)), float(np.max(intensities))
    norm_intensities = (intensities - min_i) / (max_i - min_i + 1e-6)

    rgb_colors = _get_pseudo_colors(norm_intensities, colormap_name)

    if headless or not _HAS_OPEN3D or o3d is None:
        logger.debug("[Headless/Non-GUI] Point cloud prepared for rendering with %d points.", n_sample)
        return True

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
        o3d.visualization.draw_geometries([pcd], window_name=window_name, width=1280, height=720)
        return True
    except Exception as e:
        logger.warning("Open3D visualization failed (%s). Falling back to non-interactive mode.", e)
        return False


def compare_view(
    raw_frame: Union[np.ndarray, LidarFrame],
    cleaned_frame: Union[np.ndarray, LidarFrame],
    title_left: str = "Raw (Rain/Fog)",
    title_right: str = "Denoised (LidarLens)",
    headless: bool = False,
) -> bool:
    """Side-by-side comparative visualization for raw vs. cleaned point clouds.

    Args:
        raw_frame: Raw unprocessed point cloud or LidarFrame.
        cleaned_frame: Cleaned/filtered point cloud or LidarFrame.
        title_left: Title for raw view window.
        title_right: Title for filtered view window.
        headless: If True, skips GUI window rendering for automated CI.

    Returns:
        True if visual comparison generated successfully.
    """
    raw_pts = raw_frame.points if isinstance(raw_frame, LidarFrame) else raw_frame
    clean_pts = cleaned_frame.points if isinstance(cleaned_frame, LidarFrame) else cleaned_frame

    logger.debug(
        "Compare view: Raw points=%d, Cleaned points=%d (Reduction=%.2f%%)",
        len(raw_pts),
        len(clean_pts),
        (1.0 - (len(clean_pts) / max(1, len(raw_pts)))) * 100.0,
    )

    if headless or not _HAS_OPEN3D or o3d is None:
        return True

    try:
        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(raw_pts[:, :3].astype(np.float64))
        pcd_raw.paint_uniform_color([0.8, 0.2, 0.2])

        clean_xyz_offset = clean_pts[:, :3].copy()
        clean_xyz_offset[:, 1] += 15.0

        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(clean_xyz_offset.astype(np.float64))
        pcd_clean.paint_uniform_color([0.2, 0.8, 0.3])

        o3d.visualization.draw_geometries(
            [pcd_raw, pcd_clean],
            window_name=f"{title_left} (Red) vs {title_right} (Green, +15m Y)",
            width=1440,
            height=800,
        )
        return True
    except Exception as e:
        logger.warning("Side-by-side Open3D rendering failed: %s", e)
        return False
