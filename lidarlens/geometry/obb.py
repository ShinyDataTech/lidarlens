"""LidarLens 2.5D Oriented Bounding Box (OBB) & Costmap Projection (Chapter 4).

Extracts tight 2.5D OBBs via covariance eigendecomposition (PCA) and rasterizes
polygonal footprints into 2D BEV costmaps with safety dilation.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger("lidarlens.geometry.obb")

_HAS_CV2 = False
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    cv2 = None


def get_pca_obb(cluster_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract 2.5D Oriented Bounding Box (OBB) via PCA (Chapter 4).

    Decomposes the 2D horizontal point distribution covariance matrix into
    eigenvalues and eigenvectors to find the optimal bounding heading.

    Args:
        cluster_pts: (N, 2) or (N, 3) obstacle cluster point coordinates.

    Returns:
        Tuple of (center, size, angle):
            - center: (2,) float32 BEV center (x, y).
            - size: (2,) float32 extent (length, width).
            - angle: float32 heading angle (yaw) in radians.
    """
    if len(cluster_pts) == 0:
        return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.0

    pts_2d = cluster_pts[:, :2].astype(np.float32)
    mu = np.mean(pts_2d, axis=0)
    pts_norm = pts_2d - mu

    if len(cluster_pts) < 3:
        # Fallback for degenerate single/pair points
        p_min, p_max = np.min(pts_2d, axis=0), np.max(pts_2d, axis=0)
        size = np.maximum(p_max - p_min, 0.1).astype(np.float32)
        return mu.astype(np.float32), size, 0.0

    # 1. 2D Covariance Matrix
    cov = np.cov(pts_norm, rowvar=False)

    # 2. Eigendecomposition
    evals, evecs = np.linalg.eigh(cov)

    # 3. Project points to principal axes
    proj = pts_norm @ evecs
    p_min, p_max = np.min(proj, axis=0), np.max(proj, axis=0)

    # 4. Heading angle and physical dimensions
    angle = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
    size = (p_max - p_min).astype(np.float32)
    center = (mu + (p_min + p_max) * 0.5 @ evecs.T).astype(np.float32)

    return center, size, angle


def project_to_costmap(
    center: np.ndarray,
    size: np.ndarray,
    angle: float,
    map_img: np.ndarray,
    res: float = 0.1,
) -> np.ndarray:
    """Project OBB polygon into BEV grid costmap and apply safety dilation.

    Args:
        center: (2,) OBB center coordinates (x, y) in meters.
        size: (2,) OBB length and width (dx, dy) in meters.
        angle: Heading angle (yaw) in radians.
        map_img: (H, W) uint8 raster costmap grid.
        res: Map resolution in meters per pixel.

    Returns:
        (H, W) dilated costmap with obstacle polygons filled.
    """
    c, s = float(np.cos(angle)), float(np.sin(angle))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    half = size * 0.5

    # 4 local corners
    corners_local = np.array(
        [
            [-half[0], -half[1]],
            [half[0], -half[1]],
            [half[0], half[1]],
            [-half[0], half[1]],
        ],
        dtype=np.float32,
    )

    # World grid coordinates
    world_corners = (corners_local @ R.T + center) / res

    out_map = map_img.copy()

    if _HAS_CV2 and cv2 is not None:
        cv2.fillPoly(out_map, [world_corners.astype(np.int32)], 255)
        kernel = np.ones((5, 5), np.uint8)
        dilated_map = cv2.dilate(out_map, kernel)
        return dilated_map
    else:
        # NumPy/SciPy polygon fallback
        from scipy.ndimage import binary_dilation
        # Simple bounding box approximation if OpenCV fillPoly unavailable
        min_c = np.clip(np.min(world_corners, axis=0).astype(int), 0, [out_map.shape[1] - 1, out_map.shape[0] - 1])
        max_c = np.clip(np.max(world_corners, axis=0).astype(int), 0, [out_map.shape[1] - 1, out_map.shape[0] - 1])
        out_map[min_c[1] : max_c[1] + 1, min_c[0] : max_c[0] + 1] = 255
        dilated = binary_dilation(out_map > 0, structure=np.ones((5, 5), dtype=bool))
        return (dilated * 255).astype(np.uint8)
