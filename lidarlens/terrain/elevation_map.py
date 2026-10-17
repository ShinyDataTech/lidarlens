"""LidarLens 2.5D Elevation Mapping & Traversability Analysis (Chapter 4).

Provides 100% vectorized computation of 2.5D elevation grid maps (z_map, z_max, z_min)
without nested Python loops, along with slope, step height, and traversability costmaps.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger("lidarlens.terrain.elevation_map")

_HAS_CV2 = False
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    cv2 = None


def create_elevation_map(
    points: np.ndarray,
    grid_size: float = 0.1,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_range: Tuple[float, float] = (-5.0, 5.0),
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "resolution" in kwargs:
        grid_size = float(kwargs["resolution"])
    """Construct 2.5D elevation grid map from 3D point cloud without nested loops.

    Computes the cell mean elevation (z_map) along with extreme stats (z_max, z_min)
    using vectorized NumPy index aggregation (np.maximum.at / np.bincount).

    Args:
        points: (N, 3) or (N, 4) 3D LiDAR point cloud coordinates.
        grid_size: Cell resolution in meters (e.g. 0.1 m).
        x_range: Physical X-axis coverage interval (x_min, x_max).
        y_range: Physical Y-axis coverage interval (y_min, y_max).

    Returns:
        Tuple of (z_map, z_max, z_min):
            - z_map: (rows, cols) float32 mean elevation (NaN for empty cells).
            - z_max: (rows, cols) float32 maximum elevation (-inf for empty).
            - z_min: (rows, cols) float32 minimum elevation (+inf for empty).
    """
    rows = int(np.ceil((y_range[1] - y_range[0]) / grid_size))
    cols = int(np.ceil((x_range[1] - x_range[0]) / grid_size))
    total_cells = rows * cols

    if len(points) == 0:
        z_map = np.full((rows, cols), np.nan, dtype=np.float32)
        z_max = np.full((rows, cols), -np.inf, dtype=np.float32)
        z_min = np.full((rows, cols), np.inf, dtype=np.float32)
        return z_map, z_max, z_min

    pts = points[:, :3].astype(np.float32)

    # Vectorized grid coordinate indexing
    ix = np.floor((pts[:, 0] - x_range[0]) / grid_size).astype(np.int64)
    iy = np.floor((pts[:, 1] - y_range[0]) / grid_size).astype(np.int64)

    # Valid bounding filter
    valid = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    if not np.any(valid):
        z_map = np.full((rows, cols), np.nan, dtype=np.float32)
        z_max = np.full((rows, cols), -np.inf, dtype=np.float32)
        z_min = np.full((rows, cols), np.inf, dtype=np.float32)
        return z_map, z_max, z_min

    ix_v = ix[valid]
    iy_v = iy[valid]
    z_v = pts[valid, 2]

    cell_indices = iy_v * cols + ix_v

    # 1. Vectorized z_max
    z_max_flat = np.full(total_cells, -np.inf, dtype=np.float32)
    np.maximum.at(z_max_flat, cell_indices, z_v)
    z_max = z_max_flat.reshape((rows, cols))

    # 2. Vectorized z_min
    z_min_flat = np.full(total_cells, np.inf, dtype=np.float32)
    np.minimum.at(z_min_flat, cell_indices, z_v)
    z_min = z_min_flat.reshape((rows, cols))

    # 3. Vectorized z_map (mean)
    counts = np.bincount(cell_indices, minlength=total_cells)
    sums = np.bincount(cell_indices, weights=z_v, minlength=total_cells)

    has_points = counts > 0
    z_map_flat = np.full(total_cells, np.nan, dtype=np.float32)
    z_map_flat[has_points] = sums[has_points] / counts[has_points]
    z_map = z_map_flat.reshape((rows, cols))

    return z_map, z_max, z_min


def analyze_traversability(
    z_map: np.ndarray,
    z_max: np.ndarray,
    z_min: np.ndarray,
    res: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Analyze slope and step height to generate costmaps and traversability masks.

    Calculates surface slope via Sobel gradient operators, evaluates local step
    discontinuities (z_max - z_min), applies multi-metric weighting (0.6 * slope +
    1.2 * step_height), and cleans noise via morphological opening.

    Args:
        z_map: (rows, cols) mean elevation grid.
        z_max: (rows, cols) maximum elevation grid.
        z_min: (rows, cols) minimum elevation grid.
        res: Grid cell resolution in meters.

    Returns:
        Tuple of (cost, mask):
            - cost: (rows, cols) float32 normalized cost in [0.0, 1.0].
            - mask: (rows, cols) uint8 binary traversability mask (255=passable, 0=blocked).
    """
    # 1. Fill NaNs to prevent Sobel operator boundary bleeding
    z_filled = np.nan_to_num(z_map, nan=0.0)

    # 2. Slope calculation via 2D Sobel gradient operators
    if _HAS_CV2 and cv2 is not None:
        grad_x = cv2.Sobel(z_filled, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * res)
        grad_y = cv2.Sobel(z_filled, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * res)
    else:
        from scipy.ndimage import sobel
        grad_x = sobel(z_filled, axis=1) / (8.0 * res)
        grad_y = sobel(z_filled, axis=0) / (8.0 * res)

    slope = np.sqrt(grad_x**2 + grad_y**2)

    # 3. Step height calculation for valid populated cells
    valid_mask = ~np.isnan(z_map)
    step_height = np.zeros_like(z_map, dtype=np.float32)
    step_height[valid_mask] = (z_max[valid_mask] - z_min[valid_mask]).astype(np.float32)

    # 4. Multi-metric cost fusion normalized to [0, 1]
    cost = 0.6 * slope + 1.2 * step_height
    cost = np.clip(cost, 0.0, 1.0).astype(np.float32)

    # 5. Threshold binary mask (<0.4 is traversable)
    mask = (cost < 0.4).astype(np.uint8) * 255

    # 6. Morphological opening to prune isolated false positives
    if _HAS_CV2 and cv2 is not None:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    else:
        from scipy.ndimage import binary_opening
        mask_bool = binary_opening(mask > 0, structure=np.ones((3, 3), dtype=bool))
        mask = (mask_bool * 255).astype(np.uint8)

    return cost, mask
