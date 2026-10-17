"""LidarLens 2D & 3D Occupancy Mapping (Chapter 6).

Provides ROS 2 standard 2D OccupancyGrid rasterization with topological safety
dilation, and 3D OctoMap volumetric log-odds mapping with ray-casting clearing
and temporal exponential decay.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

logger = logging.getLogger("lidarlens.mapping.occupancy")


def generate_standard_grid(
    points: np.ndarray,
    resolution: float = 0.05,
    z_range: Tuple[float, float] = (0.1, 2.0),
    inflation_radius: float = 0.2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build standardized 2D OccupancyGrid and safety inflation map (Chapter 6).

    Filters floor and ceiling points using a vertical height window, quantizes 3D
    points into discrete 2D raster cells, and performs topological binary dilation
    scaled to the robot's physical collision footprint.

    Args:
        points: (N, 3) or (N, 4) raw point cloud coordinates.
        resolution: Grid cell size in meters (e.g. 0.05 m).
        z_range: Vertical pass-through slice interval (z_min, z_max).
        inflation_radius: Collision safety dilation radius in meters.

    Returns:
        Tuple of (inflated_grid, meta):
            - inflated_grid: (height, width) int8 costmap array (100=Lethal, 0=Free).
            - meta: Dictionary containing 'origin', 'resolution', 'width', 'height'.
    """
    if len(points) == 0:
        return np.zeros((10, 10), dtype=np.int8), {
            "origin": (0.0, 0.0),
            "resolution": resolution,
            "width": 10,
            "height": 10,
        }

    # 1. Height window slice: filter ground and ceiling clutter
    mask = (points[:, 2] > z_range[0]) & (points[:, 2] < z_range[1])
    filtered_points = points[mask]

    if len(filtered_points) == 0:
        return np.zeros((10, 10), dtype=np.int8), {
            "origin": (0.0, 0.0),
            "resolution": resolution,
            "width": 10,
            "height": 10,
        }

    # 2. Origin and grid dimensions
    x_min, y_min = np.min(filtered_points[:, :2], axis=0)
    origin = (
        float(np.floor(x_min / resolution) * resolution),
        float(np.floor(y_min / resolution) * resolution),
    )

    width = int(np.ceil((np.max(filtered_points[:, 0]) - origin[0]) / resolution)) + 1
    height = int(np.ceil((np.max(filtered_points[:, 1]) - origin[1]) / resolution)) + 1

    # 3. Occupancy assignment: 100 designates lethal obstacle
    grid = np.zeros((height, width), dtype=np.int8)
    indices_x = np.clip(((filtered_points[:, 0] - origin[0]) / resolution).astype(int), 0, width - 1)
    indices_y = np.clip(((filtered_points[:, 1] - origin[1]) / resolution).astype(int), 0, height - 1)
    grid[indices_y, indices_x] = 100

    # 4. Binary morphological dilation
    struct_size = max(1, int(round(inflation_radius / resolution)))
    struct_element = np.ones((struct_size, struct_size), dtype=bool)
    inflated_mask = binary_dilation(grid > 0, structure=struct_element)
    inflated_grid = (inflated_mask * 100).astype(np.int8)

    meta = {
        "origin": origin,
        "origin_x": float(origin[0]),
        "origin_y": float(origin[1]),
        "resolution": resolution,
        "width": width,
        "height": height,
        "occupied_cells": int(np.sum(grid == 100)),
        "inflated_cells": int(np.sum(inflated_grid == 100)),
    }

    return inflated_grid, meta


class OctoMap3D:
    """Volumetric 3D Octree / Voxel log-odds occupancy mapping.

    Implements:
    - Log-odds probability updates: L(m) += l_hit / l_miss.
    - Clamping thresholds: l_min, l_max to prevent probability saturation.
    - 3D DDA/Bresenham ray-casting for clearing dynamic transient obstacles.
    - Temporal decay: L_t = L_{t-1} * exp(-lambda * delta_t) to forget stale obstacles.
    """

    def __init__(
        self,
        resolution: float = 0.1,
        prob_hit: Optional[float] = None,
        prob_miss: Optional[float] = None,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        decay_lambda: float = 0.01,
        **kwargs: Any,
    ) -> None:
        p_hit = prob_hit if prob_hit is not None else kwargs.get("p_occ", 0.70)
        p_miss = prob_miss if prob_miss is not None else kwargs.get("p_free", 0.40)
        c_min = clamp_min if clamp_min is not None else kwargs.get("l_min", -2.0)
        c_max = clamp_max if clamp_max is not None else kwargs.get("l_max", 3.5)

        self.resolution = float(resolution)
        self.prob_hit = float(p_hit)
        self.prob_miss = float(p_miss)
        self.clamp_min = float(c_min)
        self.clamp_max = float(c_max)
        self.decay_lambda = float(decay_lambda)

        # Precompute log-odds increments
        self.l_hit = float(math.log(self.prob_hit / (1.0 - self.prob_hit)))
        self.l_miss = float(math.log(self.prob_miss / (1.0 - self.prob_miss)))

        # Voxel storage: key=(vx, vy, vz) -> log_odds
        self.voxels: Dict[Tuple[int, int, int], float] = {}

    def _coord_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        return (
            int(math.floor(x / self.resolution)),
            int(math.floor(y / self.resolution)),
            int(math.floor(z / self.resolution)),
        )

    def _voxel_to_coord(self, vx: int, vy: int, vz: int) -> Tuple[float, float, float]:
        return (
            (vx + 0.5) * self.resolution,
            (vy + 0.5) * self.resolution,
            (vz + 0.5) * self.resolution,
        )

    def update_scan(
        self,
        sensor_origin: np.ndarray,
        hit_points: Optional[np.ndarray] = None,
        delta_time: float = 0.0,
        ray_cast: bool = True,
        **kwargs: Any,
    ) -> None:
        pts_in = hit_points if hit_points is not None else kwargs.get("end_points")
        if pts_in is None:
            return
        hit_points = pts_in
        """Update occupancy grid with a new scan frame.

        Args:
            sensor_origin: (3,) float array representing sensor global origin.
            hit_points: (N, 3) 3D coordinate array of beam impact reflections.
            delta_time: Elapsed time in seconds since previous update for decay.
            ray_cast: If True, executes ray-casting to clear free intermediate voxels.
        """
        # 1. Temporal exponential decay
        if delta_time > 0.0 and self.decay_lambda > 0.0:
            decay_factor = math.exp(-self.decay_lambda * delta_time)
            for k in list(self.voxels.keys()):
                self.voxels[k] *= decay_factor
                if abs(self.voxels[k]) < 0.05:
                    del self.voxels[k]

        ox, oy, oz = float(sensor_origin[0]), float(sensor_origin[1]), float(sensor_origin[2])
        orig_vox = self._coord_to_voxel(ox, oy, oz)

        # 2. Update hit endpoints and ray paths
        for pt in hit_points[:, :3]:
            hx, hy, hz = float(pt[0]), float(pt[1]), float(pt[2])
            hit_vox = self._coord_to_voxel(hx, hy, hz)

            # Ray-cast clearing between origin and endpoint
            if ray_cast:
                dx, dy, dz = hx - ox, hy - oy, oz
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                num_steps = max(1, int(dist / (self.resolution * 0.8)))
                for s in range(1, num_steps):
                    frac = s / float(num_steps)
                    rx, ry, rz = ox + frac * dx, oy + frac * dy, oz + frac * dz
                    free_vox = self._coord_to_voxel(rx, ry, rz)
                    if free_vox != hit_vox:
                        cur_l = self.voxels.get(free_vox, 0.0) + self.l_miss
                        self.voxels[free_vox] = max(self.clamp_min, min(self.clamp_max, cur_l))

            # Mark hit voxel
            cur_hit_l = self.voxels.get(hit_vox, 0.0) + self.l_hit
            self.voxels[hit_vox] = max(self.clamp_min, min(self.clamp_max, cur_hit_l))

    def get_occupied_voxels(self, prob_threshold: float = 0.5) -> np.ndarray:
        """Return array of 3D center coordinates for occupied voxels.

        Args:
            prob_threshold: Occupancy probability threshold (default 0.5 -> log_odds > 0).

        Returns:
            (M, 3) float32 coordinates of occupied voxels.
        """
        thresh_l = math.log(prob_threshold / (1.0 - prob_threshold)) if prob_threshold < 1.0 else 0.0
        occupied_coords: List[Tuple[float, float, float]] = []

        for vox, l_val in self.voxels.items():
            if l_val > thresh_l:
                occupied_coords.append(self._voxel_to_coord(vox[0], vox[1], vox[2]))

        if not occupied_coords:
            return np.empty((0, 3), dtype=np.float32)

        return np.array(occupied_coords, dtype=np.float32)

    def apply_temporal_decay(self, decay_rate: float = 0.05, dt: float = 1.0) -> None:
        """Apply exponential decay to all voxel log-odds values: L_t = L_{t-1} * exp(-decay_rate * dt)."""
        decay_factor = math.exp(-decay_rate * dt)
        for k in list(self.voxels.keys()):
            self.voxels[k] *= decay_factor
            if abs(self.voxels[k]) < 0.05:
                del self.voxels[k]

    integrate_scan = update_scan
    get_occupied_centers = get_occupied_voxels

    def to_pointcloud(self) -> np.ndarray:
        """Convert all occupied voxels to an (M, 3) point cloud."""
        return self.get_occupied_voxels()
