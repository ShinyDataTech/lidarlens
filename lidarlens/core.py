"""LidarLens Core Engine.

Provides zero-copy memory management, LidarBuffer, and LidarFrame abstractions.
Delegates to the C++20 pybind11 extension (_lidarlens_cpp) when compiled, with
a high-performance, vectorized pure-Python/NumPy fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("lidarlens.core")

# Attempt importing C++20 pybind11 extension
_HAS_CPP_EXT = False
try:
    from lidarlens import _lidarlens_cpp  # type: ignore
    _HAS_CPP_EXT = True
except ImportError:
    _lidarlens_cpp = None
    logger.debug("C++ extension _lidarlens_cpp not found; utilizing optimized NumPy fallback.")


class LidarBuffer:
    """Contiguous memory buffer for LiDAR PointXYZI data.

    Matches Chapter 2's zero-copy architecture. Exposes continuous (N, 4)
    float32 arrays directly to NumPy via buffer protocol semantics without
    intermediate heap copying.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = max(capacity, 1024)
        if _HAS_CPP_EXT and _lidarlens_cpp is not None:
            self._cpp_buf = _lidarlens_cpp.LidarBuffer(self._capacity)
            self._is_cpp = True
        else:
            self._cpp_buf = None
            self._is_cpp = False
            self._data = np.zeros((self._capacity, 4), dtype=np.float32)
            self._size = 0

    def reserve(self, n: int) -> None:
        """Reserve memory capacity for n points."""
        if self._is_cpp:
            self._cpp_buf.reserve(n)
        else:
            if n > self._capacity:
                new_data = np.zeros((n, 4), dtype=np.float32)
                if self._size > 0:
                    new_data[:self._size] = self._data[:self._size]
                self._data = new_data
                self._capacity = n

    def clear(self) -> None:
        """Reset point count to zero without freeing preallocated buffer memory."""
        if self._is_cpp:
            self._cpp_buf.clear()
        else:
            self._size = 0

    def size(self) -> int:
        """Current number of active points in the buffer."""
        if self._is_cpp:
            return self._cpp_buf.size()
        return self._size

    def __len__(self) -> int:
        return self.size()

    def empty(self) -> bool:
        """Check if buffer is empty."""
        return self.size() == 0

    def add_point(self, x: float, y: float, z: float, intensity: float = 0.0) -> None:
        """Append a single point (x, y, z, intensity) to the buffer."""
        if self._is_cpp:
            self._cpp_buf.add_point(float(x), float(y), float(z), float(intensity))
        else:
            if self._size >= self._capacity:
                self.reserve(self._capacity * 2)
            self._data[self._size] = (x, y, z, intensity)
            self._size += 1

    @property
    def capacity(self) -> int:
        """Total allocated capacity of the buffer."""
        return self._capacity

    def load_from_numpy(self, points: np.ndarray) -> None:
        """Load points from a NumPy array into the continuous buffer with zero copying if possible."""
        pts = np.ascontiguousarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Expected shape (N, 3) or (N, 4), got {pts.shape}")

        if self._is_cpp:
            self._cpp_buf.load_from_numpy(pts)
        else:
            n = pts.shape[0]
            if n > self._capacity:
                self.reserve(n)
            if pts.shape[1] == 4:
                self._data[:n] = pts
            else:
                self._data[:n, :3] = pts[:, :3]
                self._data[:n, 3] = 0.0
            self._size = n

    def copy_from(self, points: np.ndarray) -> None:
        """Alias for load_from_numpy."""
        self.load_from_numpy(points)

    def get_numpy_view(self) -> np.ndarray:
        """Return a continuous (N, 4) float32 view of the memory buffer."""
        if self._is_cpp:
            return self._cpp_buf.get_numpy_view()
        # Zero-copy view into preallocated memory slice
        return self._data[:self._size]

    @property
    def points(self) -> np.ndarray:
        """Alias for get_numpy_view()."""
        return self.get_numpy_view()

    def compute_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 3D Axis-Aligned Bounding Box (AABB) [min_xyz, max_xyz]."""
        if self._is_cpp:
            min_xyz, max_xyz = self._cpp_buf.compute_aabb()
            return np.array(min_xyz, dtype=np.float32), np.array(max_xyz, dtype=np.float32)
        view = self.get_numpy_view()
        if len(view) == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return np.min(view[:, :3], axis=0), np.max(view[:, :3], axis=0)


@dataclass
class LidarFrame:
    """Encapsulates a single LiDAR point cloud frame.

    Attributes:
        points: (N, 4) float32 array representing (x, y, z, intensity).
        timestamp: Sensor acquisition timestamp in seconds.
        frame_id: String or integer identifier.
        metadata: Custom metadata dictionary.
    """

    points: np.ndarray
    timestamp: float = 0.0
    frame_id: Union[str, int] = "lidar_frame"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.points, np.ndarray):
            self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim == 1 and self.points.size == 0:
            self.points = np.empty((0, 4), dtype=np.float32)
        elif self.points.ndim == 2:
            if self.points.shape[1] == 3:
                # Append zero intensity column
                zeros = np.zeros((self.points.shape[0], 1), dtype=np.float32)
                self.points = np.hstack([self.points.astype(np.float32), zeros])
            elif self.points.shape[1] != 4:
                raise ValueError(f"Expected (N, 3) or (N, 4) points, got {self.points.shape}")
        else:
            raise ValueError(f"Expected 2D point cloud array, got {self.points.ndim}D")

        if not self.points.flags.c_contiguous:
            self.points = np.ascontiguousarray(self.points, dtype=np.float32)

    @property
    def num_points(self) -> int:
        """Total number of points in the frame."""
        return int(self.points.shape[0])

    @property
    def is_empty(self) -> bool:
        """Check whether the frame contains no points."""
        return self.num_points == 0

    def __len__(self) -> int:
        return self.num_points

    @property
    def xyz(self) -> np.ndarray:
        """Get (N, 3) spatial coordinates."""
        return self.points[:, :3]

    @property
    def intensity(self) -> np.ndarray:
        """Get (N,) scalar reflection intensities."""
        return self.points[:, 3]

    def copy(self) -> LidarFrame:
        """Deep copy of the LiDAR frame."""
        return LidarFrame(
            points=self.points.copy(),
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_xyz(
        cls,
        xyz: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        frame_id: Union[str, int] = "lidar_frame",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LidarFrame:
        """Construct LidarFrame from xyz coordinates and optional intensities."""
        xyz_arr = np.ascontiguousarray(xyz, dtype=np.float32)
        if intensity is None:
            int_arr = np.zeros((xyz_arr.shape[0], 1), dtype=np.float32)
        else:
            int_arr = np.ascontiguousarray(intensity, dtype=np.float32).reshape(-1, 1)
        pts = np.hstack([xyz_arr, int_arr])
        return cls(points=pts, timestamp=timestamp, frame_id=frame_id, metadata=metadata or {})
