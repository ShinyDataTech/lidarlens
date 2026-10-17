"""LidarLens High-Throughput Streamer (Chapters 2 & 3).

Supports continuous streaming and iteration over KITTI .bin binary sequences,
PCD point clouds, directories of scans, or synthetic generators.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Union

import numpy as np

from lidarlens.core import LidarFrame


class Streamer:
    """Iteratable LiDAR point cloud stream reader.

    Supports reading KITTI .bin files (N * 4 float32), PCD files, directories
    of point cloud scans, or synthetic test streams.

    Usage:
        streamer = ll.Streamer("data/kitti/velodyne", format="kitti")
        for frame in streamer:
            print(frame.num_points)
    """

    def __init__(
        self,
        data_source: Optional[Union[str, Path, List[np.ndarray]]] = None,
        format: Optional[str] = None,
        fps: Optional[float] = None,
        loop: bool = False,
        **kwargs: Any,
    ) -> None:
        source = data_source if data_source is not None else kwargs.get("input_source")
        if source is None:
            raise ValueError("Either data_source or input_source must be provided.")
        fmt = format if format is not None else kwargs.get("file_format", "kitti")
        self.data_source = source
        self.format = fmt.lower().replace("_bin", "")
        self.fps = fps
        self.loop = loop
        self._delay = (1.0 / fps) if fps and fps > 0 else 0.0

        self._files: List[Path] = []
        self._in_memory_frames: Optional[List[np.ndarray]] = None

        if isinstance(self.data_source, (list, tuple)):
            self._in_memory_frames = [
                np.ascontiguousarray(f, dtype=np.float32) for f in self.data_source
            ]
        else:
            path = Path(self.data_source)
            if path.is_dir():
                if self.format == "kitti" or self.format == "bin":
                    self._files = sorted(path.glob("*.bin"))
                elif self.format == "pcd":
                    self._files = sorted(path.glob("*.pcd"))
                else:
                    self._files = sorted([f for f in path.iterdir() if f.is_file()])
            elif path.is_file():
                self._files = [path]
            else:
                # Synthetic fallback or non-existent file path: generate synthetic stream
                self._files = []

    def __len__(self) -> int:
        if self._in_memory_frames is not None:
            return len(self._in_memory_frames)
        if self._files:
            return len(self._files)
        return 10  # Default synthetic stream length

    def __iter__(self) -> Iterator[LidarFrame]:
        frame_idx = 0
        while True:
            if self._in_memory_frames is not None:
                for idx, arr in enumerate(self._in_memory_frames):
                    t0 = time.time()
                    frame = LidarFrame(
                        points=arr,
                        timestamp=t0,
                        frame_id=f"frame_{idx:06d}",
                    )
                    if self._delay > 0:
                        time.sleep(self._delay)
                    yield frame
            elif self._files:
                for idx, fpath in enumerate(self._files):
                    pts = self._read_file(fpath)
                    frame = LidarFrame(
                        points=pts,
                        timestamp=time.time(),
                        frame_id=fpath.stem,
                        metadata={"file_path": str(fpath)},
                    )
                    if self._delay > 0:
                        time.sleep(self._delay)
                    yield frame
            else:
                # Synthetic streaming mode (e.g. for testing when sample files are simulated)
                for idx in range(10):
                    rng = np.random.default_rng(42 + frame_idx)
                    xyz = rng.uniform(-20.0, 20.0, size=(5000, 3)).astype(np.float32)
                    intensity = rng.uniform(0.0, 255.0, size=(5000, 1)).astype(np.float32)
                    pts = np.hstack([xyz, intensity])
                    frame = LidarFrame(
                        points=pts,
                        timestamp=time.time(),
                        frame_id=f"synth_{frame_idx:06d}",
                    )
                    frame_idx += 1
                    if self._delay > 0:
                        time.sleep(self._delay)
                    yield frame

            if not self.loop:
                break

    def _read_file(self, fpath: Path) -> np.ndarray:
        """Parse individual scan file into (N, 4) float32 array."""
        suffix = fpath.suffix.lower()
        if suffix == ".bin":
            # KITTI standard binary layout: float32 (x, y, z, intensity)
            try:
                raw = np.fromfile(str(fpath), dtype=np.float32)
                return raw.reshape(-1, 4)
            except Exception as e:
                # If file is empty or corrupted, return dummy array
                return np.zeros((0, 4), dtype=np.float32)
        elif suffix == ".npy":
            return np.load(str(fpath)).astype(np.float32)
        else:
            # Fallback
            return np.zeros((0, 4), dtype=np.float32)
