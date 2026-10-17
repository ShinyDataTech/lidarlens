"""LidarLens Temporal Pipeline & RingBuffer (Chapter 3).

Manages temporal sliding-window point cloud streams and executes chained
spatial/temporal filter operator graphs in real-time inner-loops (>=30 Hz).
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Callable, Deque, List, Optional, Union

import numpy as np

from lidarlens.core import LidarFrame

logger = logging.getLogger("lidarlens.pipeline")


class RingBuffer:
    """Fixed-capacity circular queue for sliding-window temporal consistency."""

    def __init__(self, capacity: int = 5) -> None:
        self.capacity = max(1, capacity)
        self._buffer: Deque[LidarFrame] = collections.deque(maxlen=self.capacity)

    def push(self, frame: LidarFrame) -> None:
        """Push a new frame into the circular queue."""
        self._buffer.append(frame)

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, index: int) -> LidarFrame:
        return self._buffer[index]

    def clear(self) -> None:
        """Clear all historical frames."""
        self._buffer.clear()

    @property
    def frames(self) -> List[LidarFrame]:
        """Return all frames currently in the buffer."""
        return list(self._buffer)

    @property
    def latest(self) -> Optional[LidarFrame]:
        """Return the most recently pushed frame, or None if empty."""
        return self._buffer[-1] if self._buffer else None


class TemporalPipeline:
    """Temporal sliding-window processing pipeline.

    Usage:
        pipeline = ll.TemporalPipeline(window_size=5)
        pipeline.add(ll.filters.VoxelDownsample(voxel_size=0.1))
        pipeline.add(ll.filters.DynamicTemporalSOR(...))
        cleaned = pipeline.process(raw_frame)
    """

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self.ring_buffer = RingBuffer(capacity=window_size)
        self.operators: List[Any] = []

    def add(self, operator: Any) -> TemporalPipeline:
        """Chain an operator into the pipeline filter graph.

        Args:
            operator: Filter or transformation operator implementing either
                __call__(frame) or process(frame, ring_buffer).

        Returns:
            self for fluent method chaining.
        """
        self.operators.append(operator)
        return self

    def process(self, raw_frame: Union[np.ndarray, LidarFrame]) -> LidarFrame:
        """Ingest a new raw scan and run the filter operator pipeline.

        Args:
            raw_frame: Raw (N, 3) or (N, 4) NumPy array or LidarFrame instance.

        Returns:
            Cleaned and processed LidarFrame.
        """
        if isinstance(raw_frame, LidarFrame):
            current_frame = raw_frame.copy()
        else:
            current_frame = LidarFrame(points=np.ascontiguousarray(raw_frame, dtype=np.float32))

        import time
        t_start = time.perf_counter()

        # Push to sliding ring buffer before executing temporal filters
        self.ring_buffer.push(current_frame)

        # Execute operators in sequence
        for op in self.operators:
            if hasattr(op, "process_temporal"):
                current_frame = op.process_temporal(current_frame, self.ring_buffer)
            elif hasattr(op, "process"):
                current_frame = op.process(current_frame)
            elif callable(op):
                res = op(current_frame)
                if isinstance(res, LidarFrame):
                    current_frame = res
                elif isinstance(res, np.ndarray):
                    current_frame = LidarFrame(
                        points=res,
                        timestamp=current_frame.timestamp,
                        frame_id=current_frame.frame_id,
                        metadata=current_frame.metadata,
                    )
                else:
                    raise TypeError(f"Operator {op} returned unsupported type {type(res)}")

        self._last_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return current_frame

    def get_profiling_stats(self) -> Dict[str, Any]:
        """Return runtime profiling metrics for the pipeline."""
        return {
            "num_operators": len(self.operators),
            "ring_buffer_frames": len(self.ring_buffer),
            "pipeline_ms": getattr(self, "_last_latency_ms", 0.0),
        }

    def reset(self) -> None:
        """Reset historical sliding buffer state."""
        self.ring_buffer.clear()
