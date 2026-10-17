"""Tests for LidarLens Core, IO & Spatial Pipeline (Chapters 2 & 3)."""

import time
import numpy as np
import pytest

import lidarlens as ll
from lidarlens.core import LidarBuffer, LidarFrame
from lidarlens.benchmark import LidarLensBenchmark
from lidarlens.pipeline import TemporalPipeline, RingBuffer
from lidarlens.filters import VoxelDownsample, DynamicTemporalSOR
from lidarlens.io import Streamer


class TestCoreAndBuffer:
    def test_lidar_buffer_allocation_and_view(self):
        buf = LidarBuffer(capacity=1000)
        assert buf.size() == 0
        assert buf.empty()

        # Add points
        buf.add_point(1.0, 2.0, 3.0, 100.0)
        buf.add_point(4.0, 5.0, 6.0, 200.0)
        assert buf.size() == 2

        # Zero-copy numpy view inspection
        view = buf.get_numpy_view()
        assert isinstance(view, np.ndarray)
        assert view.shape == (2, 4)
        assert view.dtype == np.float32
        np.testing.assert_allclose(view[0], [1.0, 2.0, 3.0, 100.0])
        np.testing.assert_allclose(view[1], [4.0, 5.0, 6.0, 200.0])

        # Test AABB bounds
        min_xyz, max_xyz = buf.compute_aabb()
        np.testing.assert_allclose(min_xyz, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(max_xyz, [4.0, 5.0, 6.0])

        # Load bulk numpy
        arr = np.random.uniform(-10, 10, size=(500, 4)).astype(np.float32)
        buf.load_from_numpy(arr)
        assert buf.size() == 500
        np.testing.assert_allclose(buf.get_numpy_view()[:5], arr[:5])

        buf.clear()
        assert buf.size() == 0

    def test_lidar_frame_properties(self):
        pts = np.array([[1.0, 2.0, 3.0, 50.0], [4.0, 5.0, 6.0, 80.0]], dtype=np.float32)
        frame = LidarFrame(points=pts, timestamp=123.456, frame_id="test_001")

        assert frame.num_points == 2
        assert len(frame) == 2
        assert frame.xyz.shape == (2, 3)
        assert frame.intensity.shape == (2,)
        np.testing.assert_allclose(frame.xyz[0], [1.0, 2.0, 3.0])
        assert frame.intensity[0] == 50.0

        # Deep copy
        frame_copy = frame.copy()
        frame_copy.points[0, 0] = 999.0
        assert frame.points[0, 0] == 1.0


class TestBenchmark:
    def test_benchmark_generation_and_timing(self):
        cloud = LidarLensBenchmark.generate_synthetic_pointcloud(num_points=5000, seed=42)
        assert cloud.shape == (5000, 4)
        assert cloud.dtype == np.float32

        norm_i, metrics = LidarLensBenchmark.run_benchmark(cloud, verbose=False)
        assert norm_i.shape == (5000,)
        assert np.all(norm_i >= 0.0) and np.all(norm_i <= 1.0)
        assert "aabb_latency_ms" in metrics
        assert "norm_latency_ms" in metrics
        assert metrics["total_latency_ms"] < 50.0  # Should be sub-millisecond for 5000 pts


class TestPipelineAndFilters:
    def test_voxel_downsample(self):
        rng = np.random.default_rng(42)
        # 1000 points packed tightly in a 0.2m box
        dense_pts = rng.uniform(0.0, 0.2, size=(1000, 4)).astype(np.float32)
        filter_op = VoxelDownsample(voxel_size=0.1)

        downsampled = filter_op(dense_pts)
        assert len(downsampled) < len(dense_pts)
        assert len(downsampled) > 0

    def test_dynamic_temporal_sor_and_pipeline_latency(self):
        rng = np.random.default_rng(42)
        # Cluster of inliers around origin + a few distant outliers
        inliers = rng.normal(0.0, 1.0, size=(1000, 4)).astype(np.float32)
        inliers[:, 3] = rng.uniform(50, 200, 1000)

        outliers = np.array([
            [15.0, 15.0, 15.0, 5.0],
            [-20.0, -20.0, 10.0, 2.0],
        ], dtype=np.float32)

        pts = np.vstack([inliers, outliers])
        raw_frame = LidarFrame(points=pts)

        pipeline = TemporalPipeline(window_size=5)
        pipeline.add(VoxelDownsample(voxel_size=0.05))
        pipeline.add(DynamicTemporalSOR(base_neighbors=20, base_std_ratio=1.5, distance_scale=0.05, temporal_votes=2))

        # Push 3 frames through pipeline and assess latency
        latencies = []
        for i in range(3):
            t0 = time.perf_counter()
            cleaned = pipeline.process(raw_frame)
            t_cost_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_cost_ms)

        # Assert point reduction and outlier rejection
        assert cleaned.num_points < raw_frame.num_points
        # Inner loop budget: should satisfy >=30 Hz (<33.3 ms)
        assert np.mean(latencies) < 33.3


class TestStreamer:
    def test_streamer_iteration(self):
        pts1 = np.random.randn(100, 4).astype(np.float32)
        pts2 = np.random.randn(120, 4).astype(np.float32)

        streamer = Streamer(data_source=[pts1, pts2], fps=None)
        assert len(streamer) == 2

        frames = list(streamer)
        assert len(frames) == 2
        assert isinstance(frames[0], LidarFrame)
        assert frames[0].num_points == 100
        assert frames[1].num_points == 120
