"""LidarLens Performance Benchmarking Suite (Chapter 2).

Provides synthetic multi-million point cloud generation and throughput/latency
benchmarks for AABB bounding box searches and vectorized intensity normalization.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np


class LidarLensBenchmark:
    """Benchmark suite assessing processing latency and memory efficiency."""

    @staticmethod
    def generate_synthetic_pointcloud(
        num_points: int = 10_000_000,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate high-density synthetic LiDAR point cloud (x, y, z, intensity).

        Simulates real physical sensor properties: intensity decays inversely with
        radial distance and incorporates Gaussian receiver noise.

        Args:
            num_points: Number of 3D points to simulate (default 10,000,000).
            seed: Random seed for deterministic reproducibility.

        Returns:
            Continuous (num_points, 4) float32 NumPy array.
        """
        rng = np.random.default_rng(seed)
        xyz = rng.uniform(-50.0, 50.0, size=(num_points, 3)).astype(np.float32)

        # Intensity inverse-distance decay simulation
        dist = np.linalg.norm(xyz, axis=1)
        noise = rng.normal(0.0, 5.0, size=num_points).astype(np.float32)
        intensity = np.clip(255.0 / (dist + 1.0) + noise, 0.0, 255.0).astype(np.float32)

        return np.column_stack((xyz, intensity))

    @staticmethod
    def run_benchmark(
        cloud_data: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Evaluate AABB bounding box search and intensity normalization latency.

        Args:
            cloud_data: (N, 4) float32 point cloud array.
            verbose: If True, prints formatted timing logs to stdout.

        Returns:
            Tuple of (normalized_intensity_array, metrics_dict).
        """
        num_points = cloud_data.shape[0]

        # 1. Axis-Aligned Bounding Box (AABB) search
        t0 = time.perf_counter()
        min_bound = np.min(cloud_data[:, :3], axis=0)
        max_bound = np.max(cloud_data[:, :3], axis=0)
        t1 = time.perf_counter()
        aabb_latency_ms = (t1 - t0) * 1000.0

        # 2. Vectorized Intensity Normalization
        t2 = time.perf_counter()
        intensities = cloud_data[:, 3]
        min_i, max_i = float(np.min(intensities)), float(np.max(intensities))
        norm_intensity = (intensities - min_i) / (max_i - min_i + 1e-6)
        t3 = time.perf_counter()
        norm_latency_ms = (t3 - t2) * 1000.0

        metrics = {
            "num_points": float(num_points),
            "aabb_latency_ms": aabb_latency_ms,
            "norm_latency_ms": norm_latency_ms,
            "total_latency_ms": aabb_latency_ms + norm_latency_ms,
        }

        if verbose:
            print(f"\n=== LidarLens 性能基准测试 (数据规模: {num_points:,} 点) ===")
            print(f"1. 三维 AABB 包围盒计算耗时: {aabb_latency_ms:.2f} ms")
            print(f"   Bounds: Min {min_bound.round(2)}, Max {max_bound.round(2)}")
            print(f"2. 反射强度 (Intensity) 向量化归一化耗时: {norm_latency_ms:.2f} ms")
            print(f"   总基准耗时: {metrics['total_latency_ms']:.2f} ms")

        return norm_intensity, metrics
