"""
Chapter 2: Python 3 Development Environment & LidarLens Architecture
Code Example 2.3: Million-Scale LiDAR Benchmark & Zero-Copy Protocol Audit.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第2章 2.3.1节
"""

import sys
import time
import numpy as np
from typing import Tuple, Dict, Any

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import lidarlens as ll
from lidarlens.benchmark import LidarLensBenchmark


def run_comprehensive_benchmark(num_points: int = 1_000_000) -> Dict[str, Any]:
    """Execute full performance audit on memory throughput and vector operations."""
    print(f"Generating synthetic point cloud with {num_points:,} points...")
    t0 = time.perf_counter()
    cloud = LidarLensBenchmark.generate_synthetic_pointcloud(num_points=num_points)
    gen_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Synthetic generation completed in {gen_time_ms:.2f} ms.")

    # 1. Zero-Copy LidarBuffer Allocation and View
    t0 = time.perf_counter()
    buffer = ll.LidarBuffer(capacity=num_points)
    buffer.copy_from(cloud)
    view = buffer.get_numpy_view()
    buffer_time_ms = (time.perf_counter() - t0) * 1000.0
    
    # Verify zero-copy: modifying buffer view modifies underlying data
    original_val = view[0, 0]
    view[0, 0] += 1.0
    assert buffer.get_numpy_view()[0, 0] == original_val + 1.0, "Zero-copy view test failed"
    view[0, 0] = original_val # Restore
    # 2. Benchmark AABB & Intensity Normalization
    norm_cloud, bench_results = LidarLensBenchmark.run_benchmark(cloud, verbose=False)
    
    # 3. Memory layout footprint
    mem_mb = cloud.nbytes / (1024 * 1024)
    throughput_mpts = (num_points / 1e6) / (bench_results["total_latency_ms"] / 1000.0)

    print("\n" + "=" * 50)
    print("      LidarLens Zero-Copy Benchmark Audit Report")
    print("=" * 50)
    print(f"Point Cloud Size:       {num_points:,} points ({mem_mb:.2f} MB)")
    print(f"Buffer Ingestion Time:  {buffer_time_ms:.2f} ms")
    print(f"AABB Bounding Box Time: {bench_results['aabb_latency_ms']:.2f} ms")
    print(f"Intensity Norm Time:    {bench_results['norm_latency_ms']:.2f} ms")
    print(f"Total Pipeline Latency: {bench_results['total_latency_ms']:.2f} ms")
    print(f"Effective Throughput:   {throughput_mpts:.2f} Mpts/s")
    print("=" * 50)

    return {
        "num_points": num_points,
        "memory_mb": mem_mb,
        "buffer_time_ms": buffer_time_ms,
        "bench_results": bench_results,
        "throughput_mpts": throughput_mpts
    }


def main():
    print("=== Chapter 2: Million-Scale LiDAR Benchmark & Zero-Copy Audit ===")
    results = run_comprehensive_benchmark(num_points=1_000_000)
    assert results["throughput_mpts"] > 5.0, "Throughput fell below minimum threshold!"
    print("Chapter 2.3 verification completed successfully.\n")


if __name__ == "__main__":
    main()
