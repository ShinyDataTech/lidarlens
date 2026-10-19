"""
Chapter 3: Spatial Indexing & Real-Time Point Cloud Denoising
Code Example 3.1: 3D Morton Code Space-Filling Curve & Spatial k-d Tree Indexing.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第3章 3.1 & 3.2节
"""

import time
import numpy as np
from typing import Tuple
from scipy.spatial import cKDTree
import lidarlens as ll


def expand_bits_10bit(v: np.ndarray) -> np.ndarray:
    """Dilate 10-bit integer coordinates into 30 bits for 3D Morton encoding."""
    x = v.astype(np.uint64) & 0x3FF
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8))  & 0x0300F00F
    x = (x | (x << 4))  & 0x030C30C3
    x = (x | (x << 2))  & 0x09249249
    return x


def encode_morton_3d(xyz: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, resolution: int = 1024) -> np.ndarray:
    """Encode 3D coordinates into 30-bit Morton codes (Z-order curve) for memory locality."""
    span = np.maximum(bbox_max - bbox_min, 1e-6)
    norm = np.clip((xyz - bbox_min) / span, 0.0, 1.0)
    grid = (norm * (resolution - 1)).astype(np.uint32)
    
    code_x = expand_bits_10bit(grid[:, 0])
    code_y = expand_bits_10bit(grid[:, 1])
    code_z = expand_bits_10bit(grid[:, 2])
    
    morton_codes = (code_x << 2) | (code_y << 1) | code_z
    return morton_codes


def main():
    print("=== Chapter 3: 3D Morton Code & Spatial Indexing ===")
    
    # 1. Generate synthetic unstructured point cloud
    n_points = 50_000
    points = np.random.uniform(-30, 30, size=(n_points, 3)).astype(np.float32)
    min_b, max_b = points.min(axis=0), points.max(axis=0)
    
    # 2. Compute 3D Morton Codes
    t0 = time.perf_counter()
    codes = encode_morton_3d(points, min_b, max_b)
    morton_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Encoded {n_points:,} points to Morton codes in {morton_time_ms:.2f} ms.")
    
    # Sort points along Morton curve to achieve spatial locality
    sort_idx = np.argsort(codes)
    sorted_points = np.ascontiguousarray(points[sort_idx])
    
    # 3. Build spatial k-d Tree on raw vs Morton-ordered points
    t0 = time.perf_counter()
    tree_raw = cKDTree(points)
    build_raw_ms = (time.perf_counter() - t0) * 1000.0
    
    t0 = time.perf_counter()
    tree_morton = cKDTree(sorted_points)
    build_morton_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"k-d Tree build time (Raw):    {build_raw_ms:.2f} ms")
    print(f"k-d Tree build time (Morton): {build_morton_ms:.2f} ms")
    
    # 4. Perform k-NN query (k=16)
    queries = np.random.uniform(-20, 20, size=(1000, 3)).astype(np.float32)
    t0 = time.perf_counter()
    dists, indices = tree_morton.query(queries, k=16)
    query_ms = (time.perf_counter() - t0) * 1000.0
    print(f"1,000 k-NN queries (k=16) completed in {query_ms:.2f} ms ({query_ms/1000.0*1000:.3f} us/query).")
    
    assert indices.shape == (1000, 16)
    assert dists.shape == (1000, 16)
    print("Chapter 3.1 verification completed successfully.\n")


if __name__ == "__main__":
    main()
