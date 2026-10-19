"""
Chapter 3: Spatial Indexing & Real-Time Point Cloud Denoising
Code Example 3.2: LidarLens Real-Time Streaming & Temporal Pipeline Denoising.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第3章 3.3.1节
"""

import time
import numpy as np
import tempfile
import os
import lidarlens as ll


def generate_mock_kitti_sequence(dir_path: str, num_frames: int = 8, pts_per_frame: int = 30_000):
    """Generate synthetic KITTI .bin sequence with dynamic noise and rain reflections."""
    os.makedirs(dir_path, exist_ok=True)
    for f_idx in range(num_frames):
        pts = np.zeros((pts_per_frame, 4), dtype=np.float32)
        # 1. Roadway ground plane (dense surface)
        n_ground = int(pts_per_frame * 0.75)
        pts[:n_ground, 0] = np.random.uniform(2, 40, n_ground)
        pts[:n_ground, 1] = np.random.uniform(-10, 10, n_ground)
        pts[:n_ground, 2] = -1.7 + np.random.normal(0, 0.04, n_ground)
        pts[:n_ground, 3] = np.random.uniform(0.2, 0.8, n_ground)
        
        # 2. Side structures / cars
        n_struct = int(pts_per_frame * 0.20)
        idx_s = slice(n_ground, n_ground + n_struct)
        pts[idx_s, 0] = np.random.uniform(5, 30, n_struct)
        pts[idx_s, 1] = np.random.choice([-8.0, 8.0], n_struct) + np.random.normal(0, 0.3, n_struct)
        pts[idx_s, 2] = np.random.uniform(-1.5, 2.0, n_struct)
        pts[idx_s, 3] = np.random.uniform(0.4, 0.9, n_struct)
        
        # 3. Aerosol rain noise
        n_noise = pts_per_frame - (n_ground + n_struct)
        idx_n = slice(n_ground + n_struct, pts_per_frame)
        pts[idx_n, 0] = np.random.uniform(2, 40, n_noise)
        pts[idx_n, 1] = np.random.uniform(-15, 15, n_noise)
        pts[idx_n, 2] = np.random.uniform(3, 10, n_noise)
        pts[idx_n, 3] = np.random.uniform(0.01, 0.08, n_noise)
        
        bin_file = os.path.join(dir_path, f"{f_idx:06d}.bin")
        pts.tofile(bin_file)


def main():
    print("=== Chapter 3: LidarLens Streaming & Temporal Pipeline Denoising ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        seq_dir = os.path.join(tmpdir, "velodyne")
        generate_mock_kitti_sequence(seq_dir, num_frames=6, pts_per_frame=25_000)
        
        # 1. Initialize LidarLens Streamer
        streamer = ll.Streamer(input_source=seq_dir, file_format="kitti_bin")
        print(f"Initialized LidarLens Streamer with {len(streamer)} frames.")
        
        # 2. Build Temporal Perception Pipeline
        pipeline = ll.TemporalPipeline(window_size=5)
        # Add Voxel Downsampling (leaf_size = 0.4m for automotive real-time inner-loop)
        pipeline.add(ll.VoxelDownsample(leaf_size=0.4))
        # Add Dynamic Temporal SOR (distance_scale = 0.02, temporal_votes = 2)
        pipeline.add(ll.DynamicTemporalSOR(
            k_neighbors=16,
            base_std_ratio=1.5,
            distance_scale=0.02,
            temporal_votes=2
        ))
        
        # 3. Process Streamed Frames
        latencies = []
        for frame in streamer:
            t0 = time.perf_counter()
            cleaned_frame = pipeline.process(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            
            orig_n = frame.num_points
            clean_n = cleaned_frame.num_points
            filtered_ratio = (orig_n - clean_n) / orig_n * 100.0
            
            print(f"Frame #{frame.frame_id}: {orig_n:,} -> {clean_n:,} pts "
                  f"(filtered {filtered_ratio:.1f}%) in {elapsed_ms:.2f} ms")
            
        avg_latency_ms = float(np.mean(latencies))
        stats = pipeline.get_profiling_stats()
        print("\n" + "=" * 55)
        print("          Temporal Pipeline Profiling Statistics")
        print("=" * 55)
        print(f"Average Frame Latency:  {avg_latency_ms:.2f} ms ({1000.0/avg_latency_ms:.1f} FPS)")
        print(f"RingBuffer Cached:      {stats.get('ring_buffer_frames', 0)} frames")
        print(f"Meets >=30Hz Target:    {avg_latency_ms < 33.3}")
        print("=" * 55)
        
        # In C++ accelerated mode, latency is < 15ms. In pure Python fallback, allow up to 100ms
        assert avg_latency_ms < 100.0, f"Pipeline latency {avg_latency_ms:.2f}ms too high!"

    print("Chapter 3.2 verification completed successfully.\n")


if __name__ == "__main__":
    main()
