"""
Chapter 9: Sim-to-Real Domain Adaptation & Test-Time Adaptation
Code Example 9.2: Cascade Rain & Atmospheric Scatter Denoising Filter (Intensity + ROR + SOR).

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第9章 9.2.3节 (Para 122)
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.sim2real import rain_pointcloud_denoise, sample_marshall_palmer_rain


def generate_rain_corrupted_dataset():
    """Create a realistic mixed LiDAR scene with solid targets and rain noise."""
    rng = np.random.default_rng(42)
    
    # 1. Solid targets (Road + vehicle within typical 25m city perception zone)
    n_road = 6_000
    road_x = rng.uniform(2.0, 25.0, n_road)
    road_y = rng.uniform(-8.0, 8.0, n_road)
    road_z = rng.normal(-1.6, 0.05, n_road)
    road_int = rng.uniform(100.0, 255.0, n_road)
    road_pts = np.column_stack([road_x, road_y, road_z, road_int])
    
    # Vehicle 1 (Dense box cluster)
    n_v1 = 1_500
    v1_x = rng.uniform(8.0, 12.5, n_v1)
    v1_y = rng.uniform(-2.5, -0.7, n_v1)
    v1_z = rng.uniform(-1.5, 0.1, n_v1)
    v1_int = rng.uniform(150.0, 255.0, n_v1)
    v1_pts = np.column_stack([v1_x, v1_y, v1_z, v1_int])
    
    clean_targets = np.vstack([road_pts, v1_pts]).astype(np.float32)
    n_clean = len(clean_targets)
    
    # 2. Inject Marshall-Palmer rain (20 mm/h)
    corrupted_scene = sample_marshall_palmer_rain(
        clean_targets,
        rain_rate_mm_hr=20.0,
        max_noise_ratio=0.15,
        seed=100
    )
    
    # Target ground truth mask: first n_clean are real targets, remainder are rain noise
    is_noise_gt = np.zeros(len(corrupted_scene), dtype=bool)
    is_noise_gt[n_clean:] = True
    
    return corrupted_scene, is_noise_gt, n_clean


def main():
    print("=== Chapter 9: Cascade Rain Point Cloud Denoising Filter ===")
    
    corrupted_scene, is_noise_gt, n_clean = generate_rain_corrupted_dataset()
    n_total = len(corrupted_scene)
    n_noise = int(np.sum(is_noise_gt))
    
    print(f"Corrupted Input Scene: {n_total:,} points")
    print(f"  Real Target Points:  {n_clean:,}")
    print(f"  Injected Rain Noise: {n_noise:,} ({n_noise / n_total * 100:.1f}%)")
    
    # Execute 3-Stage Cascade Denoising:
    # 1. Intensity Threshold Filtering
    # 2. Radius Outlier Removal (ROR)
    # 3. Statistical Outlier Removal (SOR)
    t0 = time.perf_counter()
    kept_mask = rain_pointcloud_denoise(
        points_xyz=corrupted_scene[:, :3],
        intensity=corrupted_scene[:, 3],
        intensity_thr=6.0,
        ror_radius=0.7,
        ror_min_neighbors=4,
        sor_k=16,
        sor_z_thr=2.2
    )
    denoise_ms = (time.perf_counter() - t0) * 1000.0
    
    denoised_pts = corrupted_scene[kept_mask]
    n_kept = len(denoised_pts)
    
    # Evaluation Metrics
    # Noise Rejection: What % of true noise was filtered out?
    noise_rejected = np.sum(is_noise_gt & (~kept_mask))
    noise_rejection_rate = noise_rejected / n_noise
    
    # Target Preservation: What % of true targets were retained?
    targets_preserved = np.sum((~is_noise_gt) & kept_mask)
    target_retention_rate = targets_preserved / n_clean
    
    print(f"\nCascade Denoising completed in {denoise_ms:.2f} ms:")
    print(f"  Filtered Output:       {n_kept:,} points retained")
    print(f"  Noise Rejection Rate:   {noise_rejection_rate * 100:.2f}% (Target: >85%)")
    print(f"  Target Retention Rate:  {target_retention_rate * 100:.2f}% (Target: >90%)")
    
    # Assertions
    assert noise_rejection_rate >= 0.85, f"Noise rejection rate {noise_rejection_rate:.3f} below 85% requirement!"
    assert target_retention_rate >= 0.90, f"Target retention rate {target_retention_rate:.3f} below 90% requirement!"
    
    print("Rain cascade denoising algorithm verified successfully.")
    print("Chapter 9.2 verification completed successfully.\n")


if __name__ == "__main__":
    main()
