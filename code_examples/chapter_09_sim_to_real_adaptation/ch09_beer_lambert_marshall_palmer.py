"""
Chapter 9: Sim-to-Real Domain Adaptation & Test-Time Adaptation
Code Example 9.1: Beer-Lambert Atmospheric Attenuation & Marshall-Palmer Rain Simulation.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第9章 9.1 & 9.2节
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.sim2real import (
    beer_lambert_attenuation,
    sample_marshall_palmer_rain,
    inject_weather_noise
)


def generate_clean_scene(num_points: int = 20_000, seed: int = 42) -> np.ndarray:
    """Generate a clean synthetic LiDAR scan with distance-dependent reflectivity."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((num_points, 4), dtype=np.float32)
    # Target points distributed in 3D space
    r = rng.uniform(2.0, 75.0, num_points)
    az = rng.uniform(-np.pi * 0.4, np.pi * 0.4, num_points)
    el = rng.uniform(-np.radians(15.0), np.radians(8.0), num_points)
    
    pts[:, 0] = r * np.cos(el) * np.cos(az)
    pts[:, 1] = r * np.cos(el) * np.sin(az)
    pts[:, 2] = r * np.sin(el)
    # Clean high reflection targets (e.g. car body, road markers)
    pts[:, 3] = rng.uniform(50.0, 220.0, num_points).astype(np.float32)
    return pts


def main():
    print("=== Chapter 9: Physical Weather Simulation (Beer-Lambert & Marshall-Palmer) ===")
    
    clean_points = generate_clean_scene(num_points=20_000, seed=42)
    print(f"Clean Benchmark Scene: {len(clean_points):,} points")
    print(f"Initial Mean Intensity: {np.mean(clean_points[:, 3]):.2f} (std={np.std(clean_points[:, 3]):.2f})")
    
    # 1. Beer-Lambert Atmospheric Attenuation Test
    ranges = np.linalg.norm(clean_points[:, :3], axis=1)
    sigma_fog = 0.025 # Dense fog extinction coefficient (m^-1)
    
    t0 = time.perf_counter()
    attenuated_intensities = beer_lambert_attenuation(clean_points[:, 3], ranges, sigma_ext=sigma_fog)
    atten_ms = (time.perf_counter() - t0) * 1000.0
    
    print(f"\nBeer-Lambert Attenuation computed in {atten_ms:.2f} ms:")
    print(f"  Near Target (r=5m) Mean Transmittance:  {np.mean(attenuated_intensities[ranges < 10] / clean_points[ranges < 10, 3]):.4f}")
    print(f"  Far Target (r=50m) Mean Transmittance:  {np.mean(attenuated_intensities[ranges > 45] / clean_points[ranges > 45, 3]):.4f}")
    
    # Mathematical assertion: Far target transmittance must follow exp(-2 * sigma * R)
    near_trans = np.mean(attenuated_intensities[ranges < 10] / clean_points[ranges < 10, 3])
    far_trans = np.mean(attenuated_intensities[ranges > 45] / clean_points[ranges > 45, 3])
    assert near_trans > far_trans, "Beer-Lambert attenuation monotonicity violated!"
    assert far_trans < 0.20, f"Far field transmittance {far_trans:.4f} too high for dense fog extinction!"
    
    # 2. Marshall-Palmer Raindrop Size Distribution Simulation
    rain_rate = 30.0 # Heavy rain (30 mm/h)
    t0 = time.perf_counter()
    rain_points = sample_marshall_palmer_rain(
        clean_points,
        rain_rate_mm_hr=rain_rate,
        max_noise_ratio=0.15,
        seed=100
    )
    rain_ms = (time.perf_counter() - t0) * 1000.0
    
    num_rain_clutter = len(rain_points) - len(clean_points)
    print(f"\nMarshall-Palmer Rain Model (Rate={rain_rate} mm/h) computed in {rain_ms:.2f} ms:")
    print(f"  Total Scene Points: {len(rain_points):,} (Clean: {len(clean_points):,}, Injected Rain Noise: {num_rain_clutter:,})")
    
    # Noise points are appended at the end
    noise_part = rain_points[len(clean_points):]
    print(f"  Rain Clutter Mean Distance:  {np.mean(np.linalg.norm(noise_part[:, :3], axis=1)):.2f} m (near-sensor backscattering)")
    print(f"  Rain Clutter Mean Intensity: {np.mean(noise_part[:, 3]):.2f} (weak backscatter energy)")
    
    assert num_rain_clutter > 0, "No rain clutter points were injected!"
    assert np.mean(noise_part[:, 3]) < 20.0, "Rain clutter reflections unrealistically high!"
    assert np.mean(np.linalg.norm(noise_part[:, :3], axis=1)) < 20.0, "Rain clutter should predominantly cluster near sensor."
    
    print("Chapter 9.1 verification completed successfully.\n")


if __name__ == "__main__":
    main()
