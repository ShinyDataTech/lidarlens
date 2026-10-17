"""LidarLens Sim-to-Real Physical Noise Models (Chapter 9).

Simulates sensor physics discrepancies: Beer-Lambert atmospheric beam attenuation,
Marshall-Palmer raindrop size distribution backscattering, and fog/snow noise.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("lidarlens.sim2real.noise_models")


def beer_lambert_attenuation(
    intensity: np.ndarray,
    ranges: np.ndarray,
    sigma_ext: float = 0.015,
) -> np.ndarray:
    """Apply Beer-Lambert atmospheric optical transmission attenuation.

    Round-trip transmittance over distance R:
        T(R) = exp(-2 * sigma_ext * R)

    Args:
        intensity: (N,) initial reflection intensity array.
        ranges: (N,) radial distance from sensor in meters.
        sigma_ext: Atmospheric extinction coefficient (e.g. 0.015 for moderate rain, 0.05 for dense fog).

    Returns:
        (N,) attenuated reflection intensities.
    """
    transmittance = np.exp(-2.0 * sigma_ext * ranges).astype(np.float32)
    attenuated = intensity * transmittance
    return np.clip(attenuated, 0.0, 255.0).astype(np.float32)


def sample_marshall_palmer_rain(
    points: np.ndarray,
    rain_rate_mm_hr: float = 25.0,
    max_noise_ratio: float = 0.08,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Inject physically consistent raindrop backscatter noise (Chapter 9).

    Uses the Marshall-Palmer raindrop size distribution (DSD):
        N(D) = N_0 * exp(-Lambda * D)
    where Lambda = 4.1 * R_r^(-0.21) (mm^-1). Simulates near-field false-alarm
    returns with low reflection energy.

    Args:
        points: (N, 4) 3D LiDAR point cloud (x, y, z, intensity).
        rain_rate_mm_hr: Rainfall precipitation rate in mm/hour (e.g. 25.0 for heavy rain).
        max_noise_ratio: Maximum proportion of rain clutter points to inject.
        seed: Deterministic random seed.

    Returns:
        (N + M, 4) point cloud with synthesized rain clutter returns.
    """
    rng = np.random.default_rng(seed)
    n_orig = len(points)
    if n_orig == 0:
        return points

    # Marshall-Palmer slope parameter
    Lambda = 4.1 * (max(rain_rate_mm_hr, 1.0) ** -0.21)

    # Number of rain noise points to inject
    n_noise = int(n_orig * max_noise_ratio * min(1.0, rain_rate_mm_hr / 50.0))
    if n_noise <= 0:
        return points

    # Sample raindrop diameters from exponential distribution
    drop_diameters_mm = rng.exponential(scale=(1.0 / Lambda), size=n_noise)

    # Near-field spray: rain reflections cluster near sensor (1 to 20 m)
    azimuth = rng.uniform(-np.pi, np.pi, size=n_noise)
    elevation = rng.uniform(-np.radians(20.0), np.radians(10.0), size=n_noise)
    noise_r = rng.exponential(scale=6.0, size=n_noise) + 0.5  # Skewed towards 0.5-10 m
    noise_r = np.clip(noise_r, 0.5, 25.0)

    nx = noise_r * np.cos(elevation) * np.cos(azimuth)
    ny = noise_r * np.cos(elevation) * np.sin(azimuth)
    nz = noise_r * np.sin(elevation)

    # Rain backscatter intensity is minimal and correlated with drop diameter
    noise_intensity = np.clip(drop_diameters_mm * 5.0 + rng.normal(0.0, 1.0, n_noise), 0.0, 15.0).astype(np.float32)

    rain_pts = np.column_stack([nx, ny, nz, noise_intensity]).astype(np.float32)

    # Attenuate original point cloud intensities via Beer-Lambert
    orig_pts = points.copy()
    orig_r = np.linalg.norm(orig_pts[:, :3], axis=1)
    sigma_rain = 0.005 * (rain_rate_mm_hr ** 0.6)  # Extinction coefficient from rain rate
    orig_pts[:, 3] = beer_lambert_attenuation(orig_pts[:, 3], orig_r, sigma_ext=sigma_rain)

    return np.vstack([orig_pts, rain_pts])


def inject_weather_noise(
    points: np.ndarray,
    weather_type: str = "rain",
    severity: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Unified physical weather noise injection interface.

    Args:
        points: (N, 4) clean LiDAR point cloud.
        weather_type: 'rain', 'fog', or 'clear'.
        severity: Severity factor in [0.0, 1.0].
        seed: Random seed.

    Returns:
        (M, 4) degraded point cloud.
    """
    if weather_type == "clear" or severity <= 0.0:
        return points

    if weather_type == "rain":
        rain_rate = 5.0 + severity * 45.0  # 5 to 50 mm/h
        return sample_marshall_palmer_rain(points, rain_rate_mm_hr=rain_rate, seed=seed)

    elif weather_type == "fog":
        # Fog produces severe Beer-Lambert attenuation and uniform distance scattering
        pts = points.copy()
        r = np.linalg.norm(pts[:, :3], axis=1)
        sigma_fog = 0.02 + severity * 0.08
        pts[:, 3] = beer_lambert_attenuation(pts[:, 3], r, sigma_ext=sigma_fog)
        # Drop points whose attenuated intensity is below detection threshold
        survived = pts[:, 3] >= 1.0
    return points


def rain_pointcloud_denoise(
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    intensity_thr: float = 0.12,
    ror_radius: float = 0.45,
    ror_min_neighbors: int = 6,
    sor_k: int = 16,
    sor_z_thr: float = 2.2,
) -> np.ndarray:
    """Cascade rain/scatter clutter filter (Intensity threshold + ROR + SOR) (Chapter 9).

    Args:
        points_xyz: (N, 3) point positions.
        intensity: (N,) point reflection intensities.
        intensity_thr: minimum intensity threshold.
        ror_radius: Radius Outlier Removal neighborhood sphere radius (m).
        ror_min_neighbors: Minimum neighbors in sphere.
        sor_k: Statistical Outlier Removal k-NN neighbors.
        sor_z_thr: Standard deviation multiplier threshold.

    Returns:
        (N,) boolean mask indicating points to keep.
    """
    from scipy.spatial import cKDTree

    # 1. Intensity gate
    mask_int = intensity >= intensity_thr
    p1 = points_xyz[mask_int]
    if len(p1) == 0:
        return np.zeros(len(points_xyz), dtype=bool)

    # 2. Radius Outlier Removal (ROR)
    kdt1 = cKDTree(p1)
    counts = np.array([len(kdt1.query_ball_point(p, r=ror_radius)) - 1 for p in p1])
    mask_ror = counts >= ror_min_neighbors
    p2 = p1[mask_ror]
    if len(p2) == 0:
        return np.zeros(len(points_xyz), dtype=bool)

    # 3. Statistical Outlier Removal (SOR)
    kdt2 = cKDTree(p2)
    k_query = min(sor_k + 1, len(p2))
    distances, _ = kdt2.query(p2, k=k_query)
    if k_query > 1:
        mean_knn = distances[:, 1:].mean(axis=1)
        std_knn = mean_knn.std()
        if std_knn > 1e-9:
            z_score = (mean_knn - mean_knn.mean()) / std_knn
            mask_sor = z_score <= sor_z_thr
        else:
            mask_sor = np.ones(len(p2), dtype=bool)
    else:
        mask_sor = np.ones(len(p2), dtype=bool)

    idx1 = np.where(mask_int)[0]
    idx2 = idx1[mask_ror]
    idx3 = idx2[mask_sor]

    final_mask = np.zeros(len(points_xyz), dtype=bool)
    final_mask[idx3] = True
    return final_mask

