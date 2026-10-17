"""Tests for Terrain Modeling, Clustering & Geometry (Chapter 4)."""

import numpy as np
import pytest

import lidarlens as ll
from lidarlens.terrain import create_elevation_map, analyze_traversability
from lidarlens.cluster import AdaptiveDBSCAN
from lidarlens.geometry import get_pca_obb, project_to_costmap
from lidarlens.filters import RANSACGround, PatchworkPlusPlus


class TestElevationMapAndTraversability:
    def test_vectorized_elevation_map(self):
        rng = np.random.default_rng(42)
        # Points spanning [-5, 5] on X and Y, with a slope on Z
        pts = rng.uniform(-4.0, 4.0, size=(5000, 3)).astype(np.float32)
        pts[:, 2] = 0.2 * pts[:, 0]  # Inclined ramp

        z_map, z_max, z_min = create_elevation_map(
            pts, grid_size=0.2, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0)
        )

        assert z_map.shape == (50, 50)
        assert z_max.shape == (50, 50)
        assert z_min.shape == (50, 50)

        # Non-empty cells should satisfy z_min <= z_map <= z_max
        valid = ~np.isnan(z_map)
        assert np.any(valid)
        assert np.all(z_min[valid] <= z_map[valid] + 1e-4)
        assert np.all(z_map[valid] <= z_max[valid] + 1e-4)

    def test_traversability_analysis(self):
        # Create a synthetic ramp and a cliff/step
        z_map = np.zeros((40, 40), dtype=np.float32)
        z_map[:, 20:] += 1.0  # 1.0 meter step obstacle
        z_max = z_map.copy()
        z_min = z_map.copy()
        z_max[:, 20] = 1.0
        z_min[:, 20] = 0.0

        cost, mask = analyze_traversability(z_map, z_max, z_min, res=0.1)

        assert cost.shape == (40, 40)
        assert mask.shape == (40, 40)
        assert mask.dtype == np.uint8
        assert np.all(cost >= 0.0) and np.all(cost <= 1.0)
        # Flat area should be passable (mask > 0)
        assert mask[5, 5] > 0


class TestClusteringAndOBB:
    def test_adaptive_dbscan(self):
        rng = np.random.default_rng(42)
        # Cluster 1 near origin
        c1 = rng.normal(loc=[2.0, 0.0, 0.0], scale=0.2, size=(100, 3)).astype(np.float32)
        # Cluster 2 farther away
        c2 = rng.normal(loc=[15.0, 0.0, 0.0], scale=0.5, size=(100, 3)).astype(np.float32)
        pts = np.vstack([c1, c2])

        dbscan = AdaptiveDBSCAN(eps_min=0.3, alpha=0.02, min_samples=6)
        labels = dbscan.fit_predict(pts)

        assert len(labels) == 200
        # Both distinct clusters should be discovered
        unique_labels = set(labels) - {-1}
        assert len(unique_labels) >= 2

        clusters = dbscan.extract_clusters(pts)
        assert len(clusters) >= 2

    def test_pca_obb_and_costmap_projection(self):
        rng = np.random.default_rng(42)
        # Vehicle-sized bounding box: length=4.0, width=2.0 along X, Y centered at (3, 4)
        x = rng.uniform(-2.0, 2.0, size=300)
        y = rng.uniform(-1.0, 1.0, size=300)
        z = rng.uniform(-0.5, 0.5, size=300)

        # Rotate by 30 degrees (pi/6)
        theta = np.pi / 6.0
        c, s = np.cos(theta), np.sin(theta)
        x_rot = x * c - y * s + 3.0
        y_rot = x * s + y * c + 4.0

        cluster_pts = np.column_stack([x_rot, y_rot, z]).astype(np.float32)

        center, size, yaw = get_pca_obb(cluster_pts)

        # Center should be near (3, 4)
        np.testing.assert_allclose(center, [3.0, 4.0], atol=0.25)
        # Dimensions should be near 4.0 and 2.0
        dims = sorted([size[0], size[1]])
        np.testing.assert_allclose(dims, [2.0, 4.0], atol=0.4)

        # Project to costmap
        costmap = np.zeros((100, 100), dtype=np.uint8)
        dilated = project_to_costmap(center, size, yaw, costmap, res=0.1)

        assert dilated.shape == (100, 100)
        assert np.sum(dilated > 0) > 0


class TestGroundSegmentation:
    def test_ransac_ground(self):
        rng = np.random.default_rng(42)
        # Flat ground at z ≈ 0
        ground = rng.uniform(-10.0, 10.0, size=(500, 3)).astype(np.float32)
        ground[:, 2] = rng.normal(0.0, 0.02, size=500)

        # Obstacles above ground at z ≈ 1.5
        obstacles = rng.uniform(-5.0, 5.0, size=(200, 3)).astype(np.float32)
        obstacles[:, 2] = rng.uniform(1.0, 2.5, size=200)

        pts = np.vstack([ground, obstacles])

        ransac = RANSACGround(distance_threshold=0.15, max_iterations=50)
        ground_mask, plane = ransac.segment(pts)

        assert ground_mask.shape == (700,)
        # Dominant normal should be vertical (|nz| > 0.8)
        assert abs(plane[2]) > 0.8
        # Majority of ground points should be identified
        assert np.sum(ground_mask[:500]) > 450
        # Obstacle points should not be classified as ground
        assert np.sum(ground_mask[500:]) < 20

    def test_patchwork_plus_plus(self):
        rng = np.random.default_rng(42)
        ground = rng.uniform(-15.0, 15.0, size=(1000, 3)).astype(np.float32)
        ground[:, 2] = 0.0

        obstacles = rng.uniform(-10.0, 10.0, size=(300, 3)).astype(np.float32)
        obstacles[:, 2] = 1.8

        pts = np.vstack([ground, obstacles])

        patchwork = PatchworkPlusPlus(czm_rings=4, czm_sectors=16, elevation_thresh=0.3)
        g_mask, non_g_mask = patchwork.segment(pts)

        assert len(g_mask) == 1300
        assert np.sum(g_mask[:1000]) > 700
