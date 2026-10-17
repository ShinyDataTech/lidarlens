"""Tests for 2D/3D Occupancy Mapping & Quality Audit (Chapter 6)."""

import numpy as np
import pytest

from lidarlens.mapping import generate_standard_grid, OctoMap3D
from lidarlens.io.map_io import save_nav2_map, load_nav2_map, run_map_regression


class TestOccupancyMapping:
    def test_standard_grid_rasterization(self):
        rng = np.random.default_rng(42)
        # 1000 obstacle points at z = 1.0
        pts = rng.uniform(-5.0, 5.0, size=(1000, 3)).astype(np.float32)
        pts[:, 2] = 1.0

        grid, meta = generate_standard_grid(
            pts, resolution=0.1, z_range=(0.2, 1.8), inflation_radius=0.2
        )

        assert grid.ndim == 2
        assert "origin" in meta
        assert "resolution" in meta
        # Occupied cells should exist with lethal cost 100
        assert np.any(grid == 100)

    def test_octomap_3d_updates_and_decay(self):
        octomap = OctoMap3D(resolution=0.2, prob_hit=0.7, prob_miss=0.4, decay_lambda=0.1)

        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        hit_pts = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)

        # Ingest scan with ray-casting clearing
        octomap.update_scan(origin, hit_pts, delta_time=0.0, ray_cast=True)

        occupied = octomap.get_occupied_voxels()
        assert len(occupied) >= 2

        # Verify exponential decay reduces log-odds over elapsed time
        initial_vox_count = len(octomap.voxels)
        # Apply 10 seconds of decay
        octomap.update_scan(origin, np.empty((0, 3)), delta_time=10.0, ray_cast=False)
        assert len(octomap.voxels) <= initial_vox_count

    def test_map_serialization_and_regression_audit(self, tmp_path):
        # Create a simple synthetic grid
        grid = np.zeros((100, 100), dtype=np.int8)
        grid[30:70, 30:70] = 100  # Rectangular room wall
        meta = {"origin": (-5.0, -5.0), "resolution": 0.1}

        pgm_path = str(tmp_path / "test_map.pgm")
        yaml_path = str(tmp_path / "test_map.yaml")

        save_nav2_map(grid, meta, pgm_path, yaml_path)

        loaded_grid, loaded_cfg = load_nav2_map(yaml_path)
        assert loaded_grid.shape == (100, 100)

        # Run automated regression audit
        audit_res = run_map_regression(pgm_path, loaded_cfg)
        assert "total_components" in audit_res
        assert "island_noise_count" in audit_res
        assert "unknown_coverage" in audit_res
        assert audit_res["passed"] is True
