"""Tests for Odometry, Registration & Degeneracy (Chapter 5)."""

import numpy as np
import pytest

from lidarlens.registration import de_skew_pointcloud, LowLatencyGICP, detect_degeneracy


class TestDeskew:
    def test_motion_deskewing(self):
        # 100 points sampled linearly across 0.1s
        t_samples = np.linspace(0.0, 0.1, 100)
        points = np.zeros((100, 3), dtype=np.float32)
        points[:, 0] = 5.0  # Wall at x = 5.0

        # Vehicle moving forward along X with constant velocity 10 m/s
        pose_data = {
            "t": np.array([0.0, 0.05, 0.1], dtype=np.float64),
            "quat": np.array([
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64),
            "pos": np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ], dtype=np.float64),
        }

        # Reference time at end of frame
        t_ref = 0.1
        corrected = de_skew_pointcloud(points, t_samples, pose_data, t_ref)

        assert corrected.shape == (100, 3)
        # Without deskewing, early points would be shifted by vehicle displacement
        # Corrected points should align to reference time frame


class TestGICPAndDegeneracy:
    def test_low_latency_gicp_odometry(self):
        rng = np.random.default_rng(42)
        scan1 = rng.uniform(-5.0, 5.0, size=(500, 3)).astype(np.float32)

        # Scan 2: rigid translation by delta_x = 0.2, delta_y = 0.1
        scan2 = scan1 + np.array([0.2, 0.1, 0.0], dtype=np.float32)

        gicp = LowLatencyGICP(leaf_size=0.5, max_corr_dist=2.0)
        p1 = gicp.process_scan(scan1)
        p2 = gicp.process_scan(scan2)

        assert p1.shape == (4, 4)
        assert p2.shape == (4, 4)
        assert gicp.last_hessian.shape == (6, 6)

    def test_degeneracy_detection_in_tunnel(self):
        # Construct a synthetic Hessian representing a featureless tunnel along X:
        # Constraint along X translation is nearly zero (unconstrained slide)
        H = np.diag([50.0, 50.0, 50.0, 0.01, 80.0, 80.0]).astype(np.float32)

        deg_info = detect_degeneracy(H, thresh_lambda=0.5, thresh_cond=100.0)

        # Translation degeneracy should be flagged
        assert len(deg_info["trans"]) > 0
        deg_axis = deg_info["trans"][0]
        # Degenerate axis should correspond to X translation [1, 0, 0]
        assert abs(deg_axis[0]) > 0.9
        # Rotation should not be degenerate
        assert len(deg_info["rot"]) == 0
