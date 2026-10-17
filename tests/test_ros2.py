"""Tests for ROS 2 Edge Perception & Reactive Control Node (Chapter 11)."""

import numpy as np
import pytest

from lidarlens.ros2 import LidarPerceptionNode


class TestROS2PerceptionNode:
    def test_perception_node_safe_cruise(self):
        node = LidarPerceptionNode(safe_distance=1.5, roi_width=1.0, obstacle_threshold=10)

        # Points outside the forward ROI (e.g. far behind or wide lateral)
        xyz_safe = np.array([
            [-2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],     # Beyond safe_distance (1.5m)
            [1.0, 3.0, 0.0],     # Beyond roi_width / 2 (0.5m)
        ], dtype=np.float32)

        cmd, latency_ms, obstacle_count = node.process_pointcloud(xyz_safe)

        assert obstacle_count == 0
        # Cruise command
        assert cmd.linear.x == 0.5
        assert cmd.angular.z == 0.0
        assert latency_ms < 33.3  # Well within 30 Hz latency budget

    def test_perception_node_reactive_avoidance(self):
        node = LidarPerceptionNode(safe_distance=1.5, roi_width=1.0, obstacle_threshold=5)

        # 20 points directly inside forward ROI (x in (0, 1.5), y in (-0.5, 0.5))
        xyz_obstacle = np.zeros((20, 3), dtype=np.float32)
        xyz_obstacle[:, 0] = 0.8  # 0.8m ahead
        xyz_obstacle[:, 1] = 0.1  # slightly left

        cmd, latency_ms, obstacle_count = node.process_pointcloud(xyz_obstacle)

        assert obstacle_count == 20
        # Emergency stop and avoidance turn
        assert cmd.linear.x == 0.0
        assert cmd.angular.z == 0.5
        assert latency_ms < 33.3
