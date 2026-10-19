"""
Chapter 11: Edge Deployment & ROS 2 Perception Systems
Code Example 11.2: ROS 2 Edge Real-Time 3D Perception & Reactive Avoidance Control Node.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第11章 11.2节 (Para 47)
"""

import sys
import time
import numpy as np
import lidarlens as ll
from lidarlens.ros2 import LidarPerceptionNode


def generate_scenario_pointclouds():
    """Synthesize LiDAR scans for clear navigation and obstacle intrusion scenarios."""
    rng = np.random.default_rng(42)

    # Scenario A: Clear Path (Obstacles far beyond safe distance > 3.0 m)
    n_clear = 10_000
    clear_pts = np.zeros((n_clear, 3), dtype=np.float32)
    clear_pts[:, 0] = rng.uniform(3.5, 20.0, n_clear) # Ahead > 3.5m
    clear_pts[:, 1] = rng.uniform(-10.0, 10.0, n_clear)
    clear_pts[:, 2] = rng.uniform(-1.5, 1.0, n_clear)

    # Scenario B: Emergency Obstacle Intrusion (Pedestrian / Box at X=1.0m, Y=0.1m)
    # 50 points directly in the forward clearance corridor
    n_obs = 50
    obs_x = rng.uniform(0.8, 1.2, n_obs)
    obs_y = rng.uniform(-0.2, 0.2, n_obs) # Directly in vehicle path
    obs_z = rng.uniform(-0.5, 0.5, n_obs)
    obstacle_cluster = np.column_stack([obs_x, obs_y, obs_z]).astype(np.float32)

    blocked_pts = np.vstack([clear_pts, obstacle_cluster])

    return clear_pts, blocked_pts


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=== Chapter 11: ROS 2 Edge Perception & Reactive Control Closed-Loop ===")

    # 1. Instantiate ROS 2 Edge Perception Node
    # Parameters matching autonomous mobile robot hardware specifications
    node = LidarPerceptionNode(
        safe_distance=1.5,      # 1.5 meter forward emergency braking bubble
        roi_width=1.0,          # 1.0 meter robot lateral safety corridor
        obstacle_threshold=10   # Minimum 10 points to trigger avoidance
    )

    print(f"Node Initialized: [{node.node_name}]")
    print(f"  Forward Safe Distance Threshold: {node.safe_distance:.2f} m")
    print(f"  Corridor ROI Width:             {node.roi_width:.2f} m")
    print(f"  Obstacle Sensitivity Gate:       {node.obstacle_threshold} points")

    clear_scan, blocked_scan = generate_scenario_pointclouds()

    # 2. Test Scenario A: Open Corridor (Safe Cruise)
    cmd_clear, lat_clear, obs_clear = node.process_pointcloud(clear_scan)

    print(f"\nScenario A [Open Path] evaluated in {lat_clear:.3f} ms:")
    print(f"  Points inside ROI:    {obs_clear} (Threshold: {node.obstacle_threshold})")
    print(f"  Linear Velocity  cmd: {cmd_clear.linear.x:.2f} m/s (Cruising forward)")
    print(f"  Angular Velocity cmd: {cmd_clear.angular.z:.2f} rad/s")

    assert obs_clear < node.obstacle_threshold
    assert cmd_clear.linear.x == 0.5, "Robot should cruise forward when path is clear!"
    assert cmd_clear.angular.z == 0.0, "Robot should not turn when path is clear!"

    # 3. Test Scenario B: Frontal Obstacle Intrusion (Emergency Avoidance)
    cmd_blocked, lat_blocked, obs_blocked = node.process_pointcloud(blocked_scan)

    print(f"\nScenario B [Obstacle Intrusion] evaluated in {lat_blocked:.3f} ms:")
    print(f"  Points inside ROI:    {obs_blocked} (Threshold: {node.obstacle_threshold})")
    print(f"  Linear Velocity  cmd: {cmd_blocked.linear.x:.2f} m/s (Brakes engaged)")
    print(f"  Angular Velocity cmd: {cmd_blocked.angular.z:.2f} rad/s (Reactive avoidance turn)")

    assert obs_blocked >= node.obstacle_threshold
    assert cmd_blocked.linear.x == 0.0, "Robot must brake immediately upon obstacle intrusion!"
    assert cmd_blocked.angular.z > 0.0, "Robot must steer away to avoid collision!"

    # 4. Latency Audit
    print(f"\nEdge Hardware Latency Audit:")
    print(f"  Mean Perception Cycle Latency: {(lat_clear + lat_blocked) / 2.0:.3f} ms")
    assert (lat_clear + lat_blocked) / 2.0 < 5.0, "Perception loop exceeds 5ms real-time budget!"

    print("\nROS 2 perception and reactive avoidance control verified successfully.")
    print("Chapter 11.2 verification completed successfully.\n")


if __name__ == "__main__":
    main()
