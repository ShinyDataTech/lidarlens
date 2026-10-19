"""
Chapter 10: 3D Scene Graphs & Agentic Embodied AI
Code Example 10.2: Multi-Modal Spatio-Temporal Association & EWMA Feature Fusion.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第10章 10.2节 (Para 67)
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.scenegraph import SceneGraph3D, SceneNode, associate_and_update


def main():
    print("=== Chapter 10: Multi-Modal Spatio-Temporal Association & Fusion ===")

    rng = np.random.default_rng(42)

    # 1. Initialize Active Scene Graph with an existing object node
    graph = SceneGraph3D()

    # Generate synthetic 512-D CLIP visual embedding for a Wooden Chair
    base_feat_chair = rng.normal(0.0, 1.0, size=512).astype(np.float32)
    base_feat_chair /= np.linalg.norm(base_feat_chair)

    initial_chair = SceneNode(
        id="chair_01",
        label="chair",
        pose=np.array([2.50, 1.20, 0.45], dtype=np.float32),
        extent=np.array([0.6, 0.6, 0.9], dtype=np.float32),
        layer=2,
        room="living_room",
        appearance_feat=base_feat_chair.copy(),
        confidence=0.85
    )
    graph.add_node(initial_chair)
    print(f"Initial Graph Node: {initial_chair.id} at pose {initial_chair.pose}, confidence={initial_chair.confidence}")

    # 2. Case 1: Re-observation of the same chair (subtle pose shift, minor sensor noise on CLIP feat)
    rand_dir = rng.normal(0.0, 1.0, size=512).astype(np.float32)
    rand_dir /= np.linalg.norm(rand_dir)
    noisy_feat_chair = 0.95 * base_feat_chair + 0.05 * rand_dir
    noisy_feat_chair /= np.linalg.norm(noisy_feat_chair)

    obs_frame2 = SceneNode(
        id="temp_obs_02",
        label="chair",
        pose=np.array([2.54, 1.23, 0.46], dtype=np.float32), # 5cm pose measurement drift
        extent=np.array([0.62, 0.59, 0.91], dtype=np.float32),
        layer=2,
        appearance_feat=noisy_feat_chair,
        confidence=0.88
    )

    t0 = time.perf_counter()
    matched_node, is_matched = associate_and_update(
        new_obs=obs_frame2,
        graph=graph,
        mahalanobis_thresh=3.0,
        cosine_thresh=0.85,
        alpha=0.2 # EWMA rate
    )
    assoc_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nCase 1 (Same Instance Re-Observation) evaluated in {assoc_ms:.2f} ms:")
    print(f"  Match Result: is_matched = {is_matched} (Target Node ID: {matched_node.id})")
    print(f"  Updated Filtered Pose: {matched_node.pose} (Smoothly converged)")
    print(f"  Updated Accumulated Confidence: {matched_node.confidence:.2f} (Boosted from 0.85)")

    assert is_matched is True
    assert matched_node.id == "chair_01"
    assert matched_node.confidence > 0.85
    # Pose must be filtered between initial and new observation
    assert 2.50 < matched_node.pose[0] < 2.54

    # 3. Case 2: New Distinct Physical Entity Observation (Lamp at a different position)
    feat_lamp = rng.normal(0.0, 1.0, size=512).astype(np.float32)
    feat_lamp /= np.linalg.norm(feat_lamp)

    obs_lamp = SceneNode(
        id="lamp_01",
        label="floor_lamp",
        pose=np.array([4.80, -2.10, 1.50], dtype=np.float32), # 5 meters away
        extent=np.array([0.4, 0.4, 1.8], dtype=np.float32),
        layer=2,
        appearance_feat=feat_lamp,
        confidence=0.92
    )

    t0 = time.perf_counter()
    new_node, is_matched_case2 = associate_and_update(
        new_obs=obs_lamp,
        graph=graph,
        mahalanobis_thresh=3.0,
        cosine_thresh=0.85
    )
    assoc_case2_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nCase 2 (Novel Object Instance) evaluated in {assoc_case2_ms:.2f} ms:")
    print(f"  Match Result: is_matched = {is_matched_case2} (Expected: False, new node created)")
    print(f"  Graph Total Nodes: {len(graph.nodes)} nodes")

    assert is_matched_case2 is False
    assert len(graph.nodes) == 2
    assert "lamp_01" in graph.nodes

    print("\nMulti-modal spatial and appearance fusion verified successfully.")
    print("Chapter 10.2 verification completed successfully.\n")


if __name__ == "__main__":
    main()
