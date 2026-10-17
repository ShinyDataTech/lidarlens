"""Tests for 3D Scene Graphs & Spatio-Temporal Association (Chapter 10)."""

import numpy as np
import pytest

from lidarlens.scenegraph import SceneNode, SceneEdge, SceneGraph3D, associate_and_update


class TestSceneGraph:
    def test_scenegraph_5tuple_and_query(self):
        graph = SceneGraph3D()

        # Add Room node
        room_node = SceneNode(
            id="room_kitchen",
            label="kitchen",
            pose=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            layer=1,
        )
        graph.add_node(room_node)

        # Add Object nodes
        cup1 = SceneNode(
            id="cup_01",
            label="cup",
            pose=np.array([2.0, 1.0, 0.8], dtype=np.float32),
            room="kitchen",
            layer=2,
            confidence=0.85,
        )
        cup2 = SceneNode(
            id="cup_02",
            label="cup",
            pose=np.array([5.0, 3.0, 0.8], dtype=np.float32),
            room="kitchen",
            layer=2,
            confidence=0.95,
        )
        graph.add_node(cup1)
        graph.add_node(cup2)

        # Add Edge
        edge = SceneEdge(src_id="cup_01", dst_id="room_kitchen", relation="inside")
        graph.add_edge(edge)

        # Query all cups
        all_cups = graph.query_instances(label="cup")
        assert len(all_cups) == 2

        # Query nearest cup
        nearest_cup = graph.query_instances(label="cup", relation="nearest")
        assert len(nearest_cup) == 2
        assert nearest_cup[0].id == "cup_01"

    def test_associate_and_update(self):
        graph = SceneGraph3D()
        base_node = SceneNode(
            id="chair_01",
            label="chair",
            pose=np.array([2.0, 2.0, 0.5], dtype=np.float32),
            appearance_feat=np.ones(128, dtype=np.float32) / np.sqrt(128),
            confidence=0.8,
        )
        graph.add_node(base_node)

        # New observation with close pose and identical embedding
        obs = SceneNode(
            id="chair_obs",
            label="chair",
            pose=np.array([2.1, 1.9, 0.5], dtype=np.float32),
            appearance_feat=np.ones(128, dtype=np.float32) / np.sqrt(128),
            confidence=0.9,
        )

        matched_node, is_matched = associate_and_update(
            obs, graph, mahalanobis_thresh=3.0, cosine_thresh=0.85, alpha=0.2
        )

        assert is_matched is True
        assert matched_node.id == "chair_01"
        assert matched_node.confidence > 0.8
