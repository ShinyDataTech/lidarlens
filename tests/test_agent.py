"""Tests for Agentic Spatial AI & Pydantic Tool Schemas (Chapter 10)."""

import json
import numpy as np
import pytest

from lidarlens.agent import (
    QueryMapSchema,
    NavigateSchema,
    get_tool_definitions,
    RobotToolExecutor,
    agentic_spatial_loop,
    MockSpatialLLMClient,
)
from lidarlens.scenegraph import SceneNode, SceneGraph3D


class TestAgentSchemasAndExecutor:
    def test_pydantic_schemas(self):
        q = QueryMapSchema(target_name="cup", room_hint="kitchen")
        assert q.target_name == "cup"
        assert q.room_hint == "kitchen"

        n = NavigateSchema(target_x=2.5, target_y=-1.0, tolerance=0.3)
        assert n.target_x == 2.5
        assert n.tolerance == 0.3

        tools = get_tool_definitions()
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "query_semantic_map"
        assert tools[1]["function"]["name"] == "navigate_to_pose"

    def test_robot_tool_executor(self):
        graph = SceneGraph3D()
        node = SceneNode(
            id="cup_42",
            label="cup",
            pose=np.array([1.5, 2.5, 0.8], dtype=np.float32),
            room="kitchen",
            layer=2,
        )
        graph.add_node(node)

        executor = RobotToolExecutor(scene_graph=graph)

        # 1. Execute valid query
        q_args = json.dumps({"target_name": "cup", "room_hint": "kitchen"})
        res_json = executor.execute_tool("query_semantic_map", q_args)
        res = json.loads(res_json)
        assert res["status"] == "success"
        assert res["label"] == "cup"
        np.testing.assert_allclose(res["pose"], [1.5, 2.5, 0.8], atol=1e-4)

        # 2. Execute navigation
        nav_args = json.dumps({"target_x": 1.5, "target_y": 2.5, "tolerance": 0.5})
        nav_res_json = executor.execute_tool("navigate_to_pose", nav_args)
        nav_res = json.loads(nav_res_json)
        assert nav_res["status"] == "success"

        # 3. Invalid tool name handling
        err_res = json.loads(executor.execute_tool("unknown_tool", "{}"))
        assert err_res["status"] == "error"

    def test_agentic_spatial_loop_mock(self):
        graph = SceneGraph3D()
        cup = SceneNode(
            id="coffee_cup",
            label="cup",
            pose=np.array([2.5, 1.2, 0.8], dtype=np.float32),
            layer=2,
        )
        graph.add_node(cup)

        executor = RobotToolExecutor(scene_graph=graph)
        mock_llm = MockSpatialLLMClient(target_obj="cup", target_coords=(2.5, 1.2))

        final_answer = agentic_spatial_loop(
            instruction="帮我拿厨房桌子上的咖啡杯",
            llm_client=mock_llm,
            executor=executor,
            max_steps=5,
        )

        assert "已成功到达" in final_answer
        assert "cup" in final_answer
