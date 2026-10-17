"""LidarLens Robot Tool Executor (Chapter 10).

Executes spatial tool requests dispatched by LLMs, bridges high-level semantic
intent to physical hardware actions, and formats structured physical feedback.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from lidarlens.agent.schemas import NavigateSchema, QueryMapSchema
from lidarlens.scenegraph.graph import SceneGraph3D

logger = logging.getLogger("lidarlens.agent.executor")


@dataclass
class MockNavResult:
    is_successful: bool = True
    message: str = "已到达目标位置"


class MockNavClient:
    """Mock navigation client for simulation and automated testing."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def go_to_pose(self, x: float, y: float, tolerance: float = 0.5) -> MockNavResult:
        if self.fail:
            return MockNavResult(is_successful=False, message="路径被动态障碍物阻挡")
        return MockNavResult(is_successful=True, message=f"已成功到达坐标 ({x:.2f}, {y:.2f})")


class RobotToolExecutor:
    """Physical hardware tool executor (Chapter 10).

    Dispatches LLM tool calls against the 3D Scene Graph and navigation planner.
    """

    def __init__(self, scene_graph: SceneGraph3D, nav_client: Optional[Any] = None) -> None:
        self.sg = scene_graph
        self.nav = nav_client if nav_client is not None else MockNavClient()

    def execute_tool(self, tool_call_name: str, arguments_json: str) -> str:
        """Parse LLM tool call, invoke physical subsystems, and return structured JSON feedback."""
        try:
            args = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json

            if tool_call_name == "query_semantic_map":
                # 1. Validate arguments using Pydantic
                validated_args = QueryMapSchema(**args)

                # 2. Query 3D Scene Graph
                results = self.sg.query_instances(
                    label=validated_args.target_name,
                    room=validated_args.room_hint,
                    relation=validated_args.spatial_relation,
                )

                if not results:
                    return json.dumps({"status": "error", "msg": f"未找到符合描述的目标物体: {validated_args.target_name}"}, ensure_ascii=False)

                best_match = results[0]
                return json.dumps({
                    "status": "success",
                    "id": best_match.id,
                    "label": best_match.label,
                    "pose": best_match.pose.tolist(),
                    "confidence": float(best_match.confidence),
                }, ensure_ascii=False)

            elif tool_call_name == "navigate_to_pose":
                validated_args = NavigateSchema(**args)

                # 3. Call Navigation Subsystem
                nav_result = self.nav.go_to_pose(
                    x=validated_args.target_x,
                    y=validated_args.target_y,
                    tolerance=validated_args.tolerance,
                )

                if nav_result.is_successful:
                    return json.dumps({"status": "success", "msg": nav_result.message}, ensure_ascii=False)
                else:
                    return json.dumps({"status": "failed", "msg": nav_result.message}, ensure_ascii=False)

            else:
                return json.dumps({"status": "error", "msg": f"未知工具名称: {tool_call_name}"}, ensure_ascii=False)

        except Exception as e:
            logger.warning("Tool execution error: %s", e)
            return json.dumps({"status": "error", "msg": f"工具执行异常: {str(e)}"}, ensure_ascii=False)
