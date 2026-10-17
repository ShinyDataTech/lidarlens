"""LidarLens Agentic Spatial Loop (Chapter 10).

Manages the dual-speed agentic decision loop: dispatches Pydantic spatial queries,
injects physical world execution feedback, and enforces context window budgets
and anti-retry storm guards.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from lidarlens.agent.executor import RobotToolExecutor
from lidarlens.agent.schemas import get_tool_definitions

logger = logging.getLogger("lidarlens.agent.loop")


class MockLLMResponse:
    """Mock LLM response message supporting OpenAI format."""

    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[Any]] = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class MockToolCall:
    """Mock tool call container."""

    def __init__(self, call_id: str, name: str, arguments: Dict[str, Any]) -> None:
        self.id = call_id
        self.function = type("FunctionObj", (), {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)})()


class MockSpatialLLMClient:
    """Deterministic spatial LLM simulator for CI testing and offline evaluation."""

    def __init__(self, target_obj: str = "cup", target_coords: tuple = (2.5, 1.2)) -> None:
        self.target_obj = target_obj
        self.target_coords = target_coords
        self.step = 0

    class Completions:
        def __init__(self, outer: MockSpatialLLMClient) -> None:
            self.outer = outer

        def create(self, **kwargs) -> Any:
            messages = kwargs.get("messages", [])
            last_msg = messages[-1] if messages else {}

            # Step 0: Request query_semantic_map
            if self.outer.step == 0:
                self.outer.step += 1
                tc = MockToolCall("call_001", "query_semantic_map", {"target_name": self.outer.target_obj})
                msg = MockLLMResponse(content=None, tool_calls=[tc])
                return type("ChoiceContainer", (), {"choices": [type("Choice", (), {"message": msg})()]})()

            # Step 1: Request navigate_to_pose
            elif self.outer.step == 1:
                self.outer.step += 1
                tc = MockToolCall(
                    "call_002",
                    "navigate_to_pose",
                    {"target_x": self.outer.target_coords[0], "target_y": self.outer.target_coords[1], "tolerance": 0.5},
                )
                msg = MockLLMResponse(content=None, tool_calls=[tc])
                return type("ChoiceContainer", (), {"choices": [type("Choice", (), {"message": msg})()]})()

            # Step 2: Final completion confirmation
            else:
                msg = MockLLMResponse(content=f"已成功到达目标 {self.outer.target_obj} 所在位置并就绪。")
                return type("ChoiceContainer", (), {"choices": [type("Choice", (), {"message": msg})()]})()

    @property
    def chat(self) -> Any:
        return type("ChatObj", (), {"completions": self.Completions(self)})()


def agentic_spatial_loop(
    instruction: str,
    llm_client: Any,
    executor: RobotToolExecutor,
    max_steps: int = 10,
    max_history_tokens: int = 4000,
) -> str:
    """The Agentic Decision Loop (Chapter 10).

    Args:
        instruction: Natural language human command (e.g. "Go get me the coffee cup in the kitchen").
        llm_client: LLM client exposing chat.completions.create (OpenAI, Gemini, or MockSpatialLLMClient).
        executor: RobotToolExecutor connected to Scene Graph and Nav2 client.
        max_steps: Maximum loop iterations to prevent infinite retry storms.
        max_history_tokens: Approximate context window budget constraint.

    Returns:
        Final synthesized response string to the human user.
    """
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是一个服务机器人高阶具身智能体控制器。你必须严格先使用 query_semantic_map "
                "获取目标物体的精确空间坐标，随后使用 navigate_to_pose 驱动底盘前往。严禁凭空臆造物理坐标。"
            ),
        },
        {"role": "user", "content": instruction},
    ]

    step = 0
    tool_definitions = get_tool_definitions()

    while step < max_steps:
        # Context window budget trimming (prevent overflow on long loops)
        if len(messages) > 12:
            # Retain system prompt and user prompt, trim intermediate history
            messages = [messages[0], messages[1]] + messages[-8:]

        # 1. Dispatch prompt to spatial LLM
        response = llm_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
        )

        response_message = response.choices[0].message

        # 2. Check if LLM requested physical tool invocation
        tool_calls = getattr(response_message, "tool_calls", None)

        if tool_calls:
            # Append model's assistant thought to context
            messages.append({
                "role": "assistant",
                "content": getattr(response_message, "content", None),
                "tool_calls": [
                    {
                        "id": getattr(tc, "id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            })

            for tool_call in tool_calls:
                call_name = tool_call.function.name
                call_args = tool_call.function.arguments
                call_id = getattr(tool_call, "id", "tool_id_0")

                logger.info("[Agent Step %d] Invoking tool -> %s", step, call_name)

                # 3. Ground abstract intention into physical world execution
                tool_result = executor.execute_tool(
                    tool_call_name=call_name,
                    arguments_json=call_args,
                )

                logger.info("[Physical Feedback]: %s", tool_result)

                # 4. Inject physical world result back into context
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": call_name,
                    "content": tool_result,
                })
        else:
            # 5. Task complete or clarification requested
            final_reply = getattr(response_message, "content", "任务完成。")
            logger.info("[Agent Reply]: %s", final_reply)
            return final_reply or "任务执行完毕。"

        step += 1

    logger.warning("Agentic spatial loop reached maximum iterations (%d). Aborting.", max_steps)
    return "抱歉，环境过于复杂或物理执行重试超限，无法顺利完成任务。"
