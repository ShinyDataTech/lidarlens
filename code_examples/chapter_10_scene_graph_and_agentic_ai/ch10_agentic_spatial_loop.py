"""
Chapter 10: 3D Scene Graphs & Agentic Embodied AI
Code Example 10.4: Multi-Turn Agentic Spatial Loop with Real-World Physical Feedback.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第10章 10.3 & 10.4节 (Paras 222 & 263)
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.scenegraph import SceneGraph3D, SceneNode
from lidarlens.agent import (
    RobotToolExecutor,
    MockNavClient,
    MockSpatialLLMClient,
    agentic_spatial_loop
)


def build_apartment_scene_graph() -> SceneGraph3D:
    """Build test scene graph with dining table and coffee cup."""
    sg = SceneGraph3D()
    cup = SceneNode(
        id="cup_kitchen_01",
        label="cup",
        pose=np.array([3.20, 1.50, 0.85], dtype=np.float32),
        layer=2,
        room="kitchen",
        confidence=0.96
    )
    table = SceneNode(
        id="table_kitchen_01",
        label="table",
        pose=np.array([3.25, 1.50, 0.75], dtype=np.float32),
        layer=2,
        room="kitchen"
    )
    sg.add_node(cup)
    sg.add_node(table)
    return sg


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=== Chapter 10: Multi-Turn Agentic Spatial Decision Loop ===")

    # 1. Environment and Subsystem Setup
    sg = build_apartment_scene_graph()
    nav_client = MockNavClient(fail=False)
    executor = RobotToolExecutor(scene_graph=sg, nav_client=nav_client)

    # Simulator client mimicking GPT-4 / Embodied LLM with function calling
    llm_client = MockSpatialLLMClient(target_obj="cup", target_coords=(3.20, 1.50))

    human_instruction = "请前往厨房并找到桌子上的咖啡杯 (Go find the coffee cup on the kitchen table)"
    print(f"Human User Instruction: \"{human_instruction}\"")
    print("-" * 65)

    # 2. Run Autonomous Agentic Decision Loop
    t0 = time.perf_counter()
    final_reply = agentic_spatial_loop(
        instruction=human_instruction,
        llm_client=llm_client,
        executor=executor,
        max_steps=5
    )
    loop_ms = (time.perf_counter() - t0) * 1000.0

    print("-" * 65)
    print(f"Agent Final Synthesized Reply:\n  \"{final_reply}\"")
    print(f"Total Agentic Decision Loop Latency: {loop_ms:.2f} ms")

    # 3. Assertions: Task must be completed within step limits
    assert llm_client.step >= 2, "Agentic loop terminated prematurely without completing multi-turn plan!"
    assert "到达" in final_reply or "cup" in final_reply or "成功" in final_reply
    print("Multi-turn agentic loop verified with complete audit trail.")
    print("Chapter 10.4 verification completed successfully.\n")


if __name__ == "__main__":
    main()
