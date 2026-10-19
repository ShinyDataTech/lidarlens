"""
Chapter 10: 3D Scene Graphs & Agentic Embodied AI
Code Example 10.3: Pydantic Strict Spatial Tool Schemas & Hardware-Level Execution.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第10章 10.3节 (Paras 249 & 256)
"""

import json
import numpy as np
from pydantic import ValidationError
import lidarlens as ll
from lidarlens.agent import (
    QueryMapSchema,
    NavigateSchema,
    get_tool_definitions,
    RobotToolExecutor,
    MockNavClient
)
from lidarlens.scenegraph import SceneGraph3D, SceneNode


def setup_test_scene() -> SceneGraph3D:
    """Prepare a simple scene graph with target objects."""
    sg = SceneGraph3D()
    cup = SceneNode(
        id="cup_kitchen_01",
        label="cup",
        pose=np.array([3.20, 1.50, 0.85]),
        layer=2,
        room="kitchen",
        confidence=0.95
    )
    chair = SceneNode(
        id="chair_living_01",
        label="chair",
        pose=np.array([1.10, -2.40, 0.45]),
        layer=2,
        room="living_room",
        confidence=0.90
    )
    sg.add_node(cup)
    sg.add_node(chair)
    return sg


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=== Chapter 10: Pydantic Spatial Tools & Hardware Execution ===")

    # 1. Inspect LLM Function-Calling Tool Definitions
    tool_defs = get_tool_definitions()
    print(f"Registered {len(tool_defs)} Function-Calling Tools:")
    for td in tool_defs:
        func = td["function"]
        print(f"  - [{func['name']}]: {func['description']}")
    assert len(tool_defs) == 2
    assert tool_defs[0]["function"]["name"] == "query_semantic_map"
    assert tool_defs[1]["function"]["name"] == "navigate_to_pose"

    # 2. Test Strict Schema Validation
    # Valid arguments
    valid_query = QueryMapSchema(target_name="cup", room_hint="kitchen")
    assert valid_query.target_name == "cup"
    assert valid_query.room_hint == "kitchen"
    print("\nValid query schema parsed successfully.")

    # Invalid arguments (missing required target_name)
    try:
        QueryMapSchema.model_validate({"room_hint": "kitchen"})
        raise AssertionError("Validation error should have been raised for missing target_name!")
    except ValidationError as e:
        print(f"Pydantic caught invalid query correctly: {e.errors()[0]['msg']}")

    # 3. Test Physical Tool Dispatch via RobotToolExecutor
    sg = setup_test_scene()
    nav = MockNavClient()
    executor = RobotToolExecutor(scene_graph=sg, nav_client=nav)

    # Tool Call 1: Query Semantic Map
    query_call_json = json.dumps({"target_name": "cup", "room_hint": "kitchen"})
    res1_raw = executor.execute_tool("query_semantic_map", query_call_json)
    res1 = json.loads(res1_raw)
    print(f"\nExecution Result 1 [query_semantic_map]:\n  {res1}")
    assert res1["status"] == "success"
    assert res1["id"] == "cup_kitchen_01"
    assert res1["pose"] == [3.20, 1.50, 0.85]

    # Tool Call 2: Navigate to Target Pose
    target_x, target_y = res1["pose"][0], res1["pose"][1]
    nav_call_json = json.dumps({"target_x": target_x, "target_y": target_y, "tolerance": 0.5})
    res2_raw = executor.execute_tool("navigate_to_pose", nav_call_json)
    res2 = json.loads(res2_raw)
    print(f"\nExecution Result 2 [navigate_to_pose]:\n  {res2}")
    assert res2["status"] == "success"

    # Tool Call 3: Query Non-Existent Object
    query_missing = json.dumps({"target_name": "laptop"})
    res3_raw = executor.execute_tool("query_semantic_map", query_missing)
    res3 = json.loads(res3_raw)
    print(f"\nExecution Result 3 [Non-Existent Target]:\n  {res3}")
    assert res3["status"] == "error"

    print("\nStrict Pydantic schemas and tool execution verified successfully.")
    print("Chapter 10.3 verification completed successfully.\n")


if __name__ == "__main__":
    main()
