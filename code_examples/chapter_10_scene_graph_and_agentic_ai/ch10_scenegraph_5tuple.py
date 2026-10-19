"""
Chapter 10: 3D Scene Graphs & Agentic Embodied AI
Code Example 10.1: 5-Tuple 3D Scene Graph Construction & Hierarchical Spatial Querying.

Textbook Reference:
- 《Python三维空间感知与机器人视觉：从LiDAR工程到Agentic AI实战》第10章 10.1 & 10.2节
"""

import time
import numpy as np
import lidarlens as ll
from lidarlens.scenegraph import SceneGraph3D, SceneNode, SceneEdge


def build_sample_indoor_scene_graph() -> SceneGraph3D:
    """Construct a hierarchical 5-tuple 3D Scene Graph G = (V, E, A, T, P)."""
    sg = SceneGraph3D()

    # Layer 0: Building
    building = SceneNode(id="bldg_01", label="OfficeBuilding", pose=np.array([0.0, 0.0, 0.0]), layer=0)
    sg.add_node(building)

    # Layer 1: Rooms
    kitchen = SceneNode(id="room_kitchen", label="Kitchen", pose=np.array([5.0, 2.0, 0.0]), layer=1, room="kitchen")
    office = SceneNode(id="room_office", label="Office", pose=np.array([0.0, -4.0, 0.0]), layer=1, room="office")
    sg.add_node(kitchen)
    sg.add_node(office)

    # Edges: Building -> Room
    sg.add_edge(SceneEdge("bldg_01", "room_kitchen", "contains"))
    sg.add_edge(SceneEdge("bldg_01", "room_office", "contains"))

    # Layer 2: Objects
    # Kitchen Objects: Dining Table, 2 Coffee Cups, Chair
    table_k = SceneNode(
        id="table_01",
        label="table",
        pose=np.array([5.2, 2.1, 0.75]),
        extent=np.array([1.6, 0.9, 0.75]),
        layer=2,
        room="kitchen",
        attributes={"material": "wood", "color": "brown"}
    )
    cup_near = SceneNode(
        id="cup_01",
        label="cup",
        pose=np.array([5.0, 1.8, 0.85]),
        extent=np.array([0.1, 0.1, 0.15]),
        layer=2,
        room="kitchen",
        confidence=0.96,
        attributes={"color": "white", "affordance": "graspable"}
    )
    cup_far = SceneNode(
        id="cup_02",
        label="cup",
        pose=np.array([5.5, 2.4, 0.85]),
        extent=np.array([0.1, 0.1, 0.15]),
        layer=2,
        room="kitchen",
        confidence=0.88,
        attributes={"color": "blue", "affordance": "graspable"}
    )

    # Office Objects: Desk, Office Chair
    desk_off = SceneNode(
        id="desk_01",
        label="table",
        pose=np.array([0.2, -4.1, 0.75]),
        extent=np.array([1.4, 0.8, 0.75]),
        layer=2,
        room="office"
    )

    for obj in [table_k, cup_near, cup_far, desk_off]:
        sg.add_node(obj)

    # Spatial relations: Room contains table, Table supports cups
    sg.add_edge(SceneEdge("room_kitchen", "table_01", "inside"))
    sg.add_edge(SceneEdge("table_01", "cup_01", "support"))
    sg.add_edge(SceneEdge("table_01", "cup_02", "support"))
    sg.add_edge(SceneEdge("room_office", "desk_01", "inside"))

    return sg


def main():
    print("=== Chapter 10: 5-Tuple 3D Scene Graph Construction & Querying ===")

    t0 = time.perf_counter()
    sg = build_sample_indoor_scene_graph()
    build_ms = (time.perf_counter() - t0) * 1000.0

    print(f"3D Scene Graph constructed in {build_ms:.2f} ms:")
    print(f"  Total Vertices (V): {len(sg.nodes)} nodes across 3 topological layers (Building, Room, Object)")
    print(f"  Total Directed Edges (E): {len(sg.edges)} spatial relations ('contains', 'inside', 'support')")

    # 1. Query all tables across all rooms
    all_tables = sg.query_instances(label="table")
    print(f"\nQuery [label='table']: Found {len(all_tables)} instances -> {[n.id for n in all_tables]}")
    assert len(all_tables) == 2

    # 2. Query cups specifically in the kitchen
    kitchen_cups = sg.query_instances(label="cup", room="kitchen")
    print(f"Query [label='cup', room='kitchen']: Found {len(kitchen_cups)} instances -> {[n.id for n in kitchen_cups]}")
    assert len(kitchen_cups) == 2

    # 3. Disambiguate with spatial relation: 'nearest'
    nearest_cup = sg.query_instances(label="cup", room="kitchen", relation="nearest")
    print(f"Query [label='cup', relation='nearest']: Top Match -> {nearest_cup[0].id} at pose {nearest_cup[0].pose}")
    assert nearest_cup[0].id == "cup_01"

    # 4. Disambiguate with spatial relation: 'leftmost' (highest Y in kitchen)
    leftmost_cup = sg.query_instances(label="cup", room="kitchen", relation="leftmost")
    print(f"Query [label='cup', relation='leftmost']: Top Match -> {leftmost_cup[0].id} at pose {leftmost_cup[0].pose}")
    assert leftmost_cup[0].id == "cup_02"

    print("\nScene graph topology and spatial disambiguation queries verified successfully.")
    print("Chapter 10.1 verification completed successfully.\n")


if __name__ == "__main__":
    main()
