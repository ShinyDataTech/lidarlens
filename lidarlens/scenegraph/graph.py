"""LidarLens 3D Scene Graph Engine (Chapter 10).

Represents structured spatial environments as a 5-tuple G = (V, E, A, T, P):
  V: Vertices / semantic entity instances
  E: Edges / spatial & functional relations
  A: Node attributes (material, color, affordances)
  T: Topological layer hierarchy (Building -> Room -> Object)
  P: Metric 3D spatial poses and bounding extents
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("lidarlens.scenegraph.graph")


@dataclass
class SceneNode:
    """A semantic 3D entity instance in the scene graph."""

    id: str
    label: str
    pose: np.ndarray  # (3,) metric position [x, y, z]
    extent: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=np.float32))
    layer: int = 2  # 0: Building, 1: Room, 2: Object
    room: Optional[str] = None
    appearance_feat: Optional[np.ndarray] = None  # CLIP or DINOv2 512-D embedding
    confidence: float = 0.90
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pose, np.ndarray):
            self.pose = np.asarray(self.pose, dtype=np.float32)
        if not isinstance(self.extent, np.ndarray):
            self.extent = np.asarray(self.extent, dtype=np.float32)
        if self.appearance_feat is not None and not isinstance(self.appearance_feat, np.ndarray):
            self.appearance_feat = np.asarray(self.appearance_feat, dtype=np.float32)


@dataclass
class SceneEdge:
    """A directed spatial relation between two nodes."""

    src_id: str
    dst_id: str
    relation: str  # 'support', 'inside', 'adjacent', 'near', 'on'
    attributes: Dict[str, Any] = field(default_factory=dict)


class SceneGraph3D:
    """Hierarchical 3D Scene Graph representation G = (V, E, A, T, P)."""

    def __init__(self) -> None:
        self.nodes: Dict[str, SceneNode] = {}
        self.edges: List[SceneEdge] = []
        self._adj: Dict[str, List[SceneEdge]] = {}

    def add_node(self, node: SceneNode) -> None:
        """Add or update an entity node."""
        self.nodes[node.id] = node
        if node.id not in self._adj:
            self._adj[node.id] = []

    def add_edge(self, edge: SceneEdge) -> None:
        """Add a directed spatial edge between existing nodes."""
        if edge.src_id not in self.nodes or edge.dst_id not in self.nodes:
            raise KeyError(f"Both nodes {edge.src_id} and {edge.dst_id} must exist before adding edge.")
        self.edges.append(edge)
        self._adj[edge.src_id].append(edge)

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        """Lookup node by ID."""
        return self.nodes.get(node_id)

    def query_instances(
        self,
        label: Optional[str] = None,
        room: Optional[str] = None,
        relation: Optional[str] = None,
    ) -> List[SceneNode]:
        """Query target objects matching semantic, room, and spatial constraints (Chapter 10).

        Args:
            label: Semantic class (e.g. 'cup', 'chair', 'table').
            room: Room constraint (e.g. 'kitchen', 'office').
            relation: Spatial modifier (e.g. 'nearest', 'leftmost').

        Returns:
            List of matching SceneNode instances ranked by relevance/confidence.
        """
        candidates: List[SceneNode] = []

        for node in self.nodes.values():
            if node.layer != 2:
                continue  # Search only object-level instances
            if label and node.label.lower() != label.lower():
                continue
            if room and node.room and node.room.lower() != room.lower():
                continue
            candidates.append(node)

        if not candidates:
            return []

        # Spatial relation sorting
        if relation:
            rel = relation.lower()
            if rel == "nearest":
                # Sort by distance from origin
                candidates.sort(key=lambda n: float(np.linalg.norm(n.pose[:2])))
            elif rel == "leftmost":
                # Sort by positive Y in robot frame (left is +y)
                candidates.sort(key=lambda n: float(n.pose[1]), reverse=True)
            elif rel == "rightmost":
                candidates.sort(key=lambda n: float(n.pose[1]))
            elif rel == "highest":
                candidates.sort(key=lambda n: float(n.pose[2]), reverse=True)
        else:
            # Default sort by detection confidence
            candidates.sort(key=lambda n: n.confidence, reverse=True)

        return candidates
