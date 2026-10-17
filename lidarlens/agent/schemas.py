"""LidarLens Pydantic Schemas for Spatial LLM Tool Calling (Chapter 10).

Defines strict typed schemas for 3D semantic map querying and robot base navigation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryMapSchema(BaseModel):
    """Tool 1: Query 3D spatial coordinates of an object in the Scene Graph (Chapter 10)."""

    target_name: str = Field(..., description="Target object semantic category, e.g. 'cup', 'chair', 'table'")
    room_hint: Optional[str] = Field(None, description="Optional search room hint, e.g. 'kitchen', 'office'")
    spatial_relation: Optional[str] = Field(None, description="Spatial modifier constraint, e.g. 'nearest', 'leftmost'")


class NavigateSchema(BaseModel):
    """Tool 2: Command the robot mobile base to navigate to a target 2D/3D pose (Chapter 10)."""

    target_x: float = Field(..., description="Target global X coordinate in meters")
    target_y: float = Field(..., description="Target global Y coordinate in meters")
    tolerance: float = Field(0.5, description="Allowable arrival radius error tolerance in meters")


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return standard Function-Calling tool definitions for LLM APIs."""
    return [
        {
            "type": "function",
            "function": {
                "name": "query_semantic_map",
                "description": "Query the 3D scene graph for metric coordinates of a named physical object.",
                "parameters": QueryMapSchema.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "navigate_to_pose",
                "description": "Drive the robot base towards target coordinates.",
                "parameters": NavigateSchema.model_json_schema(),
            },
        },
    ]
