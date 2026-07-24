"""
Shared types for the MCPGRAM SDK. These mirror the JSON shapes returned by
GET /api/v1/tools and POST /api/v1/execute -- see mcpgram-dashboard's
app/api/v1/tools/route.ts and app/api/v1/execute/route.ts for the
server-side source of truth. Mirrors src/types.ts in the JS SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass
class ToolDefinition:
    tool_id: str
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ExecuteResult:
    status: Optional[Literal["success", "error"]]
    output: Any
    error: Optional[str]
