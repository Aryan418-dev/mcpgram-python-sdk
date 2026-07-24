"""
Shared schema-translation layer (Phase 3, step 2) -- Python side.

Every framework tool format is, underneath, the same three fields: name,
description, and a JSON Schema for the parameters -- just with different
field names and wrapping. Mirrors src/formats.ts in the JS SDK field-for-
field, so both SDKs produce identical shapes from the same tool definition.

This module only converts *shapes*. It does not execute tools or handle an
agent's tool-call round-trip -- that's the adapters (client.for_langgraph(),
client.for_crewai()).
"""
from __future__ import annotations

from typing import Any, Literal

from .types import ToolDefinition

ToolFormat = Literal["claude", "openai", "langchain", "crewai"]


def normalize_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """
    Defensively normalize a tool's JSON Schema before handing it to any
    framework. Our own connector tool definitions always set these
    correctly, but external MCP servers' schemas aren't guaranteed to --
    and most frameworks require an object schema with a `properties` key,
    even for zero-argument tools.
    """
    base = schema if isinstance(schema, dict) else {}
    return {"type": "object", "properties": {}, **base}


def format_tool(tool: ToolDefinition, fmt: ToolFormat) -> dict[str, Any]:
    """Convert one tool definition into a single target framework's shape."""
    schema = normalize_schema(tool.input_schema)

    if fmt == "claude":
        return {"name": tool.name, "description": tool.description, "input_schema": schema}

    if fmt == "openai":
        return {
            "type": "function",
            "function": {"name": tool.name, "description": tool.description, "parameters": schema},
        }

    if fmt == "langchain":
        return {"name": tool.name, "description": tool.description, "parameters": schema}

    if fmt == "crewai":
        return {"name": tool.name, "description": tool.description, "args_schema": schema}

    raise ValueError(f"Unknown tool format: {fmt}")


def format_tools(tools: list[ToolDefinition], fmt: ToolFormat) -> list[dict[str, Any]]:
    """Convert a whole list of tool definitions (e.g. a Toolset's .tools) into a target framework's shape."""
    return [format_tool(t, fmt) for t in tools]
