"""
LangGraph adapter (Phase 3, step 5 -- Python side).

Usage:
    toolset = client.use("github")
    lg = toolset.for_langgraph()

    lg.tools           # plain {name, description, parameters} dicts, no extra deps
    lg.langchain_tools # actual langchain_core.tools.StructuredTool instances,
                        # only populated if langchain-core is installed

Drop lg.langchain_tools straight into a LangGraph ToolNode or an agent's
tool list -- each one calls back into MCPGRAM's /api/v1/execute under the
hood via the same Platform.call() every other adapter uses.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..formats import format_tools
from .._jsonschema import schema_to_pydantic_model
from ..types import ExecuteResult, ToolDefinition

CallFn = Callable[[str, dict[str, Any]], Awaitable[ExecuteResult]]


def _stringify_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    import json

    try:
        return json.dumps(output, indent=2, default=str)
    except TypeError:
        return str(output)


@dataclass
class LangGraphGateway:
    tools: list[dict[str, Any]]
    langchain_tools: Optional[list[Any]]

    async def run_one(self, name: str, tool_input: dict[str, Any]) -> str:
        """Execute a single tool by name and return LangChain-ready string content."""
        raise NotImplementedError  # replaced per-instance in build_langgraph_gateway


def _try_build_langchain_tools(
    flat_tools: list[ToolDefinition], call_fn: CallFn
) -> Optional[list[Any]]:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        return None

    built: list[Any] = []
    for tool in flat_tools:
        args_model = schema_to_pydantic_model(tool.name, tool.input_schema or {"type": "object", "properties": {}})

        def _make_coroutine(tool_id: str):
            async def _coroutine(**kwargs: Any) -> str:
                result = await call_fn(tool_id, kwargs)
                if result.status == "error":
                    return f"Error: {result.error or 'Tool execution failed with no error message.'}"
                return _stringify_output(result.output)

            return _coroutine

        built.append(
            StructuredTool.from_function(
                coroutine=_make_coroutine(tool.tool_id),
                name=tool.name,
                description=tool.description,
                args_schema=args_model,
            )
        )
    return built


def build_langgraph_gateway(flat_tools: list[ToolDefinition], call_fn: CallFn) -> LangGraphGateway:
    tools = format_tools(flat_tools, "langchain")
    langchain_tools = _try_build_langchain_tools(flat_tools, call_fn)

    by_name = {t.name: t for t in flat_tools}

    async def run_one(name: str, tool_input: dict[str, Any]) -> str:
        tool = by_name.get(name)
        if tool is None:
            available = ", ".join(by_name.keys()) or "(none)"
            return f'Unknown tool "{name}". Available tools: {available}'
        result = await call_fn(tool.tool_id, tool_input)
        if result.status == "error":
            return f"Error: {result.error or 'Tool execution failed with no error message.'}"
        return _stringify_output(result.output)

    gateway = LangGraphGateway(tools=tools, langchain_tools=langchain_tools)
    gateway.run_one = run_one  # type: ignore[method-assign]
    return gateway
