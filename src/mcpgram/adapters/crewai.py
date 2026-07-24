"""
CrewAI adapter (Phase 3, step 5 -- Python side).

Usage:
    toolset = client.use("notion")
    cw = toolset.for_crewai()

    cw.tools        # plain {name, description, args_schema} dicts, no extra deps
    cw.crewai_tools # actual crewai.tools.BaseTool instances, only populated
                     # if the `crewai` package is installed

CrewAI tools are normally pydantic-backed Python classes; crewai_tools builds
one BaseTool *instance* per MCPGRAM tool from a single shared subclass (not
one dynamic class per tool -- BaseTool's own fields are pydantic v2 fields,
and overriding them via a bare `type(...)` namespace without annotations
raises PydanticUserError, since pydantic treats that as an invalid field
override rather than a plain attribute set). The tool_id and executor are
stored as private attributes instead, bound per instance in __init__.
"""
from __future__ import annotations

import asyncio
import json
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
    try:
        return json.dumps(output, indent=2, default=str)
    except TypeError:
        return str(output)


def _run_sync(call_fn: CallFn, tool_id: str, kwargs: dict[str, Any]) -> str:
    # CrewAI's BaseTool._run is synchronous; MCPGRAM's Platform.call is
    # async, so bridge with a dedicated thread when a loop is already
    # running (e.g. an async CrewAI kickoff), rather than asyncio.run()
    # which raises in that case.
    try:
        loop = asyncio.get_event_loop()
        running = loop.is_running()
    except RuntimeError:
        running = False

    if running:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            result: ExecuteResult = pool.submit(asyncio.run, call_fn(tool_id, kwargs)).result()
    else:
        result = asyncio.run(call_fn(tool_id, kwargs))

    if result.status == "error":
        return f"Error: {result.error or 'Tool execution failed with no error message.'}"
    return _stringify_output(result.output)


@dataclass
class CrewAIGateway:
    tools: list[dict[str, Any]]
    crewai_tools: Optional[list[Any]]


def _try_build_crewai_tools(flat_tools: list[ToolDefinition], call_fn: CallFn) -> Optional[list[Any]]:
    try:
        from crewai.tools import BaseTool
        from pydantic import PrivateAttr
    except ImportError:
        return None

    class _MCPGramCrewAITool(BaseTool):
        _tool_id: str = PrivateAttr()
        _call_fn: Any = PrivateAttr()

        def __init__(self, *, tool_id: str, call_fn: CallFn, **kwargs: Any):
            super().__init__(**kwargs)
            self._tool_id = tool_id
            self._call_fn = call_fn

        def _run(self, **kwargs: Any) -> str:
            return _run_sync(self._call_fn, self._tool_id, kwargs)

    built: list[Any] = []
    for tool in flat_tools:
        args_model = schema_to_pydantic_model(tool.name, tool.input_schema or {"type": "object", "properties": {}})
        built.append(
            _MCPGramCrewAITool(
                name=tool.name,
                description=tool.description,
                args_schema=args_model,
                tool_id=tool.tool_id,
                call_fn=call_fn,
            )
        )
    return built


def build_crewai_gateway(flat_tools: list[ToolDefinition], call_fn: CallFn) -> CrewAIGateway:
    tools = format_tools(flat_tools, "crewai")
    crewai_tools = _try_build_crewai_tools(flat_tools, call_fn)
    return CrewAIGateway(tools=tools, crewai_tools=crewai_tools)
