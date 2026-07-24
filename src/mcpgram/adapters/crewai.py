"""
CrewAI adapter (Phase 3, step 5 -- Python side).

Usage:
    toolset = client.use("notion")
    cw = toolset.for_crewai()

    cw.tools        # plain {name, description, args_schema} dicts, no extra deps
    cw.crewai_tools # actual crewai.tools.BaseTool instances, only populated
                     # if the `crewai` package is installed

CrewAI tools are normally pydantic-backed Python classes; crewai_tools
dynamically builds one BaseTool subclass per MCPGRAM tool, with a `_run`
that calls back into /api/v1/execute the same way every other adapter does.
"""
from __future__ import annotations

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
class CrewAIGateway:
    tools: list[dict[str, Any]]
    crewai_tools: Optional[list[Any]]


def _try_build_crewai_tools(flat_tools: list[ToolDefinition], call_fn: CallFn) -> Optional[list[Any]]:
    try:
        from crewai.tools import BaseTool
    except ImportError:
        return None

    import asyncio

    built: list[Any] = []
    for tool in flat_tools:
        args_model = schema_to_pydantic_model(tool.name, tool.input_schema or {"type": "object", "properties": {}})

        def _make_run(tool_id: str):
            def _run(**kwargs: Any) -> str:
                # CrewAI's BaseTool._run is synchronous; MCPGRAM's Platform.call
                # is async, so bridge with a dedicated event loop rather than
                # asyncio.run() (which breaks if a crew is already running
                # inside its own loop, e.g. an async CrewAI kickoff).
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            result: ExecuteResult = pool.submit(asyncio.run, call_fn(tool_id, kwargs)).result()
                    else:
                        result = loop.run_until_complete(call_fn(tool_id, kwargs))
                except RuntimeError:
                    result = asyncio.run(call_fn(tool_id, kwargs))

                if result.status == "error":
                    return f"Error: {result.error or 'Tool execution failed with no error message.'}"
                return _stringify_output(result.output)

            return _run

        tool_cls = type(
            f"MCPGRAM{tool.name.title().replace('_', '')}Tool",
            (BaseTool,),
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": args_model,
                "_run": staticmethod(_make_run(tool.tool_id)),
            },
        )
        built.append(tool_cls())
    return built


def build_crewai_gateway(flat_tools: list[ToolDefinition], call_fn: CallFn) -> CrewAIGateway:
    tools = format_tools(flat_tools, "crewai")
    crewai_tools = _try_build_crewai_tools(flat_tools, call_fn)
    return CrewAIGateway(tools=tools, crewai_tools=crewai_tools)
