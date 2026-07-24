"""
OpenAI Agents / Chat Completions adapter -- Python side (mirrors
mcpgram-sdk's src/adapters/openai.ts).

    openai_gw = await client.for_openai("notion")
    resp = openai_client.chat.completions.create(model="gpt-4o", tools=openai_gw.tools, messages=[...])
    tool_messages = await openai_gw.run(resp.choices[0].message.tool_calls)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from ..formats import format_tools
from ..types import ExecuteResult, ToolDefinition


def _stringify_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    try:
        return json.dumps(output, indent=2, default=str)
    except TypeError:
        return str(output)


@dataclass
class OpenAIGateway:
    tools: list[dict[str, Any]]
    _by_name: dict[str, ToolDefinition]
    _call: Callable[[str, dict[str, Any]], Awaitable[ExecuteResult]]

    async def run_one(self, tool_call: Any) -> dict[str, Any]:
        # tool_call may be an OpenAI SDK object (has .id/.function.name/.function.arguments)
        # or a plain dict -- support both without hard-depending on the openai package.
        call_id = getattr(tool_call, "id", None) or tool_call["id"]
        func = getattr(tool_call, "function", None) or tool_call["function"]
        name = getattr(func, "name", None) or func["name"]
        raw_args = getattr(func, "arguments", None) or func.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            args = {}

        tool = self._by_name.get(name)
        if tool is None:
            content = f'Unknown tool "{name}". It isn\'t available in this workspace.'
        else:
            try:
                result = await self._call(tool.tool_id, args)
                content = result.error if result.status == "error" and result.error else _stringify_output(result.output)
            except Exception as err:  # noqa: BLE001
                content = str(err)

        return {"role": "tool", "tool_call_id": call_id, "content": content}

    async def run(self, tool_calls: Iterable[Any]) -> list[dict[str, Any]]:
        """Executes every tool call in an assistant message's tool_calls and returns the matching `role: "tool"` messages."""
        return [await self.run_one(tc) for tc in (tool_calls or [])]


def build_openai_gateway(flat_tools: list[ToolDefinition], call_fn) -> OpenAIGateway:
    tools = format_tools(flat_tools, "openai")
    by_name = {t.name: t for t in flat_tools}
    return OpenAIGateway(tools=tools, _by_name=by_name, _call=call_fn)
