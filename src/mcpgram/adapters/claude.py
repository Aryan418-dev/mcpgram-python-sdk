"""
Claude adapter -- Python side (mirrors mcpgram-sdk's src/adapters/claude.ts).

    claude = await client.for_claude("github")
    msg = anthropic_client.messages.create(model="claude-sonnet-5", tools=claude.tools, messages=[...])
    results = await claude.run(msg.content)  # executes any tool_use blocks
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Union

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
class ClaudeGateway:
    tools: list[dict[str, Any]]
    _by_name: dict[str, ToolDefinition]
    _call: Callable[[str, dict[str, Any]], Awaitable[ExecuteResult]]

    async def run_one(self, block: dict[str, Any]) -> dict[str, Any]:
        tool = self._by_name.get(block["name"])
        if tool is None:
            return {
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": f'Unknown tool "{block["name"]}". It isn\'t available in this workspace.',
                "is_error": True,
            }
        try:
            result = await self._call(tool.tool_id, block.get("input") or {})
            if result.status == "error":
                return {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": result.error or "Tool execution failed with no error message.",
                    "is_error": True,
                }
            return {"type": "tool_result", "tool_use_id": block["id"], "content": _stringify_output(result.output)}
        except Exception as err:  # noqa: BLE001
            return {"type": "tool_result", "tool_use_id": block["id"], "content": str(err), "is_error": True}

    async def run(self, content: Union[list[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
        """Accepts either a list of tool_use blocks, or a full Claude response object (anything with a .content list)."""
        blocks = content if isinstance(content, list) else content.get("content", [])
        tool_use_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        results = []
        for block in tool_use_blocks:
            results.append(await self.run_one(block))
        return results


def build_claude_gateway(flat_tools: list[ToolDefinition], call_fn) -> ClaudeGateway:
    tools = format_tools(flat_tools, "claude")
    by_name = {t.name: t for t in flat_tools}
    return ClaudeGateway(tools=tools, _by_name=by_name, _call=call_fn)
