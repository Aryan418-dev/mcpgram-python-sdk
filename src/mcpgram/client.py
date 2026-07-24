"""
The MCPGRAM client. Wraps a workspace's API key and the two public
endpoints every native connector and external MCP server is exposed
through uniformly (see mcpgram-sdk's src/client.ts -- this mirrors it
field-for-field):

    GET  /api/v1/tools?server=<name>  -- discover tools (use(), for_claude())
    POST /api/v1/execute              -- run a tool (call())

Usage:
    client = Platform(api_key="mcpg_live_...", base_url="https://...")
    github = await client.use("github")
    result = await github.call("github_list_repos", {"per_page": 10})
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import httpx

from .errors import PlatformApiError
from .formats import format_tools
from .types import ExecuteResult, ToolDefinition


@dataclass
class Toolset:
    """
    The result of `client.use(name)` -- a bundle of tools belonging to one
    or more matching connectors/MCP servers, plus a convenience call() that
    resolves a tool by name or tool_id within this bundle.
    """

    query: str
    tools: list[ToolDefinition]
    _call: Callable[[str, dict[str, Any]], Awaitable[ExecuteResult]]

    async def call(self, tool_name_or_id: str, tool_input: Optional[dict[str, Any]] = None) -> ExecuteResult:
        by_id = {t.tool_id: t for t in self.tools}
        by_name = {t.name: t for t in self.tools}
        match = by_id.get(tool_name_or_id) or by_name.get(tool_name_or_id)
        if match is None:
            available = ", ".join(t.name for t in self.tools) or "(none)"
            raise ValueError(f'Tool "{tool_name_or_id}" not found in "{self.query}". Available tools: {available}')
        return await self._call(match.tool_id, tool_input or {})

    def for_claude(self):
        from .adapters.claude import build_claude_gateway

        return build_claude_gateway(self.tools, self._call)

    def for_openai(self):
        from .adapters.openai import build_openai_gateway

        return build_openai_gateway(self.tools, self._call)

    def for_langgraph(self):
        from .adapters.langgraph import build_langgraph_gateway

        return build_langgraph_gateway(self.tools, self._call)

    def for_crewai(self):
        from .adapters.crewai import build_crewai_gateway

        return build_crewai_gateway(self.tools, self._call)


class Platform:
    def __init__(self, api_key: str, base_url: str):
        if not api_key:
            raise ValueError("Platform requires an api_key. Create one from your workspace's API Keys page in the MCPGRAM dashboard.")
        if not base_url:
            raise ValueError("Platform requires a base_url (e.g. the URL of your MCPGRAM deployment). There's no default yet.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def _request(self, path: str, method: str = "GET", json_body: Optional[dict[str, Any]] = None) -> Any:
        res = await self._client.request(
            method,
            f"{self._base_url}{path}",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json=json_body,
        )
        try:
            body = res.json()
        except ValueError:
            body = None

        # 502 is a deliberate "tool ran but failed" response shape from
        # /api/v1/execute, not a transport-level failure -- let it through
        # so call() can return it as a normal ExecuteResult instead of raising.
        if res.status_code >= 400 and res.status_code != 502:
            retry_after = res.headers.get("retry-after")
            raise PlatformApiError(
                (body or {}).get("error", f"Request to {path} failed with status {res.status_code}") if isinstance(body, dict) else f"Request to {path} failed with status {res.status_code}",
                res.status_code,
                body,
                float(retry_after) * 1000 if retry_after else None,
            )
        return body

    async def _list_tools(self, server_filter: Optional[str] = None) -> list[ToolDefinition]:
        query = f"?server={server_filter}" if server_filter else ""
        json_body = await self._request(f"/api/v1/tools{query}")
        servers = (json_body or {}).get("servers", [])
        flat: list[ToolDefinition] = []
        for server in servers:
            for t in server.get("tools", []):
                flat.append(
                    ToolDefinition(
                        tool_id=t["tool_id"],
                        name=t["name"],
                        description=t.get("description", ""),
                        input_schema=t.get("input_schema") or {},
                    )
                )
        return flat

    async def use(self, name: str) -> Toolset:
        """
        Resolve tools for a connector or MCP server by name (case-insensitive
        substring match against the server's display name -- e.g. "github"
        matches the native connector "GitHub (native)", or any external MCP
        server you've named yourself).

        Raises if nothing matches. If multiple servers match, their tools
        are merged into one Toolset (call() still routes each tool to its
        own server under the hood).
        """
        flat_tools = await self._list_tools(name)
        if not flat_tools:
            raise ValueError(f'No connected server or connector matches "{name}". Check the name against your workspace\'s dashboard.')
        return Toolset(query=name, tools=flat_tools, _call=self.call)

    async def for_claude(self, server_filter: Optional[str] = None):
        from .adapters.claude import build_claude_gateway

        flat_tools = await self._list_tools(server_filter)
        return build_claude_gateway(flat_tools, self.call)

    async def for_openai(self, server_filter: Optional[str] = None):
        from .adapters.openai import build_openai_gateway

        flat_tools = await self._list_tools(server_filter)
        return build_openai_gateway(flat_tools, self.call)

    async def for_langgraph(self, server_filter: Optional[str] = None):
        from .adapters.langgraph import build_langgraph_gateway

        flat_tools = await self._list_tools(server_filter)
        return build_langgraph_gateway(flat_tools, self.call)

    async def for_crewai(self, server_filter: Optional[str] = None):
        from .adapters.crewai import build_crewai_gateway

        flat_tools = await self._list_tools(server_filter)
        return build_crewai_gateway(flat_tools, self.call)

    async def call(self, tool_id: str, tool_input: Optional[dict[str, Any]] = None) -> ExecuteResult:
        """Directly execute a known tool_id (bypasses use() when you already have the ID)."""
        json_body = await self._request(
            "/api/v1/execute", method="POST", json_body={"tool_id": tool_id, "input": tool_input or {}}
        )
        json_body = json_body or {}
        return ExecuteResult(status=json_body.get("status"), output=json_body.get("output"), error=json_body.get("error"))

    async def aclose(self) -> None:
        await self._client.aclose()
