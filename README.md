# mcpgram-python-sdk

Official Python SDK for MCPGRAM — one client for native connectors and
external MCP servers, with LangGraph and CrewAI adapters.

Mirrors [`@mcpgram/sdk`](https://github.com/Aryan418-dev/mcpgram-sdk) (the
JS/TS SDK) field-for-field against the same `/api/v1/tools` and
`/api/v1/execute` endpoints, so both SDKs stay interchangeable.

## Install

```bash
pip install mcpgram              # core client only
pip install "mcpgram[langgraph]" # + native LangChain/LangGraph tool objects
pip install "mcpgram[crewai]"    # + native CrewAI tool objects
pip install "mcpgram[all]"       # everything
```

## Quickstart

```python
from mcpgram import Platform

client = Platform(api_key="mcpg_live_...", base_url="https://your-app.vercel.app")

github = client.use("github")
result = github.call("github_list_repos", {"per_page": 10})
print(result.output)
```

## LangGraph

```python
toolset = client.use("github")
lg = toolset.for_langgraph()

lg.tools          # plain {name, description, parameters} dicts — zero extra deps
lg.langchain_tools # actual langchain_core.tools.StructuredTool instances
                    # (only populated if langchain-core is installed; otherwise None)
```

Drop `lg.langchain_tools` straight into a LangGraph `ToolNode` or agent's tool
list — each one calls back into MCPGRAM's `/api/v1/execute` under the hood.

## CrewAI

```python
toolset = client.use("notion")
cw = toolset.for_crewai()

cw.tools        # plain {name, description, args_schema} dicts
cw.crewai_tools # actual crewai.tools.BaseTool instances (needs `crewai` installed)
```

## Design notes

- **Zero required extras.** The `tools` list on every adapter is always
  populated — plain dicts in the target framework's field names. The native
  framework objects (`langchain_tools`, `crewai_tools`) are `None` unless the
  optional dependency is installed, so importing `mcpgram` never forces you
  to install LangChain or CrewAI you're not using.
- **Same error shape as the JS SDK.** A tool that ran but failed comes back
  as a normal `ExecuteResult(status="error", ...)`, not an exception — only
  transport/auth failures raise `PlatformApiError`.
