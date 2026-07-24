"""
Internal helper: builds a pydantic model class from a JSON Schema object
schema, for adapters that need real typed args_schema classes (LangChain's
StructuredTool, CrewAI's BaseTool) rather than plain dicts.

Deliberately small and permissive rather than a full JSON Schema
implementation -- our own connector tools and most external MCP servers only
use flat object schemas with primitive/array/object property types, and an
unrecognized keyword should degrade to `Any` rather than raise.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import create_model

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _field_type(prop_schema: dict[str, Any]) -> Any:
    return _TYPE_MAP.get(prop_schema.get("type"), Any)


def schema_to_pydantic_model(name: str, schema: dict[str, Any]) -> type:
    """Build a pydantic BaseModel subclass from a normalized JSON Schema object schema."""
    properties: dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    fields: dict[str, tuple[Any, Any]] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _field_type(prop_schema if isinstance(prop_schema, dict) else {})
        description = (prop_schema or {}).get("description") if isinstance(prop_schema, dict) else None
        if prop_name in required:
            default = ...
        else:
            py_type = Optional[py_type]
            default = None
        fields[prop_name] = (py_type, default if description is None else default)

    model_name = f"{name.replace('-', '_').replace(' ', '_').title().replace('_', '')}Args" or "ToolArgs"
    return create_model(model_name, **fields)  # type: ignore[call-overload]
