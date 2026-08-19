from __future__ import annotations

import json
from typing import Any

from mcp_server.tool_policy import ALL_TOOL_POLICIES, ToolCallerPolicy


class LangGraphToolSafetyError(RuntimeError):
    """Raised when a LangGraph node attempts a non-controller-safe MCP tool."""


def assert_langgraph_tool_safe(tool_name: str, *, allowed_tools: list[str] | tuple[str, ...]) -> None:
    """Enforce the same MCP caller policy and state allow-list used before LangGraph.

    LangGraph is an orchestration runtime, not a permission bypass. Human-only
    tools therefore remain impossible for autonomous graph nodes to execute.
    """

    policy = ALL_TOOL_POLICIES.get(str(tool_name))
    if policy is None:
        raise LangGraphToolSafetyError(f"Unknown MCP tool: {tool_name}")
    if policy.caller_policy is not ToolCallerPolicy.CONTROLLER_ALLOWED:
        raise LangGraphToolSafetyError(
            f"MCP tool {tool_name!r} is HUMAN_UI_ONLY and cannot be called by LangGraph."
        )
    if str(tool_name) not in set(allowed_tools or []):
        raise LangGraphToolSafetyError(
            f"MCP tool {tool_name!r} is not allowed in the current deterministic workflow state."
        )


def extract_mcp_structured_result(result: Any) -> dict[str, Any]:
    """Extract the existing ToolResult payload from an MCP SDK result."""

    value = getattr(result, "structured_content", None)
    if isinstance(value, dict):
        if "result" in value and isinstance(value["result"], dict):
            return dict(value["result"])
        return dict(value)

    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if not isinstance(text, str):
            continue
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}
