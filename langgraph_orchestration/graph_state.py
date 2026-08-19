from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class EDTechGraphState(TypedDict, total=False):
    """Shared LangGraph orchestration state.

    Existing Agent 1/Agent 2 artifacts + PostgreSQL remain authoritative.  This
    state stores only the orchestration view, MCP execution trace and progress
    events that LangGraph passes between nodes.
    """

    run_id: str
    user_request: str
    agent2_action: str

    workflow_state: str
    human_gate: str
    human_action_required: bool
    human_resume: dict[str, Any] | str | bool | None
    allowed_tools: list[str]
    state_reason: str

    current_node: str
    selected_route: str
    node_status: str
    stop_reason: str

    selected_tool: str
    tool_arguments: dict[str, Any]
    tool_success: bool
    tool_status: str
    tool_error: str | None
    tool_result: dict[str, Any]
    quiz_outcome: str
    quiz_mode: str

    # Notebook 06 -> MCP -> Notebook 08 visual orchestration.
    visual_handoff_path: str
    visual_tools_required: list[str]
    visual_tools_pending: list[str]
    visual_integrity_status: str
    visual_patched_manifest_path: str
    visual_tool_results: Annotated[list[dict[str, Any]], operator.add]

    # Used to keep real graph execution bounded even if an external artifact
    # fails to advance after a successful tool call.
    execution_steps: int
    max_execution_steps: int

    # Reducers make history append instead of overwrite.  Step 4 will stream
    # these same events into the right-hand Streamlit workflow panel.
    events: Annotated[list[dict[str, Any]], operator.add]
    completed_nodes: Annotated[list[str], operator.add]
    called_tools: Annotated[list[str], operator.add]
