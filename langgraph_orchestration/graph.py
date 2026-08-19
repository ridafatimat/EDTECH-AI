from __future__ import annotations

import json
from pathlib import Path

from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from orchestration.assessment_intent import AssessmentRequestKind, parse_assessment_request
from orchestration.guardrails import (
    AGENT1_CHUNK,
    AGENT1_MAP,
    AGENT1_PREPROCESS,
    AGENT2_RETRIEVAL,
    AGENT2_COMPLETE_QUIZ,
    AGENT2_MISSING_QUIZ,
    RENDER_LOGIC_VISUAL,
    RENDER_TECHNICAL_VISUAL,
    RENDER_STRUCTURED_VISUAL,
)
from .graph_state import EDTechGraphState
from .mcp_runtime import assert_langgraph_tool_safe, extract_mcp_structured_result
from .routing import LangGraphRoute, route_for_state


# Visual routing is based on Notebook 08's visual schema, not syllabus-topic names.
# LangGraph selects one of three semantic MCP tools only after Notebook 06 has
# produced a concrete visual handoff.
_VISUAL_TOOL_BY_TYPE = {
    "logic_gate_diagram": RENDER_LOGIC_VISUAL,
    "network_diagram": RENDER_TECHNICAL_VISUAL,
    "simple_flowchart": RENDER_TECHNICAL_VISUAL,
    "cpu_block_diagram": RENDER_TECHNICAL_VISUAL,
    "truth_table": RENDER_STRUCTURED_VISUAL,
    "code_block": RENDER_STRUCTURED_VISUAL,
    "trace_table": RENDER_STRUCTURED_VISUAL,
    "array_grid": RENDER_STRUCTURED_VISUAL,
    "database_table": RENDER_STRUCTURED_VISUAL,
    "memory_grid": RENDER_STRUCTURED_VISUAL,
    "binary_register": RENDER_STRUCTURED_VISUAL,
}

_VISUAL_TOOL_ORDER = [
    RENDER_LOGIC_VISUAL,
    RENDER_TECHNICAL_VISUAL,
    RENDER_STRUCTURED_VISUAL,
]

class StateProviderLike(Protocol):
    def get_snapshot(self, run_id: str): ...


class MCPClientLike(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]): ...


def _event(node: str, status: str, message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"node": node, "status": status, "message": message}
    payload.update(extra)
    return payload


def _resolve_update(state_provider: StateProviderLike, state: EDTechGraphState) -> dict[str, Any]:
    run_id = str(state.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run_id is required")

    snapshot = state_provider.get_snapshot(run_id)
    route = route_for_state(snapshot.state, snapshot.human_gate)
    return {
        "workflow_state": snapshot.state.value,
        "human_gate": snapshot.human_gate.value,
        "human_action_required": bool(snapshot.human_action_required),
        "allowed_tools": list(snapshot.allowed_tools),
        "state_reason": str(snapshot.reason),
        "selected_route": route.value,
        "current_node": "resolve_state",
        "node_status": "completed",
        "events": [
            _event(
                "resolve_state",
                "completed",
                f"Resolved {snapshot.state.value}; next LangGraph route is {route.value}.",
            )
        ],
        "completed_nodes": ["resolve_state"],
    }


def build_shadow_graph(state_provider: StateProviderLike):
    """Keep Step-1 read-only graph available as a regression/fallback probe."""

    def resolve_state_node(state: EDTechGraphState) -> dict[str, Any]:
        return _resolve_update(state_provider, state)

    def marker_node(node_name: str, *, status: str, message: str):
        def _node(state: EDTechGraphState) -> dict[str, Any]:
            return {
                "current_node": node_name,
                "node_status": status,
                "stop_reason": message,
                "events": [_event(node_name, status, message)],
                "completed_nodes": [node_name],
            }
        return _node

    def route_after_resolve(state: EDTechGraphState) -> str:
        return str(state.get("selected_route") or LangGraphRoute.BLOCKED.value)

    builder = StateGraph(EDTechGraphState)
    builder.add_node("resolve_state", resolve_state_node)
    builder.add_node(LangGraphRoute.PREPROCESS.value, marker_node(
        LangGraphRoute.PREPROCESS.value, status="planned",
        message="LangGraph selected Agent 1 preprocessing as the next stage."))
    builder.add_node(LangGraphRoute.CHUNK.value, marker_node(
        LangGraphRoute.CHUNK.value, status="planned",
        message="LangGraph selected Agent 1 semantic chunking as the next stage."))
    builder.add_node(LangGraphRoute.TOPIC_MAPPING.value, marker_node(
        LangGraphRoute.TOPIC_MAPPING.value, status="planned",
        message="LangGraph selected Agent 1 topic mapping as the next stage."))
    builder.add_node(LangGraphRoute.HUMAN_GATE.value, marker_node(
        LangGraphRoute.HUMAN_GATE.value, status="waiting_human",
        message="LangGraph reached an existing mandatory human gate and must pause."))
    builder.add_node(LangGraphRoute.TOPIC_STATUS.value, marker_node(
        LangGraphRoute.TOPIC_STATUS.value, status="waiting_user",
        message="Agent 1 mapping is resolved; inspect/edit topics before the next transition."))
    builder.add_node(LangGraphRoute.AGENT2_DECISION.value, marker_node(
        LangGraphRoute.AGENT2_DECISION.value, status="waiting_request",
        message="Approved topics are available; LangGraph is waiting for an explicit assessment request."))
    builder.add_node(LangGraphRoute.COMPLETE.value, marker_node(
        LangGraphRoute.COMPLETE.value, status="completed",
        message="The workflow is in a terminal Agent 2 state."))
    builder.add_node(LangGraphRoute.BLOCKED.value, marker_node(
        LangGraphRoute.BLOCKED.value, status="blocked",
        message="The workflow is blocked/invalid and LangGraph will not advance it."))

    builder.add_edge(START, "resolve_state")
    builder.add_conditional_edges(
        "resolve_state", route_after_resolve,
        {route.value: route.value for route in LangGraphRoute},
    )
    for route in LangGraphRoute:
        builder.add_edge(route.value, END)
    return builder.compile()


def build_execution_graph(
    state_provider: StateProviderLike,
    mcp_client: MCPClientLike,
    *,
    default_max_steps: int = 8,
    checkpointer: Any | None = None,
    native_interrupts: bool = False,
):
    """Build the real MCP execution graph.

    Step-2 behavior remains the default for backwards compatibility. When
    ``native_interrupts=True`` (Step 3), mandatory human gates use LangGraph
    ``interrupt()`` and the graph must be compiled with a checkpointer.
    """

    if native_interrupts and checkpointer is None:
        raise ValueError("native_interrupts=True requires a LangGraph checkpointer")

    async def resolve_state_node(state: EDTechGraphState) -> dict[str, Any]:
        update = _resolve_update(state_provider, state)

        # A fresh explicit Agent 2 button action may start even when the
        # deterministic snapshot already contains a previous assessment.
        # The action is supplied by Streamlit; it is not inferred by the LLM.
        explicit_action = str(state.get("agent2_action") or "").strip()
        if explicit_action in {
            "retrieve_official",
            "complete_quiz",
            "missing_quiz",
        } and str(update.get("workflow_state") or "") in {
            "TOPICS_APPROVED",
            "ASSESSMENT_REQUEST_READY",
            "ASSESSMENT_READY",
            "NO_SAFE_ASSESSMENT",
        }:
            update["selected_route"] = LangGraphRoute.AGENT2_DECISION.value

        steps = int(state.get("execution_steps") or 0)
        max_steps = int(state.get("max_execution_steps") or default_max_steps)
        if steps >= max_steps and update["selected_route"] not in {
            LangGraphRoute.COMPLETE.value,
            LangGraphRoute.HUMAN_GATE.value,
            LangGraphRoute.TOPIC_STATUS.value,
        }:
            update.update({
                "selected_route": LangGraphRoute.BLOCKED.value,
                "stop_reason": f"LangGraph execution step limit reached ({max_steps}).",
            })
        return update

    def _start_node(stage: str, message: str):
        async def _node(state: EDTechGraphState) -> dict[str, Any]:
            return {
                "current_node": stage,
                "node_status": "running",
                "stop_reason": "",
                "events": [_event(stage, "running", message)],
            }
        return _node

    async def _call_state_tool(
        state: EDTechGraphState,
        *,
        stage: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        assert_langgraph_tool_safe(tool_name, allowed_tools=list(state.get("allowed_tools") or []))
        result = await mcp_client.call_tool(tool_name, arguments)
        payload = extract_mcp_structured_result(result)
        success = bool(payload.get("success")) and not bool(getattr(result, "is_error", False))
        message = str(payload.get("message") or payload.get("error") or "")
        update: dict[str, Any] = {
            "current_node": stage,
            "node_status": "completed" if success else "failed",
            "selected_tool": tool_name,
            "tool_arguments": dict(arguments),
            "tool_success": success,
            "tool_status": str(payload.get("status") or ""),
            "tool_error": None if success else (str(payload.get("error") or message or "MCP tool failed")),
            "tool_result": payload,
            "execution_steps": int(state.get("execution_steps") or 0) + 1,
            "called_tools": [tool_name],
            "events": [
                _event(
                    stage,
                    "completed" if success else "failed",
                    message or f"{tool_name} {'completed' if success else 'failed'} through MCP.",
                    tool=tool_name,
                )
            ],
        }
        if success:
            update["completed_nodes"] = [stage]
        else:
            update["stop_reason"] = update["tool_error"]
        return update

    async def preprocess_execute(state: EDTechGraphState) -> dict[str, Any]:
        return await _call_state_tool(
            state, stage="preprocess", tool_name=AGENT1_PREPROCESS,
            arguments={"run_id": str(state["run_id"])},
        )

    async def chunk_execute(state: EDTechGraphState) -> dict[str, Any]:
        return await _call_state_tool(
            state, stage="chunk", tool_name=AGENT1_CHUNK,
            arguments={"run_id": str(state["run_id"])},
        )

    async def topic_mapping_execute(state: EDTechGraphState) -> dict[str, Any]:
        return await _call_state_tool(
            state, stage="topic_mapping", tool_name=AGENT1_MAP,
            arguments={"run_id": str(state["run_id"])},
        )

    async def human_gate_node(state: EDTechGraphState) -> dict[str, Any]:
        gate = str(state.get("human_gate") or "")
        message = f"LangGraph reached mandatory human gate {gate}; autonomous MCP execution is paused."

        if not native_interrupts:
            # Step-2 compatibility path: stop safely without a native checkpointed pause.
            return {
                "current_node": "human_gate",
                "node_status": "waiting_human",
                "human_action_required": True,
                "stop_reason": message,
                "events": [_event("human_gate", "waiting_human", message)],
                "completed_nodes": ["human_gate"],
            }

        # IMPORTANT: interrupt() comes before any side effect/update in this node.
        # LangGraph may re-run the node on resume; the actual human DB mutation
        # remains outside the graph and must already have been applied through
        # the existing HUMAN_UI_ONLY MCP path. The resume payload is therefore
        # only a signal to re-check authoritative state, never an approval itself.
        resume_value = interrupt({
            "kind": "edtech_human_gate",
            "run_id": str(state.get("run_id") or ""),
            "human_gate": gate,
            "workflow_state": str(state.get("workflow_state") or ""),
            "message": message,
            "allowed_human_actions": list(state.get("allowed_tools") or []),
        })

        return {
            "current_node": "human_gate",
            "node_status": "resumed",
            "human_action_required": False,
            "human_resume": resume_value,
            "stop_reason": "",
            "events": [_event(
                "human_gate", "resumed",
                "Human input was received; LangGraph will re-resolve authoritative workflow state."
            )],
            "completed_nodes": ["human_gate"],
        }

    async def topic_status_node(state: EDTechGraphState) -> dict[str, Any]:
        message = "Agent 1 mapping is resolved; a human may inspect/edit detected topics before handoff."
        return {
            "current_node": "topic_status",
            "node_status": "waiting_user",
            "stop_reason": message,
            "events": [_event("topic_status", "waiting_user", message)],
            "completed_nodes": ["topic_status"],
        }

    async def agent2_decision_node(state: EDTechGraphState) -> dict[str, Any]:
        request_text = str(state.get("user_request") or "")
        intent = parse_assessment_request(request_text)
        explicit_action = str(state.get("agent2_action") or "").strip()

        selected_tool = ""
        arguments: dict[str, Any] = {}
        message = ""

        if explicit_action == "retrieve_official" or (
            not explicit_action
            and intent.kind is AssessmentRequestKind.GENERATE_ASSESSMENT
        ):
            selected_tool = AGENT2_RETRIEVAL
            arguments = intent.retrieval_arguments(user_request=request_text)
            message = "Official-assessment action selected; LangGraph will invoke Notebook 05 retrieval through MCP."

        elif explicit_action == "complete_quiz" or (
            not explicit_action
            and intent.kind is AssessmentRequestKind.GENERATE_COMPLETE_QUIZ
        ):
            selected_tool = AGENT2_COMPLETE_QUIZ
            arguments = intent.complete_quiz_arguments(user_request=request_text)
            message = "Complete-quiz action selected; LangGraph will invoke Notebook 06 complete_quiz through MCP without Notebook 05."

        elif explicit_action == "missing_quiz" or (
            not explicit_action
            and intent.kind is AssessmentRequestKind.GENERATE_MISSING_QUIZ
        ):
            selected_tool = AGENT2_MISSING_QUIZ
            arguments = intent.missing_quiz_arguments(user_request=request_text)
            message = "Missing-coverage action selected; LangGraph will invoke Notebook 06 fill_shortfall using the exact current Notebook 05 run."

        if not selected_tool:
            message = "Human-approved topics exist; LangGraph is waiting for an explicit official-assessment or quiz-generation action."
            return {
                "current_node": "agent2_decision",
                "node_status": "waiting_request",
                "selected_tool": "",
                "tool_arguments": {},
                "stop_reason": message,
                "events": [_event("agent2_decision", "waiting_request", message)],
                "completed_nodes": ["agent2_decision"],
            }

        return {
            "current_node": "agent2_decision",
            "node_status": "completed",
            "selected_tool": selected_tool,
            "tool_arguments": arguments,
            "stop_reason": "",
            "events": [_event(
                "agent2_decision", "completed", message, tool=selected_tool,
            )],
            "completed_nodes": ["agent2_decision"],
        }

    async def agent2_retrieval_execute(state: EDTechGraphState) -> dict[str, Any]:
        arguments = dict(state.get("tool_arguments") or {})
        arguments["run_id"] = str(state["run_id"])
        update = await _call_state_tool(
            state, stage="agent2_retrieval", tool_name=AGENT2_RETRIEVAL,
            arguments=arguments,
        )
        update["agent2_action"] = ""
        return update

    async def agent2_complete_quiz_execute(state: EDTechGraphState) -> dict[str, Any]:
        arguments = dict(state.get("tool_arguments") or {})
        arguments["run_id"] = str(state["run_id"])
        update = await _call_state_tool(
            state, stage="agent2_complete_quiz", tool_name=AGENT2_COMPLETE_QUIZ,
            arguments=arguments,
        )
        update["agent2_action"] = ""
        if update.get("tool_success"):
            data = dict((update.get("tool_result") or {}).get("data") or {})
            update["quiz_mode"] = "complete_quiz"
            update["quiz_outcome"] = str(data.get("outcome") or "")
        return update

    async def agent2_missing_quiz_execute(state: EDTechGraphState) -> dict[str, Any]:
        arguments = dict(state.get("tool_arguments") or {})
        arguments["run_id"] = str(state["run_id"])
        update = await _call_state_tool(
            state, stage="agent2_missing_quiz", tool_name=AGENT2_MISSING_QUIZ,
            arguments=arguments,
        )
        update["agent2_action"] = ""
        if update.get("tool_success"):
            data = dict((update.get("tool_result") or {}).get("data") or {})
            update["quiz_mode"] = "fill_shortfall"
            update["quiz_outcome"] = str(data.get("outcome") or "")
        return update

    async def visual_plan_node(state: EDTechGraphState) -> dict[str, Any]:
        """Plan semantic MCP visual tools from the concrete Notebook 06 handoff."""

        quiz_mode = str(state.get("quiz_mode") or "").strip()
        tool_result = dict(state.get("tool_result") or {})
        data = dict(tool_result.get("data") or {})
        output_dir_raw = str(data.get("output_dir") or "").strip()

        if not quiz_mode or not output_dir_raw:
            message = (
                "Notebook 06 completed without a quiz output directory/quiz mode; "
                "MCP visual routing is not required."
            )
            return {
                "current_node": "visual_plan",
                "node_status": "completed",
                "visual_tools_required": [],
                "visual_tools_pending": [],
                "visual_integrity_status": "NOT_REQUIRED",
                "events": [_event("visual_plan", "completed", message)],
                "completed_nodes": ["visual_plan"],
            }

        handoff_path = Path(output_dir_raw) / "visual_tool_handoff.json"
        if not handoff_path.is_file():
            message = (
                "Notebook 06 produced no visual_tool_handoff.json; no MCP visual "
                "tools are required for this quiz action."
            )
            return {
                "current_node": "visual_plan",
                "node_status": "completed",
                "visual_handoff_path": str(handoff_path),
                "visual_tools_required": [],
                "visual_tools_pending": [],
                "visual_integrity_status": "NOT_REQUIRED",
                "events": [_event("visual_plan", "completed", message)],
                "completed_nodes": ["visual_plan"],
            }

        try:
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        except Exception as exc:
            message = (
                "Notebook 06 visual handoff could not be read: "
                f"{type(exc).__name__}: {exc}"
            )
            return {
                "current_node": "visual_plan",
                "node_status": "failed",
                "stop_reason": message,
                "visual_handoff_path": str(handoff_path),
                "visual_integrity_status": "BLOCKED",
                "events": [_event("visual_plan", "failed", message)],
            }

        questions = handoff.get("questions", [])
        if not isinstance(questions, list):
            questions = []

        required_tools: list[str] = []
        unknown_types: list[str] = []

        for question in questions:
            if not isinstance(question, dict):
                continue
            visual_type = str(
                question.get("visual_requirement", "none") or "none"
            ).strip()
            if visual_type == "none":
                continue
            tool_name = _VISUAL_TOOL_BY_TYPE.get(visual_type)
            if tool_name is None:
                unknown_types.append(visual_type)
                continue
            if tool_name not in required_tools:
                required_tools.append(tool_name)

        if unknown_types:
            message = (
                "Notebook 06 handoff contains unsupported visual type(s): "
                + ", ".join(sorted(set(unknown_types)))
            )
            return {
                "current_node": "visual_plan",
                "node_status": "failed",
                "stop_reason": message,
                "visual_handoff_path": str(handoff_path),
                "visual_integrity_status": "BLOCKED",
                "events": [_event("visual_plan", "failed", message)],
            }

        required_tools = [
            tool_name for tool_name in _VISUAL_TOOL_ORDER
            if tool_name in required_tools
        ]

        # Local capability grant. Visual tools remain absent from the generic
        # deterministic workflow-state allow-list, so the ordinary controller
        # cannot select them by itself.
        allowed_tools = list(state.get("allowed_tools") or [])
        for tool_name in required_tools:
            if tool_name not in allowed_tools:
                allowed_tools.append(tool_name)

        message = (
            "Notebook 06 visual handoff requires MCP tools: "
            + ", ".join(required_tools)
            if required_tools
            else "Notebook 06 handoff contains no required visuals."
        )

        return {
            "current_node": "visual_plan",
            "node_status": "completed",
            "visual_handoff_path": str(handoff_path),
            "visual_tools_required": required_tools,
            "visual_tools_pending": list(required_tools),
            "visual_integrity_status": "PENDING" if required_tools else "NOT_REQUIRED",
            "allowed_tools": allowed_tools,
            "events": [
                _event(
                    "visual_plan",
                    "completed",
                    message,
                    visual_tools=required_tools,
                )
            ],
            "completed_nodes": ["visual_plan"],
        }

    async def visual_render_execute(state: EDTechGraphState) -> dict[str, Any]:
        pending = list(state.get("visual_tools_pending") or [])
        if not pending:
            return {
                "current_node": "visual_render",
                "node_status": "completed",
                "visual_integrity_status": "PASS",
                "events": [
                    _event(
                        "visual_render",
                        "completed",
                        "All required MCP visual tools are complete.",
                    )
                ],
                "completed_nodes": ["visual_render"],
            }

        tool_name = str(pending[0])
        arguments = {
            "run_id": str(state["run_id"]),
            "quiz_mode": str(state.get("quiz_mode") or ""),
        }
        update = await _call_state_tool(
            state,
            stage="visual_render",
            tool_name=tool_name,
            arguments=arguments,
        )

        if not update.get("tool_success"):
            update["visual_integrity_status"] = "BLOCKED"
            return update

        remaining = pending[1:]
        data = dict(((update.get("tool_result") or {}).get("data") or {}))
        aggregate_status = str(data.get("aggregate_status") or "").strip()
        update["visual_tools_pending"] = remaining
        update["visual_integrity_status"] = (
            "PENDING" if remaining else (aggregate_status or "PASS")
        )
        update["visual_patched_manifest_path"] = str(
            data.get("patched_manifest_path") or ""
        )
        update["visual_tool_results"] = [
            {
                "tool_name": tool_name,
                "success": True,
                "data": data,
            }
        ]
        return update

    async def complete_node(state: EDTechGraphState) -> dict[str, Any]:
        message = "The workflow is in a terminal Agent 2 state."
        return {
            "current_node": "complete",
            "node_status": "completed",
            "stop_reason": message,
            "events": [_event("complete", "completed", message)],
            "completed_nodes": ["complete"],
        }

    async def blocked_node(state: EDTechGraphState) -> dict[str, Any]:
        message = str(state.get("stop_reason") or "Workflow state is invalid/blocked; LangGraph will not execute a tool.")
        return {
            "current_node": "blocked",
            "node_status": "blocked",
            "stop_reason": message,
            "events": [_event("blocked", "blocked", message)],
            "completed_nodes": ["blocked"],
        }

    def route_after_resolve(state: EDTechGraphState) -> str:
        return str(state.get("selected_route") or LangGraphRoute.BLOCKED.value)

    def route_after_tool(state: EDTechGraphState) -> str:
        return "resolve_state" if bool(state.get("tool_success")) else "blocked"

    def route_after_agent2_decision(state: EDTechGraphState) -> str:
        selected = str(state.get("selected_tool") or "")
        if selected == AGENT2_RETRIEVAL:
            return "agent2_retrieval"
        if selected == AGENT2_COMPLETE_QUIZ:
            return "agent2_complete_quiz"
        if selected == AGENT2_MISSING_QUIZ:
            return "agent2_missing_quiz"
        return "end"

    def route_after_agent2_action(state: EDTechGraphState) -> str:
        return "visual_plan" if bool(state.get("tool_success")) else "blocked"

    def route_after_visual_plan(state: EDTechGraphState) -> str:
        if str(state.get("node_status") or "") == "failed":
            return "blocked"
        return "visual_render" if list(state.get("visual_tools_pending") or []) else "end"

    def route_after_visual_render(state: EDTechGraphState) -> str:
        if not bool(state.get("tool_success", True)):
            return "blocked"
        return "visual_render" if list(state.get("visual_tools_pending") or []) else "end"

    builder = StateGraph(EDTechGraphState)
    builder.add_node("resolve_state", resolve_state_node)

    # Separate start and execute nodes intentionally produce a stream update
    # before the expensive MCP call. Step 4 will render these as ▶ running / ✓ done.
    builder.add_node("preprocess", _start_node("preprocess", "Agent 1 preprocessing started."))
    builder.add_node("preprocess_execute", preprocess_execute)
    builder.add_node("chunk", _start_node("chunk", "Agent 1 semantic chunking started."))
    builder.add_node("chunk_execute", chunk_execute)
    builder.add_node("topic_mapping", _start_node("topic_mapping", "Agent 1 topic mapping started."))
    builder.add_node("topic_mapping_execute", topic_mapping_execute)

    builder.add_node("human_gate", human_gate_node)
    builder.add_node("topic_status", topic_status_node)
    builder.add_node("agent2_decision", agent2_decision_node)
    builder.add_node("agent2_retrieval", _start_node("agent2_retrieval", "Agent 2 official retrieval started."))
    builder.add_node("agent2_retrieval_execute", agent2_retrieval_execute)
    builder.add_node("agent2_complete_quiz", _start_node("agent2_complete_quiz", "Agent 2 complete quiz generation started."))
    builder.add_node("agent2_complete_quiz_execute", agent2_complete_quiz_execute)
    builder.add_node("agent2_missing_quiz", _start_node("agent2_missing_quiz", "Agent 2 missing quiz coverage generation started."))
    builder.add_node("agent2_missing_quiz_execute", agent2_missing_quiz_execute)
    builder.add_node("visual_plan", visual_plan_node)
    builder.add_node("visual_render", visual_render_execute)
    builder.add_node("complete", complete_node)
    builder.add_node("blocked", blocked_node)

    builder.add_edge(START, "resolve_state")
    builder.add_conditional_edges(
        "resolve_state",
        route_after_resolve,
        {
            LangGraphRoute.PREPROCESS.value: "preprocess",
            LangGraphRoute.CHUNK.value: "chunk",
            LangGraphRoute.TOPIC_MAPPING.value: "topic_mapping",
            LangGraphRoute.HUMAN_GATE.value: "human_gate",
            LangGraphRoute.TOPIC_STATUS.value: "topic_status",
            LangGraphRoute.AGENT2_DECISION.value: "agent2_decision",
            LangGraphRoute.COMPLETE.value: "complete",
            LangGraphRoute.BLOCKED.value: "blocked",
        },
    )

    builder.add_edge("preprocess", "preprocess_execute")
    builder.add_conditional_edges("preprocess_execute", route_after_tool, {
        "resolve_state": "resolve_state", "blocked": "blocked"})
    builder.add_edge("chunk", "chunk_execute")
    builder.add_conditional_edges("chunk_execute", route_after_tool, {
        "resolve_state": "resolve_state", "blocked": "blocked"})
    builder.add_edge("topic_mapping", "topic_mapping_execute")
    builder.add_conditional_edges("topic_mapping_execute", route_after_tool, {
        "resolve_state": "resolve_state", "blocked": "blocked"})

    builder.add_conditional_edges("agent2_decision", route_after_agent2_decision, {
        "agent2_retrieval": "agent2_retrieval",
        "agent2_complete_quiz": "agent2_complete_quiz",
        "agent2_missing_quiz": "agent2_missing_quiz",
        "end": END,
    })
    builder.add_edge("agent2_retrieval", "agent2_retrieval_execute")
    builder.add_conditional_edges("agent2_retrieval_execute", route_after_tool, {
        "resolve_state": "resolve_state", "blocked": "blocked"})

    builder.add_edge("agent2_complete_quiz", "agent2_complete_quiz_execute")
    builder.add_conditional_edges(
        "agent2_complete_quiz_execute",
        route_after_agent2_action,
        {"visual_plan": "visual_plan", "blocked": "blocked"},
    )

    builder.add_edge("agent2_missing_quiz", "agent2_missing_quiz_execute")
    builder.add_conditional_edges(
        "agent2_missing_quiz_execute",
        route_after_agent2_action,
        {"visual_plan": "visual_plan", "blocked": "blocked"},
    )

    builder.add_conditional_edges(
        "visual_plan",
        route_after_visual_plan,
        {"visual_render": "visual_render", "end": END, "blocked": "blocked"},
    )
    builder.add_conditional_edges(
        "visual_render",
        route_after_visual_render,
        {"visual_render": "visual_render", "end": END, "blocked": "blocked"},
    )

    if native_interrupts:
        # After Command(resume=...), never trust the resume payload as proof of
        # approval. Re-resolve PostgreSQL/artifacts. If the human action was not
        # actually committed, routing lands on the same gate and interrupts again.
        builder.add_edge("human_gate", "resolve_state")
    else:
        builder.add_edge("human_gate", END)

    for node in ["topic_status", "complete", "blocked"]:
        builder.add_edge(node, END)

    return builder.compile(checkpointer=checkpointer)


def build_hitl_graph(
    state_provider: StateProviderLike,
    mcp_client: MCPClientLike,
    *,
    checkpointer: Any,
    default_max_steps: int = 8,
):
    """Build the Step-3 graph with native LangGraph HITL interrupts enabled."""

    return build_execution_graph(
        state_provider,
        mcp_client,
        default_max_steps=default_max_steps,
        checkpointer=checkpointer,
        native_interrupts=True,
    )
