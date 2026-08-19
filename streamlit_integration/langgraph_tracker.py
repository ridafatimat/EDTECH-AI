from __future__ import annotations

from copy import deepcopy
from typing import Any


STAGES: tuple[tuple[str, str], ...] = (
    ("transcript", "Transcript received"),
    ("preprocess", "Preprocessing"),
    ("chunk", "Semantic chunking"),
    ("topic_mapping", "Topic mapping"),
    ("mapping_review", "Mapping review"),
    ("topic_handoff", "Agent 1 → Agent 2 approval"),
    ("agent2_retrieval", "Official assessment retrieval"),
    ("complete_quiz", "Complete AI quiz generation"),
    ("missing_quiz", "Missing quiz coverage"),
    ("assessment_ready", "Assessment / quiz ready"),
)

_ICONS = {
    "pending": "○",
    "running": "▶",
    "completed": "✓",
    "waiting_human": "⏸",
    "waiting_request": "◌",
    "blocked": "⚠",
    "failed": "✕",
}


def empty_tracker(*, run_id: str = "") -> dict[str, Any]:
    return {
        "run_id": str(run_id or ""),
        "workflow_state": "NO_RUN" if not run_id else "",
        "human_gate": "NONE",
        "state_reason": "",
        "stages": {
            key: {"label": label, "status": "pending", "message": ""}
            for key, label in STAGES
        },
    }


def _mark(tracker: dict[str, Any], key: str, status: str, message: str = "") -> None:
    stage = tracker["stages"].get(key)
    if stage is None:
        return
    stage["status"] = status
    if message:
        stage["message"] = str(message)


def tracker_from_snapshot(snapshot: dict[str, Any] | None, *, run_id: str = "") -> dict[str, Any]:
    snapshot = dict(snapshot or {})
    tracker = empty_tracker(run_id=run_id)
    state = str(snapshot.get("state") or snapshot.get("workflow_state") or ("RAW_TRANSCRIPT_READY" if run_id else "NO_RUN"))
    gate = str(snapshot.get("human_gate") or "NONE")
    tracker["workflow_state"] = state
    tracker["human_gate"] = gate
    tracker["state_reason"] = str(snapshot.get("reason") or snapshot.get("state_reason") or "")

    if state == "NO_RUN":
        return tracker

    _mark(tracker, "transcript", "completed")

    if state in {
        "PREPROCESSING_COMPLETE", "CHUNKS_READY", "AWAITING_TOPIC_MAPPING_REVIEW",
        "AWAITING_AGENT2_TOPIC_APPROVAL", "TOPICS_APPROVED", "ASSESSMENT_REQUEST_READY",
        "ASSESSMENT_READY", "NO_SAFE_ASSESSMENT", "NO_RETAINED_TOPICS",
    }:
        _mark(tracker, "preprocess", "completed")

    if state in {
        "CHUNKS_READY", "AWAITING_TOPIC_MAPPING_REVIEW", "AWAITING_AGENT2_TOPIC_APPROVAL",
        "TOPICS_APPROVED", "ASSESSMENT_REQUEST_READY", "ASSESSMENT_READY",
        "NO_SAFE_ASSESSMENT", "NO_RETAINED_TOPICS",
    }:
        _mark(tracker, "chunk", "completed")

    if state in {
        "AWAITING_TOPIC_MAPPING_REVIEW", "AWAITING_AGENT2_TOPIC_APPROVAL", "TOPICS_APPROVED",
        "ASSESSMENT_REQUEST_READY", "ASSESSMENT_READY", "NO_SAFE_ASSESSMENT",
        "NO_RETAINED_TOPICS",
    }:
        _mark(tracker, "topic_mapping", "completed")

    if state == "AWAITING_TOPIC_MAPPING_REVIEW" or gate == "TOPIC_MAPPING_REVIEW":
        _mark(tracker, "mapping_review", "waiting_human", "Human review is required before the graph can continue.")
    elif state in {
        "AWAITING_AGENT2_TOPIC_APPROVAL", "TOPICS_APPROVED", "ASSESSMENT_REQUEST_READY",
        "ASSESSMENT_READY", "NO_SAFE_ASSESSMENT", "NO_RETAINED_TOPICS",
    }:
        _mark(tracker, "mapping_review", "completed", "Resolved or not required.")

    if state == "AWAITING_AGENT2_TOPIC_APPROVAL" or gate == "AGENT2_TOPIC_APPROVAL":
        _mark(tracker, "topic_handoff", "waiting_human", "Approve the final Agent 1 topics before Agent 2 can run.")
    elif state in {"TOPICS_APPROVED", "ASSESSMENT_REQUEST_READY", "ASSESSMENT_READY", "NO_SAFE_ASSESSMENT"}:
        _mark(tracker, "topic_handoff", "completed")

    if state in {"TOPICS_APPROVED", "ASSESSMENT_REQUEST_READY"}:
        _mark(tracker, "agent2_retrieval", "waiting_request", "Waiting for an explicit assessment request.")
    elif state in {"ASSESSMENT_READY", "NO_SAFE_ASSESSMENT"}:
        _mark(tracker, "agent2_retrieval", "completed")

    if state == "ASSESSMENT_READY":
        _mark(tracker, "assessment_ready", "completed")
    elif state == "NO_SAFE_ASSESSMENT":
        _mark(tracker, "assessment_ready", "blocked", "No safe compatible assessment was found.")

    if state in {"INVALID_STATE", "INTEGRITY_ERROR", "NO_RUN"}:
        for key, _ in STAGES:
            if tracker["stages"][key]["status"] == "pending":
                _mark(tracker, key, "blocked")
                break

    return tracker


def update_tracker_from_graph_update(
    tracker: dict[str, Any] | None,
    update: dict[str, Any] | None,
) -> dict[str, Any]:
    value = deepcopy(tracker or empty_tracker())
    update = dict(update or {})

    # Standard LangGraph ``updates`` mode shape: {node_name: {state update...}}
    for node_name, payload in update.items():
        if node_name == "__interrupt__":
            interrupts = payload if isinstance(payload, (list, tuple)) else [payload]
            for item in interrupts:
                raw = getattr(item, "value", item)
                if not isinstance(raw, dict):
                    continue
                value["workflow_state"] = str(raw.get("workflow_state") or value.get("workflow_state") or "")
                value["human_gate"] = str(raw.get("human_gate") or value.get("human_gate") or "NONE")
                gate = value["human_gate"]
                if gate == "TOPIC_MAPPING_REVIEW":
                    _mark(value, "mapping_review", "waiting_human", str(raw.get("message") or "Waiting for human review."))
                elif gate == "AGENT2_TOPIC_APPROVAL":
                    _mark(value, "mapping_review", "completed", "Resolved or not required.")
                    _mark(value, "topic_handoff", "waiting_human", str(raw.get("message") or "Waiting for human topic approval."))
            continue

        if not isinstance(payload, dict):
            continue
        current = str(payload.get("current_node") or node_name)
        status = str(payload.get("node_status") or payload.get("status") or "")
        message = ""
        events = payload.get("events") or []
        if events and isinstance(events, list) and isinstance(events[-1], dict):
            message = str(events[-1].get("message") or "")

        workflow_state = payload.get("workflow_state")
        human_gate = payload.get("human_gate")
        if workflow_state:
            value["workflow_state"] = str(workflow_state)
        if human_gate:
            value["human_gate"] = str(human_gate)
        if payload.get("state_reason"):
            value["state_reason"] = str(payload.get("state_reason"))

        stage_key = {
            "preprocess": "preprocess",
            "preprocess_execute": "preprocess",
            "chunk": "chunk",
            "chunk_execute": "chunk",
            "topic_mapping": "topic_mapping",
            "topic_mapping_execute": "topic_mapping",
            "agent2_retrieval": "agent2_retrieval",
            "agent2_retrieval_execute": "agent2_retrieval",
            "agent2_complete_quiz": "complete_quiz",
            "agent2_complete_quiz_execute": "complete_quiz",
            "agent2_missing_quiz": "missing_quiz",
            "agent2_missing_quiz_execute": "missing_quiz",
        }.get(current)
        if stage_key and status in {"running", "completed", "failed", "blocked"}:
            _mark(value, stage_key, status, message)

        if current == "agent2_decision" and status == "waiting_request":
            _mark(value, "topic_handoff", "completed")
            _mark(value, "agent2_retrieval", "waiting_request", message)
        if current in {"agent2_complete_quiz_execute", "agent2_missing_quiz_execute"} and status == "completed":
            _mark(value, "assessment_ready", "completed", message)
        if current == "complete" or value.get("workflow_state") == "ASSESSMENT_READY":
            _mark(value, "agent2_retrieval", "completed")
            _mark(value, "assessment_ready", "completed", message)

    return value


def render_tracker(placeholder: Any, tracker: dict[str, Any]) -> None:
    """Render the compact right-side Streamlit tracker into a placeholder."""
    with placeholder.container():
        import streamlit as st

        st.markdown("### LangGraph workflow")
        state = str(tracker.get("workflow_state") or "—")
        gate = str(tracker.get("human_gate") or "NONE")
        st.caption(f"State: `{state}`")
        if gate and gate != "NONE":
            st.caption(f"Human gate: `{gate}`")

        for key, _label in STAGES:
            item = tracker["stages"][key]
            status = str(item.get("status") or "pending")
            icon = _ICONS.get(status, "○")
            label = str(item.get("label") or key)
            if status == "running":
                st.markdown(f"**{icon} {label}**  \n`running`")
            elif status == "waiting_human":
                st.markdown(f"**{icon} {label}**  \n`waiting for human`")
            elif status == "waiting_request":
                st.markdown(f"{icon} {label}  \n`waiting for request`")
            elif status == "completed":
                st.markdown(f"{icon} {label}")
            elif status in {"failed", "blocked"}:
                st.markdown(f"**{icon} {label}**  \n`{status}`")
            else:
                st.markdown(f"{icon} {label}")

        reason = str(tracker.get("state_reason") or "").strip()
        if reason:
            with st.expander("Current state details", expanded=False):
                st.write(reason)
