from __future__ import annotations

from enum import Enum

from orchestration.workflow_state import HumanGate, WorkflowState


class LangGraphRoute(str, Enum):
    PREPROCESS = "preprocess"
    CHUNK = "chunk"
    TOPIC_MAPPING = "topic_mapping"
    HUMAN_GATE = "human_gate"
    TOPIC_STATUS = "topic_status"
    AGENT2_DECISION = "agent2_decision"
    COMPLETE = "complete"
    BLOCKED = "blocked"


def route_for_state(state: WorkflowState, human_gate: HumanGate) -> LangGraphRoute:
    """Deterministically map existing workflow state to a LangGraph node."""

    if human_gate is not HumanGate.NONE:
        return LangGraphRoute.HUMAN_GATE

    if state is WorkflowState.RAW_TRANSCRIPT_READY:
        return LangGraphRoute.PREPROCESS
    if state is WorkflowState.PREPROCESSING_COMPLETE:
        return LangGraphRoute.CHUNK
    if state is WorkflowState.CHUNKS_READY:
        return LangGraphRoute.TOPIC_MAPPING

    if state in {
        WorkflowState.TOPIC_MAPPING_COMPLETE,
        WorkflowState.NO_RETAINED_TOPICS,
    }:
        return LangGraphRoute.TOPIC_STATUS

    if state in {
        WorkflowState.TOPICS_APPROVED,
        WorkflowState.ASSESSMENT_REQUEST_READY,
    }:
        return LangGraphRoute.AGENT2_DECISION

    if state in {
        WorkflowState.ASSESSMENT_READY,
        WorkflowState.NO_SAFE_ASSESSMENT,
    }:
        return LangGraphRoute.COMPLETE

    return LangGraphRoute.BLOCKED
