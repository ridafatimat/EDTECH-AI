from __future__ import annotations

from .workflow_state import WorkflowState


# Agent 1 high-level tools
AGENT1_PREPROCESS = "run_agent1_preprocessing"
AGENT1_CHUNK = "run_agent1_chunking"
AGENT1_MAP = "run_agent1_topic_mapping"
GET_DETECTED_TOPICS = "get_detected_topics"
GET_PENDING_TOPIC_REVIEW = "get_pending_topic_review"
SUBMIT_TOPIC_REVIEW = "submit_topic_review"
SUBMIT_DETECTED_TOPIC_EDIT = "submit_detected_topic_edit"
SAVE_AGENT2_TOPIC_APPROVAL = "save_agent2_topic_approval"
GET_APPROVED_TOPICS = "get_approved_topics"

# Phase 8 Agent 2 tools. These mirror capabilities that exist now.
AGENT2_RETRIEVAL = "run_agent2_retrieval"
AGENT2_COMPLETE_QUIZ = "generate_agent2_complete_quiz"
AGENT2_MISSING_QUIZ = "generate_agent2_missing_quiz_coverage"
SUBMIT_AGENT2_QUIZ_REVIEW = "submit_agent2_quiz_review"
GET_AGENT2_ASSESSMENT = "get_agent2_assessment"
GET_AGENT2_MARK_SCHEMES = "get_agent2_mark_schemes"
GET_AGENT2_RENDERED_PAGES = "get_agent2_rendered_pages"

# Visual-rendering MCP tools.
#
# These are intentionally NOT added to the deterministic workflow-state
# allow-list below. The generic controller therefore never sees them as normal
# state-transition candidates. LangGraph grants them locally only after a
# successful Notebook 06 quiz action and only when the current visual handoff
# requires that renderer family.
RENDER_LOGIC_VISUAL = "render_logic_visual"
RENDER_TECHNICAL_VISUAL = "render_technical_visual"
RENDER_STRUCTURED_VISUAL = "render_structured_visual"


_ALLOWED: dict[WorkflowState, frozenset[str]] = {
    WorkflowState.NO_RUN: frozenset(),
    WorkflowState.RAW_TRANSCRIPT_READY: frozenset({AGENT1_PREPROCESS}),
    WorkflowState.PREPROCESSING_COMPLETE: frozenset({AGENT1_CHUNK}),
    WorkflowState.CHUNKS_READY: frozenset({AGENT1_MAP}),
    WorkflowState.TOPIC_MAPPING_COMPLETE: frozenset(
        {GET_DETECTED_TOPICS, GET_PENDING_TOPIC_REVIEW}
    ),
    WorkflowState.AWAITING_TOPIC_MAPPING_REVIEW: frozenset(
        {GET_DETECTED_TOPICS, GET_PENDING_TOPIC_REVIEW, SUBMIT_TOPIC_REVIEW}
    ),
    WorkflowState.REVIEW_STATE_INCONSISTENT: frozenset(
        {GET_DETECTED_TOPICS, GET_PENDING_TOPIC_REVIEW}
    ),
    WorkflowState.NO_RETAINED_TOPICS: frozenset(
        {GET_DETECTED_TOPICS, SUBMIT_DETECTED_TOPIC_EDIT}
    ),
    WorkflowState.AWAITING_AGENT2_TOPIC_APPROVAL: frozenset(
        {
            GET_DETECTED_TOPICS,
            SUBMIT_DETECTED_TOPIC_EDIT,
            SAVE_AGENT2_TOPIC_APPROVAL,
            GET_APPROVED_TOPICS,
        }
    ),
    WorkflowState.TOPICS_APPROVED: frozenset(
        {
            GET_DETECTED_TOPICS,
            GET_APPROVED_TOPICS,
            SUBMIT_DETECTED_TOPIC_EDIT,
            SAVE_AGENT2_TOPIC_APPROVAL,
            AGENT2_RETRIEVAL,
            AGENT2_COMPLETE_QUIZ,
            SUBMIT_AGENT2_QUIZ_REVIEW,
        }
    ),
    WorkflowState.ASSESSMENT_REQUEST_READY: frozenset(
        {
            GET_APPROVED_TOPICS,
            SAVE_AGENT2_TOPIC_APPROVAL,
            AGENT2_RETRIEVAL,
            AGENT2_COMPLETE_QUIZ,
        }
    ),
    WorkflowState.NO_SAFE_ASSESSMENT: frozenset(
        {
            GET_AGENT2_ASSESSMENT,
            GET_APPROVED_TOPICS,
            SAVE_AGENT2_TOPIC_APPROVAL,
            AGENT2_RETRIEVAL,
            AGENT2_COMPLETE_QUIZ,
            AGENT2_MISSING_QUIZ,
            SUBMIT_AGENT2_QUIZ_REVIEW,
        }
    ),
    WorkflowState.ASSESSMENT_READY: frozenset(
        {
            GET_AGENT2_ASSESSMENT,
            GET_AGENT2_MARK_SCHEMES,
            GET_AGENT2_RENDERED_PAGES,
            GET_APPROVED_TOPICS,
            SAVE_AGENT2_TOPIC_APPROVAL,
            AGENT2_RETRIEVAL,
            AGENT2_COMPLETE_QUIZ,
            AGENT2_MISSING_QUIZ,
            SUBMIT_AGENT2_QUIZ_REVIEW,
        }
    ),
    WorkflowState.TOOL_FAILED: frozenset(),
    WorkflowState.INVALID_STATE: frozenset(),
}


class ToolNotAllowedError(RuntimeError):
    pass


def allowed_tools_for_state(state: WorkflowState) -> tuple[str, ...]:
    return tuple(sorted(_ALLOWED.get(state, frozenset())))


def is_tool_allowed(state: WorkflowState, tool_name: str) -> bool:
    return tool_name in _ALLOWED.get(state, frozenset())


def assert_tool_allowed(state: WorkflowState, tool_name: str) -> None:
    if is_tool_allowed(state, tool_name):
        return
    allowed = ", ".join(allowed_tools_for_state(state)) or "none"
    raise ToolNotAllowedError(
        f"Tool '{tool_name}' is not allowed while workflow state is "
        f"{state.value}. Allowed tools: {allowed}."
    )