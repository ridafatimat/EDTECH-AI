from __future__ import annotations

from enum import Enum


class WorkflowState(str, Enum):
    """Controller-visible EDTech orchestration states."""

    NO_RUN = "NO_RUN"
    RAW_TRANSCRIPT_READY = "RAW_TRANSCRIPT_READY"
    PREPROCESSING_COMPLETE = "PREPROCESSING_COMPLETE"
    CHUNKS_READY = "CHUNKS_READY"
    TOPIC_MAPPING_COMPLETE = "TOPIC_MAPPING_COMPLETE"

    AWAITING_TOPIC_MAPPING_REVIEW = "AWAITING_TOPIC_MAPPING_REVIEW"
    REVIEW_STATE_INCONSISTENT = "REVIEW_STATE_INCONSISTENT"
    NO_RETAINED_TOPICS = "NO_RETAINED_TOPICS"
    AWAITING_AGENT2_TOPIC_APPROVAL = "AWAITING_AGENT2_TOPIC_APPROVAL"

    # Human-approved Agent 1 topics are available; Agent 2 may run.
    TOPICS_APPROVED = "TOPICS_APPROVED"

    # A current assessment request artifact exists but no current package yet.
    ASSESSMENT_REQUEST_READY = "ASSESSMENT_REQUEST_READY"

    # Agent 2 completed safely but did not return a suitable assessment.
    NO_SAFE_ASSESSMENT = "NO_SAFE_ASSESSMENT"

    # Agent 2 returned a current assessment package. Agent 2 currently has no
    # formal HITL/self-improving gate, so this is a normal ready state.
    ASSESSMENT_READY = "ASSESSMENT_READY"

    TOOL_FAILED = "TOOL_FAILED"
    INVALID_STATE = "INVALID_STATE"


class HumanGate(str, Enum):
    NONE = "NONE"
    TOPIC_MAPPING_REVIEW = "TOPIC_MAPPING_REVIEW"
    TOPIC_REVIEW_INTEGRITY = "TOPIC_REVIEW_INTEGRITY"
    AGENT2_TOPIC_APPROVAL = "AGENT2_TOPIC_APPROVAL"
