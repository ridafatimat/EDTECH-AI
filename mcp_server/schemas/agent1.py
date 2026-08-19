from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import RunRequest


class TopicReviewAction(str, Enum):
    APPROVE = "approve"
    CORRECT = "correct"
    REJECT = "reject"


class TopicReviewDecision(BaseModel):
    """One Module 3 mapping-review decision.

    This mirrors the existing topic_review_set_status path. Corrections require
    a human reason. Approve/reject behavior is not changed by MCP.
    """

    model_config = ConfigDict(extra="forbid")

    review_id: int = Field(gt=0)
    action: TopicReviewAction
    corrected_decision: Literal["mapped", "resolved_by_module3", "out_of_syllabus"] | None = None
    corrected_mapped_concept_id: str | None = None
    reason: str | None = None
    review_notes: str | None = None

    @model_validator(mode="after")
    def validate_correction(self) -> "TopicReviewDecision":
        if self.action is not TopicReviewAction.CORRECT:
            return self

        if not (self.reason or "").strip():
            raise ValueError("A human correction reason is required.")
        if self.corrected_decision is None:
            raise ValueError("corrected_decision is required for a correction.")
        if (
            self.corrected_decision != "out_of_syllabus"
            and not (self.corrected_mapped_concept_id or "").strip()
        ):
            raise ValueError(
                "corrected_mapped_concept_id is required for a mapped correction."
            )
        return self


class TopicReviewRequest(RunRequest):
    decisions: list[TopicReviewDecision] = Field(min_length=1)
    reviewed_by: str = Field(default="mcp_human_review", min_length=1)


class DetectedTopicEditAction(str, Enum):
    REPLACE_TOPIC = "replace_topic"
    REMOVE_TOPIC = "remove_topic"
    CHANGE_ROLE = "change_role"
    ADD_TOPIC = "add_topic"


class DetectedTopicEditRequest(RunRequest):
    """Human-authorized edit that feeds the existing self-improving memory.

    The reason is mandatory for every action because these operations change
    the final detected-topic set/role and are the reviewer-supervised learning
    path already present in Agent 1.
    """

    model_config = ConfigDict(extra="forbid")

    action: DetectedTopicEditAction
    topic_index: int | None = Field(default=None, ge=0)
    source_concept_id: str | None = None
    target_concept_id: str | None = None
    target_role: Literal["primary", "supporting"] | None = None
    source_chunk_ids: list[int] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    reviewed_by: str = Field(default="mcp_human_review", min_length=1)

    @model_validator(mode="after")
    def validate_edit(self) -> "DetectedTopicEditRequest":
        if not self.reason.strip():
            raise ValueError("A reviewer reason is required for every detected-topic edit.")

        if self.action is DetectedTopicEditAction.ADD_TOPIC:
            if not (self.target_concept_id or "").strip():
                raise ValueError("target_concept_id is required when adding a topic.")
            if self.target_role is None:
                raise ValueError("target_role is required when adding a topic.")
            if not self.source_chunk_ids:
                raise ValueError("source_chunk_ids are required when adding a topic.")
            return self

        if self.topic_index is None:
            raise ValueError("topic_index is required for edits to an existing topic.")

        if self.action is DetectedTopicEditAction.REPLACE_TOPIC and not (
            self.target_concept_id or ""
        ).strip():
            raise ValueError("target_concept_id is required when replacing a topic.")

        if self.action is DetectedTopicEditAction.CHANGE_ROLE and self.target_role is None:
            raise ValueError("target_role is required when changing a topic role.")

        return self


class ApprovedTopicSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_index: int = Field(ge=0)
    approved: bool = True
    topic: str | None = None
    role: Literal["primary", "supporting"] | None = None
    official_reference: str | None = None


class Agent2TopicApprovalRequest(RunRequest):
    """Separate Agent 1 -> Agent 2 human handoff approval."""

    selections: list[ApprovedTopicSelection] = Field(min_length=1)
    reviewed_by: str = Field(default="mcp_human_review", min_length=1)
