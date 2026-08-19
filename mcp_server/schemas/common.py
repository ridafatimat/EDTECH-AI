from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orchestration.workflow_state import WorkflowState


class ToolStatus(str, Enum):
    COMPLETED = "completed"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    BLOCKED = "blocked"
    FAILED = "failed"
    NO_CHANGE = "no_change"


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)


class ToolResult(BaseModel):
    """Common controller-facing response envelope for all MCP tools."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    status: ToolStatus
    run_id: str
    state: WorkflowState
    human_action_required: bool = False
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error: str | None = None
