from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orchestration.workflow_state import HumanGate, WorkflowState


class ControllerAction(str, Enum):
    CALL_TOOL = "call_tool"
    PAUSE_FOR_HUMAN = "pause_for_human"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    FAILED = "failed"


class ControllerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ControllerAction
    tool_name: str | None = None
    reason: str = Field(min_length=1)
    decision_source: str = Field(min_length=1)
    safe_candidates: list[str] = Field(default_factory=list)
    tool_arguments: dict[str, Any] = Field(default_factory=dict)


class ControllerStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    user_request: str
    state_before: WorkflowState
    state_after: WorkflowState
    human_gate_before: HumanGate
    human_gate_after: HumanGate
    decision: ControllerDecision
    tool_called: str | None = None
    tool_success: bool | None = None
    tool_payload: dict[str, Any] = Field(default_factory=dict)
    should_stop: bool
    stop_reason: str = ""


class ControllerRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    user_request: str
    steps: list[ControllerStepResult] = Field(default_factory=list)
    final_state: WorkflowState
    final_human_gate: HumanGate
    human_action_required: bool
    completed: bool
    stop_reason: str
