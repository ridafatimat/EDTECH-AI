from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from orchestration.guardrails import (
    AGENT1_CHUNK,
    AGENT1_MAP,
    AGENT1_PREPROCESS,
    GET_APPROVED_TOPICS,
    GET_DETECTED_TOPICS,
    GET_PENDING_TOPIC_REVIEW,
    SAVE_AGENT2_TOPIC_APPROVAL,
    SUBMIT_DETECTED_TOPIC_EDIT,
    SUBMIT_TOPIC_REVIEW,
    ToolNotAllowedError,
    assert_tool_allowed,
)
from orchestration.state_resolver import resolve_agent1_state
from mcp_server.schemas.agent1 import (
    Agent2TopicApprovalRequest,
    DetectedTopicEditRequest,
    TopicReviewRequest,
)
from mcp_server.schemas.common import RunRequest, ToolResult, ToolStatus


class Agent1Executor(Protocol):
    frontend_project_root: Path

    def execute(self, *, run_id: str, module: str) -> tuple[Path, str]: ...


class Agent1HitlExecutor(Protocol):
    frontend_project_root: Path

    def get_effective_topics(self, run_id: str) -> dict[str, Any]: ...

    def get_pending_topic_reviews(self, run_id: str) -> dict[str, Any]: ...

    def get_review_status_overrides(self, run_id: str) -> dict[int, str]: ...

    def submit_topic_review(self, request: TopicReviewRequest) -> list[dict[str, Any]]: ...

    def submit_detected_topic_edit(self, request: DetectedTopicEditRequest) -> dict[str, Any]: ...

    def get_approved_topics(self, run_id: str) -> dict[str, Any]: ...

    def save_agent2_topic_approval(
        self, request: Agent2TopicApprovalRequest
    ) -> dict[str, Any]: ...


class Agent1ToolService:
    """Controller-facing high-level wrappers for existing Agent 1 behavior."""

    def __init__(
        self,
        executor: Agent1Executor,
        hitl_executor: Agent1HitlExecutor | None = None,
    ):
        self.executor = executor
        if hitl_executor is None:
            from mcp_server.adapters.agent1_hitl_adapter import Agent1HitlAdapter

            hitl_executor = Agent1HitlAdapter(executor.frontend_project_root)
        self.hitl_executor = hitl_executor

    def _run_dir(self, run_id: str) -> Path:
        return Path(self.executor.frontend_project_root) / "runs" / run_id

    def _snapshot(self, run_id: str):
        """Resolve controller state using DB review status when available."""

        overrides: dict[int, str] | None = None
        provider = getattr(self.hitl_executor, "get_review_status_overrides", None)
        if callable(provider):
            try:
                overrides = provider(run_id)
            except Exception:
                # Fail closed: if DB reconciliation cannot be read, retain the
                # artifact state. A stale pending artifact therefore remains a
                # human gate rather than accidentally unlocking Agent 2.
                overrides = None
        return resolve_agent1_state(
            self._run_dir(run_id),
            topic_review_status_overrides=overrides,
        )

    def _blocked(self, *, run_id: str, tool_name: str) -> ToolResult | None:
        before = self._snapshot(run_id)
        try:
            assert_tool_allowed(before.state, tool_name)
        except ToolNotAllowedError as exc:
            return ToolResult(
                success=False,
                status=ToolStatus.BLOCKED,
                run_id=run_id,
                state=before.state,
                human_action_required=before.human_action_required,
                message=str(exc),
                error_code="TOOL_NOT_ALLOWED_FOR_STATE",
                error=str(exc),
                data={"allowed_tools": list(before.allowed_tools)},
            )
        return None

    def _result_after(
        self,
        *,
        run_id: str,
        data: dict[str, Any],
        message: str | None = None,
        success_status: ToolStatus | None = None,
    ) -> ToolResult:
        after = self._snapshot(run_id)
        status = success_status or (
            ToolStatus.HUMAN_REVIEW_REQUIRED
            if after.human_action_required
            else ToolStatus.COMPLETED
        )
        return ToolResult(
            success=True,
            status=status,
            run_id=run_id,
            state=after.state,
            human_action_required=after.human_action_required,
            message=message or after.reason,
            data={
                **data,
                "allowed_tools": list(after.allowed_tools),
                "human_gate": after.human_gate.value,
                "pending_topic_review_count": after.pending_topic_review_count,
                "approved_topic_count": after.approved_topic_count,
                "topic_review_integrity_issue_count": after.topic_review_integrity_issue_count,
            },
        )

    def _failed(
        self,
        *,
        run_id: str,
        tool_name: str,
        exc: Exception,
        error_code: str,
    ) -> ToolResult:
        snapshot = self._snapshot(run_id)
        return ToolResult(
            success=False,
            status=ToolStatus.FAILED,
            run_id=run_id,
            state=snapshot.state,
            human_action_required=snapshot.human_action_required,
            message=f"{tool_name} failed without changing orchestration rules.",
            error_code=error_code,
            error=f"{type(exc).__name__}: {exc}",
            data={"allowed_tools": list(snapshot.allowed_tools)},
        )

    # ------------------------------------------------------------------
    # Existing notebook stages
    # ------------------------------------------------------------------
    def _run_stage(
        self,
        *,
        request: RunRequest,
        tool_name: str,
        module: str,
    ) -> ToolResult:
        blocked = self._blocked(run_id=request.run_id, tool_name=tool_name)
        if blocked is not None:
            return blocked

        try:
            resolved_run_dir, transcript_name = self.executor.execute(
                run_id=request.run_id,
                module=module,
            )
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                tool_name=tool_name,
                exc=exc,
                error_code="AGENT1_NOTEBOOK_EXECUTION_FAILED",
            )

        # The notebook stage may have created/updated PostgreSQL review rows,
        # so resolve the post-stage state with the same DB-authoritative review
        # reconciliation used by all controller guardrails.
        after = self._snapshot(request.run_id)
        status = (
            ToolStatus.HUMAN_REVIEW_REQUIRED
            if after.human_action_required
            else ToolStatus.COMPLETED
        )
        return ToolResult(
            success=True,
            status=status,
            run_id=request.run_id,
            state=after.state,
            human_action_required=after.human_action_required,
            message=after.reason,
            data={
                "transcript_name": after.transcript_name,
                "allowed_tools": list(after.allowed_tools),
                "human_gate": after.human_gate.value,
                "pending_topic_review_count": after.pending_topic_review_count,
                "approved_topic_count": after.approved_topic_count,
                "topic_review_integrity_issue_count": after.topic_review_integrity_issue_count,
            },
        )

    def run_agent1_preprocessing(self, request: RunRequest) -> ToolResult:
        return self._run_stage(
            request=request,
            tool_name=AGENT1_PREPROCESS,
            module="preprocessing",
        )

    def run_agent1_chunking(self, request: RunRequest) -> ToolResult:
        return self._run_stage(
            request=request,
            tool_name=AGENT1_CHUNK,
            module="chunking",
        )

    def run_agent1_topic_mapping(self, request: RunRequest) -> ToolResult:
        return self._run_stage(
            request=request,
            tool_name=AGENT1_MAP,
            module="topic_mapping",
        )

    # ------------------------------------------------------------------
    # Phase 5A: real HITL / handoff wrappers
    # ------------------------------------------------------------------
    def get_detected_topics(self, request: RunRequest) -> ToolResult:
        blocked = self._blocked(run_id=request.run_id, tool_name=GET_DETECTED_TOPICS)
        if blocked is not None:
            return blocked
        try:
            data = self.hitl_executor.get_effective_topics(request.run_id)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                tool_name=GET_DETECTED_TOPICS,
                exc=exc,
                error_code="AGENT1_GET_DETECTED_TOPICS_FAILED",
            )
        return self._result_after(run_id=request.run_id, data=data)

    def get_pending_topic_review(self, request: RunRequest) -> ToolResult:
        blocked = self._blocked(
            run_id=request.run_id,
            tool_name=GET_PENDING_TOPIC_REVIEW,
        )
        if blocked is not None:
            return blocked
        try:
            data = self.hitl_executor.get_pending_topic_reviews(request.run_id)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                tool_name=GET_PENDING_TOPIC_REVIEW,
                exc=exc,
                error_code="AGENT1_GET_PENDING_REVIEW_FAILED",
            )
        return self._result_after(run_id=request.run_id, data=data)

    def submit_topic_review(self, request: TopicReviewRequest) -> ToolResult:
        blocked = self._blocked(run_id=request.run_id, tool_name=SUBMIT_TOPIC_REVIEW)
        if blocked is not None:
            return blocked
        try:
            rows = self.hitl_executor.submit_topic_review(request)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                tool_name=SUBMIT_TOPIC_REVIEW,
                exc=exc,
                error_code="AGENT1_SUBMIT_TOPIC_REVIEW_FAILED",
            )
        return self._result_after(
            run_id=request.run_id,
            data={"updated_reviews": rows, "updated_count": len(rows)},
            message=(
                "Human topic-mapping decision saved through the existing "
                "PostgreSQL review/memory path."
            ),
        )

    def submit_detected_topic_edit(self, request: DetectedTopicEditRequest) -> ToolResult:
        blocked = self._blocked(
            run_id=request.run_id,
            tool_name=SUBMIT_DETECTED_TOPIC_EDIT,
        )
        if blocked is not None:
            return blocked
        try:
            data = self.hitl_executor.submit_detected_topic_edit(request)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                tool_name=SUBMIT_DETECTED_TOPIC_EDIT,
                exc=exc,
                error_code="AGENT1_SUBMIT_DETECTED_TOPIC_EDIT_FAILED",
            )
        return self._result_after(
            run_id=request.run_id,
            data=data,
            message=(
                "Human detected-topic correction saved through Agent 1's "
                "existing reviewer-approved contextual edit-memory path."
            ),
        )

    def save_agent2_topic_approval(
        self, request: Agent2TopicApprovalRequest
    ) -> ToolResult:
        blocked = self._blocked(
            run_id=request.run_id,
            tool_name=SAVE_AGENT2_TOPIC_APPROVAL,
        )
        if blocked is not None:
            return blocked
        try:
            data = self.hitl_executor.save_agent2_topic_approval(request)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                tool_name=SAVE_AGENT2_TOPIC_APPROVAL,
                exc=exc,
                error_code="AGENT1_SAVE_AGENT2_APPROVAL_FAILED",
            )
        return self._result_after(
            run_id=request.run_id,
            data=data,
            message="Human-approved Agent 1 topics are now available to Agent 2.",
        )

    def get_approved_topics(self, request: RunRequest) -> ToolResult:
        blocked = self._blocked(run_id=request.run_id, tool_name=GET_APPROVED_TOPICS)
        if blocked is not None:
            return blocked
        try:
            data = self.hitl_executor.get_approved_topics(request.run_id)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                tool_name=GET_APPROVED_TOPICS,
                exc=exc,
                error_code="AGENT1_GET_APPROVED_TOPICS_FAILED",
            )
        return self._result_after(run_id=request.run_id, data=data)
