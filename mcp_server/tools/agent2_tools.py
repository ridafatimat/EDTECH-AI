from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from mcp_server.schemas.agent2 import (
    GenerateAgent2CompleteQuizRequest,
    GenerateAgent2MissingQuizRequest,
    GetAgent2AssessmentRequest,
    GetAgent2MarkSchemesRequest,
    GetAgent2RenderedPagesRequest,
    RunAgent2RetrievalRequest,
    SubmitAgent2QuizReviewRequest,
)
from mcp_server.schemas.common import ToolResult, ToolStatus
from orchestration.guardrails import (
    AGENT2_RETRIEVAL,
    AGENT2_COMPLETE_QUIZ,
    AGENT2_MISSING_QUIZ,
    SUBMIT_AGENT2_QUIZ_REVIEW,
    GET_AGENT2_ASSESSMENT,
    GET_AGENT2_MARK_SCHEMES,
    GET_AGENT2_RENDERED_PAGES,
    ToolNotAllowedError,
    assert_tool_allowed,
)
from orchestration.state_resolver import resolve_agent1_state
from orchestration.workflow_state import WorkflowState


class Agent2Executor(Protocol):
    frontend_project_root: Path

    def execute_retrieval(self, request: RunAgent2RetrievalRequest) -> dict[str, Any]: ...

    def execute_complete_quiz(self, request: GenerateAgent2CompleteQuizRequest) -> dict[str, Any]: ...

    def execute_missing_quiz(self, request: GenerateAgent2MissingQuizRequest) -> dict[str, Any]: ...

    def submit_quiz_review(self, request: SubmitAgent2QuizReviewRequest) -> dict[str, Any]: ...

    def get_current_assessment(self, run_id: str) -> dict[str, Any]: ...

    def fetch_mark_schemes(
        self, run_id: str, question_ids: list[str]
    ) -> dict[str, Any]: ...

    def get_rendered_question_pages(
        self, run_id: str, question_ids: list[str]
    ) -> dict[str, Any]: ...


class Agent2ToolService:
    """Thin MCP boundary over the existing Agent 2 runner/notebooks.

    No retrieval/ranking/mark-scheme/rendering algorithm is implemented here.
    """

    def __init__(self, executor: Agent2Executor):
        self.executor = executor

    def _run_dir(self, run_id: str) -> Path:
        return Path(self.executor.frontend_project_root) / "runs" / run_id

    def _snapshot(self, run_id: str):
        return resolve_agent1_state(self._run_dir(run_id))

    def _blocked(self, *, run_id: str, tool_name: str) -> ToolResult | None:
        snapshot = self._snapshot(run_id)
        try:
            assert_tool_allowed(snapshot.state, tool_name)
        except ToolNotAllowedError as exc:
            return ToolResult(
                success=False,
                status=ToolStatus.BLOCKED,
                run_id=run_id,
                state=snapshot.state,
                human_action_required=snapshot.human_action_required,
                message=str(exc),
                data={"allowed_tools": list(snapshot.allowed_tools)},
                error_code="TOOL_NOT_ALLOWED_FOR_STATE",
                error=str(exc),
            )
        return None

    def _result_after(
        self,
        *,
        run_id: str,
        data: dict[str, Any],
        message: str = "",
    ) -> ToolResult:
        snapshot = self._snapshot(run_id)
        return ToolResult(
            success=True,
            status=ToolStatus.COMPLETED,
            run_id=run_id,
            state=snapshot.state,
            human_action_required=snapshot.human_action_required,
            message=message or snapshot.reason,
            data=data,
        )

    def _failed(self, *, run_id: str, error_code: str, exc: Exception) -> ToolResult:
        snapshot = self._snapshot(run_id)
        return ToolResult(
            success=False,
            status=ToolStatus.FAILED,
            run_id=run_id,
            state=(
                snapshot.state
                if snapshot.human_action_required
                else WorkflowState.TOOL_FAILED
            ),
            human_action_required=snapshot.human_action_required,
            message=str(exc),
            data={},
            error_code=error_code,
            error=f"{type(exc).__name__}: {exc}",
        )

    def run_agent2_retrieval(
        self, request: RunAgent2RetrievalRequest
    ) -> ToolResult:
        blocked = self._blocked(run_id=request.run_id, tool_name=AGENT2_RETRIEVAL)
        if blocked is not None:
            return blocked
        try:
            data = self.executor.execute_retrieval(request)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                error_code="AGENT2_RETRIEVAL_FAILED",
                exc=exc,
            )

        assessment_generated = bool(data.get("assessment_generated"))
        question_count = int(data.get("question_count") or 0)
        if assessment_generated and question_count:
            message = (
                f"Existing Agent 2 returned {question_count} question(s). "
                "The current assessment package is ready for display."
            )
        else:
            message = (
                "Existing Agent 2 completed without a safe assessment. "
                "No weak or wrong-paper question was substituted."
            )
        return self._result_after(
            run_id=request.run_id,
            data=data,
            message=message,
        )


    def generate_agent2_complete_quiz(
        self, request: GenerateAgent2CompleteQuizRequest
    ) -> ToolResult:
        blocked = self._blocked(
            run_id=request.run_id,
            tool_name=AGENT2_COMPLETE_QUIZ,
        )
        if blocked is not None:
            return blocked
        try:
            data = self.executor.execute_complete_quiz(request)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                error_code="AGENT2_COMPLETE_QUIZ_FAILED",
                exc=exc,
            )
        outcome = str(data.get("outcome") or "").strip().upper()
        message = (
            f"Notebook 06 complete_quiz finished with outcome {outcome or 'UNKNOWN'}."
        )
        return self._result_after(run_id=request.run_id, data=data, message=message)

    def generate_agent2_missing_quiz_coverage(
        self, request: GenerateAgent2MissingQuizRequest
    ) -> ToolResult:
        blocked = self._blocked(
            run_id=request.run_id,
            tool_name=AGENT2_MISSING_QUIZ,
        )
        if blocked is not None:
            return blocked
        try:
            data = self.executor.execute_missing_quiz(request)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                error_code="AGENT2_MISSING_QUIZ_FAILED",
                exc=exc,
            )
        outcome = str(data.get("outcome") or "").strip().upper()
        message = (
            f"Notebook 06 fill_shortfall finished with outcome {outcome or 'UNKNOWN'}."
        )
        return self._result_after(run_id=request.run_id, data=data, message=message)

    def submit_agent2_quiz_review(
        self, request: SubmitAgent2QuizReviewRequest
    ) -> ToolResult:
        # This is a HUMAN_UI_ONLY tool. The caller policy is enforced at the
        # MCP boundary; the adapter merely applies the already-made decision.
        try:
            data = self.executor.submit_quiz_review(request)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                error_code="AGENT2_QUIZ_REVIEW_FAILED",
                exc=exc,
            )
        return self._result_after(
            run_id=request.run_id,
            data=data,
            message=(
                f"Human quiz review '{request.decision}' was applied to "
                f"{request.quiz_mode}."
            ),
        )

    def get_agent2_assessment(
        self, request: GetAgent2AssessmentRequest
    ) -> ToolResult:
        blocked = self._blocked(
            run_id=request.run_id,
            tool_name=GET_AGENT2_ASSESSMENT,
        )
        if blocked is not None:
            return blocked
        try:
            data = self.executor.get_current_assessment(request.run_id)
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                error_code="AGENT2_GET_ASSESSMENT_FAILED",
                exc=exc,
            )
        return self._result_after(run_id=request.run_id, data=data)

    def get_agent2_mark_schemes(
        self, request: GetAgent2MarkSchemesRequest
    ) -> ToolResult:
        blocked = self._blocked(
            run_id=request.run_id,
            tool_name=GET_AGENT2_MARK_SCHEMES,
        )
        if blocked is not None:
            return blocked
        try:
            data = self.executor.fetch_mark_schemes(
                request.run_id, request.question_ids
            )
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                error_code="AGENT2_GET_MARK_SCHEMES_FAILED",
                exc=exc,
            )
        return self._result_after(
            run_id=request.run_id,
            data=data,
            message=(
                "Returned the exact mark-scheme bundles already linked by "
                "QuestionID in the existing Agent 2 package."
            ),
        )

    def get_agent2_rendered_pages(
        self, request: GetAgent2RenderedPagesRequest
    ) -> ToolResult:
        blocked = self._blocked(
            run_id=request.run_id,
            tool_name=GET_AGENT2_RENDERED_PAGES,
        )
        if blocked is not None:
            return blocked
        try:
            data = self.executor.get_rendered_question_pages(
                request.run_id, request.question_ids
            )
        except Exception as exc:
            return self._failed(
                run_id=request.run_id,
                error_code="AGENT2_GET_RENDERED_PAGES_FAILED",
                exc=exc,
            )
        return self._result_after(
            run_id=request.run_id,
            data=data,
            message=(
                "Returned page images already produced by existing Agent 2 / "
                "Notebook 07. MCP did not rerender the source PDFs."
            ),
        )
