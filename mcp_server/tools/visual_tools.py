from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from mcp_server.schemas.common import ToolResult, ToolStatus
from mcp_server.schemas.visuals import RenderVisualsRequest
from orchestration.state_resolver import resolve_agent1_state
from orchestration.workflow_state import WorkflowState


class VisualRendererExecutor(Protocol):
    frontend_project_root: Path

    def render_logic_visual(self, request: RenderVisualsRequest) -> dict[str, Any]: ...

    def render_technical_visual(
        self, request: RenderVisualsRequest
    ) -> dict[str, Any]: ...

    def render_structured_visual(
        self, request: RenderVisualsRequest
    ) -> dict[str, Any]: ...


class VisualToolService:
    """Three high-level MCP visual contracts over Notebook 08."""

    def __init__(self, executor: VisualRendererExecutor):
        self.executor = executor

    def _run_dir(self, run_id: str) -> Path:
        return Path(self.executor.frontend_project_root) / "runs" / str(run_id)

    def _snapshot(self, run_id: str):
        return resolve_agent1_state(self._run_dir(run_id))

    def _completed(
        self,
        *,
        request: RenderVisualsRequest,
        data: dict[str, Any],
    ) -> ToolResult:
        snapshot = self._snapshot(request.run_id)
        return ToolResult(
            success=True,
            status=ToolStatus.COMPLETED,
            run_id=request.run_id,
            state=snapshot.state,
            human_action_required=snapshot.human_action_required,
            message=(
                f"{data.get('tool_name') or 'Visual MCP tool'} completed for "
                f"{data.get('selected_question_count', 0)} question(s)."
            ),
            data=data,
        )

    def _failed(
        self,
        *,
        request: RenderVisualsRequest,
        error_code: str,
        exc: Exception,
    ) -> ToolResult:
        try:
            snapshot = self._snapshot(request.run_id)
            state = (
                snapshot.state
                if snapshot.human_action_required
                else WorkflowState.TOOL_FAILED
            )
            human_required = snapshot.human_action_required
        except Exception:
            state = WorkflowState.TOOL_FAILED
            human_required = False

        return ToolResult(
            success=False,
            status=ToolStatus.FAILED,
            run_id=request.run_id,
            state=state,
            human_action_required=human_required,
            message=str(exc),
            data={},
            error_code=error_code,
            error=f"{type(exc).__name__}: {exc}",
        )

    def render_logic_visual(self, request: RenderVisualsRequest) -> ToolResult:
        try:
            data = self.executor.render_logic_visual(request)
        except Exception as exc:
            return self._failed(
                request=request,
                error_code="LOGIC_VISUAL_RENDER_FAILED",
                exc=exc,
            )
        return self._completed(request=request, data=data)

    def render_technical_visual(self, request: RenderVisualsRequest) -> ToolResult:
        try:
            data = self.executor.render_technical_visual(request)
        except Exception as exc:
            return self._failed(
                request=request,
                error_code="TECHNICAL_VISUAL_RENDER_FAILED",
                exc=exc,
            )
        return self._completed(request=request, data=data)

    def render_structured_visual(self, request: RenderVisualsRequest) -> ToolResult:
        try:
            data = self.executor.render_structured_visual(request)
        except Exception as exc:
            return self._failed(
                request=request,
                error_code="STRUCTURED_VISUAL_RENDER_FAILED",
                exc=exc,
            )
        return self._completed(request=request, data=data)
