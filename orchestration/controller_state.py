from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mcp_server.adapters.agent1_hitl_adapter import Agent1HitlAdapter
from orchestration.state_resolver import WorkflowSnapshot, resolve_agent1_state


class ControllerStateProvider(Protocol):
    def get_snapshot(self, run_id: str) -> WorkflowSnapshot: ...


class Agent1ControllerStateProvider:
    """DB-reconciled deterministic state provider for the Phase 7 controller.

    The controller never asks an LLM to infer workflow state. Existing run
    artifacts are read deterministically and live PostgreSQL review statuses
    override stale JSON review statuses when available, matching the safety
    behavior already tested by Agent1ToolService.
    """

    def __init__(self, frontend_project_root: Path | str):
        self.frontend_project_root = Path(frontend_project_root).resolve()
        self.runs_root = self.frontend_project_root / "runs"
        self.hitl = Agent1HitlAdapter(self.frontend_project_root)

    def get_snapshot(self, run_id: str) -> WorkflowSnapshot:
        run_dir = self.runs_root / str(run_id)
        overrides: dict[int, str] | None = None
        try:
            overrides = self.hitl.get_review_status_overrides(str(run_id))
        except Exception:
            # Same fail-closed behavior as the tested tool service: when live
            # review reconciliation cannot be read, retain artifact state. A
            # stale pending artifact therefore remains a human gate.
            overrides = None
        return resolve_agent1_state(
            run_dir,
            topic_review_status_overrides=overrides,
        )
