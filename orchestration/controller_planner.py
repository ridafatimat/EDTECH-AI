from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv

from mcp_server.tool_policy import ALL_TOOL_POLICIES
from orchestration.controller_models import ControllerAction, ControllerDecision
from orchestration.state_resolver import WorkflowSnapshot


class ControllerPlanner(Protocol):
    def decide(
        self,
        *,
        user_request: str,
        snapshot: WorkflowSnapshot,
        safe_candidates: tuple[str, ...],
    ) -> ControllerDecision: ...


SYSTEM_PROMPT = """
You are the conservative orchestration planner for an EDTech multi-agent system.

You do NOT perform preprocessing, chunking, topic mapping, database operations,
Qdrant search, human review, or assessment retrieval yourself. You only decide
whether the controller should call ONE tool from the SAFE CANDIDATES supplied by
Python, or finish without a tool call.

Hard rules:
1. You may choose only a tool whose exact name appears in safe_candidates.
2. Never invent a tool name.
3. Never approve/correct/reject topics, edit detected topics, or approve Agent 2
   handoff on behalf of the human.
4. Workflow state is authoritative; do not infer a different state.
5. If the user's request cannot be satisfied by the available safe candidates,
   return action="complete" and explain what is missing.
6. Prefer the least-mutating operation that directly serves the request.
7. Never start Agent 2 retrieval unless the user clearly requested an assessment/question set.
8. Return JSON only.

Required JSON:
{
  "action": "call_tool | complete",
  "tool_name": "exact safe candidate or null",
  "reason": "brief reason"
}
""".strip()


@dataclass(slots=True)
class GroqControllerPlanner:
    """LLM planner used only when deterministic orchestration is ambiguous.

    State resolution and safety filtering happen before this class is called.
    The planner never sees HUMAN_UI_ONLY tools unless a programming error occurs,
    and its output is validated again by the controller before execution.
    """

    model_name: str = field(
        default_factory=lambda: (
            os.getenv("GROQ_CONTROLLER_MODEL")
            or os.getenv("GROQ_MODEL")
            or "openai/gpt-oss-20b"
        )
    )
    api_key: str | None = None
    max_completion_tokens: int = 350
    reasoning_effort: str = "low"
    client: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        load_dotenv()
        if self.api_key is None:
            self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = str(self.model_name).strip()
        if not self.api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is required when the Phase 7 controller needs "
                "semantic tool selection among multiple safe candidates."
            )
        if not self.model_name:
            raise EnvironmentError("No Groq controller model is configured.")
        if self.client is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise EnvironmentError(
                    "The groq package is required for semantic controller planning. "
                    "Install the project/MCP requirements first."
                ) from exc
            self.client = Groq(api_key=self.api_key)

    def _call(self, payload: dict) -> dict:
        kwargs = {
            "model": self.model_name,
            "temperature": 0,
            "max_completion_tokens": int(self.max_completion_tokens),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
        }
        try:
            response = self.client.chat.completions.create(
                reasoning_effort=self.reasoning_effort,
                **kwargs,
            )
        except Exception as first_error:
            message = str(first_error).casefold()
            if not any(
                token in message
                for token in ("reasoning_effort", "reasoning effort", "unsupported", "unknown field")
            ):
                raise
            response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty controller-planner response.")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Controller planner response must be a JSON object.")
        return parsed

    def decide(
        self,
        *,
        user_request: str,
        snapshot: WorkflowSnapshot,
        safe_candidates: tuple[str, ...],
    ) -> ControllerDecision:
        tool_descriptions = {
            name: ALL_TOOL_POLICIES[name].description
            for name in safe_candidates
            if name in ALL_TOOL_POLICIES
        }
        raw = self._call(
            {
                "user_request": str(user_request),
                "workflow_state": snapshot.state.value,
                "human_gate": snapshot.human_gate.value,
                "state_reason": snapshot.reason,
                "safe_candidates": list(safe_candidates),
                "tool_descriptions": tool_descriptions,
            }
        )

        action = str(raw.get("action") or "").strip().casefold()
        tool_name = raw.get("tool_name")
        tool_name = str(tool_name).strip() if tool_name is not None else None
        reason = " ".join(str(raw.get("reason") or "").strip().split())
        if not reason:
            reason = "Planner returned no explanation."

        if action == "call_tool":
            if not tool_name or tool_name not in safe_candidates:
                return ControllerDecision(
                    action=ControllerAction.BLOCKED,
                    tool_name=None,
                    reason=(
                        "Planner attempted to select a tool outside the Python-filtered "
                        "safe candidate set; execution was blocked."
                    ),
                    decision_source="groq_output_guardrail",
                    safe_candidates=list(safe_candidates),
                )
            return ControllerDecision(
                action=ControllerAction.CALL_TOOL,
                tool_name=tool_name,
                reason=reason,
                decision_source="groq_planner",
                safe_candidates=list(safe_candidates),
            )

        return ControllerDecision(
            action=ControllerAction.COMPLETE,
            tool_name=None,
            reason=reason,
            decision_source="groq_planner",
            safe_candidates=list(safe_candidates),
        )
