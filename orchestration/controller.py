from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from mcp_server.tool_policy import (
    ALL_TOOL_POLICIES,
    ToolCallerPolicy,
    controller_callable_tools,
)
from orchestration.assessment_intent import (
    AssessmentRequestKind,
    parse_assessment_request,
)
from orchestration.controller_models import (
    ControllerAction,
    ControllerDecision,
    ControllerRunResult,
    ControllerStepResult,
)
from orchestration.controller_planner import ControllerPlanner, GroqControllerPlanner
from orchestration.controller_state import ControllerStateProvider
from orchestration.guardrails import (
    AGENT2_RETRIEVAL,
    AGENT2_COMPLETE_QUIZ,
    AGENT2_MISSING_QUIZ,
    GET_AGENT2_ASSESSMENT,
    GET_AGENT2_MARK_SCHEMES,
    GET_AGENT2_RENDERED_PAGES,
    GET_APPROVED_TOPICS,
    GET_DETECTED_TOPICS,
    GET_PENDING_TOPIC_REVIEW,
)
from orchestration.state_resolver import WorkflowSnapshot
from orchestration.workflow_state import HumanGate, WorkflowState


class MCPClientLike(Protocol):
    async def list_tools(self): ...
    async def call_tool(self, name: str, arguments: dict[str, Any]): ...


def extract_mcp_structured_result(result: Any) -> dict[str, Any]:
    value = getattr(result, "structured_content", None)
    if isinstance(value, dict):
        if "result" in value and isinstance(value["result"], dict):
            return dict(value["result"])
        return dict(value)

    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if not isinstance(text, str):
            continue
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def controller_safe_tools(
    snapshot: WorkflowSnapshot,
    *,
    discovered_tools: set[str] | None = None,
) -> tuple[str, ...]:
    """Return state-valid, MCP-discovered, controller-callable tools.

    Phase 9 extends the same hard safety boundary across Agent 1 *and* Agent 2.
    Human-only writes are removed before any planner sees candidates.
    This includes Agent 1 review/approval tools and Agent 2 quiz-review decisions.
    """

    controller_allowed = set(controller_callable_tools())
    state_allowed = set(snapshot.allowed_tools)
    safe = controller_allowed & state_allowed
    if discovered_tools is not None:
        safe &= set(discovered_tools)

    safe = {
        name
        for name in safe
        if name in ALL_TOOL_POLICIES
        and ALL_TOOL_POLICIES[name].caller_policy
        is ToolCallerPolicy.CONTROLLER_ALLOWED
    }
    return tuple(sorted(safe))


def _is_state_advancing(tool_name: str) -> bool:
    policy = ALL_TOOL_POLICIES.get(tool_name)
    return bool(policy and policy.mutates_state)


def _terminal_agent2_state(state: WorkflowState) -> bool:
    return state in {
        WorkflowState.ASSESSMENT_READY,
        WorkflowState.NO_SAFE_ASSESSMENT,
    }


@dataclass(slots=True)
class EDTechController:
    """State-aware controller over both Agent 1 and Agent 2 MCP tools.

    Phase 9 keeps workflow state deterministic, preserves all Agent 1 human
    gates, and starts Agent 2 only for an explicit official-assessment or quiz
    action. Shared UI controls are parsed into structured MCP arguments; official
    retrieval/ranking remains inside Notebook 05 and quiz generation inside Notebook 06.
    """

    state_provider: ControllerStateProvider
    planner: ControllerPlanner | None = None
    max_steps: int = 8

    def _hard_gate_decision(
        self,
        snapshot: WorkflowSnapshot,
        safe_candidates: tuple[str, ...],
    ) -> ControllerDecision | None:
        if snapshot.human_gate is HumanGate.TOPIC_MAPPING_REVIEW:
            if GET_PENDING_TOPIC_REVIEW in safe_candidates:
                return ControllerDecision(
                    action=ControllerAction.CALL_TOOL,
                    tool_name=GET_PENDING_TOPIC_REVIEW,
                    reason=(
                        "A mandatory Module 3 human review is pending. Read the "
                        "pending items for the UI, then stop for the human decision."
                    ),
                    decision_source="hard_human_gate",
                    safe_candidates=list(safe_candidates),
                )
            return ControllerDecision(
                action=ControllerAction.PAUSE_FOR_HUMAN,
                reason="A mandatory Module 3 human review is pending.",
                decision_source="hard_human_gate",
                safe_candidates=list(safe_candidates),
            )

        if snapshot.human_gate is HumanGate.TOPIC_REVIEW_INTEGRITY:
            if GET_PENDING_TOPIC_REVIEW in safe_candidates:
                return ControllerDecision(
                    action=ControllerAction.CALL_TOOL,
                    tool_name=GET_PENDING_TOPIC_REVIEW,
                    reason=(
                        "Review artifact/PostgreSQL integrity is inconsistent. Read "
                        "diagnostics only and remain fail-closed."
                    ),
                    decision_source="hard_integrity_gate",
                    safe_candidates=list(safe_candidates),
                )
            return ControllerDecision(
                action=ControllerAction.BLOCKED,
                reason="Review state is inconsistent; mutation and Agent 2 remain blocked.",
                decision_source="hard_integrity_gate",
                safe_candidates=list(safe_candidates),
            )

        if snapshot.human_gate is HumanGate.AGENT2_TOPIC_APPROVAL:
            if GET_DETECTED_TOPICS in safe_candidates:
                return ControllerDecision(
                    action=ControllerAction.CALL_TOOL,
                    tool_name=GET_DETECTED_TOPICS,
                    reason=(
                        "Agent 1 topics require explicit human handoff approval before "
                        "Agent 2. Read current topics for the UI, then pause."
                    ),
                    decision_source="hard_human_gate",
                    safe_candidates=list(safe_candidates),
                )
            return ControllerDecision(
                action=ControllerAction.PAUSE_FOR_HUMAN,
                reason="Agent 1 -> Agent 2 topic approval must be made by a human.",
                decision_source="hard_human_gate",
                safe_candidates=list(safe_candidates),
            )

        return None

    def _agent2_request_decision(
        self,
        *,
        user_request: str,
        snapshot: WorkflowSnapshot,
        safe_candidates: tuple[str, ...],
    ) -> ControllerDecision | None:
        """Handle explicit Agent 2 requests deterministically before LLM planning."""

        intent = parse_assessment_request(user_request)

        # Generating/re-generating an assessment is never inferred from a generic
        # "continue" request. The user must clearly ask for an assessment.
        if intent.kind is AssessmentRequestKind.GENERATE_ASSESSMENT:
            if AGENT2_RETRIEVAL not in safe_candidates:
                return ControllerDecision(
                    action=ControllerAction.COMPLETE,
                    reason=(
                        "The user requested an assessment, but Agent 2 retrieval is "
                        f"not valid while the workflow is {snapshot.state.value}."
                    ),
                    decision_source="agent2_request_not_ready",
                    safe_candidates=list(safe_candidates),
                )
            return ControllerDecision(
                action=ControllerAction.CALL_TOOL,
                tool_name=AGENT2_RETRIEVAL,
                tool_arguments=intent.retrieval_arguments(user_request=user_request),
                reason=(
                    "The user explicitly requested an assessment and human-approved "
                    "Agent 1 topics are available; invoke existing Agent 2 retrieval."
                ),
                decision_source="deterministic_assessment_request",
                safe_candidates=list(safe_candidates),
            )

        if intent.kind is AssessmentRequestKind.GENERATE_COMPLETE_QUIZ:
            if AGENT2_COMPLETE_QUIZ not in safe_candidates:
                return ControllerDecision(
                    action=ControllerAction.COMPLETE,
                    reason=(
                        "The user requested a complete AI quiz, but that action "
                        f"is not valid while the workflow is {snapshot.state.value}."
                    ),
                    decision_source="complete_quiz_not_ready",
                    safe_candidates=list(safe_candidates),
                )
            return ControllerDecision(
                action=ControllerAction.CALL_TOOL,
                tool_name=AGENT2_COMPLETE_QUIZ,
                tool_arguments=intent.complete_quiz_arguments(user_request=user_request),
                reason=(
                    "The user explicitly requested a complete quiz; invoke Notebook 06 "
                    "complete_quiz through MCP without Notebook 05."
                ),
                decision_source="deterministic_complete_quiz_request",
                safe_candidates=list(safe_candidates),
            )

        if intent.kind is AssessmentRequestKind.GENERATE_MISSING_QUIZ:
            if AGENT2_MISSING_QUIZ not in safe_candidates:
                return ControllerDecision(
                    action=ControllerAction.COMPLETE,
                    reason=(
                        "Missing-coverage generation requires a current Notebook 05 "
                        "assessment result."
                    ),
                    decision_source="missing_quiz_not_ready",
                    safe_candidates=list(safe_candidates),
                )
            return ControllerDecision(
                action=ControllerAction.CALL_TOOL,
                tool_name=AGENT2_MISSING_QUIZ,
                tool_arguments=intent.missing_quiz_arguments(user_request=user_request),
                reason=(
                    "The user requested only the missing quiz coverage from the current "
                    "official assessment."
                ),
                decision_source="deterministic_missing_quiz_request",
                safe_candidates=list(safe_candidates),
            )

        if intent.kind is AssessmentRequestKind.SHOW_ASSESSMENT:
            if GET_AGENT2_ASSESSMENT in safe_candidates:
                return ControllerDecision(
                    action=ControllerAction.CALL_TOOL,
                    tool_name=GET_AGENT2_ASSESSMENT,
                    reason="The user asked to view the current Agent 2 assessment.",
                    decision_source="deterministic_agent2_read",
                    safe_candidates=list(safe_candidates),
                )
            return ControllerDecision(
                action=ControllerAction.COMPLETE,
                reason="No current Agent 2 assessment is available to display yet.",
                decision_source="agent2_read_not_ready",
                safe_candidates=list(safe_candidates),
            )

        if intent.kind is AssessmentRequestKind.SHOW_MARK_SCHEMES:
            if GET_AGENT2_MARK_SCHEMES in safe_candidates and intent.question_ids:
                return ControllerDecision(
                    action=ControllerAction.CALL_TOOL,
                    tool_name=GET_AGENT2_MARK_SCHEMES,
                    tool_arguments={"question_ids": intent.question_ids},
                    reason="The user requested QuestionID-linked mark schemes.",
                    decision_source="deterministic_agent2_read",
                    safe_candidates=list(safe_candidates),
                )
            if GET_AGENT2_ASSESSMENT in safe_candidates:
                return ControllerDecision(
                    action=ControllerAction.CALL_TOOL,
                    tool_name=GET_AGENT2_ASSESSMENT,
                    reason=(
                        "A mark scheme was requested without a usable QuestionID. "
                        "Read the current assessment first so the UI can expose IDs."
                    ),
                    decision_source="deterministic_agent2_read",
                    safe_candidates=list(safe_candidates),
                )
            return ControllerDecision(
                action=ControllerAction.COMPLETE,
                reason="No current assessment/QuestionID is available for mark-scheme lookup.",
                decision_source="agent2_read_not_ready",
                safe_candidates=list(safe_candidates),
            )

        if intent.kind is AssessmentRequestKind.SHOW_RENDERED_PAGES:
            if GET_AGENT2_RENDERED_PAGES in safe_candidates and intent.question_ids:
                return ControllerDecision(
                    action=ControllerAction.CALL_TOOL,
                    tool_name=GET_AGENT2_RENDERED_PAGES,
                    tool_arguments={"question_ids": intent.question_ids},
                    reason="The user requested rendered source pages for specific QuestionIDs.",
                    decision_source="deterministic_agent2_read",
                    safe_candidates=list(safe_candidates),
                )
            if GET_AGENT2_ASSESSMENT in safe_candidates:
                return ControllerDecision(
                    action=ControllerAction.CALL_TOOL,
                    tool_name=GET_AGENT2_ASSESSMENT,
                    reason=(
                        "Rendered pages were requested without a usable QuestionID. "
                        "Read the assessment first so the UI can expose IDs."
                    ),
                    decision_source="deterministic_agent2_read",
                    safe_candidates=list(safe_candidates),
                )
            return ControllerDecision(
                action=ControllerAction.COMPLETE,
                reason="No current assessment/QuestionID is available for rendered-page lookup.",
                decision_source="agent2_read_not_ready",
                safe_candidates=list(safe_candidates),
            )

        # At the Agent 1 -> Agent 2 boundary, a generic "continue" must not be
        # interpreted as permission to generate an assessment.
        if (
            AGENT2_RETRIEVAL in safe_candidates
            and snapshot.state in {
                WorkflowState.TOPICS_APPROVED,
                WorkflowState.ASSESSMENT_REQUEST_READY,
            }
        ):
            return ControllerDecision(
                action=ControllerAction.COMPLETE,
                reason=(
                    "Human-approved topics are ready. Agent 2 will start only after "
                    "the user explicitly requests an assessment/question set."
                ),
                decision_source="assessment_request_required",
                safe_candidates=list(safe_candidates),
            )

        return None

    def choose_action(
        self,
        *,
        user_request: str,
        snapshot: WorkflowSnapshot,
        discovered_tools: set[str] | None = None,
    ) -> ControllerDecision:
        safe = controller_safe_tools(snapshot, discovered_tools=discovered_tools)

        hard_gate = self._hard_gate_decision(snapshot, safe)
        if hard_gate is not None:
            return hard_gate

        # Agent 1's early stages remain deterministic and can continue even when
        # the final user goal is an assessment. Human gates still stop the chain.
        advancing = tuple(name for name in safe if _is_state_advancing(name))
        early_states = {
            WorkflowState.RAW_TRANSCRIPT_READY,
            WorkflowState.PREPROCESSING_COMPLETE,
            WorkflowState.CHUNKS_READY,
        }
        if snapshot.state in early_states and len(advancing) == 1:
            return ControllerDecision(
                action=ControllerAction.CALL_TOOL,
                tool_name=advancing[0],
                reason=(
                    f"{snapshot.state.value} has exactly one controller-authorized "
                    "state-advancing operation."
                ),
                decision_source="deterministic_state_transition",
                safe_candidates=list(safe),
            )

        # Phase 9: explicit Agent 2 intent and arguments are handled before any
        # semantic planner. This makes the core demo deterministic and auditable.
        a2_decision = self._agent2_request_decision(
            user_request=user_request,
            snapshot=snapshot,
            safe_candidates=safe,
        )
        if a2_decision is not None:
            return a2_decision

        # Defense in depth: even the constrained semantic planner never sees
        # Agent 2 retrieval unless the deterministic parser classified the user
        # request as an explicit assessment-generation request.
        parsed_kind = parse_assessment_request(user_request).kind
        if parsed_kind is not AssessmentRequestKind.GENERATE_ASSESSMENT:
            safe = tuple(name for name in safe if name != AGENT2_RETRIEVAL)
        if parsed_kind is not AssessmentRequestKind.GENERATE_COMPLETE_QUIZ:
            safe = tuple(name for name in safe if name != AGENT2_COMPLETE_QUIZ)
        if parsed_kind is not AssessmentRequestKind.GENERATE_MISSING_QUIZ:
            safe = tuple(name for name in safe if name != AGENT2_MISSING_QUIZ)

        if not safe:
            action = (
                ControllerAction.BLOCKED
                if snapshot.state in {
                    WorkflowState.NO_RUN,
                    WorkflowState.INVALID_STATE,
                    WorkflowState.TOOL_FAILED,
                    WorkflowState.REVIEW_STATE_INCONSISTENT,
                }
                else ControllerAction.COMPLETE
            )
            return ControllerDecision(
                action=action,
                reason=(
                    "No autonomous MCP tool is valid for the current state. "
                    "A human action or no further action is required."
                ),
                decision_source="deterministic_no_safe_tool",
                safe_candidates=[],
            )

        # Remaining read-only semantic choices can use the constrained planner.
        planner = self.planner
        if planner is None:
            try:
                planner = GroqControllerPlanner()
            except Exception as exc:
                return ControllerDecision(
                    action=ControllerAction.COMPLETE,
                    reason=(
                        "Multiple read-only controller actions are available, but the "
                        f"semantic planner is unavailable: {type(exc).__name__}: {exc}"
                    ),
                    decision_source="planner_unavailable",
                    safe_candidates=list(safe),
                )

        decision = planner.decide(
            user_request=user_request,
            snapshot=snapshot,
            safe_candidates=safe,
        )

        if decision.action is ControllerAction.CALL_TOOL:
            if decision.tool_name not in safe:
                return ControllerDecision(
                    action=ControllerAction.BLOCKED,
                    reason="Planner selected a tool outside the safe candidate set.",
                    decision_source="post_planner_guardrail",
                    safe_candidates=list(safe),
                )
            policy = ALL_TOOL_POLICIES.get(str(decision.tool_name))
            if not policy or policy.caller_policy is not ToolCallerPolicy.CONTROLLER_ALLOWED:
                return ControllerDecision(
                    action=ControllerAction.BLOCKED,
                    reason="Planner selected a human-only or unknown tool; call blocked.",
                    decision_source="post_planner_guardrail",
                    safe_candidates=list(safe),
                )
        return decision

    async def step(
        self,
        *,
        client: MCPClientLike,
        run_id: str,
        user_request: str,
    ) -> ControllerStepResult:
        before = self.state_provider.get_snapshot(run_id)

        listed = await client.list_tools()
        discovered = {
            str(tool.name)
            for tool in getattr(listed, "tools", []) or []
            if getattr(tool, "name", None)
        }
        decision = self.choose_action(
            user_request=user_request,
            snapshot=before,
            discovered_tools=discovered,
        )

        if decision.action is not ControllerAction.CALL_TOOL:
            return ControllerStepResult(
                run_id=run_id,
                user_request=user_request,
                state_before=before.state,
                state_after=before.state,
                human_gate_before=before.human_gate,
                human_gate_after=before.human_gate,
                decision=decision,
                should_stop=True,
                stop_reason=decision.reason,
            )

        tool_name = str(decision.tool_name)
        arguments = {"run_id": run_id, **dict(decision.tool_arguments)}
        # Run identity always comes from controller context, never LLM/parser data.
        arguments["run_id"] = run_id
        result = await client.call_tool(tool_name, arguments)
        payload = extract_mcp_structured_result(result)
        after = self.state_provider.get_snapshot(run_id)
        tool_success = bool(payload.get("success")) and not bool(
            getattr(result, "is_error", False)
        )

        state_changed = after.state is not before.state
        terminal_agent2 = _terminal_agent2_state(after.state) and tool_name == AGENT2_RETRIEVAL
        should_stop = (
            not tool_success
            or after.human_action_required
            or terminal_agent2
            or not _is_state_advancing(tool_name)
            or not state_changed
        )
        if not tool_success:
            stop_reason = "MCP tool returned failure/error."
        elif after.human_action_required:
            stop_reason = f"Controller reached mandatory human gate {after.human_gate.value}."
        elif terminal_agent2 and after.state is WorkflowState.ASSESSMENT_READY:
            stop_reason = "Agent 2 completed and the assessment is ready for Streamlit to display."
        elif terminal_agent2 and after.state is WorkflowState.NO_SAFE_ASSESSMENT:
            stop_reason = "Agent 2 completed safely but found no compatible assessment."
        elif not _is_state_advancing(tool_name):
            stop_reason = "Read-only controller action completed; waiting for the next user/UI action."
        elif not state_changed:
            stop_reason = "Tool completed without advancing workflow state; stopping to avoid a loop."
        else:
            stop_reason = ""

        return ControllerStepResult(
            run_id=run_id,
            user_request=user_request,
            state_before=before.state,
            state_after=after.state,
            human_gate_before=before.human_gate,
            human_gate_after=after.human_gate,
            decision=decision,
            tool_called=tool_name,
            tool_success=tool_success,
            tool_payload=payload,
            should_stop=should_stop,
            stop_reason=stop_reason,
        )

    async def run_until_pause(
        self,
        *,
        client: MCPClientLike,
        run_id: str,
        user_request: str,
        max_steps: int | None = None,
    ) -> ControllerRunResult:
        limit = int(max_steps or self.max_steps)
        steps: list[ControllerStepResult] = []
        stop_reason = ""

        for _ in range(max(1, limit)):
            step = await self.step(
                client=client,
                run_id=run_id,
                user_request=user_request,
            )
            steps.append(step)
            if step.should_stop:
                stop_reason = step.stop_reason or step.decision.reason
                break
        else:
            stop_reason = f"Controller reached max_steps={limit}."

        final = self.state_provider.get_snapshot(run_id)
        return ControllerRunResult(
            run_id=run_id,
            user_request=user_request,
            steps=steps,
            final_state=final.state,
            final_human_gate=final.human_gate,
            human_action_required=final.human_action_required,
            completed=(
                not final.human_action_required
                and final.state
                not in {
                    WorkflowState.TOOL_FAILED,
                    WorkflowState.INVALID_STATE,
                    WorkflowState.REVIEW_STATE_INCONSISTENT,
                }
            ),
            stop_reason=stop_reason,
        )
