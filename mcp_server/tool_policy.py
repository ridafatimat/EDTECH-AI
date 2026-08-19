from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from orchestration.guardrails import (
    AGENT1_CHUNK,
    AGENT1_MAP,
    AGENT1_PREPROCESS,
    AGENT2_RETRIEVAL,
    AGENT2_COMPLETE_QUIZ,
    AGENT2_MISSING_QUIZ,
    SUBMIT_AGENT2_QUIZ_REVIEW,
    GET_AGENT2_ASSESSMENT,
    GET_AGENT2_MARK_SCHEMES,
    GET_AGENT2_RENDERED_PAGES,
    RENDER_LOGIC_VISUAL,
    RENDER_TECHNICAL_VISUAL,
    RENDER_STRUCTURED_VISUAL,
    GET_APPROVED_TOPICS,
    GET_DETECTED_TOPICS,
    GET_PENDING_TOPIC_REVIEW,
    SAVE_AGENT2_TOPIC_APPROVAL,
    SUBMIT_DETECTED_TOPIC_EDIT,
    SUBMIT_TOPIC_REVIEW,
)


class ToolCallerPolicy(str, Enum):
    CONTROLLER_ALLOWED = "controller_allowed"
    HUMAN_UI_ONLY = "human_ui_only"


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    name: str
    caller_policy: ToolCallerPolicy
    mutates_state: bool
    description: str


AGENT1_TOOL_POLICIES: dict[str, ToolPolicy] = {
    AGENT1_PREPROCESS: ToolPolicy(
        AGENT1_PREPROCESS, ToolCallerPolicy.CONTROLLER_ALLOWED, True,
        "Run existing Agent 1 Module 1 preprocessing.",
    ),
    AGENT1_CHUNK: ToolPolicy(
        AGENT1_CHUNK, ToolCallerPolicy.CONTROLLER_ALLOWED, True,
        "Run existing Agent 1 Module 2 semantic chunking.",
    ),
    AGENT1_MAP: ToolPolicy(
        AGENT1_MAP, ToolCallerPolicy.CONTROLLER_ALLOWED, True,
        "Run existing Agent 1 Module 3 topic mapping.",
    ),
    GET_DETECTED_TOPICS: ToolPolicy(
        GET_DETECTED_TOPICS, ToolCallerPolicy.CONTROLLER_ALLOWED, False,
        "Read current effective Agent 1 topics.",
    ),
    GET_PENDING_TOPIC_REVIEW: ToolPolicy(
        GET_PENDING_TOPIC_REVIEW, ToolCallerPolicy.CONTROLLER_ALLOWED, False,
        "Read DB-reconciled pending Module 3 reviews.",
    ),
    SUBMIT_TOPIC_REVIEW: ToolPolicy(
        SUBMIT_TOPIC_REVIEW, ToolCallerPolicy.HUMAN_UI_ONLY, True,
        "Persist an Agent 1 review decision already made by a human.",
    ),
    SUBMIT_DETECTED_TOPIC_EDIT: ToolPolicy(
        SUBMIT_DETECTED_TOPIC_EDIT, ToolCallerPolicy.HUMAN_UI_ONLY, True,
        "Persist a human-authorized Agent 1 detected-topic edit.",
    ),
    SAVE_AGENT2_TOPIC_APPROVAL: ToolPolicy(
        SAVE_AGENT2_TOPIC_APPROVAL, ToolCallerPolicy.HUMAN_UI_ONLY, True,
        "Persist the human Agent 1 -> Agent 2 topic handoff.",
    ),
    GET_APPROVED_TOPICS: ToolPolicy(
        GET_APPROVED_TOPICS, ToolCallerPolicy.CONTROLLER_ALLOWED, False,
        "Read human-approved Agent 1 topics for Agent 2.",
    ),
}


# Agent 2 exposes official retrieval plus quiz generation. Generated-quiz
# approve/regenerate/reject remains HUMAN_UI_ONLY; LangGraph cannot originate it.
AGENT2_TOOL_POLICIES: dict[str, ToolPolicy] = {
    AGENT2_RETRIEVAL: ToolPolicy(
        AGENT2_RETRIEVAL, ToolCallerPolicy.CONTROLLER_ALLOWED, True,
        "Invoke the existing Agent 2 runner and Notebook 05 retrieval pipeline.",
    ),
    AGENT2_COMPLETE_QUIZ: ToolPolicy(
        AGENT2_COMPLETE_QUIZ, ToolCallerPolicy.CONTROLLER_ALLOWED, True,
        "Run Notebook 06 in complete_quiz mode; GPT-OSS generates the entire quiz without Notebook 05 retrieval.",
    ),
    AGENT2_MISSING_QUIZ: ToolPolicy(
        AGENT2_MISSING_QUIZ, ToolCallerPolicy.CONTROLLER_ALLOWED, True,
        "Run Notebook 06 in fill_shortfall mode using the exact current Notebook 05 package and matching selected CSV.",
    ),
    SUBMIT_AGENT2_QUIZ_REVIEW: ToolPolicy(
        SUBMIT_AGENT2_QUIZ_REVIEW, ToolCallerPolicy.HUMAN_UI_ONLY, True,
        "Apply a human approve/regenerate/reject decision to the current generated quiz candidate.",
    ),
    GET_AGENT2_ASSESSMENT: ToolPolicy(
        GET_AGENT2_ASSESSMENT, ToolCallerPolicy.CONTROLLER_ALLOWED, False,
        "Read the current Agent 2 assessment package.",
    ),
    GET_AGENT2_MARK_SCHEMES: ToolPolicy(
        GET_AGENT2_MARK_SCHEMES, ToolCallerPolicy.CONTROLLER_ALLOWED, False,
        "Read mark schemes already linked by QuestionID in the current package.",
    ),
    GET_AGENT2_RENDERED_PAGES: ToolPolicy(
        GET_AGENT2_RENDERED_PAGES, ToolCallerPolicy.CONTROLLER_ALLOWED, False,
        "Read page images already produced by existing Agent 2/Notebook 07.",
    ),
}


# Visual tools are controller-callable at the MCP policy layer, but intentionally
# absent from orchestration.guardrails._ALLOWED. LangGraph grants them only
# after a successful Notebook 06 action and a concrete visual handoff plan.
VISUAL_TOOL_POLICIES: dict[str, ToolPolicy] = {
    RENDER_LOGIC_VISUAL: ToolPolicy(
        RENDER_LOGIC_VISUAL, ToolCallerPolicy.CONTROLLER_ALLOWED, True,
        "Render logic visuals from Notebook 06 through Notebook 08/SchemDraw.",
    ),
    RENDER_TECHNICAL_VISUAL: ToolPolicy(
        RENDER_TECHNICAL_VISUAL, ToolCallerPolicy.CONTROLLER_ALLOWED, True,
        "Render technical visuals from Notebook 06 through Notebook 08/Kroki.",
    ),
    RENDER_STRUCTURED_VISUAL: ToolPolicy(
        RENDER_STRUCTURED_VISUAL, ToolCallerPolicy.CONTROLLER_ALLOWED, True,
        "Render structured visuals from Notebook 06 through Notebook 08/local rendering.",
    ),
}

ALL_TOOL_POLICIES: dict[str, ToolPolicy] = {
    **AGENT1_TOOL_POLICIES,
    **AGENT2_TOOL_POLICIES,
    **VISUAL_TOOL_POLICIES,
}


def _by_policy(
    policies: dict[str, ToolPolicy], caller: ToolCallerPolicy
) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, policy in policies.items()
            if policy.caller_policy is caller
        )
    )


def controller_callable_agent1_tools() -> tuple[str, ...]:
    return _by_policy(AGENT1_TOOL_POLICIES, ToolCallerPolicy.CONTROLLER_ALLOWED)


def human_ui_only_agent1_tools() -> tuple[str, ...]:
    return _by_policy(AGENT1_TOOL_POLICIES, ToolCallerPolicy.HUMAN_UI_ONLY)


def controller_callable_agent2_tools() -> tuple[str, ...]:
    return _by_policy(AGENT2_TOOL_POLICIES, ToolCallerPolicy.CONTROLLER_ALLOWED)


def human_ui_only_agent2_tools() -> tuple[str, ...]:
    return _by_policy(AGENT2_TOOL_POLICIES, ToolCallerPolicy.HUMAN_UI_ONLY)


def controller_callable_tools() -> tuple[str, ...]:
    return _by_policy(ALL_TOOL_POLICIES, ToolCallerPolicy.CONTROLLER_ALLOWED)


def human_ui_only_tools() -> tuple[str, ...]:
    return _by_policy(ALL_TOOL_POLICIES, ToolCallerPolicy.HUMAN_UI_ONLY)


def controller_callable_visual_tools() -> tuple[str, ...]:
    return _by_policy(VISUAL_TOOL_POLICIES, ToolCallerPolicy.CONTROLLER_ALLOWED)
