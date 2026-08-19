from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from mcp_server.adapters.agent1_notebook_adapter import Agent1NotebookAdapter
from mcp_server.adapters.agent2_notebook_adapter import Agent2NotebookAdapter
from mcp_server.adapters.visual_renderer_adapter import VisualRendererAdapter
from mcp_server.schemas.agent1 import (
    Agent2TopicApprovalRequest,
    ApprovedTopicSelection,
    DetectedTopicEditAction,
    DetectedTopicEditRequest,
    TopicReviewDecision,
    TopicReviewRequest,
)
from mcp_server.schemas.agent2 import (
    GenerateAgent2CompleteQuizRequest,
    GenerateAgent2MissingQuizRequest,
    GetAgent2AssessmentRequest,
    GetAgent2MarkSchemesRequest,
    GetAgent2RenderedPagesRequest,
    RunAgent2RetrievalRequest,
    SubmitAgent2QuizReviewRequest,
)
from mcp_server.schemas.common import RunRequest, ToolResult
from mcp_server.schemas.visuals import RenderVisualsRequest
from mcp_server.tools.agent1_tools import Agent1ToolService
from mcp_server.tools.agent2_tools import Agent2ToolService
from mcp_server.tools.visual_tools import VisualToolService

logger = logging.getLogger(__name__)

SERVER_NAME = "EDTech-Multi-Agent-MCP"

SERVER_INSTRUCTIONS = """
EDTech multi-agent MCP server.

Agent 1 and Agent 2 keep their existing notebook/business logic. MCP exposes
high-level operations only. PostgreSQL, Qdrant, retrieval/ranking and QuestionID
mark-scheme linking remain inside the existing agents. Notebook 08 visual rendering
is exposed through three semantic MCP tools without moving renderer logic into MCP.

Agent 1 HITL remains mandatory:
- submit_topic_review, submit_detected_topic_edit and save_agent2_topic_approval
  represent decisions already made by a human in Streamlit.
- An autonomous controller must never originate those human decisions.

Agent 2 quiz-generation note:
- Notebook 05 remains the official AQA retrieval path.
- Notebook 06 has two explicit modes: complete_quiz (direct GPT-OSS full quiz)
  and fill_shortfall (only missing coverage after Notebook 05).
- Generated-question approve/regenerate/reject is HUMAN-UI-ONLY and must never
  be originated autonomously by LangGraph.

Visual routing note:
- LangGraph reads Notebook 06 visual_tool_handoff.json after successful quiz generation.
- It may call only render_logic_visual, render_technical_visual, or render_structured_visual.
- Notebook 08 remains the renderer implementation (SchemDraw / Kroki / local structured).
""".strip()

def _project_root() -> Path:
    # EDTECH/mcp_server/server.py -> EDTECH
    return Path(__file__).resolve().parents[1]


def resolve_agent1_frontend_root(explicit: Path | str | None = None) -> Path:
    """Resolve the existing Agent 1 Streamlit project without moving it.

    Precedence:
      1. explicit function argument
      2. EDTECH_AGENT1_FRONTEND_ROOT environment variable
      3. <EDTECH>/Agent_1/Agent1_Streamlit_Frontend
    """

    if explicit is not None:
        root = Path(explicit)
    else:
        env_value = os.getenv("EDTECH_AGENT1_FRONTEND_ROOT", "").strip()
        root = (
            Path(env_value)
            if env_value
            else _project_root() / "Agent_1" / "Agent1_Streamlit_Frontend"
        )
    return root.expanduser().resolve()


def build_agent1_tool_service(
    frontend_root: Path | str | None = None,
) -> Agent1ToolService:
    """Build the already-tested Agent 1 tool service over existing runners."""

    resolved = resolve_agent1_frontend_root(frontend_root)
    if not resolved.is_dir():
        raise FileNotFoundError(
            "Agent 1 frontend project root was not found. Expected: "
            f"{resolved}. Set EDTECH_AGENT1_FRONTEND_ROOT if your path differs."
        )
    executor = Agent1NotebookAdapter(resolved)
    return Agent1ToolService(executor)


def build_agent2_tool_service(
    frontend_root: Path | str | None = None,
) -> Agent2ToolService:
    """Build Phase 8 over the existing frontend/agent2_runner.py path."""

    resolved = resolve_agent1_frontend_root(frontend_root)
    if not resolved.is_dir():
        raise FileNotFoundError(
            "Agent 1 frontend project root was not found. Expected: "
            f"{resolved}. Set EDTECH_AGENT1_FRONTEND_ROOT if your path differs."
        )
    executor = Agent2NotebookAdapter(resolved)
    return Agent2ToolService(executor)



def build_visual_tool_service(
    frontend_root: Path | str | None = None,
) -> VisualToolService:
    """Build the Notebook 08 visual tool service."""

    resolved = resolve_agent1_frontend_root(frontend_root)
    if not resolved.is_dir():
        raise FileNotFoundError(
            "Agent 1 frontend project root was not found. Expected: "
            f"{resolved}. Set EDTECH_AGENT1_FRONTEND_ROOT if your path differs."
        )
    return VisualToolService(VisualRendererAdapter(resolved))


def create_mcp_server(
    *,
    service: Agent1ToolService | Any | None = None,
    agent2_service: Agent2ToolService | Any | None = None,
    visual_service: VisualToolService | Any | None = None,
    frontend_root: Path | str | None = None,
) -> MCPServer:
    """Create the MCP server over existing Agent 1 + Agent 2 boundaries.

    Backward-compatible test behavior: when an injected Agent 1 ``service`` is
    supplied without ``agent2_service``, only Agent 1 tools are registered.
    Production usage supplies neither, so both real services are registered.
    """

    svc = service if service is not None else build_agent1_tool_service(frontend_root)
    a2svc = agent2_service
    if a2svc is None and service is None:
        a2svc = build_agent2_tool_service(frontend_root)

    vsvc = visual_service
    if vsvc is None and service is None:
        vsvc = build_visual_tool_service(frontend_root)

    mcp = MCPServer(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)

    # ------------------------------------------------------------------
    # Controller-callable pipeline stages
    # ------------------------------------------------------------------
    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
    def run_agent1_preprocessing(run_id: str) -> ToolResult:
        """Run existing Agent 1 Module 1 preprocessing for the current run.

        Call only when deterministic workflow state says RAW_TRANSCRIPT_READY.
        No preprocessing algorithm is implemented inside MCP.
        """

        return svc.run_agent1_preprocessing(RunRequest(run_id=run_id))

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
    def run_agent1_chunking(run_id: str) -> ToolResult:
        """Run existing Agent 1 Module 2 semantic chunking for the current run.

        Call only when deterministic workflow state says PREPROCESSING_COMPLETE.
        """

        return svc.run_agent1_chunking(RunRequest(run_id=run_id))

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
    def run_agent1_topic_mapping(run_id: str) -> ToolResult:
        """Run existing Agent 1 Module 3 topic mapping for the current run.

        The result may enter a mandatory human-review state. Never continue to
        Agent 2 while that gate is unresolved.
        """

        return svc.run_agent1_topic_mapping(RunRequest(run_id=run_id))

    # ------------------------------------------------------------------
    # Read-only HITL / handoff views
    # ------------------------------------------------------------------
    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def get_detected_topics(run_id: str) -> ToolResult:
        """Read current effective official topics for an Agent 1 run.

        This is read-only and includes human-reviewed edit-memory effects that
        Agent 1 has already authorized for the current evidence.
        """

        return svc.get_detected_topics(RunRequest(run_id=run_id))

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def get_pending_topic_review(run_id: str) -> ToolResult:
        """Read live PostgreSQL-reconciled pending Module 3 review items.

        This tool does not make the review decision. If reviews are pending,
        return them to the human-facing UI and pause autonomous progression.
        """

        return svc.get_pending_topic_review(RunRequest(run_id=run_id))

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def get_approved_topics(run_id: str) -> ToolResult:
        """Read topics explicitly approved by a human for Agent 2 handoff."""

        return svc.get_approved_topics(RunRequest(run_id=run_id))

    # ------------------------------------------------------------------
    # HUMAN-UI-ONLY writes.
    # These tools accept a decision already made by a human. They are exposed
    # through MCP as the standardized execution interface, but Phase 7 must
    # exclude them from the autonomous controller's candidate tool set.
    # ------------------------------------------------------------------
    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
    def submit_topic_review(
        run_id: str,
        decisions: list[TopicReviewDecision],
        reviewed_by: str = "mcp_human_review",
    ) -> ToolResult:
        """Persist Module 3 review decisions ALREADY made by a human.

        HUMAN-UI-ONLY. Do not autonomously choose Approve/Correct/Reject.
        Corrections continue to require the existing human reason and are
        written through Agent 1's existing PostgreSQL decision/memory path.
        """

        request = TopicReviewRequest(
            run_id=run_id,
            decisions=decisions,
            reviewed_by=reviewed_by,
        )
        return svc.submit_topic_review(request)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
    def submit_detected_topic_edit(
        run_id: str,
        action: DetectedTopicEditAction,
        reason: str,
        topic_index: int | None = None,
        source_concept_id: str | None = None,
        target_concept_id: str | None = None,
        target_role: Literal["primary", "supporting"] | None = None,
        source_chunk_ids: list[int] | None = None,
        reviewed_by: str = "mcp_human_review",
    ) -> ToolResult:
        """Persist a detected-topic edit ALREADY authorized by a human.

        HUMAN-UI-ONLY. Supports add/remove/replace/change-role and always keeps
        the mandatory reviewer reason. The call goes through Agent 1's existing
        contextual edit-memory path; MCP does not implement learning itself.
        """

        request = DetectedTopicEditRequest(
            run_id=run_id,
            action=action,
            topic_index=topic_index,
            source_concept_id=source_concept_id,
            target_concept_id=target_concept_id,
            target_role=target_role,
            source_chunk_ids=source_chunk_ids or [],
            reason=reason,
            reviewed_by=reviewed_by,
        )
        return svc.submit_detected_topic_edit(request)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
    def save_agent2_topic_approval(
        run_id: str,
        selections: list[ApprovedTopicSelection],
        reviewed_by: str = "mcp_human_review",
    ) -> ToolResult:
        """Persist Agent 1 -> Agent 2 topic selections made by a human.

        HUMAN-UI-ONLY. The controller must not approve topics itself. Until this
        human handoff exists, Agent 2 remains blocked by deterministic state
        guardrails.
        """

        request = Agent2TopicApprovalRequest(
            run_id=run_id,
            selections=selections,
            reviewed_by=reviewed_by,
        )
        return svc.save_agent2_topic_approval(request)

    # ------------------------------------------------------------------
    # Phase 8: verified existing Agent 2 capabilities only.
    # ------------------------------------------------------------------
    if a2svc is not None:
        @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
        def run_agent2_retrieval(
            run_id: str,
            paper: Literal["Paper 1", "Paper 2", "Any"] = "Any",
            number_of_questions: int = 5,
            target_total_marks: int = 20,
            minimum_question_marks: int = 1,
            maximum_question_marks: int = 12,
            minimum_primary_questions: int = 1,
            minimum_supporting_questions: int = 0,
            cover_all_approved_topics: bool = True,
            include_code_questions: bool = True,
            include_visual_questions: bool = True,
            programming_language: Literal["Automatic", "Python"] = "Automatic",
            user_request: str | None = None,
        ) -> ToolResult:
            """Run existing Agent 2 Notebook 05 using human-approved Agent 1 topics.

            MCP only prepares the normal assessment_request.json and invokes the
            existing runner. Retrieval/ranking logic remains unchanged.
            """

            request = RunAgent2RetrievalRequest(
                run_id=run_id,
                paper=paper,
                number_of_questions=number_of_questions,
                target_total_marks=target_total_marks,
                minimum_question_marks=minimum_question_marks,
                maximum_question_marks=maximum_question_marks,
                minimum_primary_questions=minimum_primary_questions,
                minimum_supporting_questions=minimum_supporting_questions,
                cover_all_approved_topics=cover_all_approved_topics,
                include_code_questions=include_code_questions,
                include_visual_questions=include_visual_questions,
                programming_language=programming_language,
                user_request=user_request,
            )
            return a2svc.run_agent2_retrieval(request)

        @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
        def generate_agent2_complete_quiz(
            run_id: str,
            paper: Literal["Paper 1", "Paper 2", "Any"] = "Any",
            number_of_questions: int = 5,
            target_total_marks: int = 20,
            minimum_question_marks: int = 1,
            maximum_question_marks: int = 12,
            minimum_primary_questions: int = 1,
            minimum_supporting_questions: int = 0,
            cover_all_approved_topics: bool = True,
            include_code_questions: bool = True,
            include_visual_questions: bool = True,
            programming_language: Literal["Automatic", "Python"] = "Automatic",
            user_request: str | None = None,
        ) -> ToolResult:
            """Generate the ENTIRE quiz with Notebook 06 + GPT-OSS.

            This tool does not invoke Notebook 05 or retrieve past-paper questions.
            """

            request = GenerateAgent2CompleteQuizRequest(
                run_id=run_id,
                paper=paper,
                number_of_questions=number_of_questions,
                target_total_marks=target_total_marks,
                minimum_question_marks=minimum_question_marks,
                maximum_question_marks=maximum_question_marks,
                minimum_primary_questions=minimum_primary_questions,
                minimum_supporting_questions=minimum_supporting_questions,
                cover_all_approved_topics=cover_all_approved_topics,
                include_code_questions=include_code_questions,
                include_visual_questions=include_visual_questions,
                programming_language=programming_language,
                user_request=user_request,
            )
            return a2svc.generate_agent2_complete_quiz(request)

        @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
        def generate_agent2_missing_quiz_coverage(
            run_id: str,
            user_request: str | None = None,
        ) -> ToolResult:
            """Generate only the missing quiz coverage from the current Notebook 05 run.

            The adapter reuses the exact current assessment_request, package and
            matching selected-question CSV; weak retrieval is never loosened.
            """

            return a2svc.generate_agent2_missing_quiz_coverage(
                GenerateAgent2MissingQuizRequest(
                    run_id=run_id,
                    user_request=user_request,
                )
            )

        @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
        def submit_agent2_quiz_review(
            run_id: str,
            quiz_mode: Literal["complete_quiz", "fill_shortfall"],
            decision: Literal["approve", "regenerate", "reject"],
            reason: str,
            reviewed_by: str = "streamlit",
        ) -> ToolResult:
            """Persist/apply a generated-quiz review decision already made by a human.

            HUMAN-UI-ONLY. LangGraph autonomous nodes must never call this tool.
            """

            return a2svc.submit_agent2_quiz_review(
                SubmitAgent2QuizReviewRequest(
                    run_id=run_id,
                    quiz_mode=quiz_mode,
                    decision=decision,
                    reason=reason,
                    reviewed_by=reviewed_by,
                )
            )

        @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
        def get_agent2_assessment(run_id: str) -> ToolResult:
            """Read the current Agent 2 assessment package."""

            return a2svc.get_agent2_assessment(
                GetAgent2AssessmentRequest(run_id=run_id)
            )

        @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
        def get_agent2_mark_schemes(
            run_id: str, question_ids: list[str]
        ) -> ToolResult:
            """Read QuestionID-linked mark schemes already present in the package."""

            return a2svc.get_agent2_mark_schemes(
                GetAgent2MarkSchemesRequest(
                    run_id=run_id, question_ids=question_ids
                )
            )

        @mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
        def get_agent2_rendered_pages(
            run_id: str, question_ids: list[str]
        ) -> ToolResult:
            """Read page images already rendered by existing Agent 2/Notebook 07."""

            return a2svc.get_agent2_rendered_pages(
                GetAgent2RenderedPagesRequest(
                    run_id=run_id, question_ids=question_ids
                )
            )


    # ------------------------------------------------------------------
    # MCP-routed Notebook 08 visual tools.
    # ------------------------------------------------------------------
    if vsvc is not None:

        @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
        def render_logic_visual(
            run_id: str,
            quiz_mode: Literal["complete_quiz", "fill_shortfall"],
        ) -> ToolResult:
            """Render current logic-gate visuals through Notebook 08/SchemDraw."""

            return vsvc.render_logic_visual(
                RenderVisualsRequest(run_id=run_id, quiz_mode=quiz_mode)
            )

        @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
        def render_technical_visual(
            run_id: str,
            quiz_mode: Literal["complete_quiz", "fill_shortfall"],
        ) -> ToolResult:
            """Render current network/flowchart/CPU visuals through Notebook 08/Kroki."""

            return vsvc.render_technical_visual(
                RenderVisualsRequest(run_id=run_id, quiz_mode=quiz_mode)
            )

        @mcp.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
        def render_structured_visual(
            run_id: str,
            quiz_mode: Literal["complete_quiz", "fill_shortfall"],
        ) -> ToolResult:
            """Render current code/table/grid visuals through Notebook 08/local rendering."""

            return vsvc.render_structured_visual(
                RenderVisualsRequest(run_id=run_id, quiz_mode=quiz_mode)
            )

    return mcp


def main() -> None:
    logging.basicConfig(level=os.getenv("EDTECH_MCP_LOG_LEVEL", "INFO"))
    mcp = create_mcp_server()
    logger.info("Starting %s over stdio", SERVER_NAME)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
