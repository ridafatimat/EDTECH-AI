from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import statistics
import time
import sys
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from streamlit_integration.langgraph_bridge import (
    build_assessment_request_text,
    build_complete_quiz_request_text,
    build_missing_quiz_request_text,
    create_langgraph_run,
    langgraph_snapshot,
    run_langgraph_request,
    submit_human_agent2_quiz_review,
    submit_human_agent2_topic_approval,
    submit_human_detected_topic_edit,
    submit_human_topic_review,
)


# ============================================================
# Project paths
# ============================================================

EDTECH_ROOT = Path(__file__).resolve().parents[1]

# During frontend migration the existing runtime stays where it is.
# Streamlit UI can be removed only after the new frontend is fully verified.
BACKEND_RUNTIME_ROOT = (
    EDTECH_ROOT
    / "Agent_1"
    / "Agent1_Streamlit_Frontend"
).resolve()

RUNS_ROOT = (BACKEND_RUNTIME_ROOT / "runs").resolve()
AGENT1_CODE_ROOT = (EDTECH_ROOT / "Agent_1").resolve()
AGENT2_ROOT = (EDTECH_ROOT / "Agent2").resolve()

load_dotenv(EDTECH_ROOT / ".env", override=False)
load_dotenv(AGENT1_CODE_ROOT / ".env", override=False)
load_dotenv(BACKEND_RUNTIME_ROOT / ".env", override=False)
load_dotenv(AGENT2_ROOT / ".env", override=False)

# Notebook 06 question-level HITL uses AGENT2_DATABASE_URL. Keep the
# existing shared EDTech PostgreSQL connection as the default, exactly as the
# current Streamlit integration does.
if not str(os.getenv("AGENT2_DATABASE_URL", "") or "").strip():
    shared_database_url = str(os.getenv("DATABASE_URL", "") or "").strip()
    if shared_database_url:
        os.environ["AGENT2_DATABASE_URL"] = shared_database_url


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="EDTech Backend API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request schemas
# ============================================================

class MappingReviewBody(BaseModel):
    action: Literal["approve", "reject", "correct"]
    corrected_decision: Literal["mapped", "out_of_syllabus"] | None = None
    corrected_mapped_concept_id: str | None = None
    reason: str = ""
    review_notes: str = ""


class DetectedTopicEditBody(BaseModel):
    action: Literal[
        "change_role",
        "replace_topic",
        "remove_topic",
        "add_topic",
    ]
    reason: str
    topic_index: int | None = None
    source_concept_id: str | None = None
    target_concept_id: str | None = None
    target_role: Literal["primary", "supporting"] | None = None
    source_chunk_ids: list[int] = Field(default_factory=list)


class TopicApprovalBody(BaseModel):
    topic_indexes: list[int] = Field(default_factory=list)


class HistoricalMemoryReviewBody(BaseModel):
    decision: Literal["use_historical", "keep_fresh"]
    memory_ids: list[int] = Field(default_factory=list)
    selected_memory_id: int | None = None
    reason: str


class AssessmentStartBody(BaseModel):
    mode: Literal["retrieve_hybrid", "complete_quiz"]
    paper: Literal["Any", "Paper 1", "Paper 2"] = "Any"
    number_of_questions: int = Field(default=5, ge=1, le=30)
    target_total_marks: int = Field(default=20, ge=1, le=200)
    minimum_question_marks: int = Field(default=1, ge=1, le=50)
    maximum_question_marks: int = Field(default=12, ge=1, le=50)
    minimum_primary_questions: int = Field(default=1, ge=0, le=30)
    minimum_supporting_questions: int = Field(default=0, ge=0, le=30)
    cover_all_approved_topics: bool = True
    include_code_questions: bool = True
    include_visual_questions: bool = True
    programming_language: Literal["Automatic", "Python"] = "Automatic"
    model_key: str
    quiz_plan: Literal["plan_a", "plan_b", "plan_c"] = "plan_c"
    special_instructions: str = ""


class QuizReviewBody(BaseModel):
    quiz_mode: Literal["complete_quiz", "fill_shortfall"]
    decision: Literal["approve", "regenerate", "reject"]
    reason: str


class QuestionMarkingGuidanceBody(BaseModel):
    marks: int = Field(default=1, ge=1, le=50)
    criterion: str


class QuestionReviewBody(BaseModel):
    quiz_mode: Literal["complete_quiz", "fill_shortfall"]
    question_id: str
    plan_index: int = Field(ge=1)
    action: Literal[
        "approve",
        "edit_question",
        "edit_marking_guidance",
        "regenerate",
        "reject",
    ]
    reason: str = ""
    question_text: str | None = None
    marking_guidance: list[QuestionMarkingGuidanceBody] = Field(default_factory=list)


class RetrievalFeedbackBody(BaseModel):
    decision: Literal["relevant", "not_relevant"]
    reason: str = ""


# ============================================================
# Generic helpers
# ============================================================

def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _resolve_run_dir(run_id: str) -> Path:
    clean_run_id = str(run_id or "").strip()

    if (
        not clean_run_id
        or clean_run_id in {".", ".."}
        or "/" in clean_run_id
        or "\\" in clean_run_id
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid run_id.",
        )

    run_dir = (RUNS_ROOT / clean_run_id).resolve()

    try:
        run_dir.relative_to(RUNS_ROOT)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid run_id.",
        ) from exc

    if not run_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Run not found: {clean_run_id}",
        )

    return run_dir


def _resolve_transcript_name(run_dir: Path) -> str:
    manifest = _load_json(run_dir / "pipeline_manifest.json")

    transcript_name = str(
        manifest.get("transcript_name") or ""
    ).strip()

    if transcript_name:
        return transcript_name

    input_dir = run_dir / "input"

    if input_dir.is_dir():
        input_files = [
            path
            for path in input_dir.iterdir()
            if path.is_file()
        ]

        if len(input_files) == 1:
            return input_files[0].stem

    raise HTTPException(
        status_code=404,
        detail="Could not determine transcript name for this run.",
    )


def _resolve_output_dir(
    run_dir: Path,
) -> tuple[str, Path]:
    transcript_name = _resolve_transcript_name(run_dir)

    output_dir = (
        run_dir
        / "output"
        / transcript_name
    ).resolve()

    try:
        output_dir.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Invalid output directory for this run.",
        ) from exc

    return transcript_name, output_dir


def _api_status_path(run_dir: Path) -> Path:
    return run_dir / "api_frontend_status.json"


def _set_api_status(
    run_id: str,
    *,
    state: str,
    error: str | None = None,
    final_state: str | None = None,
    human_action_required: bool | None = None,
) -> None:
    run_dir = (RUNS_ROOT / str(run_id)).resolve()

    if not run_dir.is_dir():
        return

    current = _load_json(_api_status_path(run_dir))

    current.update(
        {
            "run_id": str(run_id),
            "state": str(state),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    if error is not None:
        current["error"] = str(error)
    elif state != "failed":
        current.pop("error", None)

    if final_state is not None:
        current["final_state"] = str(final_state)

    if human_action_required is not None:
        current["human_action_required"] = bool(
            human_action_required
        )

    _write_json(_api_status_path(run_dir), current)


def _safe_snapshot(run_id: str) -> dict[str, Any]:
    try:
        value = langgraph_snapshot(
            frontend_root=BACKEND_RUNTIME_ROOT,
            run_id=str(run_id),
        )
    except Exception:
        return {}

    return value if isinstance(value, dict) else {}


def _run_agent1_background(run_id: str) -> None:
    """
    Execute the existing LangGraph -> MCP -> Agent 1 workflow in the
    background so the browser can poll real progress while it runs.
    """
    _set_api_status(
        run_id,
        state="running",
        human_action_required=False,
    )

    try:
        result = run_langgraph_request(
            frontend_root=BACKEND_RUNTIME_ROOT,
            run_id=str(run_id),
            user_request=(
                "Process this transcript through Agent 1 using the valid "
                "next MCP tool for each state. Stop immediately at any "
                "mandatory human gate. Do not originate any human decision."
            ),
            max_steps=8,
            mode="start",
        )

        snapshot = _safe_snapshot(run_id)

        human_required = bool(
            snapshot.get("human_action_required")
            or result.get("human_action_required")
        )

        _set_api_status(
            run_id,
            state=(
                "waiting_for_human"
                if human_required
                else "complete"
            ),
            final_state=str(
                snapshot.get("state")
                or result.get("final_state")
                or ""
            ),
            human_action_required=human_required,
        )

    except Exception as exc:
        _set_api_status(
            run_id,
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            human_action_required=False,
        )


def _resume_after_human_action(run_id: str) -> dict[str, Any]:
    """
    Resume the SAME persisted LangGraph thread after a human-only write.

    If the graph is not currently paused at a human gate, no blind resume
    is attempted.
    """
    snapshot_before = _safe_snapshot(run_id)

    if not bool(snapshot_before.get("human_action_required")):
        return snapshot_before

    _set_api_status(
        run_id,
        state="running",
        human_action_required=False,
    )

    try:
        result = run_langgraph_request(
            frontend_root=BACKEND_RUNTIME_ROOT,
            run_id=str(run_id),
            user_request=(
                "Continue the existing Agent 1 workflow after the human "
                "decision that has already been persisted. Do not originate "
                "another human decision."
            ),
            max_steps=8,
            mode="resume",
        )
    except Exception as exc:
        _set_api_status(
            run_id,
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    snapshot_after = _safe_snapshot(run_id)

    human_required = bool(
        snapshot_after.get("human_action_required")
        or result.get("human_action_required")
    )

    _set_api_status(
        run_id,
        state=(
            "waiting_for_human"
            if human_required
            else "complete"
        ),
        final_state=str(
            snapshot_after.get("state")
            or result.get("final_state")
            or ""
        ),
        human_action_required=human_required,
    )

    return snapshot_after


# ============================================================
# Agent 1 read helpers
# ============================================================

def _hitl_adapter():
    # Import lazily so FastAPI itself stays a thin bridge.
    from mcp_server.adapters.agent1_hitl_adapter import (
        Agent1HitlAdapter,
    )

    return Agent1HitlAdapter(BACKEND_RUNTIME_ROOT)


def _syllabus_options() -> tuple[list[dict[str, Any]], str | None]:
    """
    Read current AQA concepts from the PostgreSQL-backed SyllabusStore.

    If storage is temporarily unavailable, topic viewing still works;
    correction dropdowns simply report the storage error.
    """
    try:
        code_root = str(AGENT1_CODE_ROOT)

        if code_root not in sys.path:
            sys.path.insert(0, code_root)

        from app.services.syllabus_store import (
            get_syllabus_store,
        )

        concepts = get_syllabus_store().get_all_concepts()

        rows = [
            {
                "concept_id": str(concept.concept_id),
                "label": str(concept.label),
                "official_reference": str(
                    concept.official_reference
                ),
                "chapter_reference": str(
                    concept.chapter_reference
                ),
                "official_title": str(
                    concept.official_title
                ),
                "domain": str(concept.domain),
                "paper": str(concept.paper),
            }
            for concept in concepts
        ]

        return rows, None

    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _semantic_payload(run_id: str) -> dict[str, Any]:
    run_dir = _resolve_run_dir(run_id)
    transcript_name, output_dir = _resolve_output_dir(run_dir)

    chunking_path = output_dir / "02_chunking.json"

    if not chunking_path.is_file():
        raise HTTPException(
            status_code=409,
            detail=(
                "Semantic chunking output is not ready for this run. "
                "02_chunking.json was not found."
            ),
        )

    module2 = _load_json(chunking_path)
    raw_chunks = module2.get("chunks")

    if not isinstance(raw_chunks, list):
        raw_chunks = []

    chunks: list[dict[str, Any]] = []

    for raw in raw_chunks:
        if not isinstance(raw, dict):
            continue

        chunk = dict(raw)

        try:
            chunk["chunk_id"] = int(chunk.get("chunk_id"))
        except (TypeError, ValueError):
            chunk["chunk_id"] = len(chunks) + 1

        chunks.append(chunk)

    return {
        "success": True,
        "run_id": str(run_id),
        "transcript_name": transcript_name,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }



def _current_aqa_spec_version() -> str:
    return (
        os.getenv("AQA_SPEC_VERSION", "").strip()
        or "AQA-8525-v1.2-2022-11-29"
    )


def _detected_topic_reuse_feedback_store():
    """
    Existing PostgreSQL exact-context reuse-feedback store used by the
    historical final-topic HITL system.
    """
    code_root = str(AGENT1_CODE_ROOT)

    if code_root not in sys.path:
        sys.path.insert(0, code_root)

    from app.services.detected_topic_edit_reuse_feedback_store import (
        DetectedTopicEditReuseFeedbackStore,
    )

    return DetectedTopicEditReuseFeedbackStore()


def _raw_module3_result_payload(
    module3_json: dict[str, Any],
) -> dict[str, Any]:
    """
    Return the untouched fresh Module 3 result from 03_topic_mapping.json.

    Current production output wraps the real Module3Result under
    ``module3_result``. Some older/test payloads are already the raw result,
    so keep a backward-compatible top-level fallback.

    This is the canonical evidence source used by the original Streamlit HITL
    bridge and DetectedTopicEditEndToEndService.
    """
    nested = module3_json.get("module3_result")

    if isinstance(nested, dict):
        return nested

    return module3_json


def _fresh_existing_topic_evidence(
    *,
    module3_json: dict[str, Any],
    source_concept_id: str | None,
) -> str:
    """
    Mirror DetectedTopicEditEndToEndService._evidence_by_concept exactly.

    IMPORTANT:
    reuse decisions are keyed from untouched fresh Module 3 evidence,
    never from the post-memory effective topic list.
    """
    concept_id = str(
        source_concept_id or ""
    ).strip()

    if not concept_id:
        return ""

    raw_module3_result = (
        _raw_module3_result_payload(
            module3_json
        )
    )

    for topic in raw_module3_result.get(
        "merged_topics",
        [],
    ) or []:
        if not isinstance(topic, dict):
            continue

        if (
            str(
                topic.get("concept_id")
                or ""
            ).strip()
            != concept_id
        ):
            continue

        evidence = topic.get("evidence") or []

        if not isinstance(evidence, list):
            evidence = [evidence]

        return "\n".join(
            str(value).strip()
            for value in evidence
            if str(value).strip()
        )

    return ""


def _fresh_add_topic_evidence(
    module3_json: dict[str, Any],
) -> str:
    """
    Mirror DetectedTopicEditEndToEndService._addition_current_evidence.
    """
    evidence: list[str] = []

    raw_module3_result = (
        _raw_module3_result_payload(
            module3_json
        )
    )

    for chunk in raw_module3_result.get(
        "chunk_results",
        [],
    ) or []:
        if not isinstance(chunk, dict):
            continue

        candidates = [
            *(chunk.get("topic_candidates") or []),
            *(chunk.get("rejected_candidates") or []),
        ]

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            values = candidate.get("evidence") or []

            if not isinstance(values, list):
                values = [values]

            evidence.extend(
                str(value).strip()
                for value in values
                if str(value).strip()
            )

    return "\n".join(evidence)


def _memory_evidence_for_current_run(
    *,
    module3_json: dict[str, Any],
    memory: dict[str, Any],
) -> str:
    action = str(
        memory.get("edit_action")
        or memory.get("action")
        or ""
    ).strip()

    if action == "add_topic":
        return _fresh_add_topic_evidence(
            module3_json
        )

    return _fresh_existing_topic_evidence(
        module3_json=module3_json,
        source_concept_id=(
            memory.get("source_concept_id")
        ),
    )



def _normalise_memory_match_text(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value or "").casefold(),
    ).strip()


def _candidate_target_signal(
    *,
    module3_json: dict[str, Any],
    target_concept_id: str,
    target_topic: str,
) -> bool:
    """
    Guard add_topic memories from unrelated historical reuse.

    An add-topic memory is only surfaced to the human when the CURRENT
    untouched Module 3 output contains a candidate/rejected-candidate signal
    for that target official concept/topic.

    This affects only which historical cards are shown in the Next.js UI.
    It does NOT change the existing deterministic memory overlay or the
    exact-context evidence used by the PostgreSQL reuse-feedback store.
    """
    target_id = str(
        target_concept_id or ""
    ).strip()

    target_name = _normalise_memory_match_text(
        target_topic
    )

    if not target_id and not target_name:
        return False

    raw_module3_result = (
        _raw_module3_result_payload(
            module3_json
        )
    )

    id_keys = (
        "concept_id",
        "mapped_concept_id",
        "target_concept_id",
        "syllabus_concept_id",
        "official_concept_id",
        "proposed_mapped_concept_id",
    )

    name_keys = (
        "topic",
        "topic_label",
        "name",
        "title",
        "mapped_topic",
        "target_topic",
        "syllabus_topic",
        "official_topic",
    )

    def record_matches(
        record: dict[str, Any],
    ) -> bool:
        for key in id_keys:
            value = str(
                record.get(key)
                or ""
            ).strip()

            if (
                target_id
                and value
                and value == target_id
            ):
                return True

        for nested_key in (
            "mapped_concept",
            "target_concept",
            "syllabus_concept",
        ):
            nested = record.get(
                nested_key
            )

            if isinstance(nested, dict):
                if record_matches(nested):
                    return True

        if target_name:
            for key in name_keys:
                value = (
                    _normalise_memory_match_text(
                        record.get(key)
                    )
                )

                if not value:
                    continue

                if (
                    value == target_name
                    or target_name in value
                    or value in target_name
                ):
                    return True

        return False

    for chunk in raw_module3_result.get(
        "chunk_results",
        [],
    ) or []:
        if not isinstance(chunk, dict):
            continue

        for key in (
            "topic_candidates",
            "rejected_candidates",
        ):
            candidates = chunk.get(
                key
            ) or []

            if not isinstance(
                candidates,
                list,
            ):
                continue

            for candidate in candidates:
                if (
                    isinstance(
                        candidate,
                        dict,
                    )
                    and record_matches(
                        candidate
                    )
                ):
                    return True

    # Some Module 3 versions keep candidate→official mapping diagnostics here.
    llm_results = raw_module3_result.get(
        "llm_results"
    )

    if not isinstance(llm_results, list):
        llm_results = module3_json.get(
            "llm_results",
            [],
        )

    for item in llm_results or []:
        if (
            isinstance(item, dict)
            and record_matches(item)
        ):
            return True

    return False



def _memory_action_label(
    memory: dict[str, Any],
) -> str:
    action = str(
        memory.get("edit_action")
        or memory.get("action")
        or ""
    ).strip()

    source_topic = str(
        memory.get("source_topic")
        or memory.get("source_concept_id")
        or "topic"
    ).strip()

    target_topic = str(
        memory.get("target_topic")
        or memory.get("target_concept_id")
        or ""
    ).strip()

    target_role = str(
        memory.get("target_role")
        or ""
    ).strip()

    if action == "remove_topic":
        return f"Remove {source_topic}"

    if action == "change_role":
        return (
            f"Change {source_topic} to "
            f"{target_role or 'the historical role'}"
        )

    if action == "replace_topic":
        return (
            f"Replace {source_topic} with "
            f"{target_topic or 'the historical topic'}"
        )

    if action == "add_topic":
        return (
            f"Add {target_topic or 'the historical topic'}"
        )

    return action.replace("_", " ").title()


def _fresh_topic_by_concept(
    module3_json: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}

    raw_module3_result = (
        _raw_module3_result_payload(
            module3_json
        )
    )

    for raw in raw_module3_result.get(
        "merged_topics",
        [],
    ) or []:
        if not isinstance(raw, dict):
            continue

        concept_id = str(
            raw.get("concept_id")
            or ""
        ).strip()

        if concept_id:
            output[concept_id] = dict(raw)

    return output


def _memory_review_payload(
    *,
    run_id: str,
    transcript_name: str,
    module3_json: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """
    Build reviewer-facing historical final-topic memory cards.

    Show:
      - ambiguous/conflicting skipped memories (mandatory human decision);
      - automatically applied memories (human may still override them).

    Strong mismatches / incompatible historical memories stay hidden because
    the deterministic comparator has already safely rejected them.
    """
    applied = [
        item
        for item in (
            runtime.get("applied") or []
        )
        if isinstance(item, dict)
    ]

    skipped = [
        item
        for item in (
            runtime.get("skipped") or []
        )
        if isinstance(item, dict)
    ]

    diagnostics = [
        str(value)
        for value in (
            runtime.get(
                "retrieval_diagnostics"
            )
            or []
        )
        if str(value).strip()
    ]

    diagnostic_by_memory: dict[int, str] = {}

    for line in diagnostics:
        match = re.match(
            r"^memory\s+(\d+):\s*(.*)$",
            line,
            flags=re.IGNORECASE,
        )

        if match:
            diagnostic_by_memory[
                int(match.group(1))
            ] = match.group(2).strip()

    runtime_rows: list[
        dict[str, Any]
    ] = []

    for item in applied:
        try:
            memory_id = int(
                item.get("memory_id")
            )
        except (TypeError, ValueError):
            continue

        runtime_rows.append(
            {
                "memory_id": memory_id,
                "runtime_status": (
                    "historical_applied"
                ),
                "runtime_reason": str(
                    item.get("explanation")
                    or (
                        "Historical edit matched "
                        "the current context."
                    )
                ).strip(),
                "source_concept_id": str(
                    item.get(
                        "source_concept_id"
                    )
                    or ""
                ).strip(),
                "target_concept_id": str(
                    item.get(
                        "target_concept_id"
                    )
                    or ""
                ).strip(),
            }
        )

    for item in skipped:
        try:
            memory_id = int(
                item.get("memory_id")
            )
        except (TypeError, ValueError):
            continue

        reason = str(
            item.get("reason")
            or ""
        ).strip()

        reason_lower = reason.casefold()

        # Strong mismatches are already safely rejected.
        if "incompatible" in reason_lower:
            continue

        # No choice is needed if fresh Module 3 already satisfies the
        # historical outcome.
        if (
            "already matches the fresh module 3 role"
            in reason_lower
        ):
            continue

        needs_decision = (
            "uncertain" in reason_lower
            or "conflict" in reason_lower
            or (
                "multiple edit-memory candidates"
                in reason_lower
            )
        )

        if not needs_decision:
            # Keep the UI conservative: only expose rows that the historical
            # audit says need or plausibly benefit from explicit human review.
            if not reason:
                continue

        runtime_rows.append(
            {
                "memory_id": memory_id,
                "runtime_status": (
                    "decision_required"
                ),
                "runtime_reason": reason,
                "source_concept_id": str(
                    item.get(
                        "source_concept_id"
                    )
                    or ""
                ).strip(),
                "target_concept_id": str(
                    item.get(
                        "target_concept_id"
                    )
                    or ""
                ).strip(),
            }
        )

    if not runtime_rows:
        return {
            "items": [],
            "pending_count": 0,
        }

    try:
        store = (
            _detected_topic_reuse_feedback_store()
        )
    except Exception as exc:
        return {
            "items": [],
            "pending_count": 0,
            "error": (
                "Historical memory store could "
                "not be loaded: "
                f"{type(exc).__name__}: {exc}"
            ),
        }

    fresh_by_concept = (
        _fresh_topic_by_concept(
            module3_json
        )
    )

    enriched: list[
        dict[str, Any]
    ] = []

    for row in runtime_rows:
        memory_id = int(
            row["memory_id"]
        )

        try:
            memory = store.memory_snapshot(
                memory_id
            )
        except Exception:
            memory = None

        if not isinstance(memory, dict):
            memory = {}

        source_concept_id = str(
            memory.get("source_concept_id")
            or row.get(
                "source_concept_id"
            )
            or ""
        ).strip()

        target_concept_id = str(
            memory.get("target_concept_id")
            or row.get(
                "target_concept_id"
            )
            or ""
        ).strip()

        edit_action = str(
            memory.get("edit_action")
            or ""
        ).strip()

        target_topic = str(
            memory.get("target_topic")
            or target_concept_id
            or ""
        ).strip()

        # Noise guard for historical add-topic memories:
        # do not ask the user about an unrelated old addition unless the
        # current Module 3 output contains an actual candidate signal for it.
        if (
            edit_action == "add_topic"
            and not _candidate_target_signal(
                module3_json=module3_json,
                target_concept_id=target_concept_id,
                target_topic=target_topic,
            )
        ):
            continue

        evidence = (
            _memory_evidence_for_current_run(
                module3_json=module3_json,
                memory={
                    **memory,
                    "source_concept_id": (
                        source_concept_id
                    ),
                },
            )
        )

        spec_version = str(
            memory.get("spec_version")
            or _current_aqa_spec_version()
        ).strip()

        saved_decision = None
        saved_reason = None

        if evidence:
            try:
                feedback = store.get_decision(
                    memory_id=memory_id,
                    current_evidence=evidence,
                    spec_version=spec_version,
                )
            except Exception:
                feedback = None

            if feedback is not None:
                saved_decision = str(
                    getattr(
                        feedback,
                        "decision",
                        "",
                    )
                    or ""
                ).strip() or None

                saved_reason = str(
                    getattr(
                        feedback,
                        "reviewer_reason",
                        "",
                    )
                    or ""
                ).strip() or None

        # Once the human has explicitly reviewed this historical memory for
        # the exact current lesson evidence, it is resolved. Keep the DB audit
        # row, but do not keep asking the user about it on the Topic Mapping UI.
        if saved_decision in {
            "approve_reuse",
            "reject_reuse",
        }:
            continue

        fresh_topic = (
            fresh_by_concept.get(
                source_concept_id
            )
            if source_concept_id
            else None
        )

        enriched.append(
            {
                **row,
                "memory_id": memory_id,
                "edit_action": edit_action,
                "source_concept_id": (
                    source_concept_id
                ),
                "source_topic": str(
                    memory.get("source_topic")
                    or (
                        fresh_topic.get("topic")
                        if isinstance(
                            fresh_topic,
                            dict,
                        )
                        else ""
                    )
                    or source_concept_id
                    or "Topic"
                ).strip(),
                "source_role": str(
                    memory.get("source_role")
                    or ""
                ).strip(),
                "target_concept_id": (
                    target_concept_id
                ),
                "target_topic": target_topic,
                "target_role": str(
                    memory.get("target_role")
                    or ""
                ).strip(),
                "reviewer_reason": str(
                    memory.get(
                        "reviewer_reason"
                    )
                    or ""
                ).strip(),
                "stored_evidence": str(
                    memory.get(
                        "stored_evidence"
                    )
                    or ""
                ).strip(),
                "historical_outcome": (
                    _memory_action_label(
                        {
                            **memory,
                            "source_concept_id": (
                                source_concept_id
                            ),
                            "target_concept_id": (
                                target_concept_id
                            ),
                        }
                    )
                ),
                "context_diagnostic": (
                    diagnostic_by_memory.get(
                        memory_id,
                        "",
                    )
                ),
                "current_evidence": evidence,
                "spec_version": spec_version,
                "saved_decision": (
                    saved_decision
                ),
                "saved_reason": saved_reason,
                "fresh_topic": (
                    dict(fresh_topic)
                    if isinstance(
                        fresh_topic,
                        dict,
                    )
                    else None
                ),
            }
        )

    # Group competing outcomes for the same source/target lesson concept.
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for item in enriched:
        action = str(
            item.get("edit_action")
            or ""
        ).strip()

        if action == "add_topic":
            key = (
                "add::"
                + str(
                    item.get(
                        "target_concept_id"
                    )
                    or item.get("memory_id")
                )
            )
        else:
            key = (
                "existing::"
                + str(
                    item.get(
                        "source_concept_id"
                    )
                    or item.get("memory_id")
                )
            )

        groups.setdefault(
            key,
            [],
        ).append(item)

    output: list[
        dict[str, Any]
    ] = []

    pending_count = 0

    for group_key, memories in groups.items():
        memories.sort(
            key=lambda item: int(
                item.get("memory_id") or 0
            )
        )

        first = memories[0]

        runtime_statuses = {
            str(
                item.get("runtime_status")
                or ""
            )
            for item in memories
        }

        decision_required = (
            "decision_required"
            in runtime_statuses
        )

        if decision_required:
            pending_count += 1

        fresh_topic = next(
            (
                item.get("fresh_topic")
                for item in memories
                if isinstance(
                    item.get("fresh_topic"),
                    dict,
                )
            ),
            None,
        )

        output.append(
            {
                "review_key": group_key,
                "topic_label": str(
                    first.get("source_topic")
                    or first.get("target_topic")
                    or "Topic"
                ),
                "status": (
                    "decision_required"
                    if decision_required
                    else "historical_applied"
                ),
                "memory_ids": [
                    int(
                        item["memory_id"]
                    )
                    for item in memories
                ],
                "memories": memories,
                "fresh_topic": fresh_topic,
                "saved_decision": next(
                    (
                        item.get(
                            "saved_decision"
                        )
                        for item in memories
                        if item.get(
                            "saved_decision"
                        )
                    ),
                    None,
                ),
            }
        )

    return {
        "items": output,
        "pending_count": pending_count,
        "error": None,
    }


def _topics_payload(run_id: str) -> dict[str, Any]:
    run_dir = _resolve_run_dir(run_id)
    transcript_name, output_dir = _resolve_output_dir(run_dir)

    module3_path = output_dir / "03_topic_mapping.json"

    if not module3_path.is_file():
        raise HTTPException(
            status_code=409,
            detail=(
                "Topic mapping output is not ready for this run. "
                "03_topic_mapping.json was not found."
            ),
        )

    module3_json = _load_json(module3_path)

    try:
        adapter = _hitl_adapter()

        effective = adapter.get_effective_topics(str(run_id))
        review_state = adapter.get_pending_topic_reviews(
            str(run_id)
        )
        approved = adapter.get_approved_topics(str(run_id))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load Agent 1 HITL state: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    topics_raw = effective.get("topics")
    if not isinstance(topics_raw, list):
        topics_raw = []

    topics: list[dict[str, Any]] = []

    for effective_index, raw in enumerate(topics_raw):
        if not isinstance(raw, dict):
            continue

        topic = dict(raw)

        # Two different indexes exist in the original runtime:
        # - detected-topic edits use the zero-based effective-list position;
        # - the Agent 1 -> Agent 2 handoff uses enumerate(..., start=1).
        # Keep both so the Next.js UI never sends a zero-based handoff index
        # to the second human gate.
        topic["effective_index"] = effective_index
        topic["topic_index"] = effective_index + 1

        # Give the frontend stable aliases without changing backend data.
        topic["role"] = str(
            topic.get("topic_role")
            or topic.get("role")
            or "supporting"
        ).strip().casefold()

        topics.append(topic)

    pending = review_state.get("pending")
    if not isinstance(pending, list):
        pending = []

    resolved = review_state.get("resolved")
    if not isinstance(resolved, list):
        resolved = []

    orphaned = review_state.get("orphaned")
    if not isinstance(orphaned, list):
        orphaned = []

    approved_topics = approved.get("topics")
    if not isinstance(approved_topics, list):
        approved_topics = []

    llm_results = module3_json.get("llm_results")
    if not isinstance(llm_results, list):
        llm_results = []

    syllabus_options, syllabus_error = _syllabus_options()
    snapshot = _safe_snapshot(str(run_id))

    snapshot_approved_count = snapshot.get(
        "approved_topic_count",
        0,
    )

    try:
        snapshot_approved_count = int(
            snapshot_approved_count or 0
        )
    except (TypeError, ValueError):
        snapshot_approved_count = 0

    memory_review = _memory_review_payload(
        run_id=str(run_id),
        transcript_name=transcript_name,
        module3_json=module3_json,
        runtime=(
            effective.get("runtime")
            if isinstance(
                effective.get("runtime"),
                dict,
            )
            else {}
        ),
    )

    return {
        "success": True,
        "run_id": str(run_id),
        "transcript_name": transcript_name,
        "topics": topics,
        "topic_count": len(topics),
        "pending_reviews": pending,
        "pending_review_count": len(pending),
        "resolved_reviews": resolved,
        "resolved_review_count": len(resolved),
        "orphaned_reviews": orphaned,
        "orphaned_review_count": len(orphaned),
        "review_status_authority": review_state.get(
            "status_authority"
        ),
        "review_db_available": bool(
            review_state.get("db_available")
        ),
        "review_reconciliation_error": review_state.get(
            "reconciliation_error"
        ),
        "llm_results": [
            dict(item)
            for item in llm_results
            if isinstance(item, dict)
        ],
        "runtime": effective.get("runtime") or {},
        "historical_memory_reviews": memory_review["items"],
        "historical_memory_review_count": len(memory_review["items"]),
        "historical_memory_pending_count": memory_review["pending_count"],
        "historical_memory_error": memory_review.get("error"),
        "spec_version": effective.get("spec_version"),
        "approved_topics": [
            dict(item)
            for item in approved_topics
            if isinstance(item, dict)
        ],
        "approved_topic_file_count": len(approved_topics),
        "agent2_handoff_ready": snapshot_approved_count > 0,
        "snapshot": snapshot,
        "syllabus_options": syllabus_options,
        "syllabus_error": syllabus_error,
    }


# ============================================================
# Agent 2 assessment / quiz migration helpers
# ============================================================

MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION = 2
USER_REGENERATION_STATE_SCHEMA_VERSION = "agent2-user-regeneration-attempts-v1.0.0"
USER_REGENERATION_DISCLAIMER = (
    "Each generated question can be regenerated by the user up to 2 times. "
    "Regenerating the whole quiz uses 1 user regeneration attempt for every "
    "generated question in that quiz. Automatic validation, quality-control, "
    "or model retry attempts are separate and do not count toward this limit. "
    "A user attempt is recorded only after a valid regenerated result is "
    "successfully committed."
)


QUIZ_GENERATION_NOTEBOOK_OPTIONS: dict[str, dict[str, Any]] = {
    "plan_a": {
        "label": "Plan A — Notebook 06 (baseline)",
        "filenames": ["06_quiz_generation.ipynb"],
        "strategy": "current_notebook_06",
    },
    "plan_b": {
        "label": "Plan B — Notebook 06B (aggressive single-batch)",
        "filenames": [
            "06B_quiz_generation_optimized.ipynb",
            "06B_quiz_generation.ipynb",
        ],
        "strategy": "plan_b_aggressive_single_batch",
    },
    "plan_c": {
        "label": "Plan C — Notebook 06C (generic adaptive batching / cost-control V3)",
        "filenames": [
            "06C_quiz_generation_question_first_UPDATED.ipynb",
            "06C_quiz_generation_GENERIC_COST_CONTROL_V3.ipynb",
            "06C_quiz_generation_GENERIC_OPTIMIZED.ipynb",
            "06C_quiz_generation_FINAL_OFFICIAL_PDF_PRESERVED.ipynb",
            "06C_quiz_generation.ipynb",
        ],
        "strategy": "plan_c_hybrid_adaptive_batching_v1",
    },
}



# ============================================================
# Agent 2 ETA from persisted assessment history
# ============================================================

_AGENT2_ETA_CACHE: dict[str, Any] = {
    "computed_at": 0.0,
    "events": [],
}


def _parse_agent2_iso_epoch(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None

    try:
        parsed = datetime.fromisoformat(
            raw.replace("Z", "+00:00")
        )
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.timestamp()


def _agent2_timing_history_path(
    run_dir: Path,
) -> Path:
    path = (
        Path(run_dir)
        / "output"
        / "integration"
        / "agent2_timing_history.jsonl"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def _agent2_process_for_stage(
    stage: str,
) -> str | None:
    return {
        "retrieving": "official_retrieval",
        "generating": "complete_quiz_generation",
        "generating_shortfall": "shortfall_generation",
    }.get(str(stage or "").strip())


def _agent2_event_metadata(
    *,
    request: dict[str, Any] | None,
    shortfall: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = request if isinstance(request, dict) else {}
    shortfall = shortfall if isinstance(shortfall, dict) else {}

    def as_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    return {
        "quiz_plan": str(request.get("quiz_plan") or ""),
        "model_key": str(request.get("model_key") or ""),
        "requested_questions": as_int(
            request.get("number_of_questions")
        ),
        "target_marks": as_int(
            request.get("target_total_marks")
        ),
        "include_visuals": bool(
            request.get("include_visual_questions", False)
        ),
        "shortfall_questions": as_int(
            shortfall.get("missing_questions")
        ),
        "shortfall_marks": as_int(
            shortfall.get("missing_marks")
        ),
    }


def _append_agent2_timing_event(
    *,
    run_dir: Path,
    process: str,
    duration_seconds: float,
    request: dict[str, Any] | None,
    shortfall: dict[str, Any] | None = None,
    stage_started_at_utc: str | None = None,
    outcome: str = "completed",
    source: str = "nextjs_runtime",
) -> None:
    duration = float(duration_seconds or 0.0)

    if not (1.0 <= duration <= 7200.0):
        return

    metadata = _agent2_event_metadata(
        request=request,
        shortfall=shortfall,
    )

    event_seed = {
        "run_id": Path(run_dir).name,
        "process": str(process),
        "stage_started_at_utc": str(
            stage_started_at_utc or ""
        ),
        "duration_seconds": round(duration, 3),
        **metadata,
    }

    event_id = hashlib.sha256(
        json.dumps(
            event_seed,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    event = {
        "schema_version": "agent2-timing-history-v1.0.0",
        "event_id": event_id,
        **event_seed,
        "recorded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "outcome": str(outcome or "completed"),
        "source": str(source or "nextjs_runtime"),
    }

    path = _agent2_timing_history_path(run_dir)

    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = ""

        if f'"event_id": "{event_id}"' in existing:
            return

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                event,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )

    _AGENT2_ETA_CACHE["computed_at"] = 0.0
    _AGENT2_ETA_CACHE["events"] = []


def _record_previous_agent2_stage_timing(
    *,
    run_dir: Path,
    existing: dict[str, Any],
    next_stage: str,
    next_status: str,
    now_iso: str,
) -> None:
    if not isinstance(existing, dict):
        return

    previous_status = str(
        existing.get("status") or ""
    ).strip().casefold()

    previous_stage = str(
        existing.get("stage") or ""
    ).strip()

    process = _agent2_process_for_stage(
        previous_stage
    )

    if (
        previous_status not in {"queued", "running"}
        or process is None
    ):
        return

    should_close = (
        previous_stage != str(next_stage or "").strip()
        or str(next_status or "").strip().casefold()
        in {"complete", "failed"}
    )

    if not should_close:
        return

    started_epoch = _parse_agent2_iso_epoch(
        existing.get("stage_started_at_utc")
        or existing.get("started_at_utc")
    )
    ended_epoch = _parse_agent2_iso_epoch(now_iso)

    if started_epoch is None or ended_epoch is None:
        return

    _append_agent2_timing_event(
        run_dir=run_dir,
        process=process,
        duration_seconds=ended_epoch - started_epoch,
        request=(
            existing.get("request")
            if isinstance(existing.get("request"), dict)
            else {}
        ),
        shortfall=(
            existing.get("shortfall")
            if isinstance(existing.get("shortfall"), dict)
            else {}
        ),
        stage_started_at_utc=str(
            existing.get("stage_started_at_utc") or ""
        ),
        outcome=(
            "completed"
            if str(next_status).strip().casefold() == "complete"
            else "stage_transition"
        ),
    )


def _agent2_manifest_mtime(
    run_dir: Path,
    quiz_mode: str,
) -> float | None:
    output_dir = (
        Path(run_dir)
        / "output"
        / "agent2_quiz"
        / str(quiz_mode)
    )

    candidates = [
        output_dir / "final_quiz_manifest.json",
        output_dir
        / "mcp_visuals"
        / "final_quiz_manifest_with_mcp_visuals.json",
    ]

    values: list[float] = []

    for path in candidates:
        try:
            if path.is_file():
                values.append(path.stat().st_mtime)
        except OSError:
            continue

    return max(values) if values else None


def _agent2_retrieval_package_mtime(
    run_dir: Path,
) -> float | None:
    output_dir = (
        Path(run_dir)
        / "output"
        / "agent2"
    )

    if not output_dir.is_dir():
        return None

    try:
        candidates = list(
            output_dir.glob(
                "agent2_assessment_package_*.json"
            )
        )
    except OSError:
        return None

    values: list[float] = []

    for path in candidates:
        try:
            values.append(path.stat().st_mtime)
        except OSError:
            continue

    return max(values) if values else None


def _legacy_agent2_timing_events(
    run_dir: Path,
) -> list[dict[str, Any]]:
    """
    Bootstrap historical ETA from older completed Agent 2 runs using only:
      - nextjs_agent2_status.json timestamps
      - retrieval package mtime
      - generated quiz manifest mtime
    """
    status = _load_json(
        Path(run_dir)
        / "output"
        / "integration"
        / "nextjs_agent2_status.json"
    )

    if not status:
        return []

    if str(
        status.get("status") or ""
    ).strip().casefold() != "complete":
        return []

    started = _parse_agent2_iso_epoch(
        status.get("started_at_utc")
    )
    if started is None:
        return []

    request = (
        status.get("request")
        if isinstance(status.get("request"), dict)
        else {}
    )
    shortfall = (
        status.get("shortfall")
        if isinstance(status.get("shortfall"), dict)
        else {}
    )

    metadata = _agent2_event_metadata(
        request=request,
        shortfall=shortfall,
    )

    mode = str(status.get("mode") or "").strip()
    events: list[dict[str, Any]] = []

    def add(
        process: str,
        start_epoch: float,
        end_epoch: float | None,
    ) -> None:
        if end_epoch is None:
            return

        duration = float(end_epoch) - float(start_epoch)

        if not (1.0 <= duration <= 7200.0):
            return

        events.append(
            {
                "schema_version": "agent2-timing-history-v1.0.0",
                "event_id": f"legacy:{run_dir.name}:{process}",
                "run_id": run_dir.name,
                "process": process,
                "duration_seconds": duration,
                **metadata,
                "outcome": "completed",
                "source": "legacy_artifact_timestamps",
            }
        )

    if mode == "complete_quiz":
        generation_end = _agent2_manifest_mtime(
            run_dir,
            "complete_quiz",
        )
        if generation_end is None:
            generation_end = _parse_agent2_iso_epoch(
                status.get("completed_at_utc")
            )

        add(
            "complete_quiz_generation",
            started,
            generation_end,
        )
        return events

    if mode != "retrieve_hybrid":
        return events

    retrieval_end = _agent2_retrieval_package_mtime(
        run_dir
    )
    if retrieval_end is None:
        return events

    add(
        "official_retrieval",
        started,
        retrieval_end,
    )

    if bool(shortfall.get("sufficient", False)):
        return events

    shortfall_end = _agent2_manifest_mtime(
        run_dir,
        "fill_shortfall",
    )

    if (
        shortfall_end is not None
        and shortfall_end > retrieval_end
    ):
        add(
            "shortfall_generation",
            retrieval_end,
            shortfall_end,
        )

    return events


def _historical_agent2_timing_events() -> list[dict[str, Any]]:
    now = time.monotonic()

    cached_at = float(
        _AGENT2_ETA_CACHE.get("computed_at") or 0.0
    )
    cached_events = _AGENT2_ETA_CACHE.get("events")

    if (
        isinstance(cached_events, list)
        and cached_events
        and now - cached_at < 30.0
    ):
        return cached_events

    events: list[dict[str, Any]] = []

    if not RUNS_ROOT.is_dir():
        return events

    run_dirs = sorted(
        (
            path
            for path in RUNS_ROOT.iterdir()
            if path.is_dir()
            and path.name.startswith("job_")
        ),
        key=lambda path: path.name,
        reverse=True,
    )[:80]

    for run_dir in run_dirs:
        timing_path = (
            run_dir
            / "output"
            / "integration"
            / "agent2_timing_history.jsonl"
        )

        recorded_processes: set[str] = set()

        if timing_path.is_file():
            try:
                lines = timing_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            except OSError:
                lines = []

            for line in lines:
                try:
                    event = json.loads(line)
                except Exception:
                    continue

                if not isinstance(event, dict):
                    continue

                try:
                    duration = float(
                        event.get("duration_seconds") or 0.0
                    )
                except (TypeError, ValueError):
                    continue

                if not (1.0 <= duration <= 7200.0):
                    continue

                process = str(
                    event.get("process") or ""
                ).strip()
                if not process:
                    continue

                event["duration_seconds"] = duration
                events.append(event)
                recorded_processes.add(process)

        for event in _legacy_agent2_timing_events(
            run_dir
        ):
            if str(
                event.get("process") or ""
            ) in recorded_processes:
                continue

            events.append(event)

    _AGENT2_ETA_CACHE["computed_at"] = now
    _AGENT2_ETA_CACHE["events"] = events
    return events


def _agent2_percentile(
    values: list[float],
    fraction: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(
        float(value)
        for value in values
    )

    index = min(
        len(ordered) - 1,
        max(
            0,
            int(round(
                (len(ordered) - 1)
                * float(fraction)
            )),
        ),
    )

    return float(ordered[index])


def _agent2_current_units(
    *,
    process: str,
    request: dict[str, Any],
    shortfall: dict[str, Any],
) -> int:
    def as_int(value: Any) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    if process == "shortfall_generation":
        return max(
            1,
            as_int(shortfall.get("missing_questions")),
        )

    if process in {
        "complete_quiz_generation",
        "official_retrieval",
    }:
        return max(
            1,
            as_int(request.get("number_of_questions")),
        )

    return 1


def _agent2_event_units(
    event: dict[str, Any],
    process: str,
) -> int:
    key = (
        "shortfall_questions"
        if process == "shortfall_generation"
        else "requested_questions"
    )

    try:
        return max(
            1,
            int(float(event.get(key) or 1)),
        )
    except (TypeError, ValueError):
        return 1


def _agent2_adjust_duration_for_size(
    *,
    duration: float,
    event_units: int,
    current_units: int,
    process: str,
) -> float:
    if (
        event_units <= 0
        or current_units <= 0
        or process == "question_regeneration"
    ):
        return float(duration)

    ratio = (
        float(current_units)
        / float(event_units)
    )

    # Runtime has large fixed notebook/model/validation overhead.
    # Only scale the variable part by question count.
    variable_share = (
        0.20
        if process == "official_retrieval"
        else 0.35
    )

    factor = (
        (1.0 - variable_share)
        + variable_share * ratio
    )

    factor = min(
        1.75,
        max(0.60, factor),
    )

    return float(duration) * factor


def _format_agent2_duration_range(
    low_seconds: float,
    high_seconds: float,
    *,
    remaining: bool,
) -> str:
    low = max(0.0, float(low_seconds))
    high = max(low, float(high_seconds))
    suffix = " remaining" if remaining else ""

    if high < 20:
        return (
            "≈ finishing soon"
            if remaining
            else "≈ under 20 sec"
        )

    if high < 60:
        low_round = max(
            10,
            int(round(low / 10.0) * 10),
        )
        high_round = max(
            low_round + 10,
            int(round(high / 10.0) * 10),
        )
        return (
            f"≈ {low_round}–{high_round} sec"
            + suffix
        )

    low_minutes = max(
        1,
        int(low // 60),
    )
    high_minutes = max(
        low_minutes,
        int((high + 59) // 60),
    )

    if low_minutes == high_minutes:
        return (
            f"≈ {high_minutes} min"
            + suffix
        )

    return (
        f"≈ {low_minutes}–{high_minutes} min"
        + suffix
    )


def _agent2_eta_estimate(
    *,
    process: str,
    request: dict[str, Any] | None,
    shortfall: dict[str, Any] | None = None,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    request = request if isinstance(request, dict) else {}
    shortfall = shortfall if isinstance(shortfall, dict) else {}

    history = _historical_agent2_timing_events()

    all_events = [
        event
        for event in history
        if str(
            event.get("process") or ""
        ).strip() == process
    ]

    using_proxy = False

    # First regeneration can still have a realistic estimate:
    # use recent complete-quiz runtimes as a conservative proxy until actual
    # question-regeneration records exist.
    if (
        process == "question_regeneration"
        and not all_events
    ):
        all_events = [
            event
            for event in history
            if str(
                event.get("process") or ""
            ).strip()
            == "complete_quiz_generation"
        ]
        using_proxy = bool(all_events)

    if not all_events:
        return {
            "eta_seconds": None,
            "eta_label": "Learning ETA from recent Agent 2 runs…",
            "eta_total_label": "Learning ETA from recent Agent 2 runs…",
            "eta_basis": "No completed timing samples are available yet",
            "eta_sample_count": 0,
            "eta_low_seconds": None,
            "eta_high_seconds": None,
            "eta_total_low_seconds": None,
            "eta_total_high_seconds": None,
            "eta_process": process,
            "eta_source": "history",
        }

    quiz_plan = str(
        request.get("quiz_plan") or ""
    ).strip()
    model_key = str(
        request.get("model_key") or ""
    ).strip()

    same_plan_model = [
        event
        for event in all_events
        if (
            (
                not quiz_plan
                or str(
                    event.get("quiz_plan") or ""
                ).strip()
                == quiz_plan
            )
            and (
                not model_key
                or str(
                    event.get("model_key") or ""
                ).strip()
                == model_key
            )
        )
    ]

    same_plan = [
        event
        for event in all_events
        if (
            not quiz_plan
            or str(
                event.get("quiz_plan") or ""
            ).strip()
            == quiz_plan
        )
    ]

    if len(same_plan_model) >= 2:
        selected = same_plan_model
        similarity_text = "same strategy + model"
    elif len(same_plan) >= 2:
        selected = same_plan
        similarity_text = "same quiz strategy"
    else:
        selected = all_events
        similarity_text = "recent Agent 2"

    current_units = _agent2_current_units(
        process=process,
        request=request,
        shortfall=shortfall,
    )

    adjusted: list[float] = []

    for event in selected:
        duration = float(
            event.get("duration_seconds") or 0.0
        )

        if using_proxy:
            # Better to be conservatively close to a real 2–3 minute notebook
            # rerun than to claim the old fixed 20–60 seconds.
            adjusted.append(duration)
            continue

        adjusted.append(
            _agent2_adjust_duration_for_size(
                duration=duration,
                event_units=_agent2_event_units(
                    event,
                    process,
                ),
                current_units=current_units,
                process=process,
            )
        )

    adjusted = [
        value
        for value in adjusted
        if 1.0 <= value <= 7200.0
    ]

    if not adjusted:
        return {
            "eta_seconds": None,
            "eta_label": "Learning ETA from recent Agent 2 runs…",
            "eta_total_label": "Learning ETA from recent Agent 2 runs…",
            "eta_basis": "Recent timing samples were not usable",
            "eta_sample_count": 0,
            "eta_low_seconds": None,
            "eta_high_seconds": None,
            "eta_total_low_seconds": None,
            "eta_total_high_seconds": None,
            "eta_process": process,
            "eta_source": "history",
        }

    median_total = float(
        statistics.median(adjusted)
    )
    upper_total = max(
        _agent2_percentile(adjusted, 0.75),
        median_total * 1.12,
        median_total + 10.0,
    )

    elapsed = max(
        0.0,
        float(elapsed_seconds or 0.0),
    )

    remaining_low = max(
        0.0,
        median_total - elapsed,
    )
    remaining_high = max(
        0.0,
        upper_total - elapsed,
    )

    sample_count = len(adjusted)

    if using_proxy:
        basis = (
            "No question-regeneration timings yet; "
            f"conservative estimate from {sample_count} "
            f"{similarity_text} complete-quiz run"
            f"{'' if sample_count == 1 else 's'}"
        )
    else:
        basis = (
            f"Based on {sample_count} "
            f"{similarity_text} timing sample"
            f"{'' if sample_count == 1 else 's'}"
        )

    return {
        "eta_seconds": int(round(remaining_high)),
        "eta_label": _format_agent2_duration_range(
            remaining_low,
            remaining_high,
            remaining=True,
        ),
        "eta_total_label": _format_agent2_duration_range(
            median_total,
            upper_total,
            remaining=False,
        ),
        "eta_basis": basis,
        "eta_sample_count": sample_count,
        "eta_low_seconds": int(round(remaining_low)),
        "eta_high_seconds": int(round(remaining_high)),
        "eta_total_low_seconds": int(round(median_total)),
        "eta_total_high_seconds": int(round(upper_total)),
        "eta_process": process,
        "eta_source": (
            "historical_proxy"
            if using_proxy
            else "historical_runs"
        ),
    }


def _agent2_running_eta(
    *,
    status: dict[str, Any],
) -> dict[str, Any]:
    current_status = str(
        status.get("status") or ""
    ).strip().casefold()

    process = _agent2_process_for_stage(
        str(status.get("stage") or "")
    )

    if (
        current_status not in {"queued", "running"}
        or process is None
    ):
        return {
            "eta_seconds": None,
            "eta_label": None,
            "eta_total_label": None,
            "eta_basis": None,
            "eta_sample_count": 0,
            "eta_low_seconds": None,
            "eta_high_seconds": None,
            "eta_total_low_seconds": None,
            "eta_total_high_seconds": None,
            "eta_process": None,
            "eta_source": None,
        }

    started = _parse_agent2_iso_epoch(
        status.get("stage_started_at_utc")
        or status.get("started_at_utc")
    )

    elapsed = (
        max(0.0, time.time() - started)
        if started is not None
        else 0.0
    )

    return _agent2_eta_estimate(
        process=process,
        request=(
            status.get("request")
            if isinstance(status.get("request"), dict)
            else {}
        ),
        shortfall=(
            status.get("shortfall")
            if isinstance(status.get("shortfall"), dict)
            else {}
        ),
        elapsed_seconds=elapsed,
    )


def _assessment_status_path(run_dir: Path) -> Path:
    return (
        Path(run_dir)
        / "output"
        / "integration"
        / "nextjs_agent2_status.json"
    )


def _agent2_current_attempt_artifact(
    path: Path,
    attempt_started: float | None,
) -> bool:
    """Return True only for a non-empty artifact from the current attempt."""
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        modified_at = float(path.stat().st_mtime)
    except OSError:
        return False

    if attempt_started is None:
        return True

    # Small tolerance for filesystem timestamp precision.
    return modified_at + 1.0 >= float(attempt_started)


def _agent2_model_call_counts(
    path: Path,
    *,
    attempt_started: float | None,
) -> tuple[int, int]:
    """
    Read Notebook 06/06B/06C's local model-call audit file.

    This does not make a provider request. It only reads an artifact the
    notebook already writes, so live progress adds zero model/API hits.
    """
    if not _agent2_current_attempt_artifact(path, attempt_started):
        return 0, 0

    payload = _load_json(path)

    raw_calls = payload.get('calls')
    if not isinstance(raw_calls, list):
        raw_calls = payload.get('model_call_usage')
    if not isinstance(raw_calls, list):
        raw_calls = []

    completed_calls = 0
    for row in raw_calls:
        if not isinstance(row, dict):
            continue

        call_status = str(row.get('status') or '').strip().upper()
        has_usage = any(
            row.get(key) not in (None, '', 0)
            for key in (
                'actual_total_tokens',
                'actual_output_tokens',
                'actual_input_tokens',
            )
        )

        if (
            call_status
            in {
                'API_RESPONSE_RECEIVED',
                'COMPLETED',
                'COMPLETE',
                'SUCCESS',
                'SUCCEEDED',
                'OK',
            }
            or has_usage
        ):
            completed_calls += 1

    raw_hits = payload.get('api_hit_count')
    if raw_hits is None:
        raw_hits = payload.get('api_hits')

    if isinstance(raw_hits, list):
        api_hits = len(raw_hits)
    else:
        try:
            api_hits = int(raw_hits or 0)
        except (TypeError, ValueError):
            api_hits = 0

    return completed_calls, api_hits


def _agent2_live_progress(
    run_dir: Path,
    status: dict[str, Any],
) -> dict[str, Any]:
    """
    Derive live Complete Quiz progress from current-run local artifacts.

    LangGraph/MCP runs synchronously inside the background task, so the
    persisted status used to sit at 25% until the whole call returned. Plan C
    already writes generation, model-usage, quality-review and finalisation
    artifacts while it works. Mapping those existing artifacts to display
    milestones gives real movement without fake timers or extra model calls.

    Completion/failure is still controlled only by the persisted workflow
    status. This helper is a display overlay for queued/running Complete Quiz.
    """
    current_state = str(status.get('status') or '').strip().casefold()
    current_mode = str(status.get('mode') or '').strip()

    if current_state not in {'queued', 'running'} or current_mode != 'complete_quiz':
        return {}

    attempt_started = _parse_agent2_iso_epoch(status.get('started_at_utc'))
    output_dir = Path(run_dir) / 'output' / 'agent2_quiz' / 'complete_quiz'

    try:
        base_progress = int(round(float(status.get('progress') or 0)))
    except (TypeError, ValueError):
        base_progress = 0

    best: dict[str, Any] = {
        'progress': max(0, min(99, base_progress)),
        'stage': str(status.get('stage') or 'generating'),
        'message': str(
            status.get('message')
            or 'Generating the complete AQA-aligned quiz.'
        ),
        'progress_source': 'workflow_status',
        'progress_detail': None,
    }

    def advance(
        progress: int,
        *,
        stage: str,
        message: str,
        detail: str | None = None,
    ) -> None:
        nonlocal best
        clean_progress = max(0, min(99, int(progress)))
        if clean_progress <= int(best.get('progress') or 0):
            return
        best = {
            'progress': clean_progress,
            'stage': stage,
            'message': message,
            'progress_source': 'current_run_artifacts',
            'progress_detail': detail,
        }

    if _agent2_current_attempt_artifact(
        output_dir / 'generation_request.json',
        attempt_started,
    ):
        advance(
            28,
            stage='planning',
            message='Preparing the quiz blueprint and generation constraints.',
            detail='Current-run generation request prepared.',
        )

    if _agent2_current_attempt_artifact(
        output_dir / 'paper_routing_preflight.json',
        attempt_started,
    ):
        advance(
            34,
            stage='planning',
            message='Validating paper routing and the quiz generation plan.',
            detail='Paper-routing preflight completed.',
        )

    # Plan C deliberately has a variable number of model calls because of
    # adaptive batching. We therefore use completed responses as milestones,
    # never as a fixed denominator that could encourage extra API calls.
    usage_path = output_dir / 'model_call_usage.json'
    if _agent2_current_attempt_artifact(usage_path, attempt_started):
        completed_calls, api_hits = _agent2_model_call_counts(
            usage_path,
            attempt_started=attempt_started,
        )
        observed_calls = max(completed_calls, api_hits)

        if observed_calls >= 3:
            call_progress = 66
        elif observed_calls == 2:
            call_progress = 58
        elif observed_calls == 1:
            call_progress = 48
        else:
            call_progress = 40

        if completed_calls > 0:
            call_label = (
                f'{completed_calls} model response'
                f"{'' if completed_calls == 1 else 's'} completed"
            )
        elif api_hits > 0:
            call_label = (
                f'{api_hits} model call'
                f"{'' if api_hits == 1 else 's'} recorded"
            )
        else:
            call_label = 'Model-call audit initialised'

        advance(
            call_progress,
            stage='generating',
            message='Generating AQA-aligned questions and marking schemes.',
            detail=call_label + '.',
        )

    if _agent2_current_attempt_artifact(
        output_dir / 'hybrid_batching_diagnostics.json',
        attempt_started,
    ):
        advance(
            68,
            stage='generating',
            message='Generation batches are complete; preparing quality checks.',
            detail='Adaptive batching diagnostics produced.',
        )

    if _agent2_current_attempt_artifact(
        output_dir / 'semantic_quality_review.json',
        attempt_started,
    ):
        advance(
            76,
            stage='quality_checks',
            message='Checking generated questions against lesson evidence and AQA scope.',
            detail='Semantic quality review produced.',
        )

    if _agent2_current_attempt_artifact(
        output_dir / 'generated_quality_human_review.json',
        attempt_started,
    ):
        advance(
            84,
            stage='quality_checks',
            message='Applying generated-question quality gates.',
            detail='Generated quality review produced.',
        )

    if _agent2_current_attempt_artifact(
        output_dir / 'generated_human_review_queue.json',
        attempt_started,
    ):
        advance(
            88,
            stage='preparing_review',
            message='Preparing candidate questions for human review.',
            detail='Human-review queue produced.',
        )

    if _agent2_current_attempt_artifact(
        output_dir / 'quiz_generation_report.txt',
        attempt_started,
    ):
        advance(
            90,
            stage='finalising',
            message='Finalising the assessment package.',
            detail='Quiz generation report produced.',
        )

    manifest_candidates = [
        output_dir / 'mcp_visuals' / 'final_quiz_manifest_with_mcp_visuals.json',
        output_dir / 'final_quiz_manifest.json',
    ]
    if any(
        _agent2_current_attempt_artifact(path, attempt_started)
        for path in manifest_candidates
    ):
        advance(
            94,
            stage='finalising',
            message='Final quiz manifest is ready; assembling output files.',
            detail='Final quiz manifest produced.',
        )

    pdf_candidates = [
        output_dir / 'Agent2_Quiz_Output_Questions_and_Marking_Schemes.pdf',
        output_dir / 'mcp_visuals' / 'Agent2_Quiz_Output_Questions_and_Marking_Schemes.pdf',
    ]
    if any(
        _agent2_current_attempt_artifact(path, attempt_started)
        for path in pdf_candidates
    ):
        advance(
            97,
            stage='finalising',
            message='Assessment files are generated; completing the workflow.',
            detail='Student-facing PDF produced.',
        )

    return best


def _write_assessment_status(
    run_dir: Path,
    *,
    status: str,
    stage: str,
    progress: int,
    message: str,
    mode: str | None = None,
    request: dict[str, Any] | None = None,
    shortfall: dict[str, Any] | None = None,
    error: str | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    path = _assessment_status_path(run_dir)
    existing = {} if reset else _load_json(path)
    now = datetime.now(timezone.utc).isoformat()

    payload: dict[str, Any] = {
        **existing,
        "status": str(status),
        "stage": str(stage),
        "progress": max(0, min(100, int(progress))),
        "message": str(message),
        "updated_at_utc": now,
        "error": error,
    }

    try:
        _record_previous_agent2_stage_timing(
            run_dir=run_dir,
            existing=existing,
            next_stage=str(stage),
            next_status=str(status),
            now_iso=now,
        )
    except Exception:
        # ETA telemetry must never break Agent 2.
        pass

    previous_stage = str(existing.get("stage") or "").strip()
    if previous_stage != str(stage).strip() or not existing.get("stage_started_at_utc"):
        payload["stage_started_at_utc"] = now

    if not existing.get("started_at_utc") and status in {"queued", "running"}:
        payload["started_at_utc"] = now
    if status in {"complete", "failed"}:
        payload["completed_at_utc"] = now
    if mode is not None:
        payload["mode"] = str(mode)
    if request is not None:
        payload["request"] = request
    if shortfall is not None:
        payload["shortfall"] = shortfall

    _write_json(path, payload)
    return payload


def _approved_topics_path(run_dir: Path) -> Path:
    return (
        Path(run_dir)
        / "output"
        / "integration"
        / "approved_topics.json"
    )


def _approved_topics_payload(run_dir: Path) -> dict[str, Any]:
    path = _approved_topics_path(run_dir)
    if not path.is_file():
        raise HTTPException(
            status_code=409,
            detail="Approve the Agent 1 topics for Agent 2 first.",
        )

    payload = _load_json(path)
    topics = payload.get("topics")
    if not isinstance(topics, list) or not topics:
        raise HTTPException(
            status_code=409,
            detail="The approved Agent 2 topic handoff contains no topics.",
        )
    return payload


def _quiz_model_config() -> dict[str, Any]:
    path = AGENT2_ROOT / "config" / "quiz_model_config.json"
    if not path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Quiz model config was not found: {path}",
        )

    payload = _load_json(path)
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise HTTPException(
            status_code=500,
            detail="quiz_model_config.json must contain a non-empty 'models' object.",
        )
    return payload


def _quiz_generation_notebook(option_key: str) -> Path | None:
    option = QUIZ_GENERATION_NOTEBOOK_OPTIONS.get(str(option_key))
    if not isinstance(option, dict):
        return None

    filenames = [
        str(value).strip()
        for value in (option.get("filenames") or [])
        if str(value).strip()
    ]

    for folder in [AGENT2_ROOT / "Notebooks", AGENT2_ROOT / "notebooks", AGENT2_ROOT]:
        for filename in filenames:
            candidate = folder / filename
            if candidate.is_file():
                return candidate.resolve()
    return None


def _save_quiz_model_selection(
    *,
    run_dir: Path,
    model_key: str,
    model_config: dict[str, Any],
) -> Path:
    models = model_config.get("models") or {}
    selected = models.get(model_key)
    if not isinstance(selected, dict):
        raise ValueError(f"Unknown quiz model key: {model_key}")

    path = (
        Path(run_dir)
        / "output"
        / "integration"
        / "quiz_model_selection.json"
    )

    _write_json(
        path,
        {
            "schema_version": "agent2-quiz-model-selection-v1.0.0",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_key": model_key,
            "display_name": selected.get("display_name"),
            "provider": selected.get("provider"),
            "model_id": selected.get("model_id"),
            "context_window_tokens": selected.get("context_window_tokens"),
            "hard_max_output_tokens": selected.get("hard_max_output_tokens"),
            "source": "nextjs_quiz_model_selector",
        },
    )
    return path


def _save_quiz_notebook_selection(
    *,
    run_dir: Path,
    option_key: str,
    notebook_path: Path,
) -> Path:
    option = QUIZ_GENERATION_NOTEBOOK_OPTIONS.get(str(option_key), {})
    path = (
        Path(run_dir)
        / "output"
        / "integration"
        / "quiz_generation_notebook_selection.json"
    )
    _write_json(
        path,
        {
            "schema_version": "agent2-quiz-notebook-selection-v1.0.0",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "option_key": str(option_key),
            "label": str(option.get("label") or option_key),
            "strategy": str(option.get("strategy") or option_key),
            "notebook_path": str(Path(notebook_path).resolve()),
            "source": "nextjs_quiz_plan_selector",
        },
    )

    # Keep the same compatibility signals used by the existing controller.
    os.environ["AGENT2_QUIZ_NOTEBOOK_PATH"] = str(Path(notebook_path).resolve())
    os.environ["AGENT2_QUIZ_GENERATION_STRATEGY"] = str(
        option.get("strategy") or option_key
    )
    return path


def _resolve_existing_file(
    value: Any,
    *,
    output_dir: Path | None = None,
) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None

    path = Path(raw)
    if path.is_file():
        return path.resolve()

    if output_dir is not None and not path.is_absolute():
        candidate = Path(output_dir) / path
        if candidate.is_file():
            return candidate.resolve()

    return None


def _current_official_package(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    output_dir = Path(run_dir) / "output" / "agent2"
    manifest = _load_json(output_dir / "agent2_execution_manifest.json")
    package_path = _resolve_existing_file(
        manifest.get("package_path"),
        output_dir=output_dir,
    )

    if package_path is None:
        current = _load_json(output_dir / "agent2_current_run.json")
        package_path = _resolve_existing_file(
            current.get("assessment_package_json"),
            output_dir=output_dir,
        )

    if package_path is None and output_dir.is_dir():
        candidates = sorted(
            output_dir.glob("agent2_assessment_package_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        package_path = candidates[0].resolve() if candidates else None

    return package_path, _load_json(package_path) if package_path else {}


def _agent2_official_shortfall(package: dict[str, Any]) -> dict[str, Any]:
    request = package.get("assessment_request") or {}
    if not isinstance(request, dict):
        request = {}

    summary = package.get("retrieval_summary") or {}
    if not isinstance(summary, dict):
        summary = {}

    questions = package.get("questions") or []
    if not isinstance(questions, list):
        questions = []

    def as_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    target_marks = as_int(request.get("target_total_marks"))
    target_questions = as_int(request.get("number_of_questions"))
    selected_questions = as_int(summary.get("selected_questions"), len(questions))

    selected_marks_raw = summary.get("selected_marks")
    if selected_marks_raw is None:
        selected_marks = 0
        for item in questions:
            if not isinstance(item, dict):
                continue
            question = item.get("question") or {}
            if not isinstance(question, dict):
                question = {}
            selected_marks += as_int(
                question.get("marks")
                or item.get("marks_numeric")
                or item.get("marks_postgres")
                or item.get("marks_retrieval")
                or item.get("maximum_marks")
            )
    else:
        selected_marks = as_int(selected_marks_raw)

    checks = [
        bool(summary.get("requested_question_count_met", selected_questions >= target_questions)),
        bool(summary.get("coverage_requirements_met", True)),
        bool(summary.get("primary_requirement_met", True)),
        bool(summary.get("supporting_requirement_met", True)),
        bool(summary.get("distinct_reference_requirement_met", True)),
    ]
    if target_marks > 0:
        checks.append(selected_marks >= target_marks)

    sufficient = all(checks)
    return {
        "sufficient": bool(sufficient),
        "missing_marks": max(0, target_marks - selected_marks),
        "missing_questions": max(0, target_questions - selected_questions),
        "selected_marks": selected_marks,
        "selected_questions": selected_questions,
        "target_marks": target_marks,
        "target_questions": target_questions,
        # Backward-compatible alias used by the Next.js assessment UI.
        "requested_questions": target_questions,
    }


# ============================================================
# Agent 2 official-retrieval HITL memory
# ============================================================

AGENT2_RETRIEVAL_HITL_PHASE2_VERSION = "agent2-retrieval-hitl-phase2-v1.1.0"
AGENT2_RETRIEVAL_HITL_PHASE4_VERSION = "agent2-retrieval-hitl-phase4-bounded-ranking-v1.0.0"
AGENT2_RETRIEVAL_MEMORY_SCHEMA_VERSION = "agent2-retrieval-memory-v1.0.0"
AGENT2_RETRIEVAL_MEMORY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE = 384


def _normalize_database_url(url: str) -> str:
    url = str(url or "").strip()
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def _agent2_retrieval_feedback_engine() -> Any:
    database_url = str(
        os.getenv("AGENT2_DATABASE_URL", "")
        or os.getenv("DATABASE_URL", "")
        or ""
    ).strip()
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL / AGENT2_DATABASE_URL is not configured for Agent 2 retrieval HITL feedback."
        )
    return create_engine(
        _normalize_database_url(database_url),
        pool_pre_ping=True,
        future=True,
    )


def _ensure_agent2_retrieval_feedback_table(engine: Any) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS agent2_retrieval_feedback (
            id BIGSERIAL PRIMARY KEY,
            feedback_event_id UUID UNIQUE NOT NULL,
            package_fingerprint TEXT NOT NULL,
            pipeline_run_id TEXT NOT NULL,
            package_generated_at_utc TIMESTAMPTZ,
            retrieval_version TEXT,
            question_id TEXT NOT NULL,
            selected_rank INTEGER,
            agent1_topic_index INTEGER,
            concept_id TEXT,
            detected_topic TEXT,
            official_reference TEXT,
            agent1_role TEXT,
            transcript_evidence TEXT,
            transcript_evidence_source TEXT,
            semantic_score DOUBLE PRECISION,
            base_final_score DOUBLE PRECISION,
            retrieval_stage TEXT,
            query_evidence_source TEXT,
            paper_code TEXT,
            question_number TEXT,
            question_marks INTEGER,
            question_text TEXT,
            decision TEXT NOT NULL CHECK (decision IN ('relevant', 'not_relevant')),
            reason TEXT,
            reviewed_by TEXT NOT NULL DEFAULT 'nextjs_human_ui',
            phase_version TEXT NOT NULL,
            memory_eligible BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_status TEXT NOT NULL DEFAULT 'not_indexed'",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_key TEXT",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_point_id UUID",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_context_hash TEXT",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_text TEXT",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_collection TEXT",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_embedding_model TEXT",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_vector_size INTEGER",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_phase_version TEXT",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_promoted_at TIMESTAMPTZ",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_error TEXT",
        "ALTER TABLE agent2_retrieval_feedback ADD COLUMN IF NOT EXISTS memory_superseded_by_feedback_id BIGINT",
        "CREATE INDEX IF NOT EXISTS ix_agent2_retrieval_feedback_package ON agent2_retrieval_feedback (package_fingerprint, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS ix_agent2_retrieval_feedback_run_question ON agent2_retrieval_feedback (pipeline_run_id, question_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS ix_agent2_retrieval_feedback_memory_status ON agent2_retrieval_feedback (memory_status, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS ix_agent2_retrieval_feedback_memory_key ON agent2_retrieval_feedback (memory_key, created_at DESC)",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _agent2_retrieval_package_fingerprint(package: dict[str, Any]) -> str:
    question_ids = [
        str(item.get("question_id") or "").strip()
        for item in (package.get("questions") or [])
        if isinstance(item, dict)
    ]
    payload = {
        "generated_at_utc": str(package.get("generated_at_utc") or ""),
        "retrieval_version": str(package.get("retrieval_version") or ""),
        "question_ids": question_ids,
        "request": package.get("request") or package.get("assessment_request") or {},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _agent2_retrieval_topic_context(*, run_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    topic = item.get("topic") or {}
    retrieval = item.get("retrieval") or {}
    if not isinstance(topic, dict):
        topic = {}
    if not isinstance(retrieval, dict):
        retrieval = {}

    context = {
        "agent1_topic_index": retrieval.get("agent1_topic_index"),
        "concept_id": None,
        "transcript_evidence": str(retrieval.get("query_evidence") or "").strip(),
        "transcript_evidence_source": str(retrieval.get("query_evidence_source") or "").strip(),
    }

    try:
        approved_payload = _approved_topics_payload(run_dir)
        approved_topics = approved_payload.get("topics") or []
    except Exception:
        approved_topics = []

    detected_topic = str(topic.get("detected_topic") or topic.get("topic") or "").strip().casefold()
    official_reference = str(topic.get("official_reference") or "").strip()

    for approved in approved_topics:
        if not isinstance(approved, dict):
            continue
        approved_topic = str(approved.get("topic") or approved.get("detected_topic") or "").strip().casefold()
        approved_reference = str(approved.get("official_reference") or "").strip()
        if approved_reference == official_reference and (not detected_topic or approved_topic == detected_topic):
            context["concept_id"] = approved.get("concept_id")
            if context["agent1_topic_index"] is None:
                context["agent1_topic_index"] = approved.get("topic_index")
            if not context["transcript_evidence"]:
                evidence_parts = [
                    str(value).strip()
                    for value in (approved.get("source_chunk_texts") or [])
                    if str(value).strip()
                ]
                context["transcript_evidence"] = "\n\n".join(evidence_parts)
                if evidence_parts:
                    context["transcript_evidence_source"] = "approved_topics_source_chunk_texts"
            break

    return context


def _latest_agent2_retrieval_feedback(*, run_dir: Path, package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    questions = [
        item for item in (package.get("questions") or [])
        if isinstance(item, dict) and str(item.get("question_id") or "").strip()
    ]
    if not questions:
        return {}

    try:
        engine = _agent2_retrieval_feedback_engine()
        _ensure_agent2_retrieval_feedback_table(engine)
    except Exception:
        return {}

    fingerprint = _agent2_retrieval_package_fingerprint(package)
    latest: dict[str, dict[str, Any]] = {}
    with engine.connect() as connection:
        for item in questions:
            question_id = str(item.get("question_id") or "").strip()
            row = connection.execute(
                text(
                    """
                    SELECT *
                    FROM agent2_retrieval_feedback
                    WHERE question_id = :question_id
                      AND package_fingerprint = :package_fingerprint
                      AND pipeline_run_id = :pipeline_run_id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                ),
                {
                    "question_id": question_id,
                    "package_fingerprint": fingerprint,
                    "pipeline_run_id": Path(run_dir).name,
                },
            ).mappings().first()
            if row is not None:
                latest[question_id] = dict(row)
    return latest


def _agent2_retrieval_memory_collection_name() -> str:
    explicit = str(os.getenv("AGENT2_RETRIEVAL_MEMORY_COLLECTION", "") or "").strip()
    if explicit:
        return explicit
    agent1_collection = str(
        os.getenv("QDRANT_COLLECTION", "aqa_gcse_computer_science_8525")
        or "aqa_gcse_computer_science_8525"
    ).strip()
    question_collection = str(os.getenv("AGENT2_QDRANT_COLLECTION", "") or "").strip() or f"{agent1_collection}_questions"
    return f"{question_collection}_retrieval_memory"


@lru_cache(maxsize=1)
def _agent2_retrieval_memory_model() -> Any:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(AGENT2_RETRIEVAL_MEMORY_MODEL)


def _agent2_retrieval_memory_client() -> tuple[Any, Any, str]:
    from qdrant_client import QdrantClient, models

    qdrant_url = str(os.getenv("QDRANT_URL", "http://localhost:6333") or "http://localhost:6333").strip()
    qdrant_api_key = str(os.getenv("QDRANT_API_KEY", "") or "").strip() or None
    qdrant_timeout = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "30") or 30)
    prefer_grpc = str(os.getenv("QDRANT_PREFER_GRPC", "false") or "false").strip().casefold() in {"1", "true", "yes"}
    kwargs: dict[str, Any] = {
        "url": qdrant_url,
        "timeout": qdrant_timeout,
        "prefer_grpc": prefer_grpc,
    }
    if qdrant_api_key is not None:
        kwargs["api_key"] = qdrant_api_key
    client = QdrantClient(**kwargs)
    return client, models, _agent2_retrieval_memory_collection_name()


def _ensure_agent2_retrieval_memory_collection() -> tuple[Any, Any, str]:
    client, models, collection_name = _agent2_retrieval_memory_client()
    try:
        exists = bool(client.collection_exists(collection_name))
    except AttributeError:
        try:
            client.get_collection(collection_name=collection_name)
            exists = True
        except Exception:
            exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
    else:
        info = client.get_collection(collection_name=collection_name)
        vectors_config = info.config.params.vectors
        vector_size = getattr(vectors_config, "size", None)
        if vector_size is None and isinstance(vectors_config, dict):
            unnamed = vectors_config.get("") or vectors_config.get("default")
            vector_size = getattr(unnamed, "size", None)
        if int(vector_size or 0) != AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE:
            raise RuntimeError(
                f"Existing retrieval-memory Qdrant collection has vector size {vector_size}; expected {AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE}."
            )

    for field_name in [
        "concept_id", "official_reference", "agent1_role", "decision", "question_id", "memory_context_hash"
    ]:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception:
            pass
    return client, models, collection_name


def _agent2_retrieval_memory_material(row: dict[str, Any]) -> dict[str, Any]:
    evidence = re.sub(r"\s+", " ", str(row.get("transcript_evidence") or "")).strip()
    if not evidence:
        raise ValueError(
            "Retrieval memory was not indexed because transcript/lesson evidence is missing."
        )
    evidence_hash = hashlib.sha256(evidence.casefold().encode("utf-8")).hexdigest()
    spec_version = str(os.getenv("AQA_SPEC_VERSION", "") or "AQA-8525-v1.2-2022-11-29").strip()
    identity = {
        "schema": AGENT2_RETRIEVAL_MEMORY_SCHEMA_VERSION,
        "spec_version": spec_version,
        "concept_id": str(row.get("concept_id") or "").strip(),
        "detected_topic": str(row.get("detected_topic") or "").strip().casefold(),
        "official_reference": str(row.get("official_reference") or "").strip(),
        "agent1_role": str(row.get("agent1_role") or "").strip().casefold(),
        "lesson_evidence_hash": evidence_hash,
        "question_id": str(row.get("question_id") or "").strip(),
    }
    memory_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    point_id = uuid.uuid5(uuid.NAMESPACE_URL, f"agent2-retrieval-memory:{memory_key}")
    memory_text = "\n".join([
        "Agent 2 retrieval feedback memory",
        f"Memory schema: {AGENT2_RETRIEVAL_MEMORY_SCHEMA_VERSION}",
        f"AQA specification: {spec_version}",
        f"Approved concept ID: {str(row.get('concept_id') or '').strip() or 'unknown'}",
        f"Approved lesson topic: {str(row.get('detected_topic') or '').strip()}",
        f"Official reference: {str(row.get('official_reference') or '').strip()}",
        f"Topic role: {str(row.get('agent1_role') or '').strip()}",
        "Lesson evidence:",
        str(row.get("transcript_evidence") or "").strip(),
        "Retrieved official question:",
        str(row.get("question_text") or "").strip(),
        f"Paper: {str(row.get('paper_code') or '').strip()}",
        f"Question number: {str(row.get('question_number') or '').strip()}",
        f"Marks: {row.get('question_marks') if row.get('question_marks') is not None else ''}",
        f"Human decision: {str(row.get('decision') or '').strip()}",
        "Human reason:",
        str(row.get("reason") or "").strip() or "No additional reason supplied.",
    ]).strip()
    return {
        "memory_key": memory_key,
        "point_id": point_id,
        "memory_text": memory_text,
        "context_hash": hashlib.sha256(memory_text.encode("utf-8")).hexdigest(),
        "evidence_hash": evidence_hash,
        "spec_version": spec_version,
    }


def _agent2_retrieval_feedback_row(feedback_id: int) -> dict[str, Any]:
    engine = _agent2_retrieval_feedback_engine()
    _ensure_agent2_retrieval_feedback_table(engine)
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT * FROM agent2_retrieval_feedback WHERE id = :feedback_id"),
            {"feedback_id": int(feedback_id)},
        ).mappings().first()
    if row is None:
        raise ValueError(f"Retrieval feedback row {feedback_id} was not found.")
    return dict(row)


def _promote_agent2_retrieval_feedback_to_qdrant(feedback_id: int) -> dict[str, Any]:
    row = _agent2_retrieval_feedback_row(feedback_id)
    material = _agent2_retrieval_memory_material(row)
    vector = _agent2_retrieval_memory_model().encode(
        [material["memory_text"]],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    if int(len(vector)) != AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE:
        raise RuntimeError(
            f"Retrieval-memory embedding has vector size {len(vector)}; expected {AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE}."
        )
    client, models, collection_name = _ensure_agent2_retrieval_memory_collection()
    payload = {
        "memory_schema_version": AGENT2_RETRIEVAL_MEMORY_SCHEMA_VERSION,
        "memory_phase_version": AGENT2_RETRIEVAL_HITL_PHASE2_VERSION,
        "memory_key": material["memory_key"],
        "memory_context_hash": material["context_hash"],
        "lesson_evidence_hash": material["evidence_hash"],
        "spec_version": material["spec_version"],
        "feedback_id": int(row["id"]),
        "feedback_event_id": str(row.get("feedback_event_id") or ""),
        "pipeline_run_id": str(row.get("pipeline_run_id") or ""),
        "question_id": str(row.get("question_id") or ""),
        "agent1_topic_index": row.get("agent1_topic_index"),
        "concept_id": str(row.get("concept_id") or ""),
        "detected_topic": str(row.get("detected_topic") or ""),
        "official_reference": str(row.get("official_reference") or ""),
        "agent1_role": str(row.get("agent1_role") or ""),
        "decision": str(row.get("decision") or ""),
        "reason": str(row.get("reason") or ""),
        "question_text": str(row.get("question_text") or ""),
        "question_number": str(row.get("question_number") or ""),
        "question_marks": row.get("question_marks"),
        "paper_code": str(row.get("paper_code") or ""),
        "semantic_score_at_feedback": row.get("semantic_score"),
        "base_final_score_at_feedback": row.get("base_final_score"),
        "reviewed_by": str(row.get("reviewed_by") or ""),
        "feedback_created_at": str(row.get("created_at") or ""),
        "embedding_model": AGENT2_RETRIEVAL_MEMORY_MODEL,
        "vector_size": AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE,
        "ranking_adjustment_enabled": True,
        "ranking_policy_version": AGENT2_RETRIEVAL_HITL_PHASE4_VERSION,
    }
    client.upsert(
        collection_name=collection_name,
        points=[models.PointStruct(id=str(material["point_id"]), vector=vector.tolist(), payload=payload)],
        wait=True,
    )

    engine = _agent2_retrieval_feedback_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE agent2_retrieval_feedback
                SET memory_eligible = FALSE,
                    memory_status = 'superseded',
                    memory_superseded_by_feedback_id = :feedback_id
                WHERE memory_key = :memory_key
                  AND id <> :feedback_id
                  AND memory_status = 'indexed'
                """
            ),
            {"feedback_id": int(row["id"]), "memory_key": material["memory_key"]},
        )
        connection.execute(
            text(
                """
                UPDATE agent2_retrieval_feedback
                SET memory_eligible = TRUE,
                    memory_status = 'indexed',
                    memory_key = :memory_key,
                    memory_point_id = CAST(:memory_point_id AS UUID),
                    memory_context_hash = :memory_context_hash,
                    memory_text = :memory_text,
                    memory_collection = :memory_collection,
                    memory_embedding_model = :memory_embedding_model,
                    memory_vector_size = :memory_vector_size,
                    memory_phase_version = :memory_phase_version,
                    memory_promoted_at = NOW(),
                    memory_error = NULL,
                    memory_superseded_by_feedback_id = NULL
                WHERE id = :feedback_id
                """
            ),
            {
                "feedback_id": int(row["id"]),
                "memory_key": material["memory_key"],
                "memory_point_id": str(material["point_id"]),
                "memory_context_hash": material["context_hash"],
                "memory_text": material["memory_text"],
                "memory_collection": collection_name,
                "memory_embedding_model": AGENT2_RETRIEVAL_MEMORY_MODEL,
                "memory_vector_size": AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE,
                "memory_phase_version": AGENT2_RETRIEVAL_HITL_PHASE2_VERSION,
            },
        )
    return {
        "status": "indexed",
        "qdrant_written": True,
        "feedback_id": int(row["id"]),
        "memory_collection": collection_name,
    }


def _persist_agent2_retrieval_feedback(
    *,
    run_dir: Path,
    package: dict[str, Any],
    item: dict[str, Any],
    decision: str,
    reason: str,
) -> dict[str, Any]:
    normalized_decision = str(decision or "").strip().casefold()
    if normalized_decision not in {"relevant", "not_relevant"}:
        raise ValueError("Decision must be Relevant or Not Relevant.")
    reason_text = str(reason or "").strip()
    if normalized_decision == "not_relevant" and not reason_text:
        raise ValueError("A reason is required for Not Relevant feedback.")

    topic = item.get("topic") or {}
    question = item.get("question") or {}
    retrieval = item.get("retrieval") or {}
    if not isinstance(topic, dict):
        topic = {}
    if not isinstance(question, dict):
        question = {}
    if not isinstance(retrieval, dict):
        retrieval = {}
    context = _agent2_retrieval_topic_context(run_dir=run_dir, item=item)
    fingerprint = _agent2_retrieval_package_fingerprint(package)

    def optional_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def optional_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    engine = _agent2_retrieval_feedback_engine()
    _ensure_agent2_retrieval_feedback_table(engine)
    with engine.begin() as connection:
        saved = connection.execute(
            text(
                """
                INSERT INTO agent2_retrieval_feedback (
                    feedback_event_id, package_fingerprint, pipeline_run_id,
                    package_generated_at_utc, retrieval_version, question_id,
                    selected_rank, agent1_topic_index, concept_id, detected_topic,
                    official_reference, agent1_role, transcript_evidence,
                    transcript_evidence_source, semantic_score, base_final_score,
                    retrieval_stage, query_evidence_source, paper_code,
                    question_number, question_marks, question_text, decision,
                    reason, reviewed_by, phase_version, memory_eligible, memory_status
                ) VALUES (
                    CAST(:feedback_event_id AS UUID), :package_fingerprint, :pipeline_run_id,
                    CAST(NULLIF(:package_generated_at_utc, '') AS TIMESTAMPTZ),
                    :retrieval_version, :question_id, :selected_rank, :agent1_topic_index,
                    :concept_id, :detected_topic, :official_reference, :agent1_role,
                    :transcript_evidence, :transcript_evidence_source, :semantic_score,
                    :base_final_score, :retrieval_stage, :query_evidence_source,
                    :paper_code, :question_number, :question_marks, :question_text,
                    :decision, :reason, 'nextjs_human_ui', :phase_version, FALSE, 'pending_index'
                ) RETURNING id
                """
            ),
            {
                "feedback_event_id": str(uuid.uuid4()),
                "package_fingerprint": fingerprint,
                "pipeline_run_id": Path(run_dir).name,
                "package_generated_at_utc": str(package.get("generated_at_utc") or "").strip(),
                "retrieval_version": str(package.get("retrieval_version") or "").strip() or None,
                "question_id": str(item.get("question_id") or question.get("question_id") or question.get("id") or "").strip(),
                "selected_rank": optional_int(item.get("rank") or retrieval.get("selected_rank")),
                "agent1_topic_index": optional_int(context.get("agent1_topic_index")),
                "concept_id": str(context.get("concept_id") or "").strip() or None,
                "detected_topic": str(topic.get("detected_topic") or topic.get("topic") or "").strip() or None,
                "official_reference": str(topic.get("official_reference") or "").strip() or None,
                "agent1_role": str(topic.get("role") or topic.get("topic_role") or "").strip().casefold() or None,
                "transcript_evidence": str(context.get("transcript_evidence") or "").strip() or None,
                "transcript_evidence_source": str(context.get("transcript_evidence_source") or "").strip() or None,
                "semantic_score": optional_float(retrieval.get("semantic_score")),
                "base_final_score": optional_float(retrieval.get("final_score")),
                "retrieval_stage": str(retrieval.get("stage") or "").strip() or None,
                "query_evidence_source": str(retrieval.get("query_evidence_source") or "").strip() or None,
                "paper_code": str(question.get("paper_code") or "").strip() or None,
                "question_number": str(question.get("number") or "").strip() or None,
                "question_marks": optional_int(question.get("marks")),
                "question_text": str(question.get("text") or question.get("question_text") or "").strip(),
                "decision": normalized_decision,
                "reason": reason_text or None,
                "phase_version": AGENT2_RETRIEVAL_HITL_PHASE2_VERSION,
            },
        ).mappings().one()
    feedback_id = int(saved["id"])
    result = {"status": "saved", "feedback_id": feedback_id, "qdrant_written": False}
    try:
        result.update(_promote_agent2_retrieval_feedback_to_qdrant(feedback_id))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE agent2_retrieval_feedback
                    SET memory_eligible = FALSE, memory_status = 'error', memory_error = :memory_error
                    WHERE id = :feedback_id
                    """
                ),
                {"feedback_id": feedback_id, "memory_error": error[:4000]},
            )
        result["memory_error"] = error
    return result


def _normalise_marking_guidance(value: Any) -> Any:
    if isinstance(value, (list, dict, str)):
        return value
    if value is None:
        return ""
    return str(value)


def _normalise_official_question(
    item: dict[str, Any],
    position: int,
) -> dict[str, Any]:
    question = item.get("question") or {}
    topic = item.get("topic") or {}
    retrieval = item.get("retrieval") or {}
    mark_scheme = item.get("mark_scheme") or {}

    if not isinstance(question, dict):
        question = {}
    if not isinstance(topic, dict):
        topic = {}
    if not isinstance(retrieval, dict):
        retrieval = {}
    if not isinstance(mark_scheme, dict):
        mark_scheme = {}

    image_paths = question.get("rendered_page_images") or item.get("rendered_page_images") or []
    if not isinstance(image_paths, list):
        image_paths = []

    try:
        marks = int(
            question.get("marks")
            or mark_scheme.get("maximum_marks")
            or 0
        )
    except (TypeError, ValueError):
        marks = 0

    return {
        "question_id": str(
            item.get("question_id")
            or question.get("question_id")
            or question.get("id")
            or f"OFFICIAL_{position:03d}"
        ),
        "source": "official",
        "topic": str(
            topic.get("detected_topic")
            or topic.get("topic")
            or ""
        ).strip(),
        "official_reference": str(topic.get("official_reference") or "").strip(),
        "role": str(topic.get("role") or topic.get("topic_role") or "").strip(),
        "marks": marks,
        "paper": str(question.get("paper_label") or question.get("paper_code") or "").strip(),
        "question_number": str(question.get("number") or "").strip(),
        "question_text": str(question.get("text") or question.get("question_text") or "").strip(),
        "context": str(question.get("context") or "").strip(),
        "marking_guidance": _normalise_marking_guidance(
            mark_scheme.get("raw_marking_guidance")
            or mark_scheme.get("marking_guidance")
            or ""
        ),
        "visual_paths": [str(value) for value in image_paths if str(value or "").strip()],
        "visual_type": "official_page_render" if image_paths else "",
        "visual_spec": {},
        "semantic_score": retrieval.get("semantic_score"),
    }


def _collect_generated_visual_paths(question: dict[str, Any]) -> list[str]:
    paths: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            lowered = value.strip().casefold()
            if lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".svg")):
                if value.strip() not in paths:
                    paths.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                add(item)

    for key in [
        "visual_path",
        "rendered_visual_path",
        "image_path",
        "final_visual_path",
        "rendered_page_images",
    ]:
        add(question.get(key))

    visual = question.get("visual") or {}
    if isinstance(visual, dict):
        for key in ["path", "visual_path", "image_path", "rendered_path", "final_path"]:
            add(visual.get(key))

    return paths


def _normalise_generated_question(
    question: dict[str, Any],
    position: int,
) -> dict[str, Any]:
    visual = question.get("visual") or {}
    if not isinstance(visual, dict):
        visual = {}
    visual_spec = visual.get("spec") or {}
    if not isinstance(visual_spec, dict):
        visual_spec = {}

    try:
        marks = int(question.get("marks") or 0)
    except (TypeError, ValueError):
        marks = 0

    try:
        plan_index = int(question.get("plan_index") or position)
    except (TypeError, ValueError):
        plan_index = position

    generated_question_id = str(
        question.get("generated_question_id")
        or question.get("question_id")
        or question.get("id")
        or f"GEN_{position:03d}"
    )

    return {
        "question_id": generated_question_id,
        "generated_question_id": generated_question_id,
        "plan_index": plan_index,
        "source": "ai_generated",
        "topic": str(question.get("topic") or question.get("detected_topic") or "").strip(),
        "official_reference": str(
            question.get("official_reference")
            or question.get("agent1_official_reference")
            or ""
        ).strip(),
        "role": str(question.get("role") or question.get("agent1_role") or "").strip(),
        "marks": marks,
        "paper": str(question.get("paper_label") or question.get("paper_code") or "").strip(),
        "question_number": str(question.get("question_number") or "").strip(),
        "question_text": str(
            question.get("question_text")
            or question.get("text")
            or ""
        ).strip(),
        "context": str(question.get("context") or "").strip(),
        "marking_guidance": _normalise_marking_guidance(question.get("marking_guidance") or []),
        "visual_paths": _collect_generated_visual_paths(question),
        "visual_type": str(visual.get("type") or question.get("visual_requirement") or "").strip(),
        "visual_spec": visual_spec,
        "semantic_score": None,
    }


def _quiz_output_dir(run_dir: Path, quiz_mode: str) -> Path:
    return Path(run_dir) / "output" / "agent2_quiz" / str(quiz_mode)


def _user_regeneration_state_path(run_dir: Path) -> Path:
    return (
        Path(run_dir)
        / "output"
        / "integration"
        / "user_regeneration_attempts.json"
    )


def _empty_user_regeneration_state() -> dict[str, Any]:
    return {
        "schema_version": USER_REGENERATION_STATE_SCHEMA_VERSION,
        "max_attempts_per_question": MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quiz_modes": {},
    }


def _load_user_regeneration_state(run_dir: Path) -> dict[str, Any]:
    path = _user_regeneration_state_path(run_dir)
    payload = _load_json(path)

    if not payload:
        return _empty_user_regeneration_state()

    quiz_modes = payload.get("quiz_modes")
    if not isinstance(quiz_modes, dict):
        payload["quiz_modes"] = {}

    payload["schema_version"] = USER_REGENERATION_STATE_SCHEMA_VERSION
    payload["max_attempts_per_question"] = (
        MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION
    )
    return payload


def _reset_user_regeneration_state(run_dir: Path) -> None:
    _write_json(
        _user_regeneration_state_path(run_dir),
        _empty_user_regeneration_state(),
    )


def _user_regeneration_question_record(
    *,
    state: dict[str, Any],
    quiz_mode: str,
    plan_index: int,
    create: bool = False,
) -> dict[str, Any]:
    quiz_modes = state.setdefault("quiz_modes", {})
    if not isinstance(quiz_modes, dict):
        quiz_modes = {}
        state["quiz_modes"] = quiz_modes

    mode_key = str(quiz_mode or "complete_quiz")
    mode_state = quiz_modes.get(mode_key)
    if not isinstance(mode_state, dict):
        mode_state = {}
        if create:
            quiz_modes[mode_key] = mode_state

    questions = mode_state.get("questions")
    if not isinstance(questions, dict):
        questions = {}
        if create:
            mode_state["questions"] = questions

    key = str(int(plan_index))
    record = questions.get(key)
    if not isinstance(record, dict):
        record = {
            "plan_index": int(plan_index),
            "used": 0,
            "history": [],
        }
        if create:
            questions[key] = record

    return record


def _user_regeneration_attempt_info(
    *,
    run_dir: Path,
    quiz_mode: str,
    plan_index: int,
) -> dict[str, Any]:
    state = _load_user_regeneration_state(run_dir)
    record = _user_regeneration_question_record(
        state=state,
        quiz_mode=quiz_mode,
        plan_index=int(plan_index),
        create=False,
    )

    try:
        used = max(0, int(record.get("used") or 0))
    except (TypeError, ValueError):
        used = 0

    used = min(
        MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION,
        used,
    )
    remaining = max(
        0,
        MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION - used,
    )

    return {
        "user_regeneration_attempts_used": used,
        "user_regeneration_attempts_remaining": remaining,
        "max_user_regeneration_attempts": (
            MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION
        ),
    }


def _current_generated_plan_indexes(
    *,
    run_dir: Path,
    quiz_mode: str,
    manifest: dict[str, Any] | None = None,
) -> list[int]:
    manifest = manifest if isinstance(manifest, dict) else _quiz_manifest(
        run_dir,
        quiz_mode,
    )

    raw_candidates = manifest.get("candidate_questions") or []
    use_candidates = isinstance(raw_candidates, list) and bool(raw_candidates)

    if use_candidates:
        rows = raw_candidates
    else:
        raw_questions = manifest.get("questions") or []
        rows = raw_questions if isinstance(raw_questions, list) else []

    indexes: list[int] = []

    for position, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            continue

        if not use_candidates:
            source_type = str(raw.get("source_type") or "").strip().casefold()
            generated_id = str(raw.get("generated_question_id") or "").strip()
            if source_type and source_type != "ai_generated_aqa_aligned" and not generated_id:
                continue

        try:
            plan_index = int(raw.get("plan_index") or position)
        except (TypeError, ValueError):
            plan_index = position

        if plan_index > 0 and plan_index not in indexes:
            indexes.append(plan_index)

    return sorted(indexes)


def _ensure_user_regeneration_allowed(
    *,
    run_dir: Path,
    quiz_mode: str,
    plan_indexes: list[int],
) -> dict[int, dict[str, Any]]:
    if not plan_indexes:
        raise HTTPException(
            status_code=409,
            detail="No generated question is available for user regeneration.",
        )

    result: dict[int, dict[str, Any]] = {}
    exhausted: list[int] = []

    for plan_index in sorted({int(value) for value in plan_indexes if int(value) > 0}):
        info = _user_regeneration_attempt_info(
            run_dir=run_dir,
            quiz_mode=quiz_mode,
            plan_index=plan_index,
        )
        result[plan_index] = info
        if int(info["user_regeneration_attempts_remaining"]) <= 0:
            exhausted.append(plan_index)

    if exhausted:
        if len(exhausted) == 1:
            detail = (
                f"Question {exhausted[0]} has already used both user regeneration "
                "attempts (2/2). Automatic system retries are separate and are "
                "not included in this limit."
            )
        else:
            joined = ", ".join(str(value) for value in exhausted)
            detail = (
                f"Whole-quiz regeneration is unavailable because question slots "
                f"{joined} have already used both user regeneration attempts (2/2). "
                "Automatic system retries are separate and are not included in "
                "this limit."
            )
        raise HTTPException(status_code=409, detail=detail)

    return result


def _record_successful_user_regeneration(
    *,
    run_dir: Path,
    quiz_mode: str,
    plan_indexes: list[int],
    scope: Literal["question", "whole_quiz"],
    reason: str,
    question_ids: dict[int, str] | None = None,
) -> dict[int, dict[str, Any]]:
    state = _load_user_regeneration_state(run_dir)
    event_id = str(uuid.uuid4())
    recorded_at = datetime.now(timezone.utc).isoformat()
    output: dict[int, dict[str, Any]] = {}

    unique_indexes = sorted({int(value) for value in plan_indexes if int(value) > 0})

    for plan_index in unique_indexes:
        record = _user_regeneration_question_record(
            state=state,
            quiz_mode=quiz_mode,
            plan_index=plan_index,
            create=True,
        )

        try:
            used = max(0, int(record.get("used") or 0))
        except (TypeError, ValueError):
            used = 0

        if used >= MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Question {plan_index} has already used both user regeneration "
                    "attempts (2/2)."
                ),
            )

        used += 1
        history = record.get("history")
        if not isinstance(history, list):
            history = []

        history.append(
            {
                "event_id": event_id,
                "scope": scope,
                "reason": str(reason or "").strip(),
                "question_id": str(
                    (question_ids or {}).get(plan_index) or ""
                ).strip(),
                "recorded_at_utc": recorded_at,
                "attempt_number": used,
            }
        )

        record.update(
            {
                "plan_index": plan_index,
                "used": used,
                "history": history,
                "updated_at_utc": recorded_at,
            }
        )

        output[plan_index] = {
            "user_regeneration_attempts_used": used,
            "user_regeneration_attempts_remaining": max(
                0,
                MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION - used,
            ),
            "max_user_regeneration_attempts": (
                MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION
            ),
        }

    state["schema_version"] = USER_REGENERATION_STATE_SCHEMA_VERSION
    state["max_attempts_per_question"] = MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION
    state["updated_at_utc"] = recorded_at
    _write_json(_user_regeneration_state_path(run_dir), state)
    return output


def _quiz_manifest_latest_mtime_ns(run_dir: Path, quiz_mode: str) -> int:
    output_dir = _quiz_output_dir(run_dir, quiz_mode)
    values: list[int] = []

    for path in [
        output_dir / "final_quiz_manifest.json",
        output_dir / "mcp_visuals" / "final_quiz_manifest_with_mcp_visuals.json",
    ]:
        try:
            if path.is_file():
                values.append(int(path.stat().st_mtime_ns))
        except OSError:
            continue

    return max(values) if values else 0


def _manifest_question_signatures(
    manifest: dict[str, Any],
) -> dict[int, str]:
    signatures: dict[int, str] = {}
    rows = manifest.get("candidate_questions") or []
    if not isinstance(rows, list) or not rows:
        rows = manifest.get("questions") or []
    if not isinstance(rows, list):
        rows = []

    for position, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            continue
        try:
            plan_index = int(raw.get("plan_index") or position)
        except (TypeError, ValueError):
            plan_index = position
        signature = _question_text_signature(raw)
        if plan_index > 0 and signature:
            signatures[plan_index] = signature

    return signatures


def _quiz_manifest(run_dir: Path, quiz_mode: str) -> dict[str, Any]:
    """
    Return the freshest valid quiz manifest.

    Initial MCP/Notebook 08 visual processing can create a newer augmented
    manifest, so that file normally wins after generation.  Question-level
    HITL regeneration/editing, however, can update final_quiz_manifest.json
    without immediately rebuilding the MCP visual manifest.  Always preferring
    the old MCP file therefore makes the UI show the pre-regeneration question.

    Choosing by modification time keeps the current candidate authoritative
    while still using the MCP-augmented manifest when it is genuinely newer.
    """
    output_dir = _quiz_output_dir(run_dir, quiz_mode)

    candidates = [
        output_dir / "final_quiz_manifest.json",
        (
            output_dir
            / "mcp_visuals"
            / "final_quiz_manifest_with_mcp_visuals.json"
        ),
    ]

    available: list[tuple[int, Path, dict[str, Any]]] = []

    for path in candidates:
        if not path.is_file():
            continue

        payload = _load_json(path)
        if not payload:
            continue

        try:
            modified_ns = int(path.stat().st_mtime_ns)
        except OSError:
            modified_ns = 0

        available.append((modified_ns, path, payload))

    if not available:
        return {}

    available.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return available[0][2]


def _manifest_pdf_paths(
    manifest: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    paths: list[str] = []

    def add(value: Any) -> None:
        resolved = _resolve_existing_file(value, output_dir=output_dir)
        if resolved is not None:
            text = str(resolved)
            if text not in paths:
                paths.append(text)

    output_files = manifest.get("output_files") or {}
    if isinstance(output_files, dict):
        add(output_files.get("questions_and_marking_schemes_pdf"))
        for key, value in output_files.items():
            if "pdf" in str(key).casefold():
                add(value)

    mcp_pdf = manifest.get("mcp_final_pdf") or {}
    if isinstance(mcp_pdf, dict):
        for key in ["path", "pdf_path", "output_path", "file"]:
            add(mcp_pdf.get(key))

    add(output_dir / "Agent2_Quiz_Output_Questions_and_Marking_Schemes.pdf")
    return paths


def _official_pdf_paths(run_dir: Path, package: dict[str, Any]) -> list[str]:
    output_dir = Path(run_dir) / "output" / "agent2"
    paths: list[Path] = []

    # Prefer Notebook 05's existing combined questions + answers PDF for the
    # standalone retrieval download. Hybrid assembly still uses the separate
    # student-question PDF directly inside Notebook 06C.
    output_files = package.get("output_files") or {}
    if isinstance(output_files, dict):
        combined_pdf = _resolve_existing_file(
            output_files.get("combined_questions_answers_audit_pdf"),
            output_dir=output_dir,
        )
        if combined_pdf is not None:
            paths.append(combined_pdf)

    # Keep the existing student-facing PDF as the normal fallback.
    if output_dir.is_dir():
        all_pdfs = sorted(
            output_dir.rglob("*.pdf"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        preferred = [
            path for path in all_pdfs
            if "student" in path.name.casefold()
            and "question" in path.name.casefold()
        ]
        if not preferred:
            preferred = [
                path for path in all_pdfs
                if "question" in path.name.casefold()
                or "assessment" in path.name.casefold()
            ]
        paths.extend(preferred[:3])

    # Some packages register explicit artifacts.
    for container_key in ["output_files", "artifacts", "source_artifacts"]:
        container = package.get(container_key) or {}
        if not isinstance(container, dict):
            continue
        for key, value in container.items():
            if "pdf" not in str(key).casefold():
                continue
            resolved = _resolve_existing_file(value, output_dir=output_dir)
            if resolved is not None and resolved not in paths:
                paths.append(resolved)

    return [str(path.resolve()) for path in paths]


def _generated_result(run_dir: Path, quiz_mode: str) -> dict[str, Any] | None:
    manifest = _quiz_manifest(run_dir, quiz_mode)
    if not manifest:
        return None

    raw_candidates = manifest.get("candidate_questions") or []
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    raw_questions = manifest.get("questions") or []
    if not isinstance(raw_questions, list):
        raw_questions = []

    candidate_questions = [
        _normalise_generated_question(item, index)
        for index, item in enumerate(raw_candidates, start=1)
        if isinstance(item, dict)
    ]

    accepted_questions: list[dict[str, Any]] = []
    for index, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            continue
        source_type = str(item.get("source_type") or "").strip().casefold()
        generated_id = str(item.get("generated_question_id") or "").strip()
        if source_type == "ai_generated_aqa_aligned" or generated_id:
            accepted_questions.append(_normalise_generated_question(item, index))

    # IMPORTANT: user-triggered regeneration attempts are intentionally kept
    # separate from Notebook 06's own validation/model retry counters.  They
    # are persisted by stable plan_index so a regenerated question can receive
    # a new question_id without resetting the user's 2-attempt allowance.
    for collection in (candidate_questions, accepted_questions):
        for question in collection:
            try:
                plan_index = int(question.get("plan_index") or 0)
            except (TypeError, ValueError):
                plan_index = 0

            if plan_index <= 0:
                continue

            question.update(
                _user_regeneration_attempt_info(
                    run_dir=run_dir,
                    quiz_mode=quiz_mode,
                    plan_index=plan_index,
                )
            )

    plan_indexes = _current_generated_plan_indexes(
        run_dir=run_dir,
        quiz_mode=quiz_mode,
        manifest=manifest,
    )

    user_attempts_by_plan_index: dict[str, dict[str, Any]] = {}
    blocked_plan_indexes: list[int] = []
    whole_quiz_remaining_values: list[int] = []

    for plan_index in plan_indexes:
        info = _user_regeneration_attempt_info(
            run_dir=run_dir,
            quiz_mode=quiz_mode,
            plan_index=plan_index,
        )
        user_attempts_by_plan_index[str(plan_index)] = info
        remaining = int(info["user_regeneration_attempts_remaining"])
        whole_quiz_remaining_values.append(remaining)
        if remaining <= 0:
            blocked_plan_indexes.append(plan_index)

    whole_quiz_attempts_remaining = (
        min(whole_quiz_remaining_values)
        if whole_quiz_remaining_values
        else 0
    )
    whole_quiz_available = bool(plan_indexes) and not blocked_plan_indexes

    output_dir = _quiz_output_dir(run_dir, quiz_mode)
    legacy_internal_used = int(manifest.get("regeneration_attempts_used") or 0)
    legacy_internal_max = int(manifest.get("max_regeneration_attempts") or 1)

    return {
        "quiz_mode": quiz_mode,
        "assessment_type": str(manifest.get("assessment_type") or ""),
        "human_review_state": str(manifest.get("generated_human_review_state") or ""),
        "generated_quality_accepted": bool(manifest.get("generated_quality_accepted")),
        "release_ready": bool(manifest.get("release_ready")),
        "candidate_questions": candidate_questions,
        "accepted_questions": accepted_questions,
        "candidate_count": int(manifest.get("candidate_question_count") or len(candidate_questions) or 0),
        "candidate_marks": int(manifest.get("candidate_total_marks") or 0),
        "accepted_count": int(manifest.get("accepted_question_count") or len(accepted_questions) or 0),
        "accepted_marks": int(manifest.get("accepted_total_marks") or 0),

        # Existing Notebook 06 / internal regeneration counters: unchanged.
        "regeneration_attempts_used": legacy_internal_used,
        "max_regeneration_attempts": legacy_internal_max,
        "internal_regeneration_attempts_used": legacy_internal_used,
        "internal_max_regeneration_attempts": legacy_internal_max,

        # New user-triggered regeneration budget: max 2 per question slot.
        "max_user_regeneration_attempts_per_question": (
            MAX_USER_REGENERATION_ATTEMPTS_PER_QUESTION
        ),
        "user_regeneration_attempts_by_plan_index": user_attempts_by_plan_index,
        "whole_quiz_user_regeneration_available": whole_quiz_available,
        "whole_quiz_user_regeneration_attempts_remaining": (
            whole_quiz_attempts_remaining
        ),
        "whole_quiz_user_regeneration_blocked_plan_indexes": (
            blocked_plan_indexes
        ),
        "user_regeneration_disclaimer": USER_REGENERATION_DISCLAIMER,

        "pdf_paths": _manifest_pdf_paths(manifest, output_dir),
        "validation": manifest.get("semantic_quality_validation") or {},
    }


def _shortfall_generation_outcome(
    *,
    run_dir: Path,
    shortfall: dict[str, Any],
    generation_started_epoch: float | None = None,
) -> dict[str, Any]:
    """
    Validate the hybrid fallback using the question-first policy.

    Policy:
      1. Reach the requested question count first, even if this makes the
         final mark total exceed the target.
      2. Once the question count is satisfied, keep generating if necessary
         until the final mark total also reaches the requested target.

    Exact equality is intentionally NOT required.  The final assessment is
    valid when both final_questions >= target_questions and
    final_marks >= target_marks.
    """
    manifest_mtime = _agent2_manifest_mtime(
        run_dir,
        "fill_shortfall",
    )

    if manifest_mtime is None:
        return {
            "success": False,
            "error": (
                "AI shortfall generation finished without creating a current "
                "fill_shortfall quiz manifest."
            ),
            "generated_questions": 0,
            "generated_marks": 0,
        }

    if (
        generation_started_epoch is not None
        and float(manifest_mtime) + 1.0 < float(generation_started_epoch)
    ):
        return {
            "success": False,
            "error": (
                "AI shortfall generation did not create a new candidate for "
                "this assessment attempt."
            ),
            "generated_questions": 0,
            "generated_marks": 0,
        }

    generated = _generated_result(
        run_dir,
        "fill_shortfall",
    )

    if not isinstance(generated, dict):
        return {
            "success": False,
            "error": "AI shortfall generation produced no readable candidate result.",
            "generated_questions": 0,
            "generated_marks": 0,
        }

    accepted = generated.get("accepted_questions") or []
    candidates = generated.get("candidate_questions") or []

    if not isinstance(accepted, list):
        accepted = []
    if not isinstance(candidates, list):
        candidates = []

    active_questions = accepted if accepted else candidates
    active_questions = [
        item
        for item in active_questions
        if isinstance(item, dict)
    ]

    generated_questions = len(active_questions)
    generated_marks = sum(
        max(0, int(item.get("marks") or 0))
        for item in active_questions
    )

    if generated_marks <= 0:
        mark_key = (
            "accepted_marks"
            if accepted
            else "candidate_marks"
        )
        try:
            generated_marks = max(
                0,
                int(generated.get(mark_key) or 0),
            )
        except (TypeError, ValueError):
            generated_marks = 0

    selected_questions = int(
        shortfall.get("selected_questions") or 0
    )
    selected_marks = int(
        shortfall.get("selected_marks") or 0
    )
    target_questions = int(
        shortfall.get("target_questions")
        or shortfall.get("requested_questions")
        or 0
    )
    target_marks = int(
        shortfall.get("target_marks") or 0
    )

    final_questions = selected_questions + generated_questions
    final_marks = selected_marks + generated_marks

    question_target_met = (
        target_questions <= 0
        or final_questions >= target_questions
    )
    mark_target_met = (
        target_marks <= 0
        or final_marks >= target_marks
    )

    review_state = str(
        generated.get("human_review_state") or ""
    ).strip().upper()

    has_candidate = generated_questions > 0
    no_candidate_state = (
        "NO_GENERATED_CANDIDATE" in review_state
    )

    success = bool(
        has_candidate
        and not no_candidate_state
        and question_target_met
        and mark_target_met
    )

    error = None

    if not success:
        remaining_questions = max(0, target_questions - final_questions)
        remaining_marks = max(0, target_marks - final_marks)

        if no_candidate_state or not has_candidate:
            if remaining_questions > 0:
                error = (
                    f"EDTech found {selected_questions} suitable official AQA question"
                    f"{'s' if selected_questions != 1 else ''} worth {selected_marks} marks, "
                    f"but could not generate the remaining {remaining_questions} question"
                    f"{'s' if remaining_questions != 1 else ''}. "
                    "The retrieved official questions are still available for review."
                )
            elif remaining_marks > 0:
                error = (
                    f"EDTech reached the requested question count, but could not generate "
                    f"the remaining {remaining_marks} mark{'s' if remaining_marks != 1 else ''} "
                    "needed to complete the assessment. The completed questions are still available below."
                )
            else:
                error = (
                    "EDTech could not create a valid AI fallback candidate. "
                    "The retrieved official questions are still available for review."
                )
        elif not question_target_met:
            error = (
                f"EDTech generated part of the fallback, but the assessment currently has "
                f"{final_questions}/{target_questions} questions. "
                f"{remaining_questions} more question{'s are' if remaining_questions != 1 else ' is'} still needed."
            )
        elif not mark_target_met:
            error = (
                f"EDTech completed the requested question count, but the assessment currently "
                f"contains {final_marks}/{target_marks} marks. "
                f"{remaining_marks} more mark{'s are' if remaining_marks != 1 else ' is'} still needed."
            )

    return {
        "success": success,
        "error": error,
        "generated_questions": generated_questions,
        "generated_marks": generated_marks,
        "final_questions": final_questions,
        "final_marks": final_marks,
        "question_target_met": question_target_met,
        "mark_target_met": mark_target_met,
        "human_review_state": review_state,
    }


def _official_result(run_dir: Path) -> dict[str, Any] | None:
    package_path, package = _current_official_package(run_dir)
    if not package_path or not package:
        return None

    questions = package.get("questions") or []
    if not isinstance(questions, list):
        questions = []

    summary = package.get("retrieval_summary") or {}
    if not isinstance(summary, dict):
        summary = {}

    normalized = [
        _normalise_official_question(item, index)
        for index, item in enumerate(questions, start=1)
        if isinstance(item, dict)
    ]

    latest_feedback = _latest_agent2_retrieval_feedback(
        run_dir=run_dir,
        package=package,
    )
    for question in normalized:
        saved_feedback = latest_feedback.get(str(question.get("question_id") or ""))
        if saved_feedback:
            question["retrieval_feedback"] = {
                "decision": str(saved_feedback.get("decision") or ""),
                "reason": str(saved_feedback.get("reason") or ""),
                "memory_status": str(saved_feedback.get("memory_status") or ""),
                "memory_eligible": bool(saved_feedback.get("memory_eligible")),
                "memory_error": str(saved_feedback.get("memory_error") or ""),
            }
        else:
            question["retrieval_feedback"] = None

    selected_marks = summary.get("selected_marks")
    if selected_marks is None:
        selected_marks = sum(int(item.get("marks") or 0) for item in normalized)

    return {
        "questions": normalized,
        "question_count": len(normalized),
        "selected_marks": int(selected_marks or 0),
        "release_status": str(
            summary.get("final_release_status")
            or summary.get("assessment_release_status")
            or package.get("run_status")
            or ""
        ),
        "pdf_paths": _official_pdf_paths(run_dir, package),
        "package_path": str(package_path),
    }


def _assessment_status_payload(run_id: str) -> dict[str, Any]:
    run_dir = _resolve_run_dir(run_id)
    status = _load_json(_assessment_status_path(run_dir))
    if not status:
        status = {
            "status": "idle",
            "stage": "ready",
            "progress": 0,
            "message": "Configure the assessment and run Agent 2.",
            "mode": None,
            "shortfall": None,
            "error": None,
        }

    current_mode = str(status.get("mode") or "").strip()
    current_state = str(status.get("status") or "").strip().casefold()
    results_ready = current_state == "complete"

    # Every new assessment attempt gets a fresh started_at_utc because the
    # start endpoint resets the status payload. Use it only to prevent older
    # artifacts from the SAME lesson run being surfaced as current results.
    attempt_started = _parse_agent2_iso_epoch(
        status.get("started_at_utc")
    )

    def artifact_is_current(mtime: float | None) -> bool:
        if mtime is None:
            return False
        if attempt_started is None:
            return True
        # Small tolerance for filesystem timestamp precision.
        return float(mtime) + 1.0 >= float(attempt_started)

    official: dict[str, Any] | None = None
    generated: dict[str, Any] | None = None

    # While a new attempt is queued/running, deliberately expose no old
    # assessment material. Once complete, expose only material relevant to
    # the current attempt. If hybrid retrieval succeeded but its AI fallback
    # failed, keep the CURRENT official retrieval visible so the user can
    # review what was successfully retrieved alongside the failure message.
    if results_ready:
        if current_mode == "complete_quiz":
            if artifact_is_current(
                _agent2_manifest_mtime(
                    run_dir,
                    "complete_quiz",
                )
            ):
                generated = _generated_result(
                    run_dir,
                    "complete_quiz",
                )

        elif current_mode == "retrieve_hybrid":
            if artifact_is_current(
                _agent2_retrieval_package_mtime(
                    run_dir
                )
            ):
                official = _official_result(run_dir)

            shortfall = (
                status.get("shortfall")
                if isinstance(
                    status.get("shortfall"),
                    dict,
                )
                else {}
            )

            # A generated section belongs to hybrid mode only when THIS
            # retrieval attempt actually needed shortfall generation.
            if (
                shortfall
                and not bool(
                    shortfall.get(
                        "sufficient",
                        False,
                    )
                )
                and artifact_is_current(
                    _agent2_manifest_mtime(
                        run_dir,
                        "fill_shortfall",
                    )
                )
            ):
                generated = _generated_result(
                    run_dir,
                    "fill_shortfall",
                )

        else:
            # Backward-compatible fallback for old completed status files
            # that predate explicit assessment modes.
            official = _official_result(run_dir)

            candidates: list[
                tuple[
                    float,
                    dict[str, Any],
                ]
            ] = []

            for quiz_mode in [
                "fill_shortfall",
                "complete_quiz",
            ]:
                item = _generated_result(
                    run_dir,
                    quiz_mode,
                )

                if item is None:
                    continue

                mtime = _agent2_manifest_mtime(
                    run_dir,
                    quiz_mode,
                )

                candidates.append(
                    (
                        float(mtime or 0.0),
                        item,
                    )
                )

            generated = (
                max(
                    candidates,
                    key=lambda pair: pair[0],
                )[1]
                if candidates
                else None
            )

    if (
        not results_ready
        and current_state == "failed"
        and current_mode == "retrieve_hybrid"
        and isinstance(status.get("shortfall"), dict)
        and artifact_is_current(
            _agent2_retrieval_package_mtime(
                run_dir
            )
        )
    ):
        official = _official_result(run_dir)

    # Keep ETA calculation on the persisted high-level stage so the existing
    # historical timing model retains its process mapping. Then overlay the
    # current-run artifact progress only for what the browser displays.
    eta = _agent2_running_eta(
        status=status
    )
    live_progress = _agent2_live_progress(
        run_dir,
        status,
    )
    display_status = {
        **status,
        **live_progress,
    }

    return {
        "success": True,
        "run_id": str(run_id),
        **display_status,
        **eta,
        "official": official,
        "generated": generated,
    }


def _validate_assessment_request(
    *,
    run_dir: Path,
    body: AssessmentStartBody,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    approved = _approved_topics_payload(run_dir)
    topics = approved.get("topics") or []

    primary_count = sum(
        str(item.get("role") or item.get("topic_role") or "").strip().casefold() == "primary"
        for item in topics
        if isinstance(item, dict)
    )
    supporting_count = sum(
        str(item.get("role") or item.get("topic_role") or "").strip().casefold() == "supporting"
        for item in topics
        if isinstance(item, dict)
    )

    errors: list[str] = []
    if body.maximum_question_marks < body.minimum_question_marks:
        errors.append("Maximum marks must be at least minimum marks.")
    if body.minimum_primary_questions > body.number_of_questions:
        errors.append("Minimum primary questions exceed total questions.")
    if body.minimum_supporting_questions > body.number_of_questions:
        errors.append("Minimum supporting questions exceed total questions.")
    if (
        body.minimum_primary_questions + body.minimum_supporting_questions
        > body.number_of_questions
    ):
        errors.append("Primary and supporting minimums together exceed total questions.")
    if body.cover_all_approved_topics and body.number_of_questions < len(topics):
        errors.append(
            f"Request at least {len(topics)} questions to cover all approved topics."
        )
    if body.minimum_supporting_questions > 0 and supporting_count == 0:
        errors.append("No approved supporting topic is available.")
    if body.minimum_primary_questions > 0 and primary_count == 0:
        errors.append("No approved primary topic is available.")
    if body.mode == "complete_quiz":
        if body.number_of_questions * body.minimum_question_marks > body.target_total_marks:
            errors.append(
                "Complete quiz cannot meet the target marks: minimum marks × questions exceeds target total marks."
            )
        if body.number_of_questions * body.maximum_question_marks < body.target_total_marks:
            errors.append(
                "Complete quiz cannot meet the target marks: maximum marks × questions is below target total marks."
            )

    model_config = _quiz_model_config()
    models = model_config.get("models") or {}
    if not str(body.model_key or "").strip() or body.model_key not in models:
        errors.append("Choose a valid quiz generation model.")

    notebook_path = _quiz_generation_notebook(body.quiz_plan)
    if notebook_path is None:
        option = QUIZ_GENERATION_NOTEBOOK_OPTIONS.get(body.quiz_plan, {})
        expected = ", ".join(str(value) for value in (option.get("filenames") or []))
        errors.append(
            "Selected quiz generation notebook was not found under Agent2/Notebooks. "
            f"Expected one of: {expected}"
        )

    if errors:
        raise HTTPException(status_code=400, detail=errors)

    assert notebook_path is not None
    return approved, model_config, notebook_path


def _run_assessment_background(run_id: str, request_data: dict[str, Any]) -> None:
    run_dir = RUNS_ROOT / str(run_id)
    try:
        body = AssessmentStartBody(**request_data)
        _, model_config, notebook_path = _validate_assessment_request(
            run_dir=run_dir,
            body=body,
        )

        _save_quiz_model_selection(
            run_dir=run_dir,
            model_key=body.model_key,
            model_config=model_config,
        )
        _save_quiz_notebook_selection(
            run_dir=run_dir,
            option_key=body.quiz_plan,
            notebook_path=notebook_path,
        )

        # Persist the current Next.js special-instructions value in this exact
        # pipeline run. Notebook 06/06B/06C already reads this per-run sidecar
        # from run/output/integration, so the instruction reaches quiz
        # generation without hardcoding any particular instruction semantics.
        #
        # Write the file even when the value is blank so a new assessment
        # attempt cannot accidentally reuse instructions from an older attempt
        # within the same lesson run.
        special_instruction_path = (
            Path(run_dir)
            / "output"
            / "integration"
            / "quiz_special_instructions.json"
        )
        _write_json(
            special_instruction_path,
            {
                "schema_version": "agent2-quiz-special-instructions-v1.0.0",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "special_instructions": body.special_instructions.strip(),
            },
        )

        common = dict(
            paper=body.paper,
            number_of_questions=int(body.number_of_questions),
            target_total_marks=int(body.target_total_marks),
            minimum_question_marks=int(body.minimum_question_marks),
            maximum_question_marks=int(body.maximum_question_marks),
            minimum_primary_questions=int(body.minimum_primary_questions),
            minimum_supporting_questions=int(body.minimum_supporting_questions),
            cover_all_approved_topics=bool(body.cover_all_approved_topics),
            include_code_questions=bool(body.include_code_questions),
            include_visual_questions=bool(body.include_visual_questions),
            programming_language=(
                None if body.programming_language == "Automatic" else body.programming_language
            ),
        )

        special_suffix = ""
        if body.special_instructions.strip():
            special_suffix = (
                "\nSpecial quiz instructions "
                "(mandatory user requirements unless impossible or conflicting "
                "with approved AQA scope / fixed deterministic constraints): "
                + body.special_instructions.strip()
            )

        if body.mode == "complete_quiz":
            _write_assessment_status(
                run_dir,
                status="running",
                stage="generating",
                progress=25,
                message="Generating the complete AQA-aligned quiz through LangGraph + MCP.",
                mode=body.mode,
                request=request_data,
            )

            run_langgraph_request(
                frontend_root=BACKEND_RUNTIME_ROOT,
                run_id=str(run_id),
                user_request=build_complete_quiz_request_text(**common) + special_suffix,
                agent2_action="complete_quiz",
                max_steps=10,
                mode="start",
                agent2_project_root=str(AGENT2_ROOT),
                agent2_notebook_path=str(notebook_path),
            )

            _write_assessment_status(
                run_dir,
                status="complete",
                stage="review",
                progress=100,
                message="Quiz generation finished. Review the candidate questions and visuals below.",
                mode=body.mode,
                request=request_data,
            )
            return

        _write_assessment_status(
            run_dir,
            status="running",
            stage="retrieving",
            progress=18,
            message="Retrieving strict quality-safe official AQA questions through Notebook 05.",
            mode=body.mode,
            request=request_data,
        )

        run_langgraph_request(
            frontend_root=BACKEND_RUNTIME_ROOT,
            run_id=str(run_id),
            user_request=build_assessment_request_text(**common),
            agent2_action="retrieve_official",
            max_steps=3,
            mode="start",
            agent2_project_root=str(AGENT2_ROOT),
        )

        _, package = _current_official_package(run_dir)
        if not package:
            raise RuntimeError(
                "Notebook 05 finished but the current-run assessment package could not be found."
            )

        shortfall = _agent2_official_shortfall(package)

        if shortfall["sufficient"]:
            _write_assessment_status(
                run_dir,
                status="complete",
                stage="review",
                progress=100,
                message="Official retrieval satisfies the requested assessment. AI fallback was not needed.",
                mode=body.mode,
                request=request_data,
                shortfall=shortfall,
            )
            return

        _write_assessment_status(
            run_dir,
            status="running",
            stage="generating_shortfall",
            progress=58,
            message=(
                "Official retrieval is short. Completing question count first, "
                "then marks if still needed: "
                f"{shortfall['missing_questions']} question(s) short and "
                f"{shortfall['missing_marks']} mark(s) below target. "
                "Final marks may exceed the target when required to satisfy "
                "the requested question count."
            ),
            mode=body.mode,
            request=request_data,
            shortfall=shortfall,
        )

        shortfall_generation_started = time.time()

        run_langgraph_request(
            frontend_root=BACKEND_RUNTIME_ROOT,
            run_id=str(run_id),
            user_request=build_missing_quiz_request_text() + special_suffix,
            agent2_action="missing_quiz",
            max_steps=10,
            mode="start",
            agent2_project_root=str(AGENT2_ROOT),
            agent2_notebook_path=str(notebook_path),
        )

        generation_outcome = _shortfall_generation_outcome(
            run_dir=run_dir,
            shortfall=shortfall,
            generation_started_epoch=shortfall_generation_started,
        )

        resolved_shortfall = {
            **shortfall,
            "generation_succeeded": bool(
                generation_outcome.get("success")
            ),
            "generated_questions": int(
                generation_outcome.get("generated_questions") or 0
            ),
            "generated_marks": int(
                generation_outcome.get("generated_marks") or 0
            ),
            "final_questions": int(
                generation_outcome.get("final_questions")
                or shortfall.get("selected_questions")
                or 0
            ),
            "final_marks": int(
                generation_outcome.get("final_marks")
                or shortfall.get("selected_marks")
                or 0
            ),
            "question_target_met": bool(
                generation_outcome.get("question_target_met", False)
            ),
            "mark_target_met": bool(
                generation_outcome.get("mark_target_met", False)
            ),
            "generation_review_state": str(
                generation_outcome.get("human_review_state") or ""
            ),
        }

        if not bool(generation_outcome.get("success")):
            final_questions = int(resolved_shortfall.get("final_questions") or 0)
            target_questions = int(resolved_shortfall.get("target_questions") or 0)
            final_marks = int(resolved_shortfall.get("final_marks") or 0)
            target_marks = int(resolved_shortfall.get("target_marks") or 0)

            if target_questions > 0 and final_questions < target_questions:
                remaining = target_questions - final_questions
                retrieved = int(resolved_shortfall.get("selected_questions") or 0)
                user_message = (
                    f"Assessment incomplete — EDTech found {retrieved} suitable official question"
                    f"{'s' if retrieved != 1 else ''}, but could not generate the remaining "
                    f"{remaining} question{'s' if remaining != 1 else ''}."
                )
            elif target_marks > 0 and final_marks < target_marks:
                remaining = target_marks - final_marks
                user_message = (
                    f"Assessment incomplete — the requested question count was reached, "
                    f"but EDTech could not generate the remaining {remaining} mark"
                    f"{'s' if remaining != 1 else ''} needed for the target."
                )
            else:
                user_message = (
                    "Assessment incomplete — official retrieval succeeded, but the AI fallback could not finish the assessment."
                )

            _write_assessment_status(
                run_dir,
                status="failed",
                stage="shortfall_failed",
                progress=100,
                message=user_message,
                mode=body.mode,
                request=request_data,
                shortfall=resolved_shortfall,
                error=str(
                    generation_outcome.get("error")
                    or "EDTech could not complete the AI fallback. The retrieved official questions are still available below."
                ),
            )
            return

        _write_assessment_status(
            run_dir,
            status="complete",
            stage="review",
            progress=100,
            message=(
                "Official retrieval and AI shortfall generation finished. "
                "Question count was prioritised first; final marks may exceed "
                "the requested target. Review the combined assessment below."
            ),
            mode=body.mode,
            request=request_data,
            shortfall=resolved_shortfall,
        )

    except Exception as exc:
        try:
            _write_assessment_status(
                run_dir,
                status="failed",
                stage="failed",
                progress=100,
                message="Agent 2 assessment workflow failed.",
                mode=request_data.get("mode"),
                request=request_data,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass


def _asset_path_for_run(run_dir: Path, raw_path: str) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Asset path is required.")

    path = Path(raw)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                run_dir / path,
                run_dir / "output" / "agent2" / path,
                run_dir / "output" / "agent2_quiz" / "complete_quiz" / path,
                run_dir / "output" / "agent2_quiz" / "fill_shortfall" / path,
                AGENT2_ROOT / path,
            ]
        )

    resolved_candidates: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_file():
            resolved_candidates.append(resolved)

    if not resolved_candidates and path.name:
        # Final current-run fallback for manifests that preserved only a basename.
        matches = list(run_dir.rglob(path.name))
        if len(matches) == 1 and matches[0].is_file():
            resolved_candidates.append(matches[0].resolve())

    if not resolved_candidates:
        raise HTTPException(status_code=404, detail=f"Asset not found: {raw}")

    resolved = resolved_candidates[0]
    allowed_roots = [run_dir.resolve(), AGENT2_ROOT.resolve()]
    if not any(
        resolved == root or root in resolved.parents
        for root in allowed_roots
    ):
        raise HTTPException(status_code=403, detail="Asset path is outside the allowed EDTech run roots.")

    return resolved




# ============================================================
# Dashboard - real run history / logging
# ============================================================

def _dashboard_run_created_at(run_dir: Path) -> str:
    """
    Prefer the timestamp embedded in our stable job id:
        job_YYYYMMDD_HHMMSS_<suffix>
    Fall back to the run directory creation/change timestamp.
    """
    match = re.match(
        r"^job_(\d{8})_(\d{6})(?:_|$)",
        run_dir.name,
    )

    if match:
        raw = f"{match.group(1)}{match.group(2)}"

        try:
            value = datetime.strptime(
                raw,
                "%Y%m%d%H%M%S",
            )
            return value.isoformat()
        except ValueError:
            pass

    try:
        return datetime.fromtimestamp(
            run_dir.stat().st_ctime
        ).isoformat()
    except OSError:
        return ""


def _dashboard_raw_topic_count(
    module3: dict[str, Any],
) -> int:
    """
    Lightweight persisted-artifact count for Dashboard.

    Module 3 artifacts have existed in a few shapes across EDTech versions.
    Count effective/detected topics without opening PostgreSQL, Qdrant,
    SyllabusStore, or the HITL adapter.

    Priority:
      1. Explicit final/effective topic lists.
      2. Direct mapped/detected topic lists.
      3. llm_results: count UNIQUE mapped official concepts only.
      4. Common nested result containers.

    Out-of-syllabus / removed / unresolved LLM results are not counted as
    identified official syllabus topics.
    """
    if not isinstance(module3, dict):
        return 0

    # --------------------------------------------------------------
    # 1) Explicit/direct list containers.
    # --------------------------------------------------------------
    direct_keys = [
        "effective_topics",
        "final_topics",
        "mapped_topics",
        "topics",
        "topic_mappings",
        "detected_topics",
    ]

    for key in direct_keys:
        value = module3.get(key)

        if isinstance(value, list):
            rows = [
                item
                for item in value
                if isinstance(item, dict)
            ]

            if rows:
                return len(rows)

    # --------------------------------------------------------------
    # 2) Module 3 LLM mapping results.
    #
    # Count unique official concepts with a mapped/accepted decision.
    # This is the important fallback for the current EDTech artifact shape.
    # --------------------------------------------------------------
    llm_results = module3.get("llm_results")

    if isinstance(llm_results, list):
        unique_concepts: set[str] = set()
        anonymous_mapped = 0

        for item in llm_results:
            if not isinstance(item, dict):
                continue

            decision = str(
                item.get("decision")
                or item.get("mapping_decision")
                or item.get("proposed_decision")
                or item.get("status")
                or ""
            ).strip().casefold()

            # Explicitly exclude non-syllabus / rejected outcomes.
            if decision in {
                "out_of_syllabus",
                "out-of-syllabus",
                "rejected",
                "reject",
                "removed",
                "remove",
                "unresolved",
                "none",
            }:
                continue

            concept_id = str(
                item.get("mapped_concept_id")
                or item.get("concept_id")
                or item.get("target_concept_id")
                or item.get("proposed_mapped_concept_id")
                or item.get("syllabus_concept_id")
                or ""
            ).strip()

            # Some historical rows use a nested mapped concept object.
            if not concept_id:
                mapped = item.get("mapped_concept")

                if isinstance(mapped, dict):
                    concept_id = str(
                        mapped.get("concept_id")
                        or mapped.get("id")
                        or ""
                    ).strip()

            if concept_id:
                unique_concepts.add(concept_id)

            # If a row clearly says it is mapped but carries no stable id,
            # retain it as an anonymous mapped result instead of losing it.
            elif decision in {
                "mapped",
                "approved",
                "accepted",
                "match",
            }:
                anonymous_mapped += 1

        llm_count = len(unique_concepts) + anonymous_mapped

        if llm_count > 0:
            return llm_count

    # --------------------------------------------------------------
    # 3) Common nested containers from older Module 3 exports.
    # --------------------------------------------------------------
    nested_keys = [
        "result",
        "results",
        "module3_result",
        "mapping_result",
        "topic_mapping_result",
        "output",
    ]

    for parent_key in nested_keys:
        parent = module3.get(parent_key)

        if not isinstance(parent, dict):
            continue

        nested_count = _dashboard_raw_topic_count(parent)

        if nested_count > 0:
            return nested_count

    return 0


def _dashboard_assessment_question_count(
    assessment: dict[str, Any],
) -> int:
    """
    Compact count from already-persisted assessment summary payload.
    """
    mode = str(
        assessment.get("mode") or ""
    ).strip()

    official = assessment.get("official")
    if not isinstance(official, dict):
        official = {}

    generated = assessment.get("generated")
    if not isinstance(generated, dict):
        generated = {}

    official_count = int(
        official.get("question_count") or 0
    )

    generated_count = int(
        generated.get("accepted_count")
        or generated.get("candidate_count")
        or generated.get("question_count")
        or 0
    )

    if mode == "complete_quiz":
        return generated_count

    if mode == "retrieve_hybrid":
        return official_count + generated_count

    return official_count or generated_count



def _dashboard_assessment_summary(
    run_dir: Path,
) -> dict[str, Any]:
    """
    Read only lightweight persisted assessment status/counts.

    This intentionally avoids loading question bodies, diagrams, PDF manifests,
    and generated quiz detail for every historical run.
    """
    status = _load_json(
        _assessment_status_path(run_dir)
    )

    mode = str(status.get("mode") or "").strip()

    _, official_package = _current_official_package(
        run_dir
    )

    official_questions = official_package.get("questions")
    if not isinstance(official_questions, list):
        official_questions = []

    official = {
        "question_count": len(
            [
                item
                for item in official_questions
                if isinstance(item, dict)
            ]
        )
    }

    generated_count = 0

    quiz_modes = (
        ["complete_quiz"]
        if mode == "complete_quiz"
        else ["fill_shortfall"]
        if mode == "retrieve_hybrid"
        else ["fill_shortfall", "complete_quiz"]
    )

    for quiz_mode in quiz_modes:
        quiz_dir = _quiz_output_dir(
            run_dir,
            quiz_mode,
        )

        manifest_candidates = [
            quiz_dir
            / "mcp_visuals"
            / "final_quiz_manifest_with_mcp_visuals.json",
            quiz_dir
            / "final_quiz_manifest.json",
        ]

        for path in manifest_candidates:
            manifest = _load_json(path)

            questions = manifest.get("questions")
            if isinstance(questions, list):
                generated_count = max(
                    generated_count,
                    len(
                        [
                            item
                            for item in questions
                            if isinstance(item, dict)
                        ]
                    ),
                )
                break

    return {
        "status": str(
            status.get("status") or "idle"
        ),
        "mode": mode or None,
        "official": official,
        "generated": {
            "question_count": generated_count,
            "accepted_count": generated_count,
        },
    }


def _dashboard_stage(
    *,
    module1_ready: bool,
    module2_ready: bool,
    module3_ready: bool,
    approved_count: int,
    assessment: dict[str, Any],
    api_status: dict[str, Any],
) -> tuple[int, str]:
    assessment_status = str(
        assessment.get("status") or "idle"
    ).strip().casefold()

    if assessment_status in {
        "queued",
        "running",
        "complete",
        "failed",
    }:
        if assessment_status == "complete":
            return 4, "Assessment complete"

        if assessment_status == "failed":
            return 4, "Assessment failed"

        return 4, "Assessment running"

    if approved_count > 0:
        return 3, "Ready for assessment"

    final_state = str(
        api_status.get("final_state") or ""
    ).strip()

    if module3_ready:
        if "TOPIC_MAPPING_REVIEW" in final_state:
            return 3, "Mapping review required"

        if "AGENT2_TOPIC_APPROVAL" in final_state:
            return 3, "Topic approval required"

        return 3, "Topic mapping complete"

    if module2_ready:
        return 2, "Topic mapping"

    if module1_ready:
        return 1, "Semantic analysis"

    background_state = str(
        api_status.get("state") or ""
    ).strip().casefold()

    if background_state == "failed":
        return 0, "Processing failed"

    return 0, "Preprocessing"


def _dashboard_run_summary(
    run_dir: Path,
) -> dict[str, Any] | None:
    run_id = run_dir.name

    try:
        transcript_name, output_dir = _resolve_output_dir(
            run_dir
        )
    except HTTPException:
        # Ignore unrelated/invalid folders in runs/.
        return None

    module1_ready = (
        _nonempty(
            output_dir / "01_cleaned_transcript.txt"
        )
        and _nonempty(
            output_dir / "01_preprocessing.json"
        )
    )

    module2_ready = _nonempty(
        output_dir / "02_chunking.json"
    )

    module3_path = output_dir / "03_topic_mapping.json"
    module3_ready = _nonempty(module3_path)

    # Dashboard must stay lightweight. Do NOT open the full HITL /
    # PostgreSQL / SyllabusStore adapter for every historical run.
    # Module 3's persisted JSON is enough for workspace counters.
    topic_count = (
        _dashboard_raw_topic_count(
            _load_json(module3_path)
        )
        if module3_ready
        else 0
    )

    approved_payload = _load_json(
        _approved_topics_path(run_dir)
    )

    approved_topics = approved_payload.get("topics")
    if not isinstance(approved_topics, list):
        approved_topics = []

    approved_count = sum(
        1
        for item in approved_topics
        if isinstance(item, dict)
    )

    # Dashboard consistency guard:
    # every Agent 2-approved topic must have been identified by Agent 1.
    # This also protects historical runs whose Module 3 JSON used an older
    # shape that the lightweight parser cannot fully reconstruct.
    topic_count = max(
        int(topic_count or 0),
        int(approved_count or 0),
    )

    try:
        assessment = _dashboard_assessment_summary(
            run_dir
        )
    except Exception:
        assessment = {
            "status": "idle",
            "mode": None,
            "official": {
                "question_count": 0,
            },
            "generated": {
                "question_count": 0,
                "accepted_count": 0,
            },
        }

    assessment_question_count = (
        _dashboard_assessment_question_count(
            assessment
        )
    )

    api_status = _load_json(
        _api_status_path(run_dir)
    )

    stage_index, status_label = _dashboard_stage(
        module1_ready=module1_ready,
        module2_ready=module2_ready,
        module3_ready=module3_ready,
        approved_count=approved_count,
        assessment=assessment,
        api_status=api_status,
    )

    assessment_status = str(
        assessment.get("status") or "idle"
    ).strip()

    assessment_mode = str(
        assessment.get("mode") or ""
    ).strip() or None

    return {
        "run_id": run_id,
        "transcript_name": transcript_name,
        "created_at": _dashboard_run_created_at(
            run_dir
        ),
        "status": status_label,
        "stage_index": stage_index,
        "module1_complete": module1_ready,
        "module2_complete": module2_ready,
        "module3_complete": module3_ready,
        "topics_identified": topic_count,
        "topics_approved": approved_count,
        "assessment_questions": (
            assessment_question_count
        ),
        "assessment_status": assessment_status,
        "assessment_mode": assessment_mode,
        "human_action_required": bool(
            api_status.get(
                "human_action_required"
            )
        ),
        "logged_to_runs": True,
    }


@app.get("/api/dashboard")
def dashboard():
    """
    Dashboard source of truth.

    Reads persisted run folders under RUNS_ROOT, so it includes:
    - runs created by the current Next.js/FastAPI frontend;
    - compatible historical runs created by the previous UI.

    It does not invent counters in React.
    """
    if not RUNS_ROOT.is_dir():
        return {
            "success": True,
            "metrics": {
                "transcripts_processed": 0,
                "topics_identified": 0,
                "topics_approved": 0,
                "assessment_questions": 0,
            },
            "latest_run": None,
            "recent_runs": [],
            "logging": {
                "enabled": True,
                "runs_root": str(RUNS_ROOT),
                "run_count": 0,
            },
        }

    run_dirs = [
        path
        for path in RUNS_ROOT.iterdir()
        if path.is_dir()
        and path.name.startswith("job_")
    ]

    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        try:
            item = _dashboard_run_summary(
                run_dir
            )
        except Exception:
            item = None

        if item is not None:
            rows.append(item)

    rows.sort(
        key=lambda item: str(
            item.get("created_at") or ""
        ),
        reverse=True,
    )

    metrics = {
        "transcripts_processed": len(rows),
        "topics_identified": sum(
            int(item.get("topics_identified") or 0)
            for item in rows
        ),
        "topics_approved": sum(
            int(item.get("topics_approved") or 0)
            for item in rows
        ),
        "assessment_questions": sum(
            int(
                item.get(
                    "assessment_questions"
                )
                or 0
            )
            for item in rows
        ),
    }

    return {
        "success": True,
        "metrics": metrics,
        "latest_run": rows[0] if rows else None,
        "recent_runs": rows[:6],
        "logging": {
            "enabled": True,
            "runs_root": str(RUNS_ROOT),
            "run_count": len(rows),
        },
    }


# ============================================================
# Health
# ============================================================

@app.get("/")
def root():
    return {
        "name": "EDTech Backend API",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "edtech-api",
        "backend_runtime_found": (
            BACKEND_RUNTIME_ROOT.is_dir()
        ),
    }


# ============================================================
# New run + REAL live stage progress
# ============================================================

@app.post("/api/runs")
async def create_run(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    filename = str(file.filename or "").strip()

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file has no filename.",
        )

    suffix = Path(filename).suffix.lower()

    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF, DOCX and TXT transcripts "
                "are supported."
            ),
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded transcript is empty.",
        )

    try:
        run_info = create_langgraph_run(
            frontend_root=BACKEND_RUNTIME_ROOT,
            filename=filename,
            content=content,
        )

        run_id = str(run_info["run_id"])

        _set_api_status(
            run_id,
            state="queued",
            human_action_required=False,
        )

        # IMPORTANT:
        # The real workflow is now asynchronous from the browser's point
        # of view. This is what makes real progress polling possible.
        background_tasks.add_task(
            _run_agent1_background,
            run_id,
        )

        snapshot = _safe_snapshot(run_id)

        return {
            "success": True,
            "run_id": run_id,
            "filename": filename,
            "run_dir": str(run_info["run_dir"]),
            "final_state": str(
                snapshot.get("state") or ""
            ),
            "human_action_required": bool(
                snapshot.get("human_action_required")
            ),
            "interrupt_count": 0,
            "snapshot": snapshot,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc



# ============================================================
# Agent 1 rough ETA from persisted run history
# ============================================================

_ETA_CACHE: dict[str, Any] = {
    "computed_at": 0.0,
    "durations": {},
}


def _run_started_epoch(run_dir: Path) -> float:
    """
    Stable run start time.

    Prefer the timestamp embedded in:
        job_YYYYMMDD_HHMMSS_<suffix>

    Fall back to directory ctime for older/non-standard runs.
    """
    match = re.match(
        r"^job_(\d{8})_(\d{6})(?:_|$)",
        run_dir.name,
    )

    if match:
        raw = f"{match.group(1)}{match.group(2)}"

        try:
            value = datetime.strptime(
                raw,
                "%Y%m%d%H%M%S",
            )
            return value.timestamp()
        except ValueError:
            pass

    try:
        return run_dir.stat().st_ctime
    except OSError:
        return time.time()


def _safe_mtime(path: Path) -> float | None:
    try:
        if path.is_file():
            return path.stat().st_mtime
    except OSError:
        pass

    return None


def _historical_agent1_stage_durations() -> dict[str, list[float]]:
    """
    Read only filesystem timestamps from recent completed runs.

    No PostgreSQL, Qdrant, model, HITL adapter, or large JSON read is done.

    Durations:
      preprocessing      run start -> 01_preprocessing.json
      semantic_chunking  Module 1   -> 02_chunking.json
      topic_mapping      Module 2   -> 03_topic_mapping.json
    """
    now = time.monotonic()

    cached_at = float(
        _ETA_CACHE.get("computed_at") or 0.0
    )

    cached = _ETA_CACHE.get("durations")

    if (
        isinstance(cached, dict)
        and cached
        and now - cached_at < 30.0
    ):
        return cached

    durations: dict[str, list[float]] = {
        "preprocessing": [],
        "semantic_chunking": [],
        "topic_mapping": [],
    }

    if not RUNS_ROOT.is_dir():
        return durations

    run_dirs = sorted(
        (
            path
            for path in RUNS_ROOT.iterdir()
            if path.is_dir()
            and path.name.startswith("job_")
        ),
        key=lambda path: path.name,
        reverse=True,
    )[:60]

    for historical_run in run_dirs:
        try:
            _, historical_output = _resolve_output_dir(
                historical_run
            )
        except Exception:
            continue

        module1_time = _safe_mtime(
            historical_output / "01_preprocessing.json"
        )
        module2_time = _safe_mtime(
            historical_output / "02_chunking.json"
        )
        module3_time = _safe_mtime(
            historical_output / "03_topic_mapping.json"
        )

        run_start = _run_started_epoch(
            historical_run
        )

        candidates = {
            "preprocessing": (
                module1_time - run_start
                if module1_time is not None
                else None
            ),
            "semantic_chunking": (
                module2_time - module1_time
                if (
                    module1_time is not None
                    and module2_time is not None
                )
                else None
            ),
            "topic_mapping": (
                module3_time - module2_time
                if (
                    module2_time is not None
                    and module3_time is not None
                )
                else None
            ),
        }

        for stage_name, duration in candidates.items():
            if duration is None:
                continue

            # Ignore corrupted timestamps / extremely long abandoned runs.
            if 1.0 <= duration <= 3600.0:
                durations[stage_name].append(
                    float(duration)
                )

    _ETA_CACHE["computed_at"] = now
    _ETA_CACHE["durations"] = durations

    return durations


def _current_stage_elapsed_seconds(
    *,
    run_dir: Path,
    output_dir: Path,
    stage: str,
) -> float:
    current_time = time.time()

    if stage == "semantic_chunking":
        stage_started = _safe_mtime(
            output_dir / "01_preprocessing.json"
        )
    elif stage == "topic_mapping":
        stage_started = _safe_mtime(
            output_dir / "02_chunking.json"
        )
    else:
        stage_started = _run_started_epoch(
            run_dir
        )

    if stage_started is None:
        stage_started = _run_started_epoch(
            run_dir
        )

    return max(
        0.0,
        current_time - float(stage_started),
    )


def _format_eta_label(
    remaining_seconds: float,
) -> str:
    seconds = max(
        0.0,
        float(remaining_seconds),
    )

    if seconds < 45:
        return "≈ less than 1 min remaining"

    if seconds < 90:
        return "≈ 1 min remaining"

    minutes = max(
        2,
        int(round(seconds / 60.0)),
    )

    if minutes == 2:
        return "≈ 1–2 min remaining"

    return f"≈ {minutes - 1}–{minutes} min remaining"


def _agent1_eta(
    *,
    run_dir: Path,
    output_dir: Path,
    stage: str,
    background_status: str,
) -> dict[str, Any]:
    if (
        background_status not in {"queued", "running"}
        or stage not in {
            "preprocessing",
            "semantic_chunking",
            "topic_mapping",
        }
    ):
        return {
            "eta_seconds": None,
            "eta_label": None,
            "eta_basis": None,
            "eta_sample_count": 0,
        }

    history = _historical_agent1_stage_durations()
    samples = list(
        history.get(stage) or []
    )

    # We need a few real completed examples before pretending to estimate.
    if len(samples) < 3:
        return {
            "eta_seconds": None,
            "eta_label": "Estimating remaining time…",
            "eta_basis": (
                "Waiting for enough recent EDTech timing history"
            ),
            "eta_sample_count": len(samples),
        }

    samples.sort()

    # Use a slightly conservative typical duration:
    # max(median, ~65th percentile).
    median_duration = float(
        statistics.median(samples)
    )

    percentile_index = min(
        len(samples) - 1,
        max(
            0,
            int(round(
                (len(samples) - 1) * 0.65
            )),
        ),
    )

    typical_duration = max(
        median_duration,
        float(samples[percentile_index]),
    )

    elapsed = _current_stage_elapsed_seconds(
        run_dir=run_dir,
        output_dir=output_dir,
        stage=stage,
    )

    remaining = typical_duration - elapsed

    stage_labels = {
        "preprocessing": "preprocessing",
        "semantic_chunking": "semantic chunking",
        "topic_mapping": "topic mapping",
    }

    basis = (
        f"Based on {len(samples)} recent "
        f"{stage_labels.get(stage, stage)} runs"
    )

    if remaining <= 0:
        return {
            "eta_seconds": None,
            "eta_label": "≈ finishing soon",
            "eta_basis": basis,
            "eta_sample_count": len(samples),
        }

    return {
        "eta_seconds": int(round(remaining)),
        "eta_label": _format_eta_label(
            remaining
        ),
        "eta_basis": basis,
        "eta_sample_count": len(samples),
    }


@app.get("/api/runs/{run_id}/progress")
def get_run_progress(run_id: str):
    """
    Stage-derived progress, not a fake timer.

    Percent changes only when real Agent 1 artifacts/state become available:
      upload/run created        -> 10%
      Module 1 complete         -> 40%
      Module 2 complete         -> 70%
      Module 3 complete/gate    -> 100%
    """
    run_dir = _resolve_run_dir(run_id)
    transcript_name, output_dir = _resolve_output_dir(run_dir)

    module1_ready = (
        _nonempty(output_dir / "01_cleaned_transcript.txt")
        and _nonempty(output_dir / "01_preprocessing.json")
    )
    module2_ready = _nonempty(
        output_dir / "02_chunking.json"
    )
    module3_ready = _nonempty(
        output_dir / "03_topic_mapping.json"
    )

    api_status = _load_json(
        _api_status_path(run_dir)
    )

    background_status = str(
        api_status.get("state") or "running"
    ).strip()

    snapshot: dict[str, Any] = {}

    if module3_ready or background_status != "running":
        snapshot = _safe_snapshot(run_id)

    if module3_ready:
        percent = 100

        gate = str(
            snapshot.get("human_gate") or ""
        ).strip()

        if gate == "TOPIC_MAPPING_REVIEW":
            stage = "human_mapping_review"
            message = "Topic mapping needs human review."
        elif gate == "AGENT2_TOPIC_APPROVAL":
            stage = "topic_approval"
            message = (
                "Agent 1 mapping is ready for final topic approval."
            )
        else:
            stage = "agent1_ready"
            message = "Agent 1 processing is complete."

    elif module2_ready:
        percent = 70
        stage = "topic_mapping"
        message = (
            "Semantic chunks are ready. Mapping lesson "
            "concepts to the AQA syllabus..."
        )

    elif module1_ready:
        percent = 40
        stage = "semantic_chunking"
        message = (
            "Preprocessing is complete. Creating semantic chunks..."
        )

    else:
        percent = 10
        stage = "preprocessing"
        message = "Cleaning and preparing the transcript..."

    error = api_status.get("error")

    if background_status == "failed":
        stage = "failed"
        message = "Agent 1 processing failed."

    eta = _agent1_eta(
        run_dir=run_dir,
        output_dir=output_dir,
        stage=stage,
        background_status=background_status,
    )

    return {
        "success": True,
        "run_id": str(run_id),
        "transcript_name": transcript_name,
        "percent": percent,
        "stage": stage,
        "message": message,
        "background_status": background_status,
        "error": error,
        "module1_ready": module1_ready,
        "module2_ready": module2_ready,
        "module3_ready": module3_ready,
        "human_action_required": bool(
            snapshot.get("human_action_required")
            or api_status.get("human_action_required")
        ),
        "human_gate": snapshot.get("human_gate"),
        "workflow_state": (
            snapshot.get("state")
            or api_status.get("final_state")
        ),
        "eta_seconds": eta["eta_seconds"],
        "eta_label": eta["eta_label"],
        "eta_basis": eta["eta_basis"],
        "eta_sample_count": eta["eta_sample_count"],
    }


# ============================================================
# Module 1 - real preprocessing
# ============================================================

@app.get("/api/runs/{run_id}/preprocessing")
def get_preprocessing(run_id: str):
    run_dir = _resolve_run_dir(run_id)
    transcript_name, output_dir = _resolve_output_dir(run_dir)

    cleaned_path = output_dir / "01_cleaned_transcript.txt"
    preprocessing_json_path = (
        output_dir / "01_preprocessing.json"
    )

    if not cleaned_path.is_file():
        raise HTTPException(
            status_code=409,
            detail=(
                "Preprocessing output is not ready for this run. "
                "01_cleaned_transcript.txt was not found."
            ),
        )

    cleaned_transcript = cleaned_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not cleaned_transcript:
        raise HTTPException(
            status_code=500,
            detail="The generated cleaned transcript is empty.",
        )

    module1 = _load_json(preprocessing_json_path)

    preprocessing_result = module1.get(
        "preprocessing_result"
    )
    if not isinstance(preprocessing_result, dict):
        preprocessing_result = {}

    deterministic_stats = preprocessing_result.get(
        "stats"
    )
    if not isinstance(deterministic_stats, dict):
        deterministic_stats = {}

    technical_result = module1.get(
        "technical_normalisation_result"
    )
    if not isinstance(technical_result, dict):
        technical_result = {}

    technical_stats = technical_result.get("stats")
    if not isinstance(technical_stats, dict):
        technical_stats = {}

    unresolved_issues = technical_result.get(
        "unresolved_issues"
    )
    if not isinstance(unresolved_issues, list):
        unresolved_issues = []

    return {
        "success": True,
        "run_id": str(run_id),
        "transcript_name": transcript_name,
        "cleaned_transcript": cleaned_transcript,
        "cleaned_word_count": len(
            cleaned_transcript.split()
        ),
        "deterministic_stats": deterministic_stats,
        "technical_stats": technical_stats,
        "unresolved_issue_count": len(
            unresolved_issues
        ),
    }


# ============================================================
# Module 2 - real semantic chunking
# ============================================================

@app.get("/api/runs/{run_id}/semantic")
def get_semantic(run_id: str):
    return _semantic_payload(run_id)


# ============================================================
# Module 3 - real topics + HITL state
# ============================================================

@app.get("/api/runs/{run_id}/topics")
def get_topics(run_id: str):
    return _topics_payload(run_id)


@app.post(
    "/api/runs/{run_id}/mapping-reviews/{review_id}"
)
def submit_mapping_review(
    run_id: str,
    review_id: int,
    body: MappingReviewBody,
):
    run_dir = _resolve_run_dir(run_id)
    transcript_name, output_dir = (
        _resolve_output_dir(run_dir)
    )

    module3_path = (
        output_dir / "03_topic_mapping.json"
    )

    if not module3_path.is_file():
        raise HTTPException(
            status_code=409,
            detail=(
                "Topic mapping output is not ready "
                "for final-topic editing."
            ),
        )

    module3_json = _load_json(
        module3_path
    )

    reason = str(body.reason or "").strip()

    if body.action == "correct":
        if not reason:
            raise HTTPException(
                status_code=400,
                detail=(
                    "A written correction reason is required."
                ),
            )

        if (
            body.corrected_decision == "mapped"
            and not str(
                body.corrected_mapped_concept_id or ""
            ).strip()
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Select the correct official AQA topic."
                ),
            )

        if body.corrected_decision is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "corrected_decision is required for "
                    "a correction."
                ),
            )

    status_by_action = {
        "approve": "approved",
        "reject": "rejected",
        "correct": "corrected",
    }

    try:
        result = submit_human_topic_review(
            frontend_root=BACKEND_RUNTIME_ROOT,
            run_id=str(run_id),
            review_id=int(review_id),
            status=status_by_action[body.action],
            reviewed_by="nextjs_human_ui",
            corrected_decision=(
                body.corrected_decision
                if body.action == "correct"
                else None
            ),
            corrected_mapped_concept_id=(
                str(
                    body.corrected_mapped_concept_id
                    or ""
                ).strip()
                or None
                if body.action == "correct"
                else None
            ),
            correction_reason=(
                reason
                if body.action == "correct"
                else None
            ),
            review_notes=(
                str(body.review_notes or "").strip()
                or (
                    reason
                    if body.action == "reject" and reason
                    else None
                )
            ),
        )

        _resume_after_human_action(str(run_id))

        payload = _topics_payload(str(run_id))
        payload["human_write_result"] = result
        return payload

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc



@app.post("/api/runs/{run_id}/memory-reviews")
def submit_historical_memory_review(
    run_id: str,
    body: HistoricalMemoryReviewBody,
):
    """
    Human authority over historical final-topic edit reuse.

    - use_historical:
        approve one selected historical memory for this exact fresh Module 3
        evidence; reject competing outcomes for the same review card.
    - keep_fresh:
        reject every represented historical memory for this exact evidence.

    Historical memories themselves are NOT deleted or modified. The decision
    is exact-context feedback stored in PostgreSQL.
    """
    run_dir = _resolve_run_dir(run_id)
    transcript_name, output_dir = (
        _resolve_output_dir(run_dir)
    )

    module3_path = (
        output_dir / "03_topic_mapping.json"
    )

    if not module3_path.is_file():
        raise HTTPException(
            status_code=409,
            detail=(
                "Topic mapping output is not ready "
                "for historical-memory review."
            ),
        )

    module3_json = _load_json(
        module3_path
    )

    reason = str(
        body.reason or ""
    ).strip()

    if not reason:
        raise HTTPException(
            status_code=400,
            detail=(
                "A written reason is required "
                "before saving this HITL decision."
            ),
        )

    memory_ids: list[int] = []

    for value in body.memory_ids:
        memory_id = int(value)

        if memory_id not in memory_ids:
            memory_ids.append(memory_id)

    if not memory_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "No historical memory IDs were supplied."
            ),
        )

    if body.decision == "use_historical":
        if body.selected_memory_id is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Select the historical outcome "
                    "to use."
                ),
            )

        selected_memory_id = int(
            body.selected_memory_id
        )

        if (
            selected_memory_id
            not in memory_ids
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "The selected memory is not part "
                    "of this review item."
                ),
            )
    else:
        selected_memory_id = None

    try:
        store = (
            _detected_topic_reuse_feedback_store()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load historical HITL "
                f"memory store: {type(exc).__name__}: {exc}"
            ),
        ) from exc

    writes: list[dict[str, Any]] = []

    for memory_id in memory_ids:
        try:
            memory = store.memory_snapshot(
                int(memory_id)
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not load historical memory "
                    f"{memory_id}: {type(exc).__name__}: {exc}"
                ),
            ) from exc

        if not isinstance(memory, dict):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Historical memory {memory_id} "
                    "was not found."
                ),
            )

        evidence = (
            _memory_evidence_for_current_run(
                module3_json=module3_json,
                memory=memory,
            )
        )

        if not evidence:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Could not reconstruct the exact "
                    f"fresh Module 3 evidence for memory {memory_id}. "
                    "No HITL decision was written."
                ),
            )

        spec_version = str(
            memory.get("spec_version")
            or _current_aqa_spec_version()
        ).strip()

        if body.decision == "keep_fresh":
            stored_decision = (
                "reject_reuse"
            )
            stored_reason = reason

        elif (
            int(memory_id)
            == int(selected_memory_id)
        ):
            stored_decision = (
                "approve_reuse"
            )
            stored_reason = reason

        else:
            # Resolve competing historical outcomes deterministically.
            stored_decision = (
                "reject_reuse"
            )
            stored_reason = (
                "Competing historical outcome "
                "rejected while resolving the same "
                "current lesson context. Reviewer reason: "
                + reason
            )

        try:
            store.record(
                memory_id=int(memory_id),
                current_evidence=evidence,
                decision=stored_decision,
                reviewer_reason=stored_reason,
                spec_version=spec_version,
                pipeline_run_id=str(run_id),
                source_transcript=transcript_name,
                source_concept_id=(
                    memory.get(
                        "source_concept_id"
                    )
                ),
                reviewed_by=(
                    "nextjs_human_ui"
                ),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Could not save historical HITL "
                    f"decision for memory {memory_id}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

        writes.append(
            {
                "memory_id": int(
                    memory_id
                ),
                "decision": stored_decision,
            }
        )

    # Re-read the effective list. The existing deterministic overlay sees the
    # newly persisted exact-context approve/reject decision immediately.
    payload = _topics_payload(
        str(run_id)
    )

    payload[
        "historical_memory_write_result"
    ] = {
        "decision": body.decision,
        "selected_memory_id": (
            selected_memory_id
        ),
        "writes": writes,
    }

    return payload



def _approve_manual_edit_for_current_context(
    *,
    result: dict[str, Any],
    run_id: str,
    run_dir: Path,
    transcript_name: str,
    module3_json: dict[str, Any],
    action: str,
    reason: str,
    source_concept_id: str | None,
) -> dict[str, Any]:
    """
    A manual final-topic edit is already an explicit human decision.

    The edit service creates reviewer-approved historical memory. This helper
    additionally records approve_reuse for the exact untouched fresh Module 3
    evidence of THIS lesson before the LangGraph runtime is resumed.

    Therefore:
      - the current lesson never asks the reviewer to approve the same manual
        correction a second time;
      - the PostgreSQL audit/history remains intact;
      - other lesson contexts still go through the normal deterministic
        comparator / HITL gate.
    """
    output = dict(result or {})
    output["exact_current_context_approval_saved"] = False

    try:
        memory_id = int(
            output.get(
                "detected_topic_edit_memory_id"
            )
        )
    except (TypeError, ValueError):
        output[
            "exact_current_context_approval_error"
        ] = (
            "The edit was saved, but no reusable "
            "memory ID was returned."
        )
        return output

    try:
        store = (
            _detected_topic_reuse_feedback_store()
        )
        memory = store.memory_snapshot(
            memory_id
        )
    except Exception as exc:
        output[
            "exact_current_context_approval_error"
        ] = (
            "The edit memory was saved, but its "
            f"context approval could not be loaded: "
            f"{type(exc).__name__}: {exc}"
        )
        return output

    if not isinstance(memory, dict):
        output[
            "exact_current_context_approval_error"
        ] = (
            "The edit was saved, but the created "
            "historical memory could not be loaded."
        )
        return output

    evidence = _memory_evidence_for_current_run(
        module3_json=module3_json,
        memory=memory,
    )

    if not evidence:
        output[
            "exact_current_context_approval_error"
        ] = (
            "The edit was saved, but the exact fresh "
            "Module 3 evidence for this lesson could "
            "not be reconstructed."
        )
        return output

    spec_version = str(
        memory.get("spec_version")
        or _current_aqa_spec_version()
    ).strip()

    try:
        store.record(
            memory_id=memory_id,
            current_evidence=evidence,
            decision="approve_reuse",
            reviewer_reason=str(
                reason or ""
            ).strip(),
            spec_version=spec_version,
            pipeline_run_id=str(run_id),
            source_transcript=transcript_name,
            source_concept_id=(
                str(
                    source_concept_id
                    or memory.get(
                        "source_concept_id"
                    )
                    or ""
                ).strip()
                or None
            ),
            reviewed_by="nextjs_human_ui",
        )
    except Exception as exc:
        output[
            "exact_current_context_approval_error"
        ] = (
            "The manual edit was saved, but the "
            "exact-context approval could not be "
            f"written: {type(exc).__name__}: {exc}"
        )
        return output

    output[
        "exact_current_context_approval_saved"
    ] = True
    output[
        "exact_current_context_memory_id"
    ] = memory_id

    return output


@app.post("/api/runs/{run_id}/topic-edits")
def submit_topic_edit(
    run_id: str,
    body: DetectedTopicEditBody,
):
    _resolve_run_dir(run_id)

    reason = str(body.reason or "").strip()

    if not reason:
        raise HTTPException(
            status_code=400,
            detail=(
                "A written reason is required for every "
                "final-topic edit."
            ),
        )

    if (
        body.action != "add_topic"
        and body.topic_index is None
    ):
        raise HTTPException(
            status_code=400,
            detail="topic_index is required for this edit.",
        )

    if (
        body.action == "change_role"
        and body.target_role is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Select the new topic role.",
        )

    if (
        body.action in {"replace_topic", "add_topic"}
        and not str(body.target_concept_id or "").strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="Select an official AQA topic.",
        )

    if (
        body.action == "add_topic"
        and body.target_role is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Select a role for the added topic.",
        )

    if (
        body.action == "add_topic"
        and not body.source_chunk_ids
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Select at least one lesson evidence chunk "
                "for the added topic."
            ),
        )

    try:
        result = submit_human_detected_topic_edit(
            frontend_root=BACKEND_RUNTIME_ROOT,
            run_id=str(run_id),
            action=body.action,
            reason=reason,
            topic_index=body.topic_index,
            source_concept_id=(
                str(body.source_concept_id or "").strip()
                or None
            ),
            target_concept_id=(
                str(body.target_concept_id or "").strip()
                or None
            ),
            target_role=body.target_role,
            source_chunk_ids=[
                int(value)
                for value in body.source_chunk_ids
            ],
            reviewed_by="nextjs_human_ui",
        )

        # The user just made this correction manually, so it is already
        # authoritative for THIS lesson. Save exact-context approve_reuse
        # before rerunning the memory overlay; otherwise the newly-created
        # memory can immediately appear as another HITL question.
        result = _approve_manual_edit_for_current_context(
            result=result,
            run_id=str(run_id),
            run_dir=run_dir,
            transcript_name=transcript_name,
            module3_json=module3_json,
            action=body.action,
            reason=reason,
            source_concept_id=(
                str(
                    body.source_concept_id
                    or ""
                ).strip()
                or None
            ),
        )

        # Re-check the same LangGraph thread after the human edit.
        _resume_after_human_action(str(run_id))

        payload = _topics_payload(str(run_id))
        payload["human_write_result"] = result
        return payload

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/runs/{run_id}/approve-topics")
def approve_topics_for_agent2(
    run_id: str,
    body: TopicApprovalBody,
):
    _resolve_run_dir(run_id)

    payload_before = _topics_payload(str(run_id))

    if payload_before["pending_review_count"] > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                "Resolve all pending Module 3 mapping reviews "
                "before approving topics for Agent 2."
            ),
        )

    topics = payload_before.get("topics") or []

    requested_indexes: list[int] = []

    for value in body.topic_indexes:
        index = int(value)

        if index not in requested_indexes:
            requested_indexes.append(index)

    if not requested_indexes:
        raise HTTPException(
            status_code=400,
            detail="Select at least one topic for Agent 2.",
        )

    selections: list[dict[str, Any]] = []

    topic_by_handoff_index: dict[int, dict[str, Any]] = {}
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        try:
            handoff_index = int(topic.get("topic_index"))
        except (TypeError, ValueError):
            continue
        topic_by_handoff_index[handoff_index] = topic

    for index in requested_indexes:
        topic = topic_by_handoff_index.get(int(index))

        if topic is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid topic index: {index}",
            )

        selections.append(
            {
                "topic_index": int(index),
                "approved": True,
                "topic": str(
                    topic.get("topic")
                    or topic.get("detected_topic")
                    or ""
                ).strip(),
                "role": str(
                    topic.get("role")
                    or topic.get("topic_role")
                    or "supporting"
                ).strip().casefold(),
                "official_reference": str(
                    topic.get("official_reference")
                    or ""
                ).strip(),
            }
        )

    try:
        result = submit_human_agent2_topic_approval(
            frontend_root=BACKEND_RUNTIME_ROOT,
            run_id=str(run_id),
            selections=selections,
            reviewed_by="nextjs_human_ui",
        )

        snapshot = _safe_snapshot(str(run_id))

        if (
            str(snapshot.get("human_gate") or "").strip()
            == "AGENT2_TOPIC_APPROVAL"
            and bool(snapshot.get("human_action_required"))
        ):
            _resume_after_human_action(str(run_id))

        payload = _topics_payload(str(run_id))
        payload["human_write_result"] = result
        return payload

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


# ============================================================
# Agent 2 assessment / quiz API
# ============================================================

@app.get("/api/runs/{run_id}/assessment/config")
def assessment_config(run_id: str):
    run_dir = _resolve_run_dir(run_id)
    approved = _approved_topics_payload(run_dir)
    topics = [
        dict(item)
        for item in (approved.get("topics") or [])
        if isinstance(item, dict)
    ]

    model_config = _quiz_model_config()
    models = model_config.get("models") or {}

    model_options = []
    for key, value in models.items():
        if not isinstance(value, dict):
            continue
        model_options.append(
            {
                "key": str(key),
                "display_name": str(value.get("display_name") or key),
                "provider": value.get("provider"),
                "model_id": value.get("model_id"),
                "context_window_tokens": value.get("context_window_tokens"),
                "hard_max_output_tokens": value.get("hard_max_output_tokens"),
            }
        )

    notebook_options = []
    for key, value in QUIZ_GENERATION_NOTEBOOK_OPTIONS.items():
        resolved = _quiz_generation_notebook(key)
        notebook_options.append(
            {
                "key": key,
                "label": str(value.get("label") or key),
                "strategy": str(value.get("strategy") or ""),
                "available": resolved is not None,
                "resolved_path": str(resolved) if resolved else None,
            }
        )

    primary_count = sum(
        str(item.get("role") or item.get("topic_role") or "").strip().casefold() == "primary"
        for item in topics
    )
    supporting_count = sum(
        str(item.get("role") or item.get("topic_role") or "").strip().casefold() == "supporting"
        for item in topics
    )

    return {
        "success": True,
        "run_id": str(run_id),
        "approved_topics": topics,
        "topic_count": len(topics),
        "primary_topic_count": primary_count,
        "supporting_topic_count": supporting_count,
        "actual_chunk_evidence_available": bool(
            approved.get("actual_chunk_evidence_available")
        ),
        "models": model_options,
        "notebook_options": notebook_options,
    }


@app.post("/api/runs/{run_id}/assessment/start")
def start_assessment(
    run_id: str,
    body: AssessmentStartBody,
    background_tasks: BackgroundTasks,
):
    run_dir = _resolve_run_dir(run_id)
    _validate_assessment_request(run_dir=run_dir, body=body)

    current = _load_json(_assessment_status_path(run_dir))
    if str(current.get("status") or "").strip().casefold() in {"queued", "running"}:
        raise HTTPException(
            status_code=409,
            detail="An Agent 2 assessment workflow is already running for this lesson.",
        )

    # A brand-new assessment starts a brand-new USER regeneration budget.
    # This file-backed counter is intentionally separate from Notebook 06's
    # internal validation/regeneration attempts.
    _reset_user_regeneration_state(run_dir)

    request_data = body.model_dump()
    _write_assessment_status(
        run_dir,
        status="queued",
        stage="queued",
        progress=5,
        message="Agent 2 request queued.",
        mode=body.mode,
        request=request_data,
        error=None,
        reset=True,
    )

    background_tasks.add_task(
        _run_assessment_background,
        str(run_id),
        request_data,
    )
    return _assessment_status_payload(str(run_id))


@app.get("/api/runs/{run_id}/assessment/status")
def assessment_status(run_id: str):
    return _assessment_status_payload(str(run_id))


@app.post("/api/runs/{run_id}/retrieval-feedback/{question_id}")
def retrieval_feedback(run_id: str, question_id: str, body: RetrievalFeedbackBody):
    run_dir = _resolve_run_dir(run_id)
    package_path, package = _current_official_package(run_dir)
    if not package_path or not package:
        raise HTTPException(status_code=409, detail="No current official retrieval package is available.")

    target: dict[str, Any] | None = None
    for item in (package.get("questions") or []):
        if not isinstance(item, dict):
            continue
        item_id = str(
            item.get("question_id")
            or (item.get("question") or {}).get("question_id")
            or (item.get("question") or {}).get("id")
            or ""
        ).strip()
        if item_id == str(question_id).strip():
            target = item
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Official question was not found in the current retrieval package.")

    try:
        result = _persist_agent2_retrieval_feedback(
            run_dir=run_dir,
            package=package,
            item=target,
            decision=body.decision,
            reason=body.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    payload = _assessment_status_payload(str(run_id))
    payload["retrieval_feedback_write_result"] = result
    return payload




def _agent2_question_actions_path(run_dir: Path, quiz_mode: str) -> Path:
    path = (
        Path(run_dir)
        / "output"
        / "integration"
        / f"agent2_question_review_actions_{quiz_mode}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_agent2_question_actions(
    *,
    run_dir: Path,
    quiz_mode: str,
    actions: list[dict[str, Any]],
) -> Path:
    """Write the same question-action payload consumed by Notebook 06 HITL."""
    path = _agent2_question_actions_path(run_dir, quiz_mode)
    _write_json(
        path,
        {
            "schema_version": "agent2-nextjs-question-actions-v1.0.0",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "quiz_mode": quiz_mode,
            "actions": actions,
        },
    )
    return path



def _persist_agent2_question_approval_fallback(
    *,
    run_dir: Path,
    quiz_mode: str,
    action: dict[str, Any],
) -> dict[str, Any]:
    """
    Persist an approve-only question decision when the current MCP/controller
    bridge does not support Notebook 06's neutral `pending` outer decision.

    This approves ONLY the selected generated question for HITL audit. It never
    approves/releases the entire quiz. Approvals are not promoted to Qdrant
    generation memory; corrective actions still go through Notebook 06.
    """
    database_url = str(
        os.getenv("AGENT2_DATABASE_URL", "")
        or os.getenv("DATABASE_URL", "")
        or ""
    ).strip()

    if not database_url:
        raise RuntimeError(
            "Could not persist question approval because DATABASE_URL / "
            "AGENT2_DATABASE_URL is not configured."
        )

    manifest = _quiz_manifest(run_dir, quiz_mode)
    candidate_questions = manifest.get("candidate_questions") or []
    if not isinstance(candidate_questions, list):
        candidate_questions = []

    question_id = str(action.get("question_id") or "").strip()
    try:
        plan_index = int(action.get("plan_index") or 0)
    except (TypeError, ValueError):
        plan_index = 0

    question: dict[str, Any] = {}
    for item in candidate_questions:
        if not isinstance(item, dict):
            continue

        item_id = str(
            item.get("generated_question_id")
            or item.get("question_id")
            or ""
        ).strip()

        try:
            item_plan_index = int(item.get("plan_index") or 0)
        except (TypeError, ValueError):
            item_plan_index = 0

        if (
            (question_id and item_id == question_id)
            or (plan_index > 0 and item_plan_index == plan_index)
        ):
            question = dict(item)
            break

    recorded_at = datetime.now(timezone.utc).isoformat()

    generated_human_review = manifest.get("generated_human_review") or {}
    if not isinstance(generated_human_review, dict):
        generated_human_review = {}

    event = {
        "generation_request_fingerprint": str(
            manifest.get("generation_request_fingerprint") or ""
        ),
        "final_payload_fingerprint": str(
            generated_human_review.get("payload_fingerprint") or ""
        ),
        "question_id": question_id,
        "plan_index": plan_index,
        "topic": str(question.get("topic") or ""),
        "official_reference": str(
            question.get("official_reference") or ""
        ),
        "role": str(question.get("role") or ""),
        "assigned_pattern": str(
            question.get("assessment_pattern")
            or question.get("assigned_pattern")
            or ""
        ),
        "action": "approve",
        "reason": str(action.get("reason") or ""),
        "before_question": question,
        "after_question": question,
        "pipeline_version": str(
            generated_human_review.get("pipeline_version") or ""
        ),
        "recorded_at_utc": recorded_at,
    }

    event_id = hashlib.sha256(
        json.dumps(
            event,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    event["event_id"] = event_id

    ddl = """
    CREATE TABLE IF NOT EXISTS agent2_question_feedback (
        id BIGSERIAL PRIMARY KEY,
        event_id TEXT UNIQUE NOT NULL,
        generation_request_fingerprint TEXT,
        payload_fingerprint TEXT,
        question_id TEXT,
        plan_index INTEGER,
        topic TEXT,
        official_reference TEXT,
        role TEXT,
        assigned_pattern TEXT,
        action TEXT NOT NULL,
        reason TEXT,
        before_question JSONB,
        after_question JSONB,
        model_key TEXT,
        model_id TEXT,
        pipeline_version TEXT,
        memory_eligible BOOLEAN NOT NULL DEFAULT FALSE,
        recorded_at_utc TIMESTAMPTZ NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """

    insert_sql = text(
        """
        INSERT INTO agent2_question_feedback (
            event_id,
            generation_request_fingerprint,
            payload_fingerprint,
            question_id,
            plan_index,
            topic,
            official_reference,
            role,
            assigned_pattern,
            action,
            reason,
            before_question,
            after_question,
            model_key,
            model_id,
            pipeline_version,
            memory_eligible,
            recorded_at_utc
        ) VALUES (
            :event_id,
            :generation_request_fingerprint,
            :payload_fingerprint,
            :question_id,
            :plan_index,
            :topic,
            :official_reference,
            :role,
            :assigned_pattern,
            :action,
            :reason,
            CAST(:before_question AS JSONB),
            CAST(:after_question AS JSONB),
            :model_key,
            :model_id,
            :pipeline_version,
            FALSE,
            CAST(:recorded_at_utc AS TIMESTAMPTZ)
        )
        ON CONFLICT (event_id) DO NOTHING
        """
    )

    engine = create_engine(
        _normalize_database_url(database_url),
        pool_pre_ping=True,
        future=True,
    )

    with engine.begin() as connection:
        connection.execute(text(ddl))
        result = connection.execute(
            insert_sql,
            {
                "event_id": event_id,
                "generation_request_fingerprint": event[
                    "generation_request_fingerprint"
                ],
                "payload_fingerprint": event[
                    "final_payload_fingerprint"
                ],
                "question_id": question_id,
                "plan_index": plan_index,
                "topic": event["topic"],
                "official_reference": event["official_reference"],
                "role": event["role"],
                "assigned_pattern": event["assigned_pattern"],
                "action": "approve",
                "reason": event["reason"],
                "before_question": json.dumps(
                    question,
                    ensure_ascii=False,
                    default=str,
                ),
                "after_question": json.dumps(
                    question,
                    ensure_ascii=False,
                    default=str,
                ),
                "model_key": str(
                    manifest.get("selected_model_key") or ""
                ),
                "model_id": str(
                    manifest.get("generation_model") or ""
                ),
                "pipeline_version": event["pipeline_version"],
                "recorded_at_utc": recorded_at,
            },
        )
        persisted = max(0, int(result.rowcount or 0))

    feedback_path = (
        _quiz_output_dir(run_dir, quiz_mode)
        / "generated_human_review_feedback.jsonl"
    )

    existing_ids: set[str] = set()
    if feedback_path.is_file():
        try:
            lines = feedback_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError:
            lines = []

        for line in lines:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            existing_id = str(payload.get("event_id") or "")
            if existing_id:
                existing_ids.add(existing_id)

    if event_id not in existing_ids:
        feedback_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with feedback_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    return {
        "status": "QUESTION_APPROVAL_PERSISTED",
        "persisted": persisted,
        "event_id": event_id,
        "qdrant_promoted": False,
        "bridge_fallback": True,
    }


def _call_agent2_question_review_bridge(
    *,
    run_dir: Path,
    run_id: str,
    quiz_mode: str,
    action: dict[str, Any],
    reason: str,
) -> Any:
    """
    Forward one question-level action through the existing LangGraph/MCP bridge.

    This mirrors the working Streamlit Notebook 06 integration:
      * approve -> outer decision 'pending' (never approve the whole quiz)
      * corrective/reject/regenerate -> outer decision 'regenerate'
      * question action JSON/path is supplied through both supported kwargs and
        the existing AGENT2_QUIZ_REVIEW_ACTIONS_* compatibility environment.
    """
    actions = [dict(action)]
    action_path = _write_agent2_question_actions(
        run_dir=run_dir,
        quiz_mode=quiz_mode,
        actions=actions,
    )

    if not str(os.getenv("AGENT2_DATABASE_URL", "") or "").strip():
        database_url = str(os.getenv("DATABASE_URL", "") or "").strip()
        if database_url:
            os.environ["AGENT2_DATABASE_URL"] = database_url

    previous_actions_path = os.environ.get("AGENT2_QUIZ_REVIEW_ACTIONS_PATH")
    previous_actions_json = os.environ.get("AGENT2_QUIZ_REVIEW_ACTIONS_JSON")

    os.environ["AGENT2_QUIZ_REVIEW_ACTIONS_PATH"] = str(action_path.resolve())
    os.environ["AGENT2_QUIZ_REVIEW_ACTIONS_JSON"] = json.dumps(
        actions,
        ensure_ascii=False,
    )

    signature = inspect.signature(submit_human_agent2_quiz_review)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    kwargs: dict[str, Any] = {
        "frontend_root": BACKEND_RUNTIME_ROOT,
        "run_id": str(run_id),
        "quiz_mode": quiz_mode,
        "decision": "pending" if action.get("action") == "approve" else "regenerate",
        "reason": reason,
        "reviewed_by": "nextjs_human_ui",
    }

    def add_optional(names: list[str], value: Any) -> None:
        for name in names:
            if name in parameters or accepts_var_kwargs:
                kwargs[name] = value
                return

    add_optional(
        [
            "question_actions",
            "question_level_actions",
            "review_actions",
            "actions",
        ],
        actions,
    )
    add_optional(
        [
            "question_actions_path",
            "question_level_actions_path",
            "review_actions_path",
        ],
        str(action_path.resolve()),
    )

    if not accepts_var_kwargs:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }

    try:
        return submit_human_agent2_quiz_review(**kwargs)
    except Exception as exc:
        # The current MCP tool only accepts whole-quiz
        # approve/regenerate/reject. Question-level approve deliberately uses
        # neutral "pending", so fall back to the same safe audit persistence
        # already used by the working Streamlit integration.
        if str(action.get("action") or "").strip().casefold() == "approve":
            persisted = _persist_agent2_question_approval_fallback(
                run_dir=run_dir,
                quiz_mode=quiz_mode,
                action=action,
            )
            persisted["original_bridge_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return persisted
        raise
    finally:
        if previous_actions_path is None:
            os.environ.pop("AGENT2_QUIZ_REVIEW_ACTIONS_PATH", None)
        else:
            os.environ["AGENT2_QUIZ_REVIEW_ACTIONS_PATH"] = previous_actions_path

        if previous_actions_json is None:
            os.environ.pop("AGENT2_QUIZ_REVIEW_ACTIONS_JSON", None)
        else:
            os.environ["AGENT2_QUIZ_REVIEW_ACTIONS_JSON"] = previous_actions_json



def _question_from_quiz_manifest(
    manifest: dict[str, Any],
    *,
    question_id: str,
    plan_index: int,
) -> dict[str, Any] | None:
    """
    Locate the selected generated question in the current candidate manifest.

    Prefer stable question_id, with plan_index as the deterministic fallback.
    """
    clean_question_id = str(question_id or "").strip()

    containers: list[list[Any]] = []

    for key in ("candidate_questions", "questions"):
        value = manifest.get(key) or []
        if isinstance(value, list):
            containers.append(value)

    for items in containers:
        for position, raw in enumerate(items, start=1):
            if not isinstance(raw, dict):
                continue

            item_id = str(
                raw.get("generated_question_id")
                or raw.get("question_id")
                or raw.get("id")
                or ""
            ).strip()

            try:
                item_plan_index = int(
                    raw.get("plan_index")
                    or position
                )
            except (TypeError, ValueError):
                item_plan_index = position

            if clean_question_id and item_id == clean_question_id:
                return dict(raw)

            if int(plan_index) > 0 and item_plan_index == int(plan_index):
                return dict(raw)

    return None


def _question_text_signature(question: dict[str, Any] | None) -> str:
    if not isinstance(question, dict):
        return ""

    value = str(
        question.get("question_text")
        or question.get("text")
        or ""
    )

    return re.sub(
        r"\\s+",
        " ",
        value,
    ).strip().casefold()




def _human_regeneration_commit_gate_path(
    run_dir: Path,
    quiz_mode: str,
) -> Path:
    return (
        _quiz_output_dir(
            run_dir,
            quiz_mode,
        )
        / "human_targeted_regeneration_commit_gate.json"
    )


def _human_regeneration_commit_gate_mtime_ns(
    run_dir: Path,
    quiz_mode: str,
) -> int:
    path = _human_regeneration_commit_gate_path(
        run_dir,
        quiz_mode,
    )

    try:
        return (
            int(path.stat().st_mtime_ns)
            if path.is_file()
            else 0
        )
    except OSError:
        return 0


def _fresh_human_regeneration_commit_gate(
    *,
    run_dir: Path,
    quiz_mode: str,
    previous_mtime_ns: int,
    plan_index: int,
) -> dict[str, Any] | None:
    path = _human_regeneration_commit_gate_path(
        run_dir,
        quiz_mode,
    )

    try:
        current_mtime_ns = (
            int(path.stat().st_mtime_ns)
            if path.is_file()
            else 0
        )
    except OSError:
        return None

    if (
        current_mtime_ns <= 0
        or current_mtime_ns
        <= int(previous_mtime_ns or 0)
    ):
        return None

    payload = _load_json(path)

    rows = payload.get("questions") or []
    if not isinstance(rows, list):
        rows = []

    report = next(
        (
            item
            for item in rows
            if (
                isinstance(item, dict)
                and int(
                    item.get("plan_index")
                    or 0
                )
                == int(plan_index)
            )
        ),
        None,
    )

    if not isinstance(report, dict):
        return {
            "status": str(
                payload.get("status")
                or ""
            ),
            "committable": bool(
                payload.get("committed")
            ),
            "issues": [],
            "warnings": [],
            "updated_at_utc": payload.get(
                "updated_at_utc"
            ),
        }

    return {
        **report,
        "gate_status": str(
            payload.get("status")
            or report.get("status")
            or ""
        ),
        "gate_committed": bool(
            payload.get("committed")
        ),
        "updated_at_utc": payload.get(
            "updated_at_utc"
        ),
    }


def _regeneration_commit_gate_error_detail(
    report: dict[str, Any],
) -> str:
    issues = [
        str(value).strip()
        for value in (
            report.get("issues")
            or []
        )
        if str(value).strip()
    ]

    return (
        "Regeneration incomplete: "
        + (
            " ".join(issues)
            if issues
            else (
                "the regenerated question bundle did not pass "
                "post-regeneration validation."
            )
        )
    )




@app.get("/api/runs/{run_id}/assessment/eta")
def get_agent2_eta(
    run_id: str,
    process: Literal[
        "official_retrieval",
        "complete_quiz_generation",
        "shortfall_generation",
        "question_regeneration",
    ] = Query(...),
):
    run_dir = _resolve_run_dir(run_id)

    status = _load_json(
        _assessment_status_path(run_dir)
    )

    request = (
        status.get("request")
        if isinstance(status.get("request"), dict)
        else {}
    )
    shortfall = (
        status.get("shortfall")
        if isinstance(status.get("shortfall"), dict)
        else {}
    )

    estimate = _agent2_eta_estimate(
        process=process,
        request=request,
        shortfall=shortfall,
        elapsed_seconds=0.0,
    )

    return {
        "success": True,
        "run_id": str(run_id),
        "process": process,
        **estimate,
    }



@app.post("/api/runs/{run_id}/question-review")
def question_review(run_id: str, body: QuestionReviewBody):
    run_dir = _resolve_run_dir(run_id)

    question_id = str(body.question_id or "").strip()
    if not question_id:
        raise HTTPException(status_code=400, detail="question_id is required.")

    reason = str(body.reason or "").strip()
    if body.action != "approve" and not reason:
        raise HTTPException(
            status_code=400,
            detail="A short written reason is required for this question action.",
        )

    if body.action == "approve" and not reason:
        reason = f"Question {question_id} approved in the Next.js human review."

    if body.action == "edit_question" and not str(body.question_text or "").strip():
        raise HTTPException(
            status_code=400,
            detail="Corrected question text cannot be blank.",
        )

    if body.action == "edit_marking_guidance" and not body.marking_guidance:
        raise HTTPException(
            status_code=400,
            detail="Add at least one marking-guidance row.",
        )

    # ------------------------------------------------------------------
    # NEW USER regeneration budget
    # ------------------------------------------------------------------
    # This check is ONLY for a human pressing Regenerate in the UI.
    # Notebook/model validation retries use their own existing counters and
    # never touch this state.
    user_attempt_before: dict[str, Any] | None = None
    if body.action == "regenerate":
        allowed = _ensure_user_regeneration_allowed(
            run_dir=run_dir,
            quiz_mode=body.quiz_mode,
            plan_indexes=[int(body.plan_index)],
        )
        user_attempt_before = allowed.get(int(body.plan_index))

    # Capture the pre-action candidate and commit-gate timestamp.
    # The timestamp prevents a stale report from an older HITL attempt being
    # mistaken for the result of this request.
    before_commit_gate_mtime_ns = (
        _human_regeneration_commit_gate_mtime_ns(
            run_dir,
            body.quiz_mode,
        )
        if body.action == "regenerate"
        else 0
    )

    before_manifest = _quiz_manifest(
        run_dir,
        body.quiz_mode,
    )
    before_question = _question_from_quiz_manifest(
        before_manifest,
        question_id=question_id,
        plan_index=int(body.plan_index),
    )
    before_signature = _question_text_signature(
        before_question
    )

    action_payload: dict[str, Any] = {
        "question_id": question_id,
        "plan_index": int(body.plan_index),
        "action": body.action,
        "reason": reason,
    }

    if body.action == "edit_question":
        action_payload["question_text"] = str(body.question_text or "").strip()

    if body.action == "edit_marking_guidance":
        rows = [
            {
                "marks": int(row.marks),
                "criterion": str(row.criterion or "").strip(),
            }
            for row in body.marking_guidance
            if str(row.criterion or "").strip()
        ]
        if not rows:
            raise HTTPException(
                status_code=400,
                detail="Add at least one valid marking-guidance row.",
            )
        action_payload["marking_guidance"] = rows

    review_bridge_started_epoch = time.time()

    try:
        result = _call_agent2_question_review_bridge(
            run_dir=run_dir,
            run_id=str(run_id),
            quiz_mode=body.quiz_mode,
            action=action_payload,
            reason=reason,
        )
    except Exception as exc:
        if body.action == "regenerate":
            commit_gate = _fresh_human_regeneration_commit_gate(
                run_dir=run_dir,
                quiz_mode=body.quiz_mode,
                previous_mtime_ns=before_commit_gate_mtime_ns,
                plan_index=int(body.plan_index),
            )

            if (
                isinstance(commit_gate, dict)
                and not bool(
                    commit_gate.get(
                        "committable",
                        commit_gate.get(
                            "gate_committed",
                            False,
                        ),
                    )
                )
            ):
                raise HTTPException(
                    status_code=409,
                    detail=_regeneration_commit_gate_error_detail(
                        commit_gate
                    ),
                ) from exc

        # Failed backend/model/system retries do NOT consume a user attempt.
        raise HTTPException(
            status_code=500,
            detail=(
                "Question-level review failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc
    finally:
        if body.action == "regenerate":
            try:
                timing_status = _load_json(
                    _assessment_status_path(run_dir)
                )

                _append_agent2_timing_event(
                    run_dir=run_dir,
                    process="question_regeneration",
                    duration_seconds=(
                        time.time()
                        - review_bridge_started_epoch
                    ),
                    request=(
                        timing_status.get("request")
                        if isinstance(
                            timing_status.get("request"),
                            dict,
                        )
                        else {}
                    ),
                    shortfall=(
                        timing_status.get("shortfall")
                        if isinstance(
                            timing_status.get("shortfall"),
                            dict,
                        )
                        else {}
                    ),
                    stage_started_at_utc=(
                        datetime.fromtimestamp(
                            review_bridge_started_epoch,
                            tz=timezone.utc,
                        ).isoformat()
                    ),
                    outcome="attempt_finished",
                    source="question_review_bridge",
                )
            except Exception:
                pass

    changed: bool | None = None
    user_attempt_after: dict[str, Any] | None = None
    commit_gate: dict[str, Any] | None = None

    if body.action == "regenerate":
        commit_gate = _fresh_human_regeneration_commit_gate(
            run_dir=run_dir,
            quiz_mode=body.quiz_mode,
            previous_mtime_ns=before_commit_gate_mtime_ns,
            plan_index=int(body.plan_index),
        )

        if (
            isinstance(commit_gate, dict)
            and not bool(
                commit_gate.get(
                    "committable",
                    commit_gate.get(
                        "gate_committed",
                        False,
                    ),
                )
            )
        ):
            raise HTTPException(
                status_code=409,
                detail=_regeneration_commit_gate_error_detail(
                    commit_gate
                ),
            )

        after_manifest = _quiz_manifest(
            run_dir,
            body.quiz_mode,
        )
        after_question = _question_from_quiz_manifest(
            after_manifest,
            question_id=question_id,
            plan_index=int(body.plan_index),
        )
        after_signature = _question_text_signature(
            after_question
        )

        if not after_signature:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Regeneration finished but the regenerated question "
                    "could not be found in the latest quiz manifest. "
                    "No user regeneration attempt was consumed."
                ),
            )

        changed = (
            not before_signature
            or after_signature != before_signature
        )

        if not changed:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Regeneration completed, but the model returned the same "
                    "question text. The question was not marked as successfully "
                    "regenerated and no user regeneration attempt was consumed. "
                    "Try again with a more specific reason."
                ),
            )

        # Only NOW, after a changed question has passed the existing commit
        # gate, consume exactly one USER attempt for this stable question slot.
        recorded = _record_successful_user_regeneration(
            run_dir=run_dir,
            quiz_mode=body.quiz_mode,
            plan_indexes=[int(body.plan_index)],
            scope="question",
            reason=reason,
            question_ids={
                int(body.plan_index): str(
                    (after_question or {}).get("generated_question_id")
                    or (after_question or {}).get("question_id")
                    or question_id
                )
            },
        )
        user_attempt_after = recorded.get(int(body.plan_index))

    # Re-read AFTER successful attempt persistence so the UI receives the new
    # per-question remaining count in the same response.
    payload = _assessment_status_payload(str(run_id))
    payload["human_write_result"] = result
    payload["question_review_result"] = {
        "question_id": question_id,
        "plan_index": int(body.plan_index),
        "action": body.action,
        "quiz_mode": body.quiz_mode,
        "question_changed": changed,
        "user_regeneration_attempt_before": user_attempt_before,
        "user_regeneration_attempt_after": user_attempt_after,
        "regeneration_commit_gate": commit_gate,
    }
    return payload


@app.post("/api/runs/{run_id}/quiz-review")
def quiz_review(run_id: str, body: QuizReviewBody):
    run_dir = _resolve_run_dir(run_id)
    reason = str(body.reason or "").strip()
    if not reason:
        raise HTTPException(
            status_code=400,
            detail="A written review reason is required.",
        )

    if not str(os.getenv("AGENT2_DATABASE_URL", "") or "").strip():
        database_url = str(os.getenv("DATABASE_URL", "") or "").strip()
        if database_url:
            os.environ["AGENT2_DATABASE_URL"] = database_url

    affected_plan_indexes: list[int] = []
    before_manifest_mtime_ns = 0
    before_signatures: dict[int, str] = {}
    user_attempts_before: dict[int, dict[str, Any]] = {}

    if body.decision == "regenerate":
        before_manifest = _quiz_manifest(
            run_dir,
            body.quiz_mode,
        )
        affected_plan_indexes = _current_generated_plan_indexes(
            run_dir=run_dir,
            quiz_mode=body.quiz_mode,
            manifest=before_manifest,
        )

        # Whole-quiz regeneration is allowed only if EVERY generated question
        # still has at least one of its two USER attempts remaining.
        user_attempts_before = _ensure_user_regeneration_allowed(
            run_dir=run_dir,
            quiz_mode=body.quiz_mode,
            plan_indexes=affected_plan_indexes,
        )
        before_manifest_mtime_ns = _quiz_manifest_latest_mtime_ns(
            run_dir,
            body.quiz_mode,
        )
        before_signatures = _manifest_question_signatures(before_manifest)

    try:
        result = submit_human_agent2_quiz_review(
            frontend_root=BACKEND_RUNTIME_ROOT,
            run_id=str(run_id),
            quiz_mode=body.quiz_mode,
            decision=body.decision,
            reason=reason,
            reviewed_by="nextjs_human_ui",
        )
    except Exception as exc:
        # A failed whole-quiz regeneration does not consume any USER attempts.
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    user_attempts_after: dict[int, dict[str, Any]] = {}
    changed_plan_indexes: list[int] = []

    if body.decision == "regenerate":
        after_manifest = _quiz_manifest(
            run_dir,
            body.quiz_mode,
        )
        after_manifest_mtime_ns = _quiz_manifest_latest_mtime_ns(
            run_dir,
            body.quiz_mode,
        )
        after_signatures = _manifest_question_signatures(after_manifest)

        missing_slots = [
            plan_index
            for plan_index in affected_plan_indexes
            if not after_signatures.get(plan_index)
        ]
        if missing_slots:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Whole-quiz regeneration finished without a valid regenerated "
                    "question for slot(s) "
                    + ", ".join(str(value) for value in missing_slots)
                    + ". No user regeneration attempts were consumed."
                ),
            )

        changed_plan_indexes = [
            plan_index
            for plan_index in affected_plan_indexes
            if after_signatures.get(plan_index)
            != before_signatures.get(plan_index)
        ]

        # Require evidence that a fresh regenerated quiz was actually committed.
        # This prevents a review-only/no-op response from consuming the user's
        # limited allowance.
        if (
            after_manifest_mtime_ns <= before_manifest_mtime_ns
            and not changed_plan_indexes
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Whole-quiz regeneration did not produce a fresh committed "
                    "quiz. No user regeneration attempts were consumed. Try "
                    "again with a more specific regeneration instruction."
                ),
            )

        question_ids: dict[int, str] = {}
        for plan_index in affected_plan_indexes:
            question = _question_from_quiz_manifest(
                after_manifest,
                question_id="",
                plan_index=plan_index,
            )
            question_ids[plan_index] = str(
                (question or {}).get("generated_question_id")
                or (question or {}).get("question_id")
                or ""
            ).strip()

        # One successful whole-quiz regeneration consumes ONE user attempt for
        # EACH generated question slot. Internal notebook/model retries remain
        # completely separate.
        user_attempts_after = _record_successful_user_regeneration(
            run_dir=run_dir,
            quiz_mode=body.quiz_mode,
            plan_indexes=affected_plan_indexes,
            scope="whole_quiz",
            reason=reason,
            question_ids=question_ids,
        )

    payload = _assessment_status_payload(str(run_id))
    payload["human_write_result"] = result
    payload["quiz_review_result"] = {
        "quiz_mode": body.quiz_mode,
        "decision": body.decision,
        "affected_plan_indexes": affected_plan_indexes,
        "changed_plan_indexes": changed_plan_indexes,
        "user_regeneration_attempts_before": user_attempts_before,
        "user_regeneration_attempts_after": user_attempts_after,
    }
    return payload


@app.get("/api/runs/{run_id}/asset")
def run_asset(
    run_id: str,
    path: str = Query(..., min_length=1),
):
    run_dir = _resolve_run_dir(run_id)
    resolved = _asset_path_for_run(run_dir, path)
    return FileResponse(
        path=str(resolved),
        filename=resolved.name if resolved.suffix.casefold() == ".pdf" else None,
    )