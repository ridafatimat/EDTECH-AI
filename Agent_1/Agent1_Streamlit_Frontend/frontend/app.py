from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from pipeline_runner import create_pipeline_run, run_pipeline, rerun_module3
from agent2_runner import (
    agent2_project_candidates,
    resolve_agent2_notebook,
    resolve_agent2_project_root,
    run_agent2_notebook,
)


from detected_topic_edit_runtime import apply_detected_topic_edit_runtime
from detected_topic_edit_capture import persist_detected_topic_edit_memory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Notebook 06 v2.34+ stores question-level Agent 2 feedback under
# AGENT2_DATABASE_URL. Reuse the project's existing DATABASE_URL unless an
# Agent 2-specific connection has explicitly been configured.
if not str(os.getenv("AGENT2_DATABASE_URL", "") or "").strip():
    _shared_database_url = str(os.getenv("DATABASE_URL", "") or "").strip()
    if _shared_database_url:
        os.environ["AGENT2_DATABASE_URL"] = _shared_database_url

# Phase 10 — shared EDTECH controller/MCP bridge.
# Streamlit remains the UI; orchestration is delegated to the controller.
EDTECH_ROOT = PROJECT_ROOT.parents[1]
if str(EDTECH_ROOT) not in sys.path:
    sys.path.insert(0, str(EDTECH_ROOT))

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
from streamlit_integration.langgraph_tracker import (
    empty_tracker,
    render_tracker,
    tracker_from_snapshot,
    update_tracker_from_graph_update,
)

PHASE10_STREAMLIT_MCP_BRIDGE = True
LANGGRAPH_STEP4_STREAMLIT = True
LANGGRAPH_TRACKER_PLACEHOLDER = None


def _step4_render_tracker_state(tracker: dict[str, Any]) -> None:
    st.session_state["langgraph_tracker"] = tracker
    if LANGGRAPH_TRACKER_PLACEHOLDER is not None:
        render_tracker(LANGGRAPH_TRACKER_PLACEHOLDER, tracker)


def _step4_on_graph_update(update: dict[str, Any]) -> None:
    tracker = update_tracker_from_graph_update(
        st.session_state.get("langgraph_tracker"),
        update,
    )
    _step4_render_tracker_state(tracker)


def _step4_refresh_tracker(run_id: str | None) -> dict[str, Any]:
    if not run_id:
        tracker = empty_tracker()
        _step4_render_tracker_state(tracker)
        return tracker
    try:
        snapshot = langgraph_snapshot(frontend_root=PROJECT_ROOT, run_id=run_id)
        tracker = tracker_from_snapshot(snapshot, run_id=run_id)
    except Exception as exc:
        tracker = st.session_state.get("langgraph_tracker") or empty_tracker(run_id=run_id)
        tracker["state_reason"] = f"Tracker snapshot unavailable: {exc}"
    _step4_render_tracker_state(tracker)
    return tracker


def _step4_schedule_langgraph_resume(run_id: str) -> None:
    st.session_state["langgraph_resume_pending_run_id"] = str(run_id)


def _agent1_code_root(project_root: Path) -> Path:
    """
    Return the folder that contains the existing ``app`` package.

    Streamlit lives in:
        Agent_1/Agent1_Streamlit_Frontend/frontend/app.py

    The database package lives in:
        Agent_1/app/
    """
    candidates = [
        project_root,
        project_root.parent,
    ]

    for candidate in candidates:
        if (candidate / "app" / "db").is_dir():
            return candidate

    raise RuntimeError(
        "Could not find the Agent 1 app/db package. Expected it beside "
        "Agent1_Streamlit_Frontend."
    )


def _matching_record_id(value: Any, record_id: int) -> bool:
    try:
        return int(value) == record_id
    except (TypeError, ValueError):
        return False


def _update_review_status_in_json(
    value: Any,
    *,
    record_id: int,
    status: str,
) -> int:
    """Update matching review records inside a Module 1 JSON payload."""

    changed = 0

    if isinstance(value, dict):
        possible_ids = (
            value.get("memory_id"),
            value.get("record_id"),
            value.get("correction_id"),
            value.get("id"),
        )

        if any(
            _matching_record_id(candidate, record_id)
            for candidate in possible_ids
        ):
            value["memory_status"] = status
            value["review_status"] = status

            if "status" in value:
                value["status"] = status

            changed += 1

        for child in value.values():
            changed += _update_review_status_in_json(
                child,
                record_id=record_id,
                status=status,
            )

    elif isinstance(value, list):
        for child in value:
            changed += _update_review_status_in_json(
                child,
                record_id=record_id,
                status=status,
            )

    return changed


def _persist_run_review_status(
    *,
    run_dir: Path | None,
    record_id: int,
    status: str,
) -> int:
    """
    Keep the completed run JSON aligned with PostgreSQL.

    PostgreSQL remains the source of truth. Updating the saved run JSON makes
    Approved/Rejected remain visible after a browser refresh.
    """

    if run_dir is None:
        return 0

    run_dir = Path(run_dir)
    output_dir = run_dir / "output"

    if not output_dir.is_dir():
        return 0

    changed_records = 0

    for json_path in output_dir.glob("*/01_preprocessing.json"):
        try:
            payload = json.loads(
                json_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue

        changed = _update_review_status_in_json(
            payload,
            record_id=record_id,
            status=status,
        )

        if changed:
            json_path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            changed_records += changed

    return changed_records


def human_review_set_status(
    record_id: int,
    status: str,
    *,
    project_root: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Approve or reject a technical-correction memory record.

    The existing PostgreSQL repository owns the real ``set_status`` operation;
    the frontend calls it through this application-layer helper.
    """

    normalised_status = str(status).strip().casefold()

    if normalised_status not in {"approved", "rejected"}:
        raise ValueError(
            "Human-review status must be 'approved' or 'rejected'."
        )

    root = Path(project_root or PROJECT_ROOT).resolve()
    code_root = _agent1_code_root(root)

    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    load_dotenv(code_root / ".env", override=False)
    load_dotenv(root / ".env", override=False)

    try:
        from app.db.repositories.technical_correction_repository import (
            PostgreSQLTechnicalCorrectionRepository,
        )
        from app.db.session import session_scope
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the existing Agent 1 PostgreSQL correction "
            "repository. Confirm that Agent_1/app/db exists and that the "
            "virtual environment contains the project requirements."
        ) from exc

    with session_scope() as session:
        repository = PostgreSQLTechnicalCorrectionRepository(session)
        repository.set_status(
            int(record_id),
            normalised_status,
        )

    updated_json_records = _persist_run_review_status(
        run_dir=run_dir,
        record_id=int(record_id),
        status=normalised_status,
    )

    return {
        "record_id": int(record_id),
        "status": normalised_status,
        "updated_json_records": updated_json_records,
    }



def _normalize_database_url(url: str) -> str:
    url = str(url or "").strip()
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def _topic_review_engine() -> Any:
    """Create the PostgreSQL engine used by Module 3 topic review."""

    database_url = _normalize_database_url(
        os.getenv("DATABASE_URL", "")
    )
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured for topic human review."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
        future=True,
    )


def _persist_topic_review_status_in_json(
    *,
    run_dir: Path,
    record_id: int,
    status: str,
    corrected_decision: str | None = None,
    corrected_mapped_concept_id: str | None = None,
    correction_reason: str | None = None,
) -> int:
    """Keep the current run JSON aligned with PostgreSQL review state."""

    changed_records = 0
    output_root = Path(run_dir) / "output"

    for json_path in output_root.glob("*/03_topic_mapping.json"):
        try:
            payload = json.loads(
                json_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue

        changed = 0

        review_items = payload.get("topic_review_items", [])
        if isinstance(review_items, list):
            for item in review_items:
                if (
                    isinstance(item, dict)
                    and _matching_record_id(item.get("id"), record_id)
                ):
                    item["status"] = status
                    if status == "corrected":
                        item["corrected_decision"] = corrected_decision
                        item["corrected_mapped_concept_id"] = (
                            corrected_mapped_concept_id
                        )
                        item["correction_reason"] = correction_reason
                    changed += 1

        llm_results = payload.get("llm_results", [])
        if isinstance(llm_results, list):
            for item in llm_results:
                if (
                    isinstance(item, dict)
                    and _matching_record_id(
                        item.get("review_id"),
                        record_id,
                    )
                ):
                    item["review_status"] = status
                    if status == "corrected":
                        item["decision"] = corrected_decision
                        item["mapped_concept_id"] = (
                            corrected_mapped_concept_id
                        )
                        item["reason"] = correction_reason
                    changed += 1

        if changed:
            json_path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            changed_records += changed

    return changed_records

def topic_review_set_status(
    *,
    record_id: int,
    status: str,
    run_dir: Path,
    reviewed_by: str = "streamlit",
    corrected_decision: str | None = None,
    corrected_mapped_concept_id: str | None = None,
    correction_reason: str | None = None,
    review_notes: str | None = None,
) -> dict[str, Any]:
    """Persist human Module 3 review, then schedule the same LangGraph thread to resume."""
    result = submit_human_topic_review(
        frontend_root=PROJECT_ROOT,
        run_id=Path(run_dir).name,
        review_id=int(record_id),
        status=status,
        reviewed_by=reviewed_by,
        corrected_decision=corrected_decision,
        corrected_mapped_concept_id=corrected_mapped_concept_id,
        correction_reason=correction_reason,
        review_notes=review_notes,
    )
    _step4_schedule_langgraph_resume(Path(run_dir).name)
    return result



st.set_page_config(
    page_title="Agent 1 Transcript Pipeline",
    page_icon="📘",
    layout="wide",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def deep_get(data: Any, *keys: str, default: Any = None) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def download_file(path: Path, label: str) -> None:
    if path.is_file():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=(
                "application/pdf"
                if path.suffix.casefold() == ".pdf"
                else "application/json"
                if path.suffix.casefold() == ".json"
                else "text/plain"
            ),
            use_container_width=True,
            key=f"download_{path}",
        )


def display_pdf(path: Path) -> None:
    if not path.is_file():
        st.info("PDF is not available.")
        return
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    components.html(
        f'<iframe src="data:application/pdf;base64,{encoded}" '
        'width="100%" height="720" type="application/pdf"></iframe>',
        height=740,
        scrolling=True,
    )



def first_present(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-empty value found under the supplied keys."""

    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def normalise_review_status(value: Any) -> str:
    """Map backend status names to the topic-review UI states."""

    status = str(value or "pending").strip().casefold()
    if status in {"candidate", "pending", "awaiting_review", "needs_review"}:
        return "pending"
    if status in {"approved", "accept", "accepted"}:
        return "approved"
    if status in {"corrected", "correct", "human_corrected"}:
        return "corrected"
    if status in {"rejected", "reject", "declined"}:
        return "rejected"
    return status

def extract_review_items(
    module1_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Return only real correction-memory records that can be reviewed.

    Deterministic spoken-code corrections such as ``array dot length`` ->
    ``array.length`` are useful audit entries, but they are not PostgreSQL
    correction-memory candidates and therefore intentionally have no
    ``memory_id``. They must not appear in the Human Review panel.
    """

    technical_result = deep_get(
        module1_json,
        "technical_normalisation_result",
        default={},
    ) or {}

    explicit_sources: list[tuple[Any, str | None]] = [
        (module1_json.get("review_items"), None),
        (module1_json.get("human_review_items"), None),
        (module1_json.get("correction_memory_items"), None),
        (technical_result.get("review_items"), None),
        (technical_result.get("human_review_items"), None),
        (technical_result.get("correction_memory_items"), None),
        (module1_json.get("pending_review_items"), "pending"),
        (module1_json.get("approved_review_items"), "approved"),
        (module1_json.get("rejected_review_items"), "rejected"),
        (technical_result.get("pending_review_items"), "pending"),
        (technical_result.get("approved_review_items"), "approved"),
        (technical_result.get("rejected_review_items"), "rejected"),
    ]

    collected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def add_items(
        value: Any,
        *,
        forced_status: str | None = None,
    ) -> None:
        if not isinstance(value, list):
            return

        for raw_item in value:
            if not isinstance(raw_item, dict):
                continue

            item = dict(raw_item)
            if forced_status is not None:
                item["status"] = forced_status

            record_id = review_record_id(item)
            if record_id is None or record_id in seen_ids:
                continue

            seen_ids.add(record_id)
            collected.append(item)

    # Prefer explicit human-review lists when Module 1 provides them.
    for value, forced_status in explicit_sources:
        add_items(value, forced_status=forced_status)

    # Backward-compatible fallback for older Module 1 output:
    # include only corrections linked to a real PostgreSQL memory record.
    corrections = technical_result.get("corrections")
    if isinstance(corrections, list):
        for raw_item in corrections:
            if not isinstance(raw_item, dict):
                continue

            source = str(raw_item.get("source") or "").strip().casefold()
            record_id = review_record_id(raw_item)

            if (
                record_id is None
                or record_id in seen_ids
                or source == "spoken_code_rule"
            ):
                continue

            seen_ids.add(record_id)
            collected.append(dict(raw_item))

    return collected


def review_record_id(item: dict[str, Any]) -> int | None:
    value = first_present(item, "memory_id", "record_id", "correction_id", "id")

    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def review_session_key(run_dir: Path, record_id: int) -> str:
    return f"human_review::{run_dir.resolve()}::{record_id}"


def review_item_status(
    item: dict[str, Any],
    *,
    run_dir: Path,
    record_id: int | None,
) -> str:
    stored_status = normalise_review_status(
        first_present(
            item,
            "memory_status",
            "review_status",
            "status",
            default="pending",
        )
    )

    if record_id is None:
        return stored_status

    return normalise_review_status(
        st.session_state.get(
            review_session_key(run_dir, record_id),
            stored_status,
        )
    )


def call_human_review_set_status(
    *,
    record_id: int,
    status: str,
    run_dir: Path,
) -> Any:
    """
    Call pipeline_runner.human_review_set_status using its declared signature.

    This supports the common argument names used by the frontend/backend
    boundary while still failing clearly if the helper exposes an unexpected
    required parameter.
    """

    allowed_statuses = {"approved", "rejected"}
    if status not in allowed_statuses:
        raise ValueError(
            f"Unsupported review status {status!r}. "
            f"Expected one of {sorted(allowed_statuses)}."
        )

    values_by_name: dict[str, Any] = {
        "record_id": record_id,
        "memory_id": record_id,
        "correction_id": record_id,
        "item_id": record_id,
        "status": status,
        "new_status": status,
        "project_root": PROJECT_ROOT,
        "run_dir": run_dir,
    }

    signature = inspect.signature(human_review_set_status)
    positional_args: list[Any] = []
    keyword_args: dict[str, Any] = {}

    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        if parameter.name not in values_by_name:
            if parameter.default is inspect.Parameter.empty:
                raise RuntimeError(
                    "human_review_set_status() has an unsupported required "
                    f"parameter: {parameter.name!r}. Update the adapter in "
                    "frontend/app.py to map that parameter."
                )
            continue

        value = values_by_name[parameter.name]

        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_args.append(value)
        else:
            keyword_args[parameter.name] = value

    return human_review_set_status(*positional_args, **keyword_args)


def review_item_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "Record ID": item.get("_record_id"),
        "Original": first_present(
            item,
            "original",
            "original_phrase",
            "original_span",
            default="",
        ),
        "Suggested correction": first_present(
            item,
            "replacement",
            "corrected_phrase",
            "suggested_replacement",
            default="",
        ),
        "Status": item.get("_display_status"),
        "Confidence": first_present(item, "confidence", default=None),
    }


def render_human_review(
    review_items: list[dict[str, Any]],
    *,
    run_dir: Path,
    review_message: str | None = None,
    review_data_present: bool = False,
) -> None:
    """Render persistent Pending, Approved and Rejected review sections."""

    st.markdown("---")
    st.subheader("Human Review — Technical Corrections")
    st.caption(
        "Pending corrections can be approved or rejected. Approved corrections "
        "become reusable correction-memory records."
    )

    flash_message = st.session_state.pop("human_review_flash", None)
    if flash_message:
        st.success(flash_message)

    grouped: dict[str, list[dict[str, Any]]] = {
        "pending": [],
        "approved": [],
        "rejected": [],
    }

    for item in review_items:
        record_id = review_record_id(item)
        status = review_item_status(
            item,
            run_dir=run_dir,
            record_id=record_id,
        )
        if status not in grouped:
            status = "pending"

        grouped[status].append(
            {
                **item,
                "_record_id": record_id,
                "_display_status": status,
            }
        )

    pending_items = grouped["pending"]
    approved_items = grouped["approved"]
    rejected_items = grouped["rejected"]

    pending_col, approved_col, rejected_col = st.columns(3)
    pending_col.metric("Pending", len(pending_items))
    approved_col.metric("Approved", len(approved_items))
    rejected_col.metric("Rejected", len(rejected_items))

    pending_tab, approved_tab, rejected_tab = st.tabs(
        [
            f"Pending ({len(pending_items)})",
            f"Approved ({len(approved_items)})",
            f"Rejected ({len(rejected_items)})",
        ]
    )

    with pending_tab:
        if not pending_items:
            if not review_items and review_data_present:
                st.info(
                    review_message
                    or "No technical words to correct."
                )
            else:
                st.info(
                    "No pending corrections for this transcript run."
                )

        for index, item in enumerate(pending_items, start=1):
            record_id = item.get("_record_id")
            original = str(
                first_present(
                    item,
                    "original",
                    "original_phrase",
                    "original_span",
                    default="",
                )
            )
            replacement = str(
                first_present(
                    item,
                    "replacement",
                    "corrected_phrase",
                    "suggested_replacement",
                    default="",
                )
            )
            confidence = first_present(item, "confidence", default=None)
            reason = first_present(item, "reason", default="")
            context = first_present(
                item,
                "sentence_text",
                "context",
                "context_keywords",
                default="",
            )
            source_model = first_present(
                item,
                "source_model",
                "model",
                default=None,
            )

            title = f"Pending correction {index}"
            if record_id is not None:
                title += f" — record {record_id}"

            with st.expander(title, expanded=True):
                metadata_parts = ["Status: pending"]
                if confidence is not None:
                    try:
                        metadata_parts.append(
                            f"Confidence: {float(confidence):.2f}"
                        )
                    except (TypeError, ValueError):
                        metadata_parts.append(f"Confidence: {confidence}")
                if source_model:
                    metadata_parts.append(f"Model: {source_model}")
                st.caption(" | ".join(metadata_parts))

                left, right = st.columns(2)
                with left:
                    st.markdown("**Original**")
                    st.code(original or "Not provided", language=None)
                with right:
                    st.markdown("**Suggested correction**")
                    st.code(replacement or "Not provided", language=None)

                if reason:
                    st.markdown("**Reason**")
                    st.write(reason)

                if context:
                    st.markdown("**Transcript context**")
                    if isinstance(context, (list, tuple, set)):
                        st.write(", ".join(str(value) for value in context))
                    else:
                        st.write(context)

                if record_id is None:
                    st.error(
                        "This item has no memory_id/record_id, so its status "
                        "cannot be updated."
                    )
                    continue

                approve_col, reject_col = st.columns(2)
                approve_clicked = approve_col.button(
                    "Approve",
                    key=f"approve_review_{run_dir.name}_{record_id}",
                    type="primary",
                    use_container_width=True,
                )
                reject_clicked = reject_col.button(
                    "Reject",
                    key=f"reject_review_{run_dir.name}_{record_id}",
                    use_container_width=True,
                )

                selected_status = (
                    "approved"
                    if approve_clicked
                    else "rejected"
                    if reject_clicked
                    else None
                )

                if selected_status is not None:
                    try:
                        call_human_review_set_status(
                            record_id=record_id,
                            status=selected_status,
                            run_dir=run_dir,
                        )
                    except Exception as exc:
                        st.error(
                            f"Could not set record {record_id} to "
                            f"{selected_status}: {exc}"
                        )
                    else:
                        st.session_state[
                            review_session_key(run_dir, record_id)
                        ] = selected_status
                        st.session_state["human_review_flash"] = (
                            f"Record {record_id} marked as {selected_status}."
                        )
                        st.rerun()

    with approved_tab:
        if approved_items:
            st.dataframe(
                pd.DataFrame(review_item_row(item) for item in approved_items),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No approved corrections for this transcript run.")

    with rejected_tab:
        if rejected_items:
            st.dataframe(
                pd.DataFrame(review_item_row(item) for item in rejected_items),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No rejected corrections for this transcript run.")

    if not review_items and not review_data_present:
        st.warning(
            "The review UI is active, but this appears to be an older "
            "Module 1 output without the review_items field."
        )



def _topic_resolution_source_label(value: Any) -> str:
    labels = {
        "memory": "PostgreSQL memory",
        "llm": "Groq LLM",
        "module3": "Resolved by Module 3",
        "deterministic": "Deterministic / unresolved",
    }
    key = str(value or "").strip().casefold()
    return labels.get(key, str(value or "Unknown"))


def _topic_review_row(item: dict[str, Any]) -> dict[str, Any]:
    status = normalise_review_status(item.get("status"))
    final_decision = (
        item.get("corrected_decision")
        if status == "corrected"
        else item.get("decision")
    )
    final_concept_id = (
        item.get("corrected_mapped_concept_id")
        if status == "corrected"
        else item.get("mapped_concept_id")
    )

    corrected_label = None
    if status == "corrected" and final_concept_id:
        for candidate in item.get("qdrant_candidates") or []:
            if (
                isinstance(candidate, dict)
                and str(candidate.get("concept_id")) == str(final_concept_id)
            ):
                corrected_label = candidate.get("label")
                break

    return {
        "Review ID": item.get("id"),
        "Detected topic": item.get("rough_topic"),
        "Proposed topic": item.get("mapped_topic"),
        "Final topic": corrected_label or final_concept_id or (
            "Out of syllabus" if final_decision == "out_of_syllabus" else None
        ),
        "Official reference": item.get("official_reference"),
        "Decision": final_decision,
        "Confidence": item.get("confidence"),
        "Status": status,
        "Correction reason": item.get("correction_reason"),
    }



def _current_aqa_spec_version() -> str:
    return (
        os.getenv("AQA_SPEC_VERSION", "").strip()
        or "AQA-8525-v1.2-2022-11-29"
    )


def _detected_topic_reuse_feedback_store() -> Any:
    """Load the final-topic reuse-feedback store without changing pipeline wiring."""
    code_root = _agent1_code_root(PROJECT_ROOT)
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from app.services.detected_topic_edit_reuse_feedback_store import (
        DetectedTopicEditReuseFeedbackStore,
    )

    return DetectedTopicEditReuseFeedbackStore()


def _final_topic_stable_lesson_context(
    *,
    run_dir: Path,
    transcript_name: str,
) -> str:
    """Return a stable exact-lesson context independent of Module 3 output.

    Exact human corrections must survive later runs of the same transcript even
    if Module 3 extracts slightly different candidate/evidence strings.  The
    cleaned transcript is deterministic input-side context, so it is a safer
    exact lesson fingerprint than model-derived candidate evidence.
    """
    output_dir = Path(run_dir) / "output" / transcript_name
    cleaned_path = output_dir / "01_cleaned_transcript.txt"
    cleaned = ""
    if cleaned_path.is_file():
        try:
            cleaned = cleaned_path.read_text(encoding="utf-8").strip()
        except OSError:
            cleaned = ""

    if not cleaned:
        chunk_payload = load_json(output_dir / "02_chunking.json")
        pieces: list[str] = []
        for chunk in chunk_payload.get("chunks", []) or []:
            if not isinstance(chunk, dict):
                continue
            text_value = str(chunk.get("text") or "").strip()
            if text_value:
                pieces.append(text_value)
        cleaned = "\n".join(pieces).strip()

    if not cleaned:
        return ""
    return "LESSON_CONTEXT_V1\n" + cleaned


def _bridge_stable_lesson_feedback_to_backend_evidence(
    *,
    run_dir: Path,
    transcript_name: str,
    raw_module3_result: dict[str, Any],
) -> dict[str, Any]:
    """Mirror stable same-transcript decisions onto current backend evidence.

    The backend comparator intentionally keys exact approve/reject decisions to
    its current evidence representation.  That representation can drift slightly
    across reruns even for the same transcript.  Human corrections are therefore
    first stored against a stable cleaned-transcript context and, on each render,
    mirrored to whatever canonical evidence the backend produced for this run.

    This is exact-context reuse only. Different transcripts do not share the
    stable lesson hash, so normal deterministic comparison remains unchanged.
    """
    stable_context = _final_topic_stable_lesson_context(
        run_dir=run_dir,
        transcript_name=transcript_name,
    )
    if not stable_context:
        return {"checked": 0, "bridged": 0, "skipped": 0}

    store = _detected_topic_reuse_feedback_store()
    try:
        rows = store.feedback_for_evidence(
            current_evidence=stable_context,
            spec_version=_current_aqa_spec_version(),
        )
    except Exception:
        return {"checked": 0, "bridged": 0, "skipped": 0}

    checked = bridged = skipped = 0
    for row in rows:
        checked += 1
        try:
            memory_id = int(row.get("memory_id"))
            memory = store.memory_snapshot(memory_id)
        except Exception:
            memory = None
        if not memory:
            skipped += 1
            continue

        action = str(memory.get("edit_action") or "").strip()
        if action == "add_topic":
            backend_evidence = _final_topic_addition_current_evidence(
                raw_module3_result
            )
        else:
            backend_evidence = _final_topic_existing_current_evidence(
                raw_module3_result=raw_module3_result,
                source_concept_id=memory.get("source_concept_id"),
            )

        if not backend_evidence:
            skipped += 1
            continue

        try:
            store.record(
                memory_id=memory_id,
                current_evidence=backend_evidence,
                decision=str(row.get("decision") or ""),
                reviewer_reason=str(row.get("reviewer_reason") or ""),
                spec_version=str(
                    row.get("spec_version")
                    or memory.get("spec_version")
                    or _current_aqa_spec_version()
                ),
                pipeline_run_id=Path(run_dir).name,
                source_transcript=transcript_name,
                source_concept_id=memory.get("source_concept_id"),
                reviewed_by=str(row.get("reviewed_by") or "streamlit"),
            )
        except Exception:
            skipped += 1
            continue
        bridged += 1

    return {"checked": checked, "bridged": bridged, "skipped": skipped}


def _bootstrap_legacy_manual_add_memory_for_same_lesson(
    *,
    run_dir: Path,
    transcript_name: str,
) -> dict[str, Any]:
    """One-time safe migration for pre-V8 manual add-topic memories.

    Older UI versions could create a reviewer-approved add-topic memory without
    a durable same-transcript approval.  We only bootstrap such a memory when
    its stored human-selected evidence is a long exact normalized substring of
    the current cleaned transcript. This is deliberately stricter than semantic
    similarity and does not generalize the memory to different lesson wording.
    """
    stable_context = _final_topic_stable_lesson_context(
        run_dir=run_dir,
        transcript_name=transcript_name,
    )
    if not stable_context:
        return {"checked": 0, "bootstrapped": 0, "skipped": 0}

    store = _detected_topic_reuse_feedback_store()
    normalized_lesson = store.normalize_evidence(
        stable_context.replace("LESSON_CONTEXT_V1\n", "", 1)
    )
    checked = bootstrapped = skipped = 0

    try:
        memories = store.reusable_add_memories(
            spec_version=_current_aqa_spec_version()
        )
    except Exception:
        return {"checked": 0, "bootstrapped": 0, "skipped": 0}

    for memory in memories:
        checked += 1
        try:
            memory_id = int(memory.get("memory_id"))
        except (TypeError, ValueError):
            skipped += 1
            continue

        try:
            existing = store.get_decision(
                memory_id=memory_id,
                current_evidence=stable_context,
                spec_version=str(
                    memory.get("spec_version")
                    or _current_aqa_spec_version()
                ),
            )
        except Exception:
            existing = None
        if existing is not None:
            continue

        stored_evidence = store.normalize_evidence(
            str(memory.get("stored_evidence") or "")
        )
        # Require substantial verbatim lesson evidence; do not bootstrap from
        # short generic snippets such as "input/output" or "if statement".
        if len(stored_evidence) < 200 or stored_evidence not in normalized_lesson:
            skipped += 1
            continue

        reason = str(memory.get("reviewer_reason") or "").strip()
        if not reason:
            skipped += 1
            continue

        try:
            store.record(
                memory_id=memory_id,
                current_evidence=stable_context,
                decision="approve_reuse",
                reviewer_reason=reason,
                spec_version=str(
                    memory.get("spec_version")
                    or _current_aqa_spec_version()
                ),
                pipeline_run_id=Path(run_dir).name,
                source_transcript=transcript_name,
                source_concept_id=None,
                reviewed_by="streamlit_legacy_exact_evidence_bootstrap",
            )
        except Exception:
            skipped += 1
            continue
        bootstrapped += 1

    return {
        "checked": checked,
        "bootstrapped": bootstrapped,
        "skipped": skipped,
    }


def _final_topic_existing_current_evidence(
    *,
    raw_module3_result: dict[str, Any],
    source_concept_id: str | None,
) -> str:
    """Return the canonical fresh-Module-3 evidence used by the backend gate.

    Important: HITL feedback must be keyed from the untouched fresh Module 3
    result, not from the post-memory effective topic list. This mirrors
    DetectedTopicEditEndToEndService._evidence_by_concept exactly, so a human
    decision written in Streamlit can be found again by the runtime overlay on
    the current run and on a later identical run.
    """
    concept_id = str(source_concept_id or "").strip()
    if not concept_id:
        return ""

    for topic in raw_module3_result.get("merged_topics", []) or []:
        if not isinstance(topic, dict):
            continue
        if str(topic.get("concept_id") or "").strip() != concept_id:
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


def _final_topic_addition_current_evidence(
    raw_module3_result: dict[str, Any],
) -> str:
    """Mirror DetectedTopicEditEndToEndService._addition_current_evidence."""
    evidence: list[str] = []

    for chunk in raw_module3_result.get("chunk_results", []) or []:
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



def _canonicalize_current_run_hitl_feedback(
    *,
    run_dir: Path,
    transcript_name: str,
    raw_module3_result: dict[str, Any],
) -> dict[str, int]:
    """Repair/normalize saved HITL decisions onto backend-canonical evidence.

    The human decision for the current run is authoritative. Older UI builds
    sometimes stored the decision using evidence reconstructed from the
    post-overlay topic list, while the backend lookup uses untouched fresh
    Module 3 evidence. That can leave a valid PostgreSQL decision invisible to
    the runtime overlay.

    This migration is deliberately narrow and generic:
    - it reads only decisions already made by the human for THIS run;
    - it does not invent or change approve/reject choices;
    - it rewrites the same choice against the canonical evidence text used by
      the backend gate;
    - historical edit memories themselves are not modified;
    - future identical runs can therefore find the same exact-context decision.
    """
    try:
        store = _detected_topic_reuse_feedback_store()
        feedback_rows = store.feedback_for_run(Path(run_dir).name)
    except Exception:
        return {"checked": 0, "canonicalized": 0, "skipped": 0}

    checked = 0
    canonicalized = 0
    skipped = 0
    seen_memory_ids: set[int] = set()

    # feedback_for_run is newest-first. Only the latest explicit decision for
    # each memory is authoritative for the current run.
    for row in feedback_rows:
        try:
            memory_id = int(row.get("memory_id"))
        except (TypeError, ValueError):
            skipped += 1
            continue

        if memory_id in seen_memory_ids:
            continue
        seen_memory_ids.add(memory_id)
        checked += 1

        try:
            memory = store.memory_snapshot(memory_id)
        except Exception:
            memory = None
        if not memory:
            skipped += 1
            continue

        action = str(memory.get("edit_action") or "").strip()
        if action == "add_topic":
            canonical_evidence = _final_topic_addition_current_evidence(
                raw_module3_result
            )
        else:
            canonical_evidence = _final_topic_existing_current_evidence(
                raw_module3_result=raw_module3_result,
                source_concept_id=memory.get("source_concept_id"),
            )

        if not canonical_evidence:
            skipped += 1
            continue

        try:
            canonical_hash = store.evidence_hash(canonical_evidence)
        except Exception:
            skipped += 1
            continue

        if str(row.get("current_evidence_hash") or "") == canonical_hash:
            continue

        try:
            store.record(
                memory_id=memory_id,
                current_evidence=canonical_evidence,
                decision=str(row.get("decision") or ""),
                reviewer_reason=str(row.get("reviewer_reason") or ""),
                spec_version=str(
                    row.get("spec_version")
                    or memory.get("spec_version")
                    or _current_aqa_spec_version()
                ),
                pipeline_run_id=Path(run_dir).name,
                source_transcript=transcript_name,
                source_concept_id=memory.get("source_concept_id"),
                reviewed_by=str(row.get("reviewed_by") or "streamlit"),
            )
        except Exception:
            skipped += 1
            continue

        canonicalized += 1

    return {
        "checked": checked,
        "canonicalized": canonicalized,
        "skipped": skipped,
    }


def _materialize_exact_approved_add_topic_memories(
    *,
    run_dir: Path,
    transcript_name: str,
    raw_module3_result: dict[str, Any],
    edit_memory_runtime: dict[str, Any],
    merged_topics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Materialize exact-human-approved missed topics outside strict Module3 scores.

    Module 3 may entirely miss a concept, which means there are no current
    candidate ranking/confidence metrics to copy into a real MergedTopic.  The
    existing manual-addition path already represents such reviewer-approved
    topics with official AQA metadata and null model scores.  Reused add_topic
    memories follow that same representation, but ONLY when the current
    evidence has an exact approve_reuse decision.

    This keeps automatic reuse fail-closed while allowing the same human
    decision to survive Streamlit reruns and later identical transcript runs.
    """
    applied_rows = [
        item
        for item in (edit_memory_runtime.get("applied") or [])
        if isinstance(item, dict) and item.get("action") == "add_topic"
    ]
    if not applied_rows:
        return {"applied_memory_ids": [], "skipped_memory_ids": []}

    try:
        store = _detected_topic_reuse_feedback_store()
        current_evidence = _final_topic_addition_current_evidence(
            raw_module3_result
        )
        catalogue_by_id = {
            str(item.get("concept_id") or "").strip(): item
            for item in _official_aqa_topic_options()
            if isinstance(item, dict)
        }
    except Exception:
        return {"applied_memory_ids": [], "skipped_memory_ids": []}

    present_concept_ids = {
        str(topic.get("concept_id") or "").strip()
        for topic in merged_topics
        if isinstance(topic, dict)
    }
    applied_memory_ids: list[int] = []
    skipped_memory_ids: list[int] = []

    for row in applied_rows:
        try:
            memory_id = int(row.get("memory_id"))
        except (TypeError, ValueError):
            continue

        try:
            memory = store.memory_snapshot(memory_id)
        except Exception:
            memory = None
        if not memory or str(memory.get("edit_action") or "") != "add_topic":
            skipped_memory_ids.append(memory_id)
            continue

        target_concept_id = str(
            memory.get("target_concept_id")
            or row.get("target_concept_id")
            or ""
        ).strip()
        if not target_concept_id:
            skipped_memory_ids.append(memory_id)
            continue

        # A regular add_topic with real current candidate metrics may already
        # have been materialized by the backend adapter. Never duplicate it.
        if target_concept_id in present_concept_ids:
            continue

        try:
            feedback = store.get_decision(
                memory_id=memory_id,
                current_evidence=current_evidence,
                spec_version=str(
                    memory.get("spec_version") or _current_aqa_spec_version()
                ),
            )
        except Exception:
            feedback = None

        if feedback is None or feedback.decision != "approve_reuse":
            skipped_memory_ids.append(memory_id)
            continue

        official = catalogue_by_id.get(target_concept_id)
        if not official:
            skipped_memory_ids.append(memory_id)
            continue

        source_chunk_ids = [
            int(value)
            for value in (memory.get("source_chunk_ids") or [])
            if str(value).lstrip("-").isdigit()
        ]

        evidence_text = ""
        if source_chunk_ids:
            try:
                evidence_text = _detected_topic_evidence_from_chunks(
                    run_dir=run_dir,
                    transcript_name=transcript_name,
                    source_chunk_ids=source_chunk_ids,
                )
            except Exception:
                evidence_text = ""
        if not str(evidence_text).strip():
            evidence_text = str(memory.get("stored_evidence") or "").strip()

        role = str(memory.get("target_role") or "supporting").strip().casefold()
        if role not in {"primary", "supporting"}:
            role = "supporting"

        merged_topics.append(
            {
                "concept_id": target_concept_id,
                "topic": official.get("topic") or memory.get("target_topic"),
                "domain": official.get("domain"),
                "official_reference": official.get("official_reference"),
                "chapter_reference": official.get("chapter_reference"),
                "official_title": official.get("official_title"),
                "paper": official.get("paper"),
                "source_pages": official.get("source_pages") or [],
                # Deliberately null: the concept was missed by fresh Module 3,
                # so no current model ranking/confidence metrics exist.
                "confidence": None,
                "ranking_score": None,
                "topic_role": role,
                "source_chunk_ids": source_chunk_ids,
                "support_span_count": None,
                "mean_semantic_score": None,
                "mean_keyword_score": None,
                "mean_salience_score": None,
                "coverage_score": None,
                "evidence": [evidence_text] if evidence_text else [],
                "supporting_candidate_count": None,
                "human_edited": True,
                "memory_applied": True,
                "human_added_topic": True,
                "human_edit_action": "add_topic",
                "human_edit_reason": feedback.reviewer_reason,
                "detected_topic_edit_memory_id": memory_id,
            }
        )
        present_concept_ids.add(target_concept_id)
        applied_memory_ids.append(memory_id)

    return {
        "applied_memory_ids": applied_memory_ids,
        "skipped_memory_ids": skipped_memory_ids,
    }


def _final_topic_identity(topic: dict[str, Any] | None) -> str:
    """Stable UI identity for comparing fresh and effective topic lists."""
    if not isinstance(topic, dict):
        return ""
    concept_id = str(topic.get("concept_id") or "").strip()
    if concept_id:
        return f"concept:{concept_id}"
    reference = str(topic.get("official_reference") or "").strip()
    label = str(topic.get("topic") or "").strip().casefold()
    return f"fallback:{reference}:{label}" if (reference or label) else ""


def _final_topic_diff_summary(
    *,
    raw_module3_result: dict[str, Any],
    effective_topics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare untouched fresh Module 3 topics with the live effective topic list."""
    fresh_topics = [
        dict(item)
        for item in (raw_module3_result.get("merged_topics") or [])
        if isinstance(item, dict)
    ]
    effective = [dict(item) for item in effective_topics if isinstance(item, dict)]

    fresh_by_key = {
        _final_topic_identity(item): item
        for item in fresh_topics
        if _final_topic_identity(item)
    }
    effective_by_key = {
        _final_topic_identity(item): item
        for item in effective
        if _final_topic_identity(item)
    }

    removed_keys = set(fresh_by_key) - set(effective_by_key)
    added_keys = set(effective_by_key) - set(fresh_by_key)
    shared_keys = set(fresh_by_key) & set(effective_by_key)
    role_changed_keys = {
        key
        for key in shared_keys
        if str(fresh_by_key[key].get("topic_role") or "").strip().casefold()
        != str(effective_by_key[key].get("topic_role") or "").strip().casefold()
    }

    return {
        "fresh_topics": fresh_topics,
        "fresh_by_key": fresh_by_key,
        "effective_by_key": effective_by_key,
        "removed_keys": removed_keys,
        "added_keys": added_keys,
        "role_changed_keys": role_changed_keys,
        "changed_keys": removed_keys | added_keys | role_changed_keys,
    }


def _final_topic_feedback_summary(
    *,
    run_dir: Path,
    edit_memory_runtime: dict[str, Any],
    merged_topics: list[dict[str, Any]],
) -> dict[str, Any]:
    """UI-only provenance summary for final-topic memory / human activity."""
    feedback_rows: list[dict[str, Any]] = []
    try:
        feedback_rows = _detected_topic_reuse_feedback_store().feedback_for_run(
            Path(run_dir).name
        )
    except Exception:
        feedback_rows = []

    human_review_keys: set[str] = set()
    decision_by_concept: dict[str, str] = {}
    feedback_store = None
    memory_cache: dict[int, dict[str, Any] | None] = {}
    try:
        feedback_store = _detected_topic_reuse_feedback_store()
    except Exception:
        feedback_store = None

    for item in feedback_rows:
        concept_id = str(item.get("source_concept_id") or "").strip()
        memory_id = str(item.get("memory_id") or "").strip()

        # add_topic feedback has no source_concept_id. Resolve its target
        # concept so one human decision counts once and the added row receives
        # the correct HITL status instead of being tracked only as memory:N.
        if not concept_id and memory_id and feedback_store is not None:
            try:
                memory_id_int = int(memory_id)
                if memory_id_int not in memory_cache:
                    memory_cache[memory_id_int] = feedback_store.memory_snapshot(
                        memory_id_int
                    )
                memory = memory_cache.get(memory_id_int) or {}
                concept_id = str(
                    memory.get("target_concept_id")
                    or memory.get("source_concept_id")
                    or ""
                ).strip()
            except Exception:
                concept_id = ""

        key = f"concept:{concept_id}" if concept_id else f"memory:{memory_id}"
        if key:
            human_review_keys.add(key)
        if concept_id:
            decision_by_concept[concept_id] = str(item.get("decision") or "")

    audit = _final_topic_hitl_audit_data(
        edit_memory_runtime=edit_memory_runtime,
        merged_topics=merged_topics,
    )
    auto_applied_keys: set[str] = set()
    for row in audit.get("applied_rows", []) or []:
        concept_id = str(
            row.get("_source_concept_id")
            or row.get("_target_concept_id")
            or ""
        ).strip()
        memory_id = str(row.get("Memory ID") or "").strip()
        key = f"concept:{concept_id}" if concept_id else f"memory:{memory_id}"
        if key:
            auto_applied_keys.add(key)

    manual_topic_keys = {
        _final_topic_identity(topic)
        for topic in merged_topics
        if isinstance(topic, dict)
        and bool(topic.get("human_edited") or topic.get("memory_applied"))
        and _final_topic_identity(topic)
    }

    affected_keys = human_review_keys | auto_applied_keys | manual_topic_keys
    return {
        "feedback_rows": feedback_rows,
        "human_review_keys": human_review_keys,
        "auto_applied_keys": auto_applied_keys,
        "manual_topic_keys": manual_topic_keys,
        "affected_keys": affected_keys,
        "affected_count": len(affected_keys),
        "human_decision_count": len(human_review_keys),
        "decision_by_concept": decision_by_concept,
    }


def _final_topic_status_label(
    topic: dict[str, Any],
    *,
    diff: dict[str, Any],
    feedback_summary: dict[str, Any],
) -> str:
    key = _final_topic_identity(topic)
    concept_id = str(topic.get("concept_id") or "").strip()
    decision = str(
        (feedback_summary.get("decision_by_concept") or {}).get(concept_id) or ""
    )

    if key in diff.get("added_keys", set()):
        return "Added vs fresh Module 3"
    if key in diff.get("role_changed_keys", set()):
        return "Role updated by HITL / memory"
    if decision == "reject_reuse":
        return "Human reviewed — fresh result kept"
    if decision == "approve_reuse":
        return "Human approved historical edit"
    if bool(topic.get("human_edited") or topic.get("memory_applied")):
        return "Human / memory edited"
    return "Fresh / unchanged"


def render_live_effective_topic_list(
    *,
    merged_topics: list[dict[str, Any]],
    raw_module3_result: dict[str, Any],
    edit_memory_runtime: dict[str, Any],
    run_dir: Path,
) -> None:
    """Live HITL-side view; updates automatically after every saved decision."""
    primary = [t for t in merged_topics if t.get("topic_role") == "primary"]
    supporting = [t for t in merged_topics if t.get("topic_role") == "supporting"]
    diff = _final_topic_diff_summary(
        raw_module3_result=raw_module3_result,
        effective_topics=merged_topics,
    )
    feedback = _final_topic_feedback_summary(
        run_dir=run_dir,
        edit_memory_runtime=edit_memory_runtime,
        merged_topics=merged_topics,
    )

    st.markdown("#### Live effective topic list")
    st.caption(
        "This list refreshes automatically after every HITL choice. Removed topics "
        "disappear, approved additions appear, and role changes update here immediately — "
        "no manual rerun is required."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Effective topics", len(merged_topics))
    c2.metric("Primary", len(primary))
    c3.metric("Supporting", len(supporting))
    c4.metric("HITL decisions", feedback.get("human_decision_count", 0))
    c5.metric("Topics changed", len(diff.get("changed_keys", set())))

    if merged_topics:
        rows = []
        for topic in merged_topics:
            rows.append(
                {
                    "Topic": topic.get("topic"),
                    "Role": topic.get("topic_role"),
                    "Official reference": topic.get("official_reference"),
                    "HITL status": _final_topic_status_label(
                        topic,
                        diff=diff,
                        feedback_summary=feedback,
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    change_rows: list[dict[str, Any]] = []
    for key in sorted(diff.get("removed_keys", set())):
        topic = diff["fresh_by_key"].get(key, {})
        change_rows.append(
            {
                "Change": "Removed",
                "Topic": topic.get("topic"),
                "Before": topic.get("topic_role"),
                "After": "—",
            }
        )
    for key in sorted(diff.get("added_keys", set())):
        topic = diff["effective_by_key"].get(key, {})
        change_rows.append(
            {
                "Change": "Added",
                "Topic": topic.get("topic"),
                "Before": "—",
                "After": topic.get("topic_role"),
            }
        )
    for key in sorted(diff.get("role_changed_keys", set())):
        before = diff["fresh_by_key"].get(key, {})
        after = diff["effective_by_key"].get(key, {})
        change_rows.append(
            {
                "Change": "Role updated",
                "Topic": after.get("topic") or before.get("topic"),
                "Before": before.get("topic_role"),
                "After": after.get("topic_role"),
            }
        )

    if change_rows:
        with st.expander("Changes vs fresh Module 3", expanded=True):
            st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)


def _final_topic_memory_outcome_label(memory: dict[str, Any]) -> str:
    action = str(memory.get("edit_action") or "").strip()
    source_topic = str(memory.get("source_topic") or "").strip()
    target_topic = str(memory.get("target_topic") or "").strip()
    source_role = str(memory.get("source_role") or "").strip()
    target_role = str(memory.get("target_role") or "").strip()

    if action == "remove_topic":
        return f"Remove {source_topic or memory.get('source_concept_id')}"
    if action == "change_role":
        return (
            f"{source_topic or memory.get('source_concept_id')}: "
            f"{source_role or '?'} → {target_role or '?'}"
        )
    if action == "replace_topic":
        return (
            f"Replace {source_topic or memory.get('source_concept_id')} with "
            f"{target_topic or memory.get('target_concept_id')}"
        )
    if action == "add_topic":
        return (
            f"Add {target_topic or memory.get('target_concept_id')} "
            f"as {target_role or 'supporting'}"
        )
    return action or "Historical edit"


def _append_topic_decision_log(
    *,
    run_dir: Path,
    transcript_name: str,
    normalized_topic: str,
    source_chunk_ids: list[int],
    decision_stage: str,
    action: str,
    decision: str | None = None,
    mapped_concept_id: str | None = None,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
    actor_type: str = "human",
    decided_by: str = "streamlit",
) -> None:
    """Append one reviewer/UI decision without promoting mapping memory."""

    statement = text(
        """
        INSERT INTO topic_mapping_decision_log (
            pipeline_run_id,
            normalized_topic,
            source_transcript,
            source_chunk_ids,
            decision_stage,
            actor_type,
            action,
            decision,
            mapped_concept_id,
            reason,
            decided_by,
            details,
            spec_version
        )
        VALUES (
            :pipeline_run_id,
            :normalized_topic,
            :source_transcript,
            CAST(:source_chunk_ids AS jsonb),
            :decision_stage,
            :actor_type,
            :action,
            :decision,
            :mapped_concept_id,
            :reason,
            :decided_by,
            CAST(:details AS jsonb),
            :spec_version
        )
        """
    )

    engine = _topic_review_engine()
    with engine.begin() as connection:
        connection.execute(
            statement,
            {
                "pipeline_run_id": Path(run_dir).name,
                "normalized_topic": str(normalized_topic).strip().casefold(),
                "source_transcript": transcript_name,
                "source_chunk_ids": json.dumps(source_chunk_ids),
                "decision_stage": decision_stage,
                "actor_type": actor_type,
                "action": action,
                "decision": decision,
                "mapped_concept_id": mapped_concept_id,
                "reason": reason,
                "decided_by": decided_by,
                "details": json.dumps(details or {}, ensure_ascii=False),
                "spec_version": _current_aqa_spec_version(),
            },
        )


def _topic_label_override_path(
    *,
    run_dir: Path,
    transcript_name: str,
) -> Path:
    return (
        Path(run_dir)
        / "output"
        / transcript_name
        / "topic_label_overrides.json"
    )


def _save_topic_label_override(
    *,
    run_dir: Path,
    transcript_name: str,
    source_chunk_ids: list[int],
    action: str,
    rough_topic: str | None = None,
    reason: str | None = None,
) -> Path:
    """Persist reviewer guidance consumed only by a Module 3 rerun."""

    normalised_action = str(action).strip().casefold()
    if normalised_action not in {"label", "ignore"}:
        raise ValueError("Topic-label override action must be label or ignore.")

    label = str(rough_topic or "").strip()
    if normalised_action == "label" and not label:
        raise ValueError("Enter a specific rough topic label.")

    override_path = _topic_label_override_path(
        run_dir=run_dir,
        transcript_name=transcript_name,
    )
    override_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "transcript": transcript_name,
        "overrides": {},
    }
    if override_path.is_file():
        try:
            existing = json.loads(override_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload.update(existing)
        except (OSError, json.JSONDecodeError):
            pass

    overrides = payload.get("overrides")
    if not isinstance(overrides, dict):
        overrides = {}
        payload["overrides"] = overrides

    timestamp = datetime.now(timezone.utc).isoformat()
    for raw_chunk_id in source_chunk_ids:
        try:
            chunk_id = int(raw_chunk_id)
        except (TypeError, ValueError):
            continue
        overrides[str(chunk_id)] = {
            "action": normalised_action,
            "rough_topic": label if normalised_action == "label" else None,
            "reason": str(reason or "").strip() or None,
            "reviewed_by": "streamlit",
            "updated_at": timestamp,
        }

    override_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return override_path


def _normalise_topic_label_memory_evidence(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    return value


def _save_topic_label_memory(
    *,
    transcript_name: str,
    source_chunk_ids: list[int],
    evidence_text: str,
    action: str,
    assigned_rough_topic: str | None = None,
    reason: str | None = None,
) -> int:
    """Persist one reviewer-validated missing-label decision for future runs."""

    normalised_action = str(action or "").strip().casefold()
    if normalised_action not in {"label", "ignore"}:
        raise ValueError("Topic-label memory action must be label or ignore.")

    evidence_text = str(evidence_text or "").strip()
    if not evidence_text:
        raise ValueError("Topic-label memory requires transcript evidence.")

    assigned_label = str(assigned_rough_topic or "").strip() or None
    if normalised_action == "label" and not assigned_label:
        raise ValueError("A specific rough topic label is required.")
    if normalised_action == "ignore":
        assigned_label = None

    normalised_evidence = _normalise_topic_label_memory_evidence(evidence_text)
    evidence_hash = hashlib.sha256(
        normalised_evidence.encode("utf-8")
    ).hexdigest()
    spec_version = _current_aqa_spec_version()
    memory_key = hashlib.sha256(
        f"{spec_version}|{evidence_hash}".encode("utf-8")
    ).hexdigest()

    statement = text(
        """
        INSERT INTO topic_label_memory (
            memory_key,
            evidence_hash,
            evidence_text,
            action,
            assigned_rough_topic,
            reason,
            source_transcript,
            source_chunk_ids,
            spec_version,
            reviewer_approved,
            validation_status,
            reviewed_by,
            reviewed_at
        )
        VALUES (
            :memory_key,
            :evidence_hash,
            :evidence_text,
            :action,
            :assigned_rough_topic,
            :reason,
            :source_transcript,
            CAST(:source_chunk_ids AS jsonb),
            :spec_version,
            TRUE,
            'validated',
            'streamlit',
            NOW()
        )
        ON CONFLICT (memory_key) DO UPDATE SET
            evidence_text = EXCLUDED.evidence_text,
            action = EXCLUDED.action,
            assigned_rough_topic = EXCLUDED.assigned_rough_topic,
            reason = EXCLUDED.reason,
            source_transcript = EXCLUDED.source_transcript,
            source_chunk_ids = EXCLUDED.source_chunk_ids,
            spec_version = EXCLUDED.spec_version,
            reviewer_approved = TRUE,
            validation_status = 'validated',
            reviewed_by = 'streamlit',
            reviewed_at = NOW(),
            updated_at = NOW()
        RETURNING id
        """
    )

    engine = _topic_review_engine()
    with engine.begin() as connection:
        memory_id = connection.execute(
            statement,
            {
                "memory_key": memory_key,
                "evidence_hash": evidence_hash,
                "evidence_text": evidence_text,
                "action": normalised_action,
                "assigned_rough_topic": assigned_label,
                "reason": str(reason or "").strip() or None,
                "source_transcript": transcript_name,
                "source_chunk_ids": json.dumps(source_chunk_ids),
                "spec_version": spec_version,
            },
        ).scalar_one()

    return int(memory_id)


def render_needs_topic_label_review(
    *,
    llm_results: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    run_dir: Path,
    transcript_name: str,
) -> None:
    """Resolve generic semantic residue before it enters memory/Qdrant."""

    needs_label = [
        item
        for item in llm_results
        if isinstance(item, dict)
        and str(item.get("review_status") or "").casefold()
        == "needs_topic_label"
    ]
    if not needs_label:
        return

    chunk_text_by_id: dict[int, str] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        try:
            chunk_id = int(chunk.get("chunk_id"))
        except (TypeError, ValueError):
            continue
        chunk_text_by_id[chunk_id] = str(chunk.get("text") or "").strip()

    st.markdown("---")
    st.subheader("Human Review — Missing Rough Topic Label")
    st.caption(
        "These chunks are clearly Computer Science, but Agent 1 could not "
        "derive a safe specific rough-topic label. Reviewer label/ignore "
        "decisions are stored in PostgreSQL evidence memory, so future "
        "similar chunks consult that memory before asking again. A human "
        "label then continues through the normal mapping memory → Qdrant → "
        "confidence/Groq → human-review path."
    )

    for index, item in enumerate(needs_label, start=1):
        raw_ids = item.get("source_chunk_ids") or []
        source_chunk_ids: list[int] = []
        for raw_id in raw_ids:
            try:
                value = int(raw_id)
            except (TypeError, ValueError):
                continue
            if value not in source_chunk_ids:
                source_chunk_ids.append(value)

        key_suffix = "_".join(str(value) for value in source_chunk_ids) or str(index)
        label_memory_evidence = "\n\n".join(
            chunk_text_by_id.get(chunk_id, "").strip()
            for chunk_id in source_chunk_ids
            if chunk_text_by_id.get(chunk_id, "").strip()
        ).strip()
        if not label_memory_evidence:
            label_memory_evidence = str(item.get("evidence") or item.get("reason") or "").strip()

        with st.expander(
            f"Unlabelled CS content {index} — source chunk(s) {source_chunk_ids or 'unknown'}",
            expanded=True,
        ):
            if item.get("reason"):
                st.markdown("**Why Agent 1 stopped here**")
                st.write(item.get("reason"))

            for chunk_id in source_chunk_ids:
                text_value = chunk_text_by_id.get(chunk_id)
                if text_value:
                    st.markdown(f"**Chunk {chunk_id} evidence**")
                    st.write(text_value)

            label = st.text_input(
                "Assign a specific rough topic label",
                key=f"rough_topic_label_{run_dir.name}_{key_suffix}",
                placeholder="e.g. Searching and sorting algorithms",
            ).strip()
            label_note = st.text_area(
                "Reviewer note (optional)",
                key=f"rough_topic_label_note_{run_dir.name}_{key_suffix}",
                height=90,
            ).strip()

            assign_clicked = st.button(
                "Save label and rerun Module 3",
                key=f"save_topic_label_{run_dir.name}_{key_suffix}",
                type="primary",
                use_container_width=True,
            )

            st.markdown("**Or, if this chunk adds no new lesson topic:**")
            ignore_reason = st.text_area(
                "Why should this be treated as already covered / no additional topic?",
                key=f"ignore_topic_reason_{run_dir.name}_{key_suffix}",
                height=90,
            ).strip()
            ignore_clicked = st.button(
                "Mark as no additional topic and rerun Module 3",
                key=f"ignore_topic_label_{run_dir.name}_{key_suffix}",
                use_container_width=True,
            )

            if assign_clicked:
                if not label:
                    st.error("Enter a specific rough topic label first.")
                else:
                    try:
                        label_memory_id = _save_topic_label_memory(
                            transcript_name=transcript_name,
                            source_chunk_ids=source_chunk_ids,
                            evidence_text=label_memory_evidence,
                            action="label",
                            assigned_rough_topic=label,
                            reason=label_note,
                        )
                        _save_topic_label_override(
                            run_dir=run_dir,
                            transcript_name=transcript_name,
                            source_chunk_ids=source_chunk_ids,
                            action="label",
                            rough_topic=label,
                            reason=label_note,
                        )
                        _append_topic_decision_log(
                            run_dir=run_dir,
                            transcript_name=transcript_name,
                            normalized_topic=label,
                            source_chunk_ids=source_chunk_ids,
                            decision_stage="topic_label_review",
                            action="assign_topic_label",
                            decision="needs_review",
                            reason=label_note or None,
                            details={
                                "previous_label": item.get("rough_topic"),
                                "assigned_label": label,
                                "topic_label_memory_id": label_memory_id,
                            },
                        )
                        with st.spinner(
                            "Re-running only Module 3. Module 1 and Module 2 are not rerun."
                        ):
                            rerun_module3(
                                project_root=PROJECT_ROOT,
                                run_dir=run_dir,
                                transcript_name=transcript_name,
                            )
                    except Exception as exc:
                        st.error(f"Could not rerun Module 3: {exc}")
                    else:
                        st.session_state["topic_review_flash"] = (
                            f"Rough topic '{label}' saved. Module 3 reran through the normal "
                            "memory/Qdrant/Groq review path."
                        )
                        st.rerun()

            if ignore_clicked:
                if not ignore_reason:
                    st.error(
                        "A reason is required before suppressing a possible CS topic."
                    )
                else:
                    try:
                        label_memory_id = _save_topic_label_memory(
                            transcript_name=transcript_name,
                            source_chunk_ids=source_chunk_ids,
                            evidence_text=label_memory_evidence,
                            action="ignore",
                            reason=ignore_reason,
                        )
                        _save_topic_label_override(
                            run_dir=run_dir,
                            transcript_name=transcript_name,
                            source_chunk_ids=source_chunk_ids,
                            action="ignore",
                            reason=ignore_reason,
                        )
                        _append_topic_decision_log(
                            run_dir=run_dir,
                            transcript_name=transcript_name,
                            normalized_topic=str(
                                item.get("rough_topic")
                                or "unmapped computer science content"
                            ),
                            source_chunk_ids=source_chunk_ids,
                            decision_stage="topic_label_review",
                            action="no_additional_topic",
                            decision="no_additional_topic",
                            reason=ignore_reason,
                            details={
                                "previous_label": item.get("rough_topic"),
                                "topic_label_memory_id": label_memory_id,
                            },
                        )
                        with st.spinner(
                            "Re-running only Module 3. Module 1 and Module 2 are not rerun."
                        ):
                            rerun_module3(
                                project_root=PROJECT_ROOT,
                                run_dir=run_dir,
                                transcript_name=transcript_name,
                            )
                    except Exception as exc:
                        st.error(f"Could not rerun Module 3: {exc}")
                    else:
                        st.session_state["topic_review_flash"] = (
                            "Chunk marked as no additional topic. Module 3 reran safely."
                        )
                        st.rerun()


@st.cache_data(show_spinner=False)
def _official_aqa_topic_options() -> list[dict[str, Any]]:
    """Return the existing Agent 1 official catalogue for reviewer dropdowns."""

    code_root = _agent1_code_root(PROJECT_ROOT)
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from app.services.syllabus_store import get_syllabus_store

    concepts = get_syllabus_store().get_all_concepts()

    options = [
        {
            "concept_id": concept.concept_id,
            "topic": concept.label,
            "domain": concept.domain,
            "official_reference": concept.official_reference,
            "chapter_reference": concept.chapter_reference,
            "official_title": concept.official_title,
            "paper": concept.paper,
            "source_pages": list(concept.source_pages),
        }
        for concept in concepts
    ]
    options.sort(
        key=lambda item: (
            str(item.get("official_reference") or ""),
            str(item.get("topic") or ""),
        )
    )
    return options



def _detected_topic_evidence_from_chunks(
    *,
    run_dir: Path,
    transcript_name: str,
    source_chunk_ids: list[int],
) -> str:
    """Recover the current transcript evidence represented by source chunks."""

    chunk_path = (
        Path(run_dir)
        / "output"
        / transcript_name
        / "02_chunking.json"
    )
    chunk_payload = load_json(chunk_path)

    chunk_text_by_id: dict[int, str] = {}
    for chunk in chunk_payload.get("chunks", []) or []:
        if not isinstance(chunk, dict):
            continue
        try:
            chunk_id = int(chunk.get("chunk_id"))
        except (TypeError, ValueError):
            continue

        text_value = str(chunk.get("text") or "").strip()
        if text_value:
            chunk_text_by_id[chunk_id] = text_value

    evidence_parts = [
        chunk_text_by_id[chunk_id]
        for chunk_id in source_chunk_ids
        if chunk_text_by_id.get(chunk_id)
    ]
    evidence_text = "\n\n".join(evidence_parts).strip()

    if not evidence_text:
        raise ValueError(
            "Could not recover transcript evidence for this human correction. "
            "The edit was not saved because reusable HITL memory must include "
            "the evidence that justified the reason."
        )

    return evidence_text


def _promote_detected_topic_replacement_to_memory(
    *,
    run_dir: Path,
    transcript_name: str,
    original_topic: dict[str, Any],
    replacement_concept_id: str,
    reason: str,
    source_chunk_ids: list[int],
) -> int:
    """
    Persist an explicit official-topic replacement as a human correction.

    Role changes and removals remain current-run curation only. Replacing an
    official AQA mapping is different: the reviewer is explicitly saying that
    the system mapped the evidence to the wrong official concept. That
    correction therefore follows the same PostgreSQL human-review -> audit ->
    reusable-memory path as the normal Correct action.
    """

    reason_text = str(reason or "").strip()
    if not reason_text:
        raise ValueError("A reason is required for a reusable mapping correction.")

    original_label = str(original_topic.get("topic") or "").strip()
    original_concept_id = str(original_topic.get("concept_id") or "").strip() or None
    normalized_topic = original_label.casefold()
    if not normalized_topic:
        raise ValueError("The detected topic has no reusable topic label.")

    # Reconstruct the exact transcript evidence represented by this retained
    # topic. Memory compatibility should learn from evidence, not from a topic
    # label alone.
    chunk_path = (
        Path(run_dir)
        / "output"
        / transcript_name
        / "02_chunking.json"
    )
    chunk_payload = load_json(chunk_path)
    chunk_text_by_id: dict[int, str] = {}
    for chunk in chunk_payload.get("chunks", []) or []:
        if not isinstance(chunk, dict):
            continue
        try:
            chunk_id = int(chunk.get("chunk_id"))
        except (TypeError, ValueError):
            continue
        text_value = str(chunk.get("text") or "").strip()
        if text_value:
            chunk_text_by_id[chunk_id] = text_value

    evidence_parts = [
        chunk_text_by_id[chunk_id]
        for chunk_id in source_chunk_ids
        if chunk_text_by_id.get(chunk_id)
    ]
    evidence_text = "\n\n".join(evidence_parts).strip()
    if not evidence_text:
        raise ValueError(
            "Could not recover source-chunk evidence for this correction; "
            "the mapping was not added to reusable memory."
        )

    evidence_hash = hashlib.sha256(
        evidence_text.encode("utf-8")
    ).hexdigest()
    spec_version = _current_aqa_spec_version()
    cache_key = hashlib.sha256(
        (
            "detected-topic-correction-v1|"
            + spec_version
            + "|"
            + transcript_name
            + "|"
            + normalized_topic
            + "|"
            + evidence_hash
        ).encode("utf-8")
    ).hexdigest()

    catalogue_by_id = {
        str(item["concept_id"]): item
        for item in _official_aqa_topic_options()
    }
    replacement = catalogue_by_id.get(str(replacement_concept_id))
    if replacement is None:
        raise ValueError("Select a valid official AQA replacement topic.")

    candidate_ids = []
    for concept_id in (original_concept_id, str(replacement_concept_id)):
        if concept_id and concept_id not in candidate_ids:
            candidate_ids.append(concept_id)

    qdrant_candidates = []
    for concept_id in candidate_ids:
        item = catalogue_by_id.get(concept_id)
        if item is None:
            continue
        qdrant_candidates.append(
            {
                "concept_id": concept_id,
                "label": item.get("topic"),
                "official_reference": item.get("official_reference"),
                "source": "detected_topic_editor",
            }
        )

    confidence = original_topic.get("confidence")
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0

    insert_statement = text(
        """
        INSERT INTO topic_human_review (
            cache_key,
            normalized_topic,
            original_topic,
            evidence_hash,
            evidence_text,
            source_transcript,
            source_chunk_ids,
            memory_lookup_result,
            candidate_concept_ids,
            qdrant_candidates,
            proposed_decision,
            proposed_mapped_concept_id,
            confidence,
            confidence_band,
            reason,
            model_name,
            prompt_version,
            status,
            corrected_decision,
            corrected_mapped_concept_id,
            correction_reason,
            review_notes,
            reviewed_by,
            reviewed_at,
            spec_version,
            created_at,
            updated_at
        ) VALUES (
            :cache_key,
            :normalized_topic,
            :original_topic,
            :evidence_hash,
            :evidence_text,
            :source_transcript,
            CAST(:source_chunk_ids AS jsonb),
            'manual_edit',
            CAST(:candidate_concept_ids AS jsonb),
            CAST(:qdrant_candidates AS jsonb),
            'mapped',
            :proposed_mapped_concept_id,
            :confidence,
            'human_review',
            'Reviewer identified an incorrect retained official AQA mapping.',
            'detected_topic_editor',
            'detected-topic-correction-v1',
            'pending',
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            :spec_version,
            NOW(),
            NOW()
        )
        ON CONFLICT (
            source_transcript,
            normalized_topic,
            spec_version
        ) WHERE source_transcript IS NOT NULL
        DO UPDATE SET
            cache_key = EXCLUDED.cache_key,
            original_topic = EXCLUDED.original_topic,
            evidence_hash = EXCLUDED.evidence_hash,
            evidence_text = EXCLUDED.evidence_text,
            source_chunk_ids = EXCLUDED.source_chunk_ids,
            memory_lookup_result = EXCLUDED.memory_lookup_result,
            candidate_concept_ids = EXCLUDED.candidate_concept_ids,
            qdrant_candidates = EXCLUDED.qdrant_candidates,
            proposed_decision = EXCLUDED.proposed_decision,
            proposed_mapped_concept_id = EXCLUDED.proposed_mapped_concept_id,
            confidence = EXCLUDED.confidence,
            confidence_band = EXCLUDED.confidence_band,
            reason = EXCLUDED.reason,
            model_name = EXCLUDED.model_name,
            prompt_version = EXCLUDED.prompt_version,
            status = 'pending',
            corrected_decision = NULL,
            corrected_mapped_concept_id = NULL,
            correction_reason = NULL,
            review_notes = NULL,
            reviewed_by = NULL,
            reviewed_at = NULL,
            updated_at = NOW()
        RETURNING id
        """
    )

    engine = _topic_review_engine()
    with engine.begin() as connection:
        review_id = connection.execute(
            insert_statement,
            {
                "cache_key": cache_key,
                "normalized_topic": normalized_topic,
                "original_topic": original_label,
                "evidence_hash": evidence_hash,
                "evidence_text": evidence_text,
                "source_transcript": transcript_name,
                "source_chunk_ids": json.dumps(source_chunk_ids),
                "candidate_concept_ids": json.dumps(candidate_ids),
                "qdrant_candidates": json.dumps(qdrant_candidates),
                "proposed_mapped_concept_id": original_concept_id,
                "confidence": confidence_value,
                "spec_version": spec_version,
            },
        ).scalar_one()

    # Updating pending -> corrected invokes the existing PostgreSQL trigger,
    # which writes human_review + memory_promotion audit rows and stores a
    # reviewer-approved human_corrected memory entry with reviewer_reason.
    topic_review_set_status(
        record_id=int(review_id),
        status="corrected",
        run_dir=run_dir,
        reviewed_by="streamlit",
        corrected_decision="mapped",
        corrected_mapped_concept_id=str(replacement_concept_id),
        correction_reason=reason_text,
        review_notes=(
            "Correction originated from the final detected-topic editor."
        ),
    )
    return int(review_id)


def _record_manual_final_topic_edit_exact_context_approval(
    *,
    result: dict[str, Any],
    run_dir: Path,
    transcript_name: str,
    action: str,
    reason: str,
    current_evidence: str,
    source_concept_id: str | None,
) -> dict[str, Any]:
    """Bind a manual final-topic correction to the exact lesson context.

    Creating reviewer-approved historical edit memory is only half of the
    self-improving path.  The reviewer has also explicitly approved that new
    memory for THIS lesson, so the same transcript must not have to ask the
    reviewer for the same correction again on a later run.

    This records approve_reuse for the newly-created memory against the same
    canonical evidence representation used by the backend reuse gate.  It does
    not bypass deterministic validation on different lesson evidence.
    """
    output = dict(result or {})
    output["exact_context_approval_saved"] = False

    try:
        memory_id = int(output.get("detected_topic_edit_memory_id"))
    except (TypeError, ValueError):
        output["exact_context_approval_error"] = (
            "The edit was saved, but no reusable memory ID was returned."
        )
        return output

    evidence_text = str(current_evidence or "").strip()
    if not evidence_text:
        output["exact_context_approval_error"] = (
            "The edit memory was saved, but canonical current lesson evidence "
            "could not be reconstructed for exact-context approval."
        )
        return output

    try:
        store = _detected_topic_reuse_feedback_store()
        store.record(
            memory_id=memory_id,
            current_evidence=evidence_text,
            decision="approve_reuse",
            reviewer_reason=str(reason or "").strip(),
            spec_version=_current_aqa_spec_version(),
            pipeline_run_id=Path(run_dir).name,
            source_transcript=transcript_name,
            source_concept_id=(
                str(source_concept_id).strip() if source_concept_id else None
            ),
            reviewed_by="streamlit",
        )
    except Exception as exc:
        output["exact_context_approval_error"] = str(exc)
        return output

    output["exact_context_approval_saved"] = True
    return output



def _persist_detected_topic_edit(
    *,
    run_dir: Path,
    transcript_name: str,
    topic_index: int,
    action: str,
    source_concept_id: str | None = None,
    replacement_concept_id: str | None = None,
    new_role: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Persist a manual edit and approve its reuse for this exact lesson."""
    # Bind the explicit human correction to the stable cleaned-transcript
    # context, not to model-derived Module 3 evidence that may drift between
    # reruns of the exact same lesson.
    current_evidence = _final_topic_stable_lesson_context(
        run_dir=run_dir,
        transcript_name=transcript_name,
    )

    result = submit_human_detected_topic_edit(
        frontend_root=PROJECT_ROOT,
        run_id=Path(run_dir).name,
        action=action,
        reason=str(reason or ""),
        topic_index=int(topic_index),
        source_concept_id=source_concept_id,
        target_concept_id=replacement_concept_id,
        target_role=new_role,
        reviewed_by="streamlit",
    )

    result = _record_manual_final_topic_edit_exact_context_approval(
        result=result,
        run_dir=run_dir,
        transcript_name=transcript_name,
        action=action,
        reason=str(reason or ""),
        current_evidence=current_evidence,
        source_concept_id=source_concept_id,
    )
    _step4_schedule_langgraph_resume(Path(run_dir).name)
    return result



def _persist_detected_topic_addition(
    *,
    run_dir: Path,
    transcript_name: str,
    concept_id: str,
    role: str,
    source_chunk_ids: list[int],
    reason: str,
    existing_concept_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Persist a manual add and approve its reuse for this exact lesson."""
    # A manual add is an explicit human correction for this lesson. Store its
    # exact approval against the stable cleaned-transcript context so a later
    # run of the same transcript can reproduce it even if Module 3 candidate
    # evidence changes slightly.
    current_evidence = _final_topic_stable_lesson_context(
        run_dir=run_dir,
        transcript_name=transcript_name,
    )

    result = submit_human_detected_topic_edit(
        frontend_root=PROJECT_ROOT,
        run_id=Path(run_dir).name,
        action="add_topic",
        reason=str(reason or ""),
        target_concept_id=concept_id,
        target_role=role,
        source_chunk_ids=[int(value) for value in source_chunk_ids],
        reviewed_by="streamlit",
    )

    result = _record_manual_final_topic_edit_exact_context_approval(
        result=result,
        run_dir=run_dir,
        transcript_name=transcript_name,
        action="add_topic",
        reason=str(reason or ""),
        current_evidence=current_evidence,
        source_concept_id=None,
    )
    _step4_schedule_langgraph_resume(Path(run_dir).name)
    return result




def _restore_original_detected_topics(
    *,
    run_dir: Path,
    transcript_name: str,
) -> bool:
    json_path = (
        Path(run_dir)
        / "output"
        / transcript_name
        / "03_topic_mapping.json"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    module3_result = payload.get("module3_result")
    if not isinstance(module3_result, dict):
        return False
    original = module3_result.get("merged_topics_original")
    if not isinstance(original, list):
        return False
    module3_result["merged_topics"] = json.loads(
        json.dumps(original, ensure_ascii=False)
    )
    payload.setdefault("topic_output_edits", []).append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "restore_original",
            "reviewed_by": "streamlit",
        }
    )
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _append_topic_decision_log(
        run_dir=run_dir,
        transcript_name=transcript_name,
        normalized_topic="detected topics",
        source_chunk_ids=[],
        decision_stage="detected_topic_review",
        action="restore_original",
        decision="restored",
        details={"restored_topic_count": len(original)},
    )
    return True


def render_detected_topic_editor(
    *,
    merged_topics: list[dict[str, Any]],
    run_dir: Path,
    transcript_name: str,
) -> None:
    """Human-in-the-loop curation of the final post-Module-3 topic list."""

    if not merged_topics:
        return

    with st.expander("Review / edit detected official topics", expanded=False):
        st.caption(
            "This is the HITL/self-improving final-topic layer. Module 3's "
            "existing detection, scoring and ranking are not changed. When you "
            "change, replace, remove, or add a topic, a reason is required and "
            "the change + reason + transcript evidence are stored as contextual "
            "edit memory. On future transcripts a deterministic context-first "
            "comparison checks stored evidence + reviewer reason against current "
            "evidence. Strong matches may auto-reuse the approved edit, strong "
            "mismatches reject it, and ambiguous cases are left unchanged for "
            "explicit human review. Groq is not used for automatic final-topic "
            "edit-memory decisions."
        )

        # Existing detected/effective topics remain editable exactly here.
        # Current-run manually added topics are displayed downstream, but are
        # deliberately not treated as fresh Module 3 detections in this selector.
        editable_indices = [
            index
            for index, topic in enumerate(merged_topics)
            if not bool(topic.get("human_added_topic"))
        ]

        if editable_indices:
            st.markdown("#### Edit an existing detected topic")
            selected_index = st.selectbox(
                "Choose a detected topic",
                options=editable_indices,
                format_func=lambda index: (
                    f"{merged_topics[index].get('topic')} "
                    f"({merged_topics[index].get('official_reference')}) — "
                    f"{merged_topics[index].get('topic_role')}"
                ),
                key=f"detected_topic_editor_select_{run_dir.name}",
            )
            selected = merged_topics[int(selected_index)]

            action_label = st.radio(
                "What do you want to change?",
                options=[
                    "Change primary/supporting role",
                    "Replace with another official AQA topic",
                    "Remove from this run",
                ],
                key=f"detected_topic_editor_action_{run_dir.name}",
            )

            action_map = {
                "Change primary/supporting role": "change_role",
                "Replace with another official AQA topic": "replace_topic",
                "Remove from this run": "remove_topic",
            }
            action = action_map[action_label]
            new_role: str | None = None
            replacement_concept_id: str | None = None

            if action in {"change_role", "replace_topic"}:
                current_role = str(
                    selected.get("topic_role") or "supporting"
                ).casefold()
                role_options = ["primary", "supporting"]
                role_index = (
                    role_options.index(current_role)
                    if current_role in role_options
                    else 1
                )
                new_role = st.selectbox(
                    "Role",
                    options=role_options,
                    index=role_index,
                    key=f"detected_topic_editor_role_{run_dir.name}",
                )

            if action == "replace_topic":
                catalogue = _official_aqa_topic_options()
                catalogue_by_id = {
                    str(item["concept_id"]): item
                    for item in catalogue
                }
                concept_ids = list(catalogue_by_id)
                current_id = str(selected.get("concept_id") or "")
                default_index = (
                    concept_ids.index(current_id)
                    if current_id in concept_ids
                    else 0
                )
                replacement_concept_id = st.selectbox(
                    "Correct official AQA topic",
                    options=concept_ids,
                    index=default_index,
                    format_func=lambda concept_id: (
                        f"{catalogue_by_id[concept_id]['topic']} "
                        f"({catalogue_by_id[concept_id]['official_reference']})"
                    ),
                    key=f"detected_topic_editor_replacement_{run_dir.name}",
                )

            reason = st.text_area(
                "Reason (required)",
                key=f"detected_topic_editor_reason_{run_dir.name}",
                height=100,
                help=(
                    "Explain why this final-topic correction is correct for the "
                    "current transcript. This rationale is stored and revalidated "
                    "against future transcript evidence before automatic reuse."
                ),
            ).strip()

            save_clicked = st.button(
                "Save detected-topic edit + learn reason",
                key=f"detected_topic_editor_save_{run_dir.name}",
                type="primary",
                use_container_width=True,
            )

            if save_clicked:
                try:
                    result = _persist_detected_topic_edit(
                        run_dir=run_dir,
                        transcript_name=transcript_name,
                        topic_index=int(selected_index),
                        action=action,
                        source_concept_id=str(selected.get("concept_id") or ""),
                        replacement_concept_id=replacement_concept_id,
                        new_role=new_role,
                        reason=reason,
                    )
                except Exception as exc:
                    st.error(f"Could not save detected-topic edit: {exc}")
                else:
                    memory_id = result.get("detected_topic_edit_memory_id")
                    if result.get("exact_context_approval_saved"):
                        st.session_state["topic_review_flash"] = (
                            "Detected-topic correction saved and approved for this exact "
                            "lesson context (memory ID "
                            f"{memory_id}). The same transcript can reuse it on a later run."
                        )
                    else:
                        st.session_state["topic_review_flash"] = (
                            "Detected-topic correction and historical memory were saved "
                            f"(memory ID {memory_id}), but exact-context reuse approval "
                            "could not be stored: "
                            f"{result.get('exact_context_approval_error') or 'unknown error'}"
                        )
                    st.rerun()

        st.divider()
        st.markdown("#### Add a missed official AQA topic")
        st.caption(
            "Use this only when an official topic is genuinely taught but missing "
            "from the final list. Select the transcript chunks that justify it."
        )

        catalogue = _official_aqa_topic_options()
        catalogue_by_id = {
            str(item["concept_id"]): item
            for item in catalogue
        }
        existing_concept_ids = {
            str(topic.get("concept_id") or "").strip()
            for topic in merged_topics
            if str(topic.get("concept_id") or "").strip()
        }
        addable_concept_ids = [
            concept_id
            for concept_id in catalogue_by_id
            if concept_id not in existing_concept_ids
        ]

        if addable_concept_ids:
            add_concept_id = st.selectbox(
                "Missing official AQA topic",
                options=addable_concept_ids,
                format_func=lambda concept_id: (
                    f"{catalogue_by_id[concept_id]['topic']} "
                    f"({catalogue_by_id[concept_id]['official_reference']})"
                ),
                key=f"detected_topic_editor_add_concept_{run_dir.name}",
            )
            add_role = st.selectbox(
                "Role for added topic",
                options=["primary", "supporting"],
                index=1,
                key=f"detected_topic_editor_add_role_{run_dir.name}",
            )

            chunk_payload = load_json(
                Path(run_dir)
                / "output"
                / transcript_name
                / "02_chunking.json"
            )
            chunk_text_by_id: dict[int, str] = {}
            for chunk in chunk_payload.get("chunks", []) or []:
                if not isinstance(chunk, dict):
                    continue
                try:
                    chunk_id = int(chunk.get("chunk_id"))
                except (TypeError, ValueError):
                    continue
                chunk_text_by_id[chunk_id] = str(chunk.get("text") or "").strip()

            chunk_ids = sorted(chunk_text_by_id)
            add_source_chunks = st.multiselect(
                "Evidence chunks (required)",
                options=chunk_ids,
                format_func=lambda chunk_id: (
                    f"Chunk {chunk_id}: "
                    f"{chunk_text_by_id.get(chunk_id, '')[:90]}"
                ),
                key=f"detected_topic_editor_add_chunks_{run_dir.name}",
            )
            add_reason = st.text_area(
                "Reason for adding this topic (required)",
                key=f"detected_topic_editor_add_reason_{run_dir.name}",
                height=100,
            ).strip()

            if st.button(
                "Add topic + learn reason",
                key=f"detected_topic_editor_add_save_{run_dir.name}",
                use_container_width=True,
            ):
                try:
                    result = _persist_detected_topic_addition(
                        run_dir=run_dir,
                        transcript_name=transcript_name,
                        concept_id=add_concept_id,
                        role=add_role,
                        source_chunk_ids=[int(value) for value in add_source_chunks],
                        reason=add_reason,
                        existing_concept_ids=existing_concept_ids,
                    )
                except Exception as exc:
                    st.error(f"Could not add detected topic: {exc}")
                else:
                    memory_id = result.get("detected_topic_edit_memory_id")
                    if result.get("exact_context_approval_saved"):
                        st.session_state["topic_review_flash"] = (
                            "Missing topic added and approved for this exact lesson "
                            "context (memory ID "
                            f"{memory_id}). The same transcript can reuse it on a later run."
                        )
                    else:
                        st.session_state["topic_review_flash"] = (
                            "Missing topic and historical memory were saved "
                            f"(memory ID {memory_id}), but exact-context reuse approval "
                            "could not be stored: "
                            f"{result.get('exact_context_approval_error') or 'unknown error'}"
                        )
                    st.rerun()
        else:
            st.info("All official AQA catalogue concepts are already present.")

        json_path = (
            Path(run_dir)
            / "output"
            / transcript_name
            / "03_topic_mapping.json"
        )
        try:
            current_payload = json.loads(json_path.read_text(encoding="utf-8"))
            has_original = isinstance(
                deep_get(
                    current_payload,
                    "module3_result",
                    "merged_topics_original",
                    default=None,
                ),
                list,
            )
        except (OSError, json.JSONDecodeError):
            has_original = False

        if has_original:
            st.caption(
                "Restore original Module 3 topic list only resets this run's "
                "current-run curation; it does not silently delete reviewer-approved "
                "learning records from PostgreSQL."
            )
            if st.button(
                "Restore original Module 3 topic list",
                key=f"restore_detected_topics_{run_dir.name}",
                use_container_width=True,
            ):
                try:
                    restored = _restore_original_detected_topics(
                        run_dir=run_dir,
                        transcript_name=transcript_name,
                    )
                except Exception as exc:
                    st.error(f"Could not restore original topics: {exc}")
                else:
                    if restored:
                        st.session_state["topic_review_flash"] = (
                            "Original Module 3 topic list restored for this run."
                        )
                        st.rerun()


def render_topic_mapping_review(
    review_items: list[dict[str, Any]],
    *,
    run_dir: Path,
) -> None:
    """Render Module 3 proposals and persist Approve/Correct/Reject."""

    st.markdown("---")
    st.subheader("Human Review — Topic Mapping")
    st.caption(
        "A fresh mapping is not reusable memory until a reviewer approves "
        "or corrects it. Corrections require a reason; rejected proposals "
        "are logged but are never promoted to memory."
    )

    grouped: dict[str, list[dict[str, Any]]] = {
        "pending": [],
        "approved": [],
        "corrected": [],
        "rejected": [],
    }

    for raw_item in review_items:
        if not isinstance(raw_item, dict):
            continue
        status = normalise_review_status(
            raw_item.get("status", "pending")
        )
        if status not in grouped:
            status = "pending"
        grouped[status].append({**raw_item, "status": status})

    pending = grouped["pending"]
    approved = grouped["approved"]
    corrected = grouped["corrected"]
    rejected = grouped["rejected"]

    pcol, acol, ccol, rcol = st.columns(4)
    pcol.metric("Pending", len(pending))
    acol.metric("Approved", len(approved))
    ccol.metric("Corrected", len(corrected))
    rcol.metric("Rejected", len(rejected))

    pending_tab, approved_tab, corrected_tab, rejected_tab = st.tabs(
        [
            f"Pending ({len(pending)})",
            f"Approved ({len(approved)})",
            f"Corrected ({len(corrected)})",
            f"Rejected ({len(rejected)})",
        ]
    )

    with pending_tab:
        if not pending:
            st.info("No pending topic mappings for this run.")

        for index, item in enumerate(pending, start=1):
            record_id = item.get("id")
            detected_topic = item.get("rough_topic") or "Unknown topic"
            mapped_topic = item.get("mapped_topic") or "Not mapped"
            official_reference = item.get("official_reference") or "None"
            decision = item.get("decision") or "needs_review"
            confidence = item.get("confidence")
            reason = item.get("reason") or ""
            source_chunks = item.get("source_chunk_ids") or []

            title = f"Topic proposal {index}: {detected_topic}"
            if record_id is not None:
                title += f" — review {record_id}"

            with st.expander(title, expanded=True):
                metadata = [f"Decision: {decision}", "Source: Groq/Qdrant"]
                if confidence is not None:
                    try:
                        metadata.append(f"Confidence: {float(confidence):.2f}")
                    except (TypeError, ValueError):
                        metadata.append(f"Confidence: {confidence}")
                st.caption(" | ".join(metadata))

                left, right = st.columns(2)
                with left:
                    st.markdown("**Detected rough topic**")
                    st.write(detected_topic)
                with right:
                    st.markdown("**Proposed official mapping**")
                    st.write(mapped_topic)
                    st.caption(f"Official reference: {official_reference}")

                if source_chunks:
                    st.markdown("**Source chunks**")
                    st.write(source_chunks)
                if reason:
                    st.markdown("**System reason**")
                    st.write(reason)

                can_approve = (
                    decision == "out_of_syllabus"
                    or item.get("mapped_concept_id") is not None
                )

                if record_id is None:
                    st.error(
                        "This topic proposal has no review ID and cannot be updated."
                    )
                    continue

                if not can_approve:
                    st.warning(
                        "This proposal cannot be approved as-is because it has "
                        "no valid official mapping. You can correct or reject it."
                    )

                approve_col, reject_col = st.columns(2)
                approve_clicked = approve_col.button(
                    "Approve as suggested",
                    key=f"approve_topic_review_{run_dir.name}_{record_id}",
                    type="primary",
                    use_container_width=True,
                    disabled=not can_approve,
                )
                reject_clicked = reject_col.button(
                    "Reject",
                    key=f"reject_topic_review_{run_dir.name}_{record_id}",
                    use_container_width=True,
                )

                with st.expander("Correct this mapping", expanded=False):
                    correction_mode = st.radio(
                        "Correct decision",
                        options=["Official AQA topic", "Out of syllabus"],
                        key=f"correct_mode_{run_dir.name}_{record_id}",
                        horizontal=True,
                    )

                    corrected_concept_id: str | None = None
                    if correction_mode == "Official AQA topic":
                        candidates = [
                            candidate
                            for candidate in (item.get("qdrant_candidates") or [])
                            if isinstance(candidate, dict)
                            and candidate.get("concept_id")
                        ]
                        candidate_by_id = {
                            str(candidate["concept_id"]): candidate
                            for candidate in candidates
                        }
                        candidate_ids = list(candidate_by_id)

                        if candidate_ids:
                            corrected_concept_id = st.selectbox(
                                "Correct official topic",
                                options=candidate_ids,
                                format_func=lambda concept_id: (
                                    f"{candidate_by_id[concept_id].get('label', concept_id)} "
                                    f"({candidate_by_id[concept_id].get('official_reference', 'no ref')})"
                                ),
                                key=f"correct_concept_{run_dir.name}_{record_id}",
                            )
                        else:
                            corrected_concept_id = st.text_input(
                                "Correct official AQA concept ID",
                                key=f"correct_concept_manual_{run_dir.name}_{record_id}",
                                help=(
                                    "This is a legacy review row without a stored "
                                    "Qdrant shortlist, so enter the official concept ID."
                                ),
                            ).strip() or None

                    correction_reason = st.text_area(
                        "Why is the system suggestion wrong? (required)",
                        key=f"correction_reason_{run_dir.name}_{record_id}",
                    ).strip()

                    correct_clicked = st.button(
                        "Save correction and add to memory",
                        key=f"correct_topic_review_{run_dir.name}_{record_id}",
                        use_container_width=True,
                    )

                if approve_clicked:
                    try:
                        topic_review_set_status(
                            record_id=int(record_id),
                            status="approved",
                            run_dir=run_dir,
                        )
                    except Exception as exc:
                        st.error(f"Could not approve topic review {record_id}: {exc}")
                    else:
                        st.session_state["topic_review_flash"] = (
                            f"Topic review {record_id} approved and promoted to memory."
                        )
                        st.rerun()

                if reject_clicked:
                    try:
                        topic_review_set_status(
                            record_id=int(record_id),
                            status="rejected",
                            run_dir=run_dir,
                        )
                    except Exception as exc:
                        st.error(f"Could not reject topic review {record_id}: {exc}")
                    else:
                        st.session_state["topic_review_flash"] = (
                            f"Topic review {record_id} rejected."
                        )
                        st.rerun()

                if correct_clicked:
                    corrected_decision = (
                        "mapped"
                        if correction_mode == "Official AQA topic"
                        else "out_of_syllabus"
                    )
                    if not correction_reason:
                        st.error("A correction reason is required.")
                    elif (
                        corrected_decision == "mapped"
                        and not corrected_concept_id
                    ):
                        st.error("Select or enter the correct official AQA concept.")
                    else:
                        try:
                            topic_review_set_status(
                                record_id=int(record_id),
                                status="corrected",
                                run_dir=run_dir,
                                corrected_decision=corrected_decision,
                                corrected_mapped_concept_id=corrected_concept_id,
                                correction_reason=correction_reason,
                            )
                        except Exception as exc:
                            st.error(
                                f"Could not correct topic review {record_id}: {exc}"
                            )
                        else:
                            st.session_state["topic_review_flash"] = (
                                f"Topic review {record_id} corrected and promoted "
                                "to reviewer-approved memory."
                            )
                            st.rerun()

    with approved_tab:
        if approved:
            st.success("Approved records are reusable PostgreSQL mapping memory.")
            st.dataframe(
                pd.DataFrame(_topic_review_row(item) for item in approved),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No approved topic mappings for this run.")

    with corrected_tab:
        if corrected:
            st.success(
                "Corrected records are reusable memory together with the "
                "reviewer's correction reason."
            )
            st.dataframe(
                pd.DataFrame(_topic_review_row(item) for item in corrected),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No corrected topic mappings for this run.")

    with rejected_tab:
        if rejected:
            st.dataframe(
                pd.DataFrame(_topic_review_row(item) for item in rejected),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No rejected topic mappings for this run.")

def _normalise_chunk_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_agent2_topic_handoff(
    *,
    merged_topics: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build the Agent 1 -> Agent 2 topic payload without changing any
    notebook logic.

    Module 3 already provides the retained official topics and their source
    chunk IDs. This frontend helper only joins those IDs with the existing
    Module 2 chunk text so Agent 2 can use actual transcript evidence.
    """

    chunk_text_by_id: dict[int, str] = {}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        chunk_id = _normalise_chunk_id(
            chunk.get("chunk_id")
        )

        if chunk_id is None:
            continue

        chunk_text_by_id[chunk_id] = str(
            chunk.get("text")
            or ""
        ).strip()

    payload: list[dict[str, Any]] = []

    for topic_index, topic in enumerate(
        merged_topics,
        start=1,
    ):
        if not isinstance(topic, dict):
            continue

        raw_chunk_ids = (
            topic.get("source_chunk_ids")
            or topic.get("source_chunks")
            or []
        )

        source_chunks = []

        for raw_chunk_id in raw_chunk_ids:
            chunk_id = _normalise_chunk_id(
                raw_chunk_id
            )

            if (
                chunk_id is not None
                and chunk_id not in source_chunks
            ):
                source_chunks.append(chunk_id)

        source_chunk_texts = [
            chunk_text_by_id[chunk_id]
            for chunk_id in source_chunks
            if chunk_text_by_id.get(chunk_id)
        ]

        topic_name = str(
            topic.get("topic")
            or topic.get("detected_topic")
            or ""
        ).strip()

        role = str(
            topic.get("topic_role")
            or topic.get("role")
            or "supporting"
        ).strip().casefold()

        if role not in {
            "primary",
            "supporting",
        }:
            role = "supporting"

        payload.append(
            {
                "topic_index": topic_index,
                "concept_id": topic.get("concept_id"),
                "topic": topic_name,
                "detected_topic": topic_name,
                "role": role,
                "topic_role": role,
                "official_reference": str(
                    topic.get("official_reference")
                    or ""
                ).strip(),
                "official_title": topic.get(
                    "official_title"
                ),
                "chapter_reference": topic.get(
                    "chapter_reference"
                ),
                "domain": topic.get("domain"),
                "paper": topic.get("paper"),
                "confidence": topic.get("confidence"),
                "ranking_score": topic.get(
                    "ranking_score"
                ),
                "source_chunks": source_chunks,
                "source_chunk_texts": (
                    source_chunk_texts
                ),
                "source_chunk_text_count": len(
                    source_chunk_texts
                ),
                "source_chunk_count": len(
                    source_chunks
                ),
                "missing_source_chunk_ids": [
                    chunk_id
                    for chunk_id in source_chunks
                    if not chunk_text_by_id.get(
                        chunk_id
                    )
                ],
                "evidence": topic.get("evidence") or [],
            }
        )

    return payload


def _agent2_handoff_paths(
    run_dir: Path,
) -> tuple[Path, Path]:
    integration_dir = (
        Path(run_dir)
        / "output"
        / "integration"
    )

    integration_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        integration_dir
        / "agent1_topics_with_evidence.json",
        integration_dir
        / "approved_topics.json",
    )


def _write_agent1_topic_handoff(
    *,
    run_dir: Path,
    transcript_name: str,
    topics: list[dict[str, Any]],
    approved_only: bool,
) -> Path:
    """Write normal handoff; human-approved handoff goes through MCP then resumes LangGraph."""
    all_topics_path, approved_topics_path = _agent2_handoff_paths(run_dir)

    if approved_only:
        selections = []
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            selections.append({
                "topic_index": int(topic.get("topic_index")),
                "approved": True,
                "topic": str(topic.get("topic") or topic.get("detected_topic") or "").strip(),
                "role": str(topic.get("role") or topic.get("topic_role") or "supporting").strip().casefold(),
                "official_reference": str(topic.get("official_reference") or "").strip(),
            })
        result = submit_human_agent2_topic_approval(
            frontend_root=PROJECT_ROOT,
            run_id=Path(run_dir).name,
            selections=selections,
            reviewed_by="streamlit",
        )

        # Keep the LangGraph HITL architecture intact.
        #
        # Resume the persisted graph only when it is actually paused at the
        # Agent 2 topic-approval gate. A later "Update Approved Topics" action
        # may happen while the workflow is already in an assessment/result
        # state; that is a human MCP update, not a checkpoint that should be
        # blindly resumed.
        try:
            approval_snapshot = langgraph_snapshot(
                frontend_root=PROJECT_ROOT,
                run_id=Path(run_dir).name,
            )
        except Exception:
            approval_snapshot = {}

        approval_gate = str(
            approval_snapshot.get(
                "human_gate",
                "",
            )
            or ""
        ).strip()

        if approval_gate == "AGENT2_TOPIC_APPROVAL":
            _step4_schedule_langgraph_resume(
                Path(run_dir).name
            )
        else:
            # LangGraph is still authoritative; refresh its deterministic
            # snapshot without executing a non-existent checkpoint resume.
            try:
                _step4_refresh_tracker(
                    Path(run_dir).name
                )
            except Exception:
                pass

        return Path(
            str(
                result.get("path")
                or approved_topics_path
            )
        )

    payload = {
        "schema_version": "agent1-agent2-topic-handoff-v1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": Path(run_dir).name,
        "transcript": transcript_name,
        "source": {
            "module2_output": "02_chunking.json",
            "module3_output": "03_topic_mapping.json",
            "notebook_logic_changed": False,
            "handoff_built_by": "streamlit_frontend_langgraph_step4",
        },
        "approved_only": False,
        "topic_count": len(topics),
        "actual_chunk_evidence_available": bool(
            topics and all(topic.get("source_chunk_texts") for topic in topics)
        ),
        "topics": topics,
    }
    all_topics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return all_topics_path




def _existing_approved_topic_lookup(
    approved_topics_path: Path,
) -> dict[int, dict[str, Any]]:
    existing = load_json(
        approved_topics_path
    )

    topics = existing.get("topics", [])

    if not isinstance(topics, list):
        return {}

    lookup: dict[
        int,
        dict[str, Any],
    ] = {}

    for item in topics:
        if not isinstance(item, dict):
            continue

        topic_index = _normalise_chunk_id(
            item.get("topic_index")
        )

        if topic_index is not None:
            lookup[topic_index] = item

    return lookup


def render_agent2_topic_approval(
    *,
    run_dir: Path,
    transcript_name: str,
    merged_topics: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """
    Render the Agent 2 handoff approval screen.

    This is separate from Module 3's mapping-memory human review. It decides
    which already-retained Agent 1 topics should be passed to Agent 2 for the
    current assessment request.
    """

    st.subheader(
        "Agent 2 Handoff — Topic Approval"
    )

    st.caption(
        "This is the Agent 2 human HITL handoff. The decision is saved through "
        "the HUMAN_UI_ONLY MCP tool, and the persisted LangGraph thread resumes "
        "from the Agent 2 topic-approval gate when that gate is active."
    )

    topic_payload = build_agent2_topic_handoff(
        merged_topics=merged_topics,
        chunks=chunks,
    )

    (
        all_topics_path,
        approved_topics_path,
    ) = _agent2_handoff_paths(run_dir)

    _write_agent1_topic_handoff(
        run_dir=run_dir,
        transcript_name=transcript_name,
        topics=topic_payload,
        approved_only=False,
    )

    if not topic_payload:
        st.warning(
            "Agent 1 did not retain an official topic, so there is nothing "
            "to approve for Agent 2."
        )
        download_file(
            all_topics_path,
            "Download Agent 1 handoff JSON",
        )
        return

    existing_lookup = (
        _existing_approved_topic_lookup(
            approved_topics_path
        )
    )

    approval_already_saved = bool(
        approved_topics_path.is_file()
        and existing_lookup
    )

    if approval_already_saved:
        st.success(
            f"{len(existing_lookup)} topic(s) are already approved for Agent 2."
        )
        st.caption(
            "The approval is saved. You can go to Agent 2 Assessment / Quiz "
            "Filters now. Use 'Update Approved Topics' only if you actually "
            "want to change the handoff."
        )

    editor_rows = []

    for topic in topic_payload:
        topic_index = int(
            topic["topic_index"]
        )

        previous = existing_lookup.get(
            topic_index,
            {},
        )

        editor_rows.append(
            {
                "_topic_index": topic_index,
                "Approve": bool(previous)
                if approved_topics_path.is_file()
                else True,
                "Topic": previous.get(
                    "topic",
                    topic["topic"],
                ),
                "Role": previous.get(
                    "role",
                    topic["role"],
                ),
                "Official reference": previous.get(
                    "official_reference",
                    topic[
                        "official_reference"
                    ],
                ),
                "Confidence": topic.get(
                    "confidence"
                ),
                "Ranking score": topic.get(
                    "ranking_score"
                ),
                "Source chunks": ", ".join(
                    str(value)
                    for value in topic.get(
                        "source_chunks",
                        [],
                    )
                ),
                "Chunk texts": len(
                    topic.get(
                        "source_chunk_texts",
                        [],
                    )
                ),
            }
        )

    edited_df = st.data_editor(
        pd.DataFrame(editor_rows),
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key=(
            "agent2_topic_approval_editor_"
            f"{Path(run_dir).name}"
        ),
        column_config={
            "_topic_index": None,
            "Approve": st.column_config.CheckboxColumn(
                "Approve",
                help=(
                    "Only approved topics are sent to Agent 2."
                ),
                default=True,
            ),
            "Topic": st.column_config.TextColumn(
                "Topic",
                help=(
                    "Editable lesson-topic label used as Agent 2 query evidence."
                ),
                required=True,
            ),
            "Role": st.column_config.SelectboxColumn(
                "Role",
                options=[
                    "primary",
                    "supporting",
                ],
                required=True,
            ),
            "Official reference": st.column_config.TextColumn(
                "Official reference",
                help=(
                    "AQA official reference already produced by Agent 1."
                ),
                required=True,
            ),
            "Confidence": st.column_config.NumberColumn(
                "Confidence",
                format="%.4f",
            ),
            "Ranking score": st.column_config.NumberColumn(
                "Ranking score",
                format="%.4f",
            ),
            "Source chunks": st.column_config.TextColumn(
                "Source chunks"
            ),
            "Chunk texts": st.column_config.NumberColumn(
                "Chunk texts"
            ),
        },
        disabled=[
            "Confidence",
            "Ranking score",
            "Source chunks",
            "Chunk texts",
        ],
    )

    selected_rows = edited_df[
        edited_df["Approve"].astype(bool)
    ].copy()

    primary_count = int(
        selected_rows[
            "Role"
        ].astype(str).str.casefold().eq(
            "primary"
        ).sum()
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Detected topics",
        len(topic_payload),
    )
    c2.metric(
        "Approved for Agent 2",
        len(selected_rows),
    )
    c3.metric(
        "Primary topics",
        primary_count,
    )

    with st.expander(
        "Preview actual transcript evidence sent with each topic"
    ):
        for topic in topic_payload:
            st.markdown(
                f"**{topic['topic']} — "
                f"{topic['official_reference']}**"
            )

            evidence_text = "\n\n".join(
                topic.get(
                    "source_chunk_texts",
                    [],
                )
            )

            if evidence_text:
                st.text_area(
                    "Source chunk text",
                    evidence_text,
                    height=180,
                    key=(
                        "agent2_topic_evidence_"
                        f"{Path(run_dir).name}_"
                        f"{topic['topic_index']}"
                    ),
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.warning(
                    "No Module 2 chunk text could be linked to this topic."
                )

    approval_button_label = (
        "Update Approved Topics"
        if approval_already_saved
        else "Approve Topics and Continue to Agent 2"
    )

    approve_clicked = st.button(
        approval_button_label,
        type="primary",
        use_container_width=True,
        key=(
            "save_agent2_approved_topics_"
            f"{Path(run_dir).name}"
        ),
    )

    if approve_clicked:
        errors = []

        if selected_rows.empty:
            errors.append(
                "Approve at least one topic."
            )

        if (
            not selected_rows.empty
            and primary_count < 1
        ):
            errors.append(
                "At least one approved topic must have the primary role."
            )

        invalid_references = []

        for value in selected_rows[
            "Official reference"
        ].tolist():
            reference = str(value or "").strip()

            if not re.fullmatch(
                r"\d+(?:\.\d+){1,3}",
                reference,
            ):
                invalid_references.append(
                    reference or "<blank>"
                )

        if invalid_references:
            errors.append(
                "Invalid official reference(s): "
                + ", ".join(
                    invalid_references
                )
            )

        if errors:
            for error in errors:
                st.error(error)
        else:
            topic_by_index = {
                int(topic["topic_index"]): topic
                for topic in topic_payload
            }

            approved_topics = []

            for _, edited_row in (
                selected_rows.iterrows()
            ):
                topic_index = int(
                    edited_row[
                        "_topic_index"
                    ]
                )

                topic = dict(
                    topic_by_index[
                        topic_index
                    ]
                )

                edited_topic = str(
                    edited_row["Topic"]
                    or ""
                ).strip()

                edited_role = str(
                    edited_row["Role"]
                    or ""
                ).strip().casefold()

                edited_reference = str(
                    edited_row[
                        "Official reference"
                    ]
                    or ""
                ).strip()

                topic.update(
                    {
                        "topic": edited_topic,
                        "detected_topic": (
                            edited_topic
                        ),
                        "role": edited_role,
                        "topic_role": (
                            edited_role
                        ),
                        "official_reference": (
                            edited_reference
                        ),
                        "approved_for_agent2": (
                            True
                        ),
                    }
                )

                approved_topics.append(
                    topic
                )

            output_path = (
                _write_agent1_topic_handoff(
                    run_dir=run_dir,
                    transcript_name=(
                        transcript_name
                    ),
                    topics=approved_topics,
                    approved_only=True,
                )
            )

            st.session_state[
                "agent2_approved_topics_path"
            ] = str(output_path)

            st.success(
                f"Saved {len(approved_topics)} approved topic(s) for Agent 2."
            )

            st.caption(
                "The human approval was persisted through MCP. If LangGraph was "
                "paused at the Agent 2 topic-approval HITL gate, the same "
                "checkpointed LangGraph thread will resume and re-resolve the "
                "authoritative state. No Agent 2 retrieval/quiz runs until you "
                "press an explicit Agent 2 action button."
            )

    left, right = st.columns(2)

    with left:
        download_file(
            all_topics_path,
            "Download all topics with evidence",
        )

    with right:
        if approved_topics_path.is_file():
            download_file(
                approved_topics_path,
                "Download approved Agent 2 topics",
            )


def _agent2_assessment_request_path(run_dir: Path) -> Path:
    integration_dir = Path(run_dir) / "output" / "integration"
    integration_dir.mkdir(parents=True, exist_ok=True)
    return integration_dir / "assessment_request.json"


def _default_agent2_project_root() -> str:
    for candidate in agent2_project_candidates(PROJECT_ROOT):
        if candidate.is_dir():
            return str(candidate)
    return str(PROJECT_ROOT.parent.parent / "Agent2")


def _agent2_result_manifest_path(run_dir: Path) -> Path:
    return Path(run_dir) / "output" / "agent2" / "agent2_execution_manifest.json"


def _latest_matching_file(folder: Path, pattern: str) -> Path | None:
    """
    Legacy helper retained for older Agent 2 runs.

    New Agent 2 rendering prefers files tied to the current package timestamp,
    so a previous assessment is never shown as the result of a newer run.
    """
    paths = sorted(
        Path(folder).glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return paths[0] if paths else None


def _agent2_frontend_attempt_path(run_dir: Path) -> Path:
    output_dir = Path(run_dir) / "output" / "agent2"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "agent2_frontend_last_attempt.json"


def _write_agent2_frontend_attempt(
    *,
    run_dir: Path,
    status: str,
    assessment_request_path: Path | None = None,
    assessment_request: dict[str, Any] | None = None,
    manifest_path: Path | None = None,
    package_path: Path | None = None,
    error: str | None = None,
) -> Path:
    """
    Persist the latest frontend execution state.

    This prevents a failed/new no-result attempt from silently falling back to
    an older successful Agent 2 assessment after a Streamlit rerun/refresh.
    """
    path = _agent2_frontend_attempt_path(run_dir)

    existing = load_json(path) if path.is_file() else {}

    if str(status).casefold() == "running":
        started_at_utc = datetime.now(timezone.utc).isoformat()
    else:
        started_at_utc = existing.get("started_at_utc")

    payload = {
        "status": str(status),
        "started_at_utc": started_at_utc,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "assessment_request_path": (
            str(Path(assessment_request_path).resolve())
            if assessment_request_path is not None
            else existing.get("assessment_request_path")
        ),
        "assessment_request": (
            assessment_request
            if assessment_request is not None
            else existing.get("assessment_request")
        ),
        "manifest_path": (
            str(Path(manifest_path).resolve())
            if manifest_path is not None
            else None
        ),
        "package_path": (
            str(Path(package_path).resolve())
            if package_path is not None
            else None
        ),
        "error": error,
    }

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _resolve_agent2_file(
    value: Any,
    *,
    output_dir: Path | None = None,
) -> Path | None:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    path = Path(raw)

    if path.is_file():
        return path

    if output_dir is not None and not path.is_absolute():
        candidate = Path(output_dir) / path
        if candidate.is_file():
            return candidate

    return None


def _agent2_package_timestamp(package_path: Path) -> str | None:
    match = re.search(
        r"agent2_assessment_package_(\d{8}_\d{6})\.json$",
        package_path.name,
    )
    return match.group(1) if match else None


def _agent2_current_download_files(
    *,
    package_path: Path,
    package: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """
    Return only files belonging to the currently displayed Agent 2 package.

    The timestamp tied to the package is preferred over "latest file" lookup,
    which previously allowed stale Paper 2 artifacts to appear after a Paper 1
    no-result run.
    """
    collected: list[Path] = []

    def add(path: Path | None) -> None:
        if (
            path is not None
            and path.is_file()
            and path not in collected
        ):
            collected.append(path)

    add(package_path)

    output_files = package.get("output_files", {}) or {}
    if isinstance(output_files, dict):
        for value in output_files.values():
            if isinstance(value, (str, Path)):
                add(
                    _resolve_agent2_file(
                        value,
                        output_dir=output_dir,
                    )
                )

    add(
        _resolve_agent2_file(
            manifest.get("release_readiness_path"),
            output_dir=output_dir,
        )
    )

    timestamp = _agent2_package_timestamp(package_path)

    if timestamp:
        for path in sorted(output_dir.glob(f"*{timestamp}*")):
            add(path)

    # Keep downloads useful, but skip internal images/execution notebooks.
    allowed_suffixes = {
        ".json",
        ".csv",
        ".txt",
        ".md",
        ".pdf",
    }

    return [
        path
        for path in collected
        if path.suffix.casefold() in allowed_suffixes
    ]


def _agent2_message_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if not isinstance(value, dict):
        return str(value).strip()

    message = str(
        value.get("message")
        or value.get("user_message")
        or value.get("detail")
        or value.get("reason")
        or ""
    ).strip()

    topic = str(
        value.get("topic")
        or value.get("detected_topic")
        or ""
    ).strip()

    if topic and message:
        return f"{topic}: {message}"
    if message:
        return message

    return json.dumps(value, ensure_ascii=False)


def _collect_agent2_user_messages(
    *payloads: dict[str, Any],
) -> list[str]:
    messages: list[str] = []

    for payload in payloads:
        if not isinstance(payload, dict):
            continue

        raw_messages = payload.get("user_messages", []) or []

        if not isinstance(raw_messages, list):
            raw_messages = [raw_messages]

        for raw_message in raw_messages:
            message = _agent2_message_text(raw_message)
            if message and message not in messages:
                messages.append(message)

    return messages


def _write_agent2_assessment_request(
    *,
    run_dir: Path,
    transcript_name: str,
    assessment_request: dict[str, Any],
    approved_topics: list[dict[str, Any]],
) -> Path:
    output_path = _agent2_assessment_request_path(run_dir)
    topic_names = [
        str(topic.get("topic") or topic.get("detected_topic") or "").strip()
        for topic in approved_topics
    ]
    topic_names = [value for value in topic_names if value]
    payload = {
        "schema_version": "agent2-assessment-request-v1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": Path(run_dir).name,
        "transcript": transcript_name,
        "lesson_summary": (
            "The approved lesson topics are: " + ", ".join(topic_names) + "."
        ),
        "assessment_request": assessment_request,
        "notebook_logic_changed": False,
        "execution_method": "temporary_parameterized_copy",
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _release_status_banner(status: str, blockers: list[Any]) -> None:
    normalized = str(status or "unknown").strip().casefold()
    message = f"Assessment status: {status}"
    if normalized == "ready_for_release":
        st.success(message)
    elif normalized == "evaluation_ready":
        st.info(message)
    else:
        st.warning(message)
    if blockers:
        st.caption(
            "Release blocker(s): "
            + ", ".join(str(value) for value in blockers)
        )


def _render_phase3_structured_mark_scheme(structured: dict[str, Any]) -> None:
    if not isinstance(structured, dict):
        st.info("No Phase 3 structured mark scheme is available.")
        return
    left, middle, right = st.columns(3)
    left.metric("Cleanup status", structured.get("cleanup_status", "unknown"))
    middle.metric("Rule confidence", structured.get("rule_confidence", "N/A"))
    right.metric("Blocks", structured.get("block_count", 0))
    sections = [
        ("Marking points", structured.get("marking_points", [])),
        ("Acceptable answers", structured.get("acceptable_answers", [])),
        ("Rejected answers", structured.get("rejected_answers", [])),
        ("Additional guidance", structured.get("additional_guidance", [])),
        ("Worked examples", structured.get("worked_examples", [])),
        ("Assessment objectives", structured.get("assessment_objectives", [])),
    ]
    for title, value in sections:
        st.markdown(f"**{title}**")
        if value:
            st.json(value)
        else:
            st.caption("None")
    review_reasons = structured.get("review_reasons", [])
    if review_reasons:
        st.warning(
            "Phase 3 review reason(s): "
            + ", ".join(str(value) for value in review_reasons)
        )


AGENT2_RETRIEVAL_HITL_PHASE1_VERSION = "agent2-retrieval-hitl-phase1-v1.0.0"
AGENT2_RETRIEVAL_HITL_PHASE2_VERSION = "agent2-retrieval-hitl-phase2-v1.1.0"
AGENT2_RETRIEVAL_HITL_PHASE3_VERSION = "agent2-retrieval-hitl-phase3-recall-v1.0.0"
AGENT2_RETRIEVAL_HITL_PHASE4_VERSION = "agent2-retrieval-hitl-phase4-bounded-ranking-v1.0.0"
AGENT2_RETRIEVAL_MEMORY_SCHEMA_VERSION = "agent2-retrieval-memory-v1.0.0"
AGENT2_RETRIEVAL_MEMORY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE = 384


def _agent2_retrieval_package_fingerprint(
    package: dict[str, Any],
) -> str:
    """Stable identity for one concrete Notebook 05 assessment package."""
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
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _agent2_retrieval_feedback_engine() -> Any:
    database_url = str(
        os.getenv("AGENT2_DATABASE_URL", "")
        or os.getenv("DATABASE_URL", "")
        or ""
    ).strip()
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL / AGENT2_DATABASE_URL is not configured for "
            "Agent 2 retrieval HITL feedback."
        )
    return create_engine(
        _normalize_database_url(database_url),
        pool_pre_ping=True,
        future=True,
    )


def _ensure_agent2_retrieval_feedback_table(engine: Any) -> None:
    """Create/migrate retrieval feedback storage for Phase 1 + Phase 2 memory."""
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
            decision TEXT NOT NULL
                CHECK (decision IN ('relevant', 'not_relevant')),
            reason TEXT,
            reviewed_by TEXT NOT NULL DEFAULT 'streamlit',
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
        """
        CREATE INDEX IF NOT EXISTS ix_agent2_retrieval_feedback_package
        ON agent2_retrieval_feedback (package_fingerprint, created_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_agent2_retrieval_feedback_run_question
        ON agent2_retrieval_feedback (pipeline_run_id, question_id, created_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_agent2_retrieval_feedback_memory_status
        ON agent2_retrieval_feedback (memory_status, created_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_agent2_retrieval_feedback_memory_key
        ON agent2_retrieval_feedback (memory_key, created_at DESC)
        """,
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _agent2_retrieval_topic_context(
    *,
    run_dir: Path,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Recover the Agent 1 topic identity/evidence used for this retrieval."""
    topic = item.get("topic") or {}
    retrieval = item.get("retrieval") or {}
    if not isinstance(topic, dict):
        topic = {}
    if not isinstance(retrieval, dict):
        retrieval = {}

    context = {
        "agent1_topic_index": retrieval.get("agent1_topic_index"),
        "concept_id": None,
        "transcript_evidence": str(
            retrieval.get("query_evidence") or ""
        ).strip(),
        "transcript_evidence_source": str(
            retrieval.get("query_evidence_source") or ""
        ).strip(),
    }

    # Backward-compatible fallback for packages created before Notebook 05
    # exposed query_evidence directly in the assessment package.
    try:
        _, approved_topics_path = _agent2_handoff_paths(run_dir)
        approved_payload = load_json(approved_topics_path)
        approved_topics = approved_payload.get("topics") or []
    except Exception:
        approved_topics = []

    detected_topic = str(topic.get("detected_topic") or "").strip().casefold()
    official_reference = str(topic.get("official_reference") or "").strip()

    for approved in approved_topics:
        if not isinstance(approved, dict):
            continue
        approved_topic = str(
            approved.get("topic") or approved.get("detected_topic") or ""
        ).strip().casefold()
        approved_reference = str(
            approved.get("official_reference") or ""
        ).strip()
        if (
            approved_reference == official_reference
            and (not detected_topic or approved_topic == detected_topic)
        ):
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
                    context["transcript_evidence_source"] = (
                        "approved_topics_source_chunk_texts"
                    )
            break

    return context


def _agent2_retrieval_context_identity(
    *,
    run_dir: Path,
    item: dict[str, Any],
) -> dict[str, str]:
    """Build the exact lesson/question identity used for safe historical HITL reuse."""
    topic = item.get("topic") or {}
    if not isinstance(topic, dict):
        topic = {}

    context = _agent2_retrieval_topic_context(
        run_dir=run_dir,
        item=item,
    )
    evidence = _normalise_agent2_retrieval_memory_evidence(
        context.get("transcript_evidence")
    )
    evidence_hash = (
        hashlib.sha256(evidence.casefold().encode("utf-8")).hexdigest()
        if evidence
        else ""
    )

    return {
        "question_id": str(item.get("question_id") or "").strip(),
        "concept_id": str(context.get("concept_id") or "").strip(),
        "detected_topic": str(
            topic.get("detected_topic") or topic.get("topic") or ""
        ).strip().casefold(),
        "official_reference": str(
            topic.get("official_reference") or ""
        ).strip(),
        "agent1_role": str(
            topic.get("role") or topic.get("topic_role") or ""
        ).strip().casefold(),
        "lesson_evidence_hash": evidence_hash,
    }


def _agent2_retrieval_feedback_matches_context(
    *,
    row: dict[str, Any],
    current_identity: dict[str, str],
) -> bool:
    """Fail closed unless historical feedback matches the exact lesson context."""
    row_evidence = _normalise_agent2_retrieval_memory_evidence(
        row.get("transcript_evidence")
    )
    if not row_evidence:
        return False
    row_evidence_hash = hashlib.sha256(
        row_evidence.casefold().encode("utf-8")
    ).hexdigest()

    required_pairs = [
        (str(row.get("question_id") or "").strip(), current_identity["question_id"]),
        (str(row.get("detected_topic") or "").strip().casefold(), current_identity["detected_topic"]),
        (str(row.get("official_reference") or "").strip(), current_identity["official_reference"]),
        (str(row.get("agent1_role") or "").strip().casefold(), current_identity["agent1_role"]),
        (row_evidence_hash, current_identity["lesson_evidence_hash"]),
    ]
    if any(not left or not right or left != right for left, right in required_pairs):
        return False

    # Concept ID is an additional guard whenever both the old and current
    # payloads expose it. Older Phase 1 rows may legitimately have it blank.
    old_concept_id = str(row.get("concept_id") or "").strip()
    current_concept_id = str(current_identity.get("concept_id") or "").strip()
    if old_concept_id and current_concept_id and old_concept_id != current_concept_id:
        return False

    return True


def _agent2_latest_retrieval_feedback(
    *,
    run_dir: Path,
    package: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Return the latest compatible decision for every current selected question.

    Preference order:
    1. exact current package/run feedback;
    2. latest historical feedback for the exact same question + lesson context.

    Package timestamps therefore do not force the reviewer to label the same
    question again after a Notebook 05 rerun, while different lesson evidence
    remains fail-closed.
    """
    fingerprint = _agent2_retrieval_package_fingerprint(package)
    engine = _agent2_retrieval_feedback_engine()
    _ensure_agent2_retrieval_feedback_table(engine)

    questions = [
        item
        for item in (package.get("questions") or [])
        if isinstance(item, dict) and str(item.get("question_id") or "").strip()
    ]
    latest: dict[str, dict[str, Any]] = {}

    with engine.connect() as connection:
        for item in questions:
            current_identity = _agent2_retrieval_context_identity(
                run_dir=run_dir,
                item=item,
            )
            question_id = current_identity["question_id"]
            if not current_identity["lesson_evidence_hash"]:
                # No evidence = no historical contextual reuse. Exact current
                # package feedback can still be displayed if it exists.
                candidates = connection.execute(
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
                ).mappings().all()
            else:
                # Pull recent events for this stable official-question ID, then
                # perform the strict evidence/topic/role compatibility check in
                # Python. Current-package rows are preferred before history.
                candidates = connection.execute(
                    text(
                        """
                        SELECT *
                        FROM agent2_retrieval_feedback
                        WHERE question_id = :question_id
                        ORDER BY
                            CASE
                                WHEN package_fingerprint = :package_fingerprint
                                 AND pipeline_run_id = :pipeline_run_id
                                THEN 0 ELSE 1
                            END,
                            created_at DESC,
                            id DESC
                        LIMIT 200
                        """
                    ),
                    {
                        "question_id": question_id,
                        "package_fingerprint": fingerprint,
                        "pipeline_run_id": Path(run_dir).name,
                    },
                ).mappings().all()

            for candidate_row in candidates:
                row = dict(candidate_row)
                is_current_package = (
                    str(row.get("package_fingerprint") or "") == fingerprint
                    and str(row.get("pipeline_run_id") or "") == Path(run_dir).name
                )
                if is_current_package or _agent2_retrieval_feedback_matches_context(
                    row=row,
                    current_identity=current_identity,
                ):
                    row["context_match_source"] = (
                        "current_package"
                        if is_current_package
                        else "historical_exact_context"
                    )
                    row["historical_context_reused"] = not is_current_package
                    latest[question_id] = row
                    break

    return latest


def _agent2_retrieval_memory_load_env() -> None:
    """Load Agent 2 Qdrant settings without overriding already-set environment."""
    for candidate in agent2_project_candidates(PROJECT_ROOT):
        try:
            candidate = Path(candidate)
        except Exception:
            continue
        if candidate.is_dir():
            load_dotenv(candidate / ".env", override=False)
            break


def _agent2_retrieval_memory_collection_name() -> str:
    _agent2_retrieval_memory_load_env()
    explicit = str(
        os.getenv("AGENT2_RETRIEVAL_MEMORY_COLLECTION", "") or ""
    ).strip()
    if explicit:
        return explicit

    agent1_collection = str(
        os.getenv(
            "QDRANT_COLLECTION",
            "aqa_gcse_computer_science_8525",
        )
        or "aqa_gcse_computer_science_8525"
    ).strip()
    question_collection = str(
        os.getenv("AGENT2_QDRANT_COLLECTION", "") or ""
    ).strip() or f"{agent1_collection}_questions"
    return f"{question_collection}_retrieval_memory"


@st.cache_resource(show_spinner=False)
def _agent2_retrieval_memory_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for Phase 2 retrieval memory."
        ) from exc
    return SentenceTransformer(model_name)


def _agent2_retrieval_memory_client() -> tuple[Any, Any, str]:
    """Return Qdrant client, models module and dedicated retrieval-memory collection."""
    _agent2_retrieval_memory_load_env()
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is required for Phase 2 retrieval memory."
        ) from exc

    qdrant_url = str(
        os.getenv("QDRANT_URL", "http://localhost:6333")
        or "http://localhost:6333"
    ).strip()
    qdrant_api_key = str(os.getenv("QDRANT_API_KEY", "") or "").strip() or None
    qdrant_timeout = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "30") or 30)
    prefer_grpc = str(
        os.getenv("QDRANT_PREFER_GRPC", "false") or "false"
    ).strip().casefold() in {"1", "true", "yes"}

    kwargs: dict[str, Any] = {
        "url": qdrant_url,
        "timeout": qdrant_timeout,
        "prefer_grpc": prefer_grpc,
    }
    if qdrant_api_key is not None:
        kwargs["api_key"] = qdrant_api_key

    client = QdrantClient(**kwargs)
    collection_name = _agent2_retrieval_memory_collection_name()
    return client, models, collection_name


def _ensure_agent2_retrieval_memory_collection() -> tuple[Any, Any, str]:
    """Create/validate the separate 384-d cosine retrieval-memory collection."""
    client, models, collection_name = _agent2_retrieval_memory_client()

    if hasattr(client, "collection_exists"):
        exists = bool(client.collection_exists(collection_name))
    else:
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
                "Existing retrieval-memory Qdrant collection has vector size "
                f"{vector_size}; expected {AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE}. "
                "It was not recreated automatically."
            )

    # Payload indexes are optional accelerators for Phase 3 lookup. Index creation
    # is best-effort because older/local Qdrant versions may reject duplicate calls.
    payload_indexes = {
        "concept_id": models.PayloadSchemaType.KEYWORD,
        "official_reference": models.PayloadSchemaType.KEYWORD,
        "agent1_role": models.PayloadSchemaType.KEYWORD,
        "decision": models.PayloadSchemaType.KEYWORD,
        "question_id": models.PayloadSchemaType.KEYWORD,
        "memory_context_hash": models.PayloadSchemaType.KEYWORD,
    }
    for field_name, field_schema in payload_indexes.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except Exception:
            pass

    return client, models, collection_name


def _normalise_agent2_retrieval_memory_evidence(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _agent2_retrieval_memory_material(row: dict[str, Any]) -> dict[str, Any]:
    """Build stable contextual memory text/key from one PostgreSQL feedback event."""
    evidence = str(row.get("transcript_evidence") or "").strip()
    if not evidence:
        raise ValueError(
            "Retrieval memory was not indexed because transcript/lesson evidence "
            "is missing. Phase 2 intentionally refuses topic-label-only memory."
        )

    normalized_evidence = _normalise_agent2_retrieval_memory_evidence(evidence)
    evidence_hash = hashlib.sha256(
        normalized_evidence.casefold().encode("utf-8")
    ).hexdigest()
    spec_version = str(
        os.getenv("AQA_SPEC_VERSION", "")
        or "AQA-8525-v1.2-2022-11-29"
    ).strip()

    memory_identity = {
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
        json.dumps(
            memory_identity,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    point_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"agent2-retrieval-memory:{memory_key}",
    )

    memory_text = "\n".join(
        [
            "Agent 2 retrieval feedback memory",
            f"Memory schema: {AGENT2_RETRIEVAL_MEMORY_SCHEMA_VERSION}",
            f"AQA specification: {spec_version}",
            f"Approved concept ID: {str(row.get('concept_id') or '').strip() or 'unknown'}",
            f"Approved lesson topic: {str(row.get('detected_topic') or '').strip()}",
            f"Official reference: {str(row.get('official_reference') or '').strip()}",
            f"Topic role: {str(row.get('agent1_role') or '').strip()}",
            "Lesson evidence:",
            evidence,
            "Retrieved official question:",
            str(row.get("question_text") or "").strip(),
            f"Paper: {str(row.get('paper_code') or '').strip()}",
            f"Question number: {str(row.get('question_number') or '').strip()}",
            f"Marks: {row.get('question_marks') if row.get('question_marks') is not None else ''}",
            f"Human decision: {str(row.get('decision') or '').strip()}",
            "Human reason:",
            str(row.get("reason") or "").strip() or "No additional reason supplied.",
        ]
    ).strip()
    context_hash = hashlib.sha256(
        memory_text.encode("utf-8")
    ).hexdigest()

    return {
        "memory_key": memory_key,
        "point_id": point_id,
        "memory_text": memory_text,
        "context_hash": context_hash,
        "evidence_hash": evidence_hash,
        "spec_version": spec_version,
    }


def _agent2_retrieval_feedback_row(feedback_id: int) -> dict[str, Any]:
    engine = _agent2_retrieval_feedback_engine()
    _ensure_agent2_retrieval_feedback_table(engine)
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT *
                FROM agent2_retrieval_feedback
                WHERE id = :feedback_id
                """
            ),
            {"feedback_id": int(feedback_id)},
        ).mappings().first()
    if row is None:
        raise ValueError(f"Retrieval feedback row {feedback_id} was not found.")
    return dict(row)


def _mark_agent2_retrieval_memory_error(
    *,
    feedback_id: int,
    error: str,
) -> None:
    engine = _agent2_retrieval_feedback_engine()
    _ensure_agent2_retrieval_feedback_table(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE agent2_retrieval_feedback
                SET
                    memory_eligible = FALSE,
                    memory_status = 'error',
                    memory_error = :memory_error,
                    memory_phase_version = :memory_phase_version
                WHERE id = :feedback_id
                """
            ),
            {
                "feedback_id": int(feedback_id),
                "memory_error": str(error)[:4000],
                "memory_phase_version": AGENT2_RETRIEVAL_HITL_PHASE2_VERSION,
            },
        )


def _promote_agent2_retrieval_feedback_to_qdrant(
    *,
    feedback_id: int,
) -> dict[str, Any]:
    """Embed/index one human feedback row. No retrieval score is read or changed."""
    row = _agent2_retrieval_feedback_row(int(feedback_id))

    try:
        material = _agent2_retrieval_memory_material(row)
        model = _agent2_retrieval_memory_model(
            AGENT2_RETRIEVAL_MEMORY_MODEL
        )
        vector = model.encode(
            [material["memory_text"]],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        if int(len(vector)) != AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE:
            raise RuntimeError(
                "Retrieval-memory embedding has vector size "
                f"{len(vector)}; expected {AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE}."
            )

        client, models, collection_name = (
            _ensure_agent2_retrieval_memory_collection()
        )
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
            "package_fingerprint": str(row.get("package_fingerprint") or ""),
            "question_id": str(row.get("question_id") or ""),
            "selected_rank": row.get("selected_rank"),
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
            # Phase 4 may use this point on a FUTURE Notebook 05 run, but only
            # after the Phase 3 compatibility guard passes.
            "ranking_adjustment_enabled": True,
            "ranking_policy_version": AGENT2_RETRIEVAL_HITL_PHASE4_VERSION,
        }
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(material["point_id"]),
                    vector=vector.tolist(),
                    payload=payload,
                )
            ],
            wait=True,
        )

        engine = _agent2_retrieval_feedback_engine()
        _ensure_agent2_retrieval_feedback_table(engine)
        with engine.begin() as connection:
            # PostgreSQL keeps the full append-only feedback history. Qdrant keeps
            # only the latest decision for the same exact context/question memory key.
            connection.execute(
                text(
                    """
                    UPDATE agent2_retrieval_feedback
                    SET
                        memory_eligible = FALSE,
                        memory_status = 'superseded',
                        memory_superseded_by_feedback_id = :feedback_id
                    WHERE memory_key = :memory_key
                      AND id <> :feedback_id
                      AND memory_status = 'indexed'
                    """
                ),
                {
                    "feedback_id": int(row["id"]),
                    "memory_key": material["memory_key"],
                },
            )
            connection.execute(
                text(
                    """
                    UPDATE agent2_retrieval_feedback
                    SET
                        memory_eligible = TRUE,
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
            "feedback_id": int(row["id"]),
            "memory_eligible": True,
            "memory_key": material["memory_key"],
            "memory_point_id": str(material["point_id"]),
            "memory_collection": collection_name,
            "embedding_model": AGENT2_RETRIEVAL_MEMORY_MODEL,
            "vector_size": AGENT2_RETRIEVAL_MEMORY_VECTOR_SIZE,
            "ranking_adjustment_enabled": True,
            "ranking_policy_version": AGENT2_RETRIEVAL_HITL_PHASE4_VERSION,
        }
    except Exception as exc:
        _mark_agent2_retrieval_memory_error(
            feedback_id=int(feedback_id),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _sync_agent2_retrieval_feedback_memory(
    *,
    run_dir: Path,
    package: dict[str, Any],
) -> dict[str, Any]:
    """Backfill the latest exact-context feedback, including safe historical rows."""
    compatible = _agent2_latest_retrieval_feedback(
        run_dir=run_dir,
        package=package,
    )
    rows = [
        row
        for row in compatible.values()
        if isinstance(row, dict) and row.get("id") is not None
    ]

    indexed = 0
    skipped = 0
    historical_matches = 0
    errors: list[dict[str, Any]] = []
    for row in rows:
        if bool(row.get("historical_context_reused")):
            historical_matches += 1
        if bool(row.get("memory_eligible")) and str(
            row.get("memory_status") or ""
        ).strip().casefold() == "indexed":
            skipped += 1
            continue
        try:
            _promote_agent2_retrieval_feedback_to_qdrant(
                feedback_id=int(row["id"])
            )
        except Exception as exc:
            errors.append(
                {
                    "feedback_id": int(row["id"]),
                    "question_id": str(row.get("question_id") or ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            indexed += 1

    return {
        "status": "complete" if not errors else "partial",
        "matched_feedback_rows": len(rows),
        "historical_exact_context_matches": historical_matches,
        "indexed": indexed,
        "already_indexed": skipped,
        "errors": errors,
        "memory_collection": _agent2_retrieval_memory_collection_name(),
        "ranking_adjustment_enabled": True,
        "ranking_policy_version": AGENT2_RETRIEVAL_HITL_PHASE4_VERSION,
    }


def _persist_agent2_retrieval_feedback(
    *,
    run_dir: Path,
    package: dict[str, Any],
    item: dict[str, Any],
    decision: str,
    reason: str,
    reviewed_by: str = "streamlit",
) -> dict[str, Any]:
    """Append one human relevance decision, then index contextual Phase 2 memory."""
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

    context = _agent2_retrieval_topic_context(
        run_dir=run_dir,
        item=item,
    )
    fingerprint = _agent2_retrieval_package_fingerprint(package)
    feedback_event_id = uuid.uuid4()

    engine = _agent2_retrieval_feedback_engine()
    _ensure_agent2_retrieval_feedback_table(engine)

    insert_sql = text(
        """
        INSERT INTO agent2_retrieval_feedback (
            feedback_event_id,
            package_fingerprint,
            pipeline_run_id,
            package_generated_at_utc,
            retrieval_version,
            question_id,
            selected_rank,
            agent1_topic_index,
            concept_id,
            detected_topic,
            official_reference,
            agent1_role,
            transcript_evidence,
            transcript_evidence_source,
            semantic_score,
            base_final_score,
            retrieval_stage,
            query_evidence_source,
            paper_code,
            question_number,
            question_marks,
            question_text,
            decision,
            reason,
            reviewed_by,
            phase_version,
            memory_eligible,
            memory_status
        ) VALUES (
            CAST(:feedback_event_id AS UUID),
            :package_fingerprint,
            :pipeline_run_id,
            CAST(NULLIF(:package_generated_at_utc, '') AS TIMESTAMPTZ),
            :retrieval_version,
            :question_id,
            :selected_rank,
            :agent1_topic_index,
            :concept_id,
            :detected_topic,
            :official_reference,
            :agent1_role,
            :transcript_evidence,
            :transcript_evidence_source,
            :semantic_score,
            :base_final_score,
            :retrieval_stage,
            :query_evidence_source,
            :paper_code,
            :question_number,
            :question_marks,
            :question_text,
            :decision,
            :reason,
            :reviewed_by,
            :phase_version,
            FALSE,
            'pending_index'
        )
        RETURNING id, feedback_event_id, created_at
        """
    )

    def _optional_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _optional_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    with engine.begin() as connection:
        saved = connection.execute(
            insert_sql,
            {
                "feedback_event_id": str(feedback_event_id),
                "package_fingerprint": fingerprint,
                "pipeline_run_id": Path(run_dir).name,
                "package_generated_at_utc": str(
                    package.get("generated_at_utc") or ""
                ).strip(),
                "retrieval_version": str(
                    package.get("retrieval_version") or ""
                ).strip() or None,
                "question_id": str(item.get("question_id") or "").strip(),
                "selected_rank": _optional_int(
                    item.get("rank") or retrieval.get("selected_rank")
                ),
                "agent1_topic_index": _optional_int(
                    context.get("agent1_topic_index")
                ),
                "concept_id": str(
                    context.get("concept_id") or ""
                ).strip() or None,
                "detected_topic": str(
                    topic.get("detected_topic") or ""
                ).strip() or None,
                "official_reference": str(
                    topic.get("official_reference") or ""
                ).strip() or None,
                "agent1_role": str(
                    topic.get("role") or ""
                ).strip().casefold() or None,
                "transcript_evidence": str(
                    context.get("transcript_evidence") or ""
                ).strip() or None,
                "transcript_evidence_source": str(
                    context.get("transcript_evidence_source") or ""
                ).strip() or None,
                "semantic_score": _optional_float(
                    retrieval.get("semantic_score")
                ),
                "base_final_score": _optional_float(
                    retrieval.get("final_score")
                ),
                "retrieval_stage": str(
                    retrieval.get("stage") or ""
                ).strip() or None,
                "query_evidence_source": str(
                    retrieval.get("query_evidence_source") or ""
                ).strip() or None,
                "paper_code": str(question.get("paper_code") or "").strip() or None,
                "question_number": str(question.get("number") or "").strip() or None,
                "question_marks": _optional_int(question.get("marks")),
                "question_text": str(question.get("text") or "").strip(),
                "decision": normalized_decision,
                "reason": reason_text or None,
                "reviewed_by": reviewed_by,
                "phase_version": AGENT2_RETRIEVAL_HITL_PHASE2_VERSION,
            },
        ).mappings().one()

    result = {
        "status": "saved",
        "feedback_id": int(saved["id"]),
        "feedback_event_id": str(saved["feedback_event_id"]),
        "created_at": str(saved["created_at"]),
        "decision": normalized_decision,
        "memory_eligible": False,
        "qdrant_written": False,
        "ranking_adjustment_enabled": True,
        "ranking_policy_version": AGENT2_RETRIEVAL_HITL_PHASE4_VERSION,
    }

    try:
        memory_result = _promote_agent2_retrieval_feedback_to_qdrant(
            feedback_id=int(saved["id"])
        )
    except Exception as exc:
        result["memory_status"] = "error"
        result["memory_error"] = f"{type(exc).__name__}: {exc}"
    else:
        result.update(
            {
                "memory_status": memory_result["status"],
                "memory_eligible": True,
                "qdrant_written": True,
                "memory_point_id": memory_result["memory_point_id"],
                "memory_collection": memory_result["memory_collection"],
            }
        )

    return result


def _render_agent2_retrieval_feedback(
    *,
    run_dir: Path,
    package: dict[str, Any],
    item: dict[str, Any],
    position: int,
    existing_feedback: dict[str, Any] | None,
) -> None:
    """Compact human feedback UI for Notebook 05 retrieval learning."""
    existing = existing_feedback or {}
    existing_decision = str(existing.get("decision") or "").strip().casefold()
    decision_options = ["relevant", "not_relevant"]
    default_index = (
        decision_options.index(existing_decision)
        if existing_decision in decision_options
        else 0
    )

    st.markdown("#### Retrieval feedback")

    if existing_decision:
        label = "Relevant" if existing_decision == "relevant" else "Not Relevant"
        historical_reuse = bool(existing.get("historical_context_reused"))
        suffix = " · reused from the same lesson context" if historical_reuse else ""
        st.caption(f"Saved decision: {label}{suffix}")

    question_id = str(item.get("question_id") or position)
    form_key = f"agent2_retrieval_feedback_{Path(run_dir).name}_{position}_{question_id}"

    with st.form(form_key):
        decision = st.radio(
            "Does this question match what was taught in this lesson?",
            options=decision_options,
            index=default_index,
            format_func=lambda value: "Relevant" if value == "relevant" else "Not Relevant",
            horizontal=True,
        )
        reason = st.text_area(
            "Reason" + (
                " (required)" if decision == "not_relevant" else " (optional)"
            ),
            value=str(existing.get("reason") or ""),
            height=80,
            help="This reason is saved with the lesson context so future retrievals can learn from it.",
        ).strip()
        save_feedback = st.form_submit_button(
            "Save feedback",
            type="primary",
            use_container_width=True,
        )

    if not save_feedback:
        return

    if decision == "not_relevant" and not reason:
        st.error("Please add a reason for Not Relevant feedback.")
        return

    try:
        result = _persist_agent2_retrieval_feedback(
            run_dir=run_dir,
            package=package,
            item=item,
            decision=decision,
            reason=reason,
        )
    except Exception as exc:
        st.error(f"Could not save retrieval feedback: {exc}")
        return

    if result.get("qdrant_written"):
        st.session_state["agent2_retrieval_feedback_flash"] = (
            "Feedback saved. Future retrievals with a compatible lesson context will learn from this decision."
        )
    else:
        st.session_state["agent2_retrieval_feedback_flash"] = (
            "Feedback was saved, but retrieval-memory indexing could not complete. "
            f"The audit record is safe. Error: {result.get('memory_error') or 'unknown error'}"
        )
    st.rerun()


def _agent2_phase3_memory_diag(item: dict[str, Any]) -> dict[str, Any]:
    retrieval = item.get("retrieval") or {}
    if not isinstance(retrieval, dict):
        return {}
    diagnostic = retrieval.get("memory_recall_phase3") or {}
    return diagnostic if isinstance(diagnostic, dict) else {}


def _agent2_phase4_memory_diag(item: dict[str, Any]) -> dict[str, Any]:
    retrieval = item.get("retrieval") or {}
    if not isinstance(retrieval, dict):
        return {}
    diagnostic = retrieval.get("memory_ranking_phase4") or {}
    return diagnostic if isinstance(diagnostic, dict) else {}


def _render_agent2_retrieval_learning_effect(*, item: dict[str, Any]) -> None:
    """Show only the user-facing effect of previously saved retrieval feedback."""
    diagnostic = _agent2_phase4_memory_diag(item)
    if not diagnostic:
        return

    applied = bool(diagnostic.get("applied"))
    decision = str(diagnostic.get("decision") or "").strip().casefold()
    adjustment = float(diagnostic.get("ranking_adjustment") or 0.0)

    if applied and decision == "relevant":
        st.success(
            f"Previous feedback from a matching lesson context supported this question "
            f"({adjustment:+.4f} retrieval adjustment)."
        )
    elif applied and decision == "not_relevant":
        st.warning(
            f"Previous feedback from a matching lesson context reduced this question's "
            f"retrieval score ({adjustment:+.4f})."
        )

    if applied and diagnostic.get("reason"):
        st.caption(f"Previous feedback reason: {diagnostic.get('reason')}")


def render_agent2_assessment_results(*, run_dir: Path) -> None:
    """
    Render only the most recent Agent 2 attempt for this Agent 1 run.

    Expected retrieval limitations (partial coverage, wrong-paper-only topics,
    or zero quality-safe questions) are displayed as user-friendly warnings.
    Genuine execution errors remain red failures.
    """
    attempt_path = _agent2_frontend_attempt_path(run_dir)
    attempt = load_json(attempt_path) if attempt_path.is_file() else {}
    attempt_status = str(attempt.get("status") or "").strip().casefold()

    if attempt_status == "running":
        st.info("The latest Agent 2 assessment request is still running.")
        return

    if attempt_status == "failed":
        st.markdown("---")
        st.subheader("Agent 2 Assessment")
        st.error(
            "The latest Agent 2 execution failed. A previous successful "
            "assessment is hidden so it cannot be mistaken for the current "
            "request."
        )
        error_text = str(attempt.get("error") or "").strip()
        if error_text:
            with st.expander("Latest Agent 2 error details"):
                st.code(error_text)
        return

    manifest_path = _agent2_result_manifest_path(run_dir)
    if not manifest_path.is_file():
        return

    manifest = load_json(manifest_path)

    output_dir = Path(
        manifest.get(
            "output_dir",
            Path(run_dir) / "output" / "agent2",
        )
    )

    package_path = _resolve_agent2_file(
        manifest.get("package_path"),
        output_dir=output_dir,
    )

    # Safe fallback for Notebook 05 v2.7+ no-assessment runs.
    if package_path is None:
        current_run_path = output_dir / "agent2_current_run.json"
        current_run = load_json(current_run_path)
        package_path = _resolve_agent2_file(
            current_run.get("assessment_package_json"),
            output_dir=output_dir,
        )

    if package_path is None:
        st.error(
            "The latest Agent 2 execution completed, but its current-run "
            "assessment package could not be found."
        )
        return

    package = load_json(package_path)
    questions = package.get("questions", []) or []

    if not isinstance(questions, list):
        questions = []

    retrieval_feedback_error: str | None = None
    try:
        retrieval_feedback_lookup = _agent2_latest_retrieval_feedback(
            run_dir=run_dir,
            package=package,
        )
    except Exception as exc:
        retrieval_feedback_lookup = {}
        retrieval_feedback_error = str(exc)

    selection_summary = package.get("selection_summary", {}) or {}
    retrieval_summary = package.get("retrieval_summary", {}) or {}

    package_output_files = package.get("output_files", {}) or {}
    release_path = None

    if isinstance(package_output_files, dict):
        release_path = _resolve_agent2_file(
            package_output_files.get("release_readiness_json"),
            output_dir=output_dir,
        )

    if release_path is None:
        release_path = _resolve_agent2_file(
            manifest.get("release_readiness_path"),
            output_dir=output_dir,
        )

    release_payload = (
        load_json(release_path)
        if release_path is not None
        else {}
    )

    # Later/more specific payloads override earlier summary values.
    summary: dict[str, Any] = {}
    if isinstance(selection_summary, dict):
        summary.update(selection_summary)
    if isinstance(retrieval_summary, dict):
        summary.update(retrieval_summary)
    if isinstance(release_payload, dict):
        summary.update(release_payload)

    run_status = str(
        package.get("run_status")
        or summary.get("run_status")
        or "success"
    ).strip()

    assessment_generated_raw = package.get("assessment_generated")
    assessment_generated = (
        bool(questions)
        if assessment_generated_raw is None
        else bool(assessment_generated_raw)
    )

    user_messages = _collect_agent2_user_messages(
        package,
        selection_summary
        if isinstance(selection_summary, dict)
        else {},
        retrieval_summary
        if isinstance(retrieval_summary, dict)
        else {},
        release_payload
        if isinstance(release_payload, dict)
        else {},
    )

    st.markdown("---")
    st.subheader("Agent 2 Assessment Result")

    release_status = summary.get(
        "final_release_status",
        summary.get(
            "assessment_release_status",
            run_status,
        ),
    )

    blockers = summary.get("release_blockers", []) or []
    if not isinstance(blockers, list):
        blockers = [blockers]

    _release_status_banner(
        str(release_status),
        blockers,
    )

    normalized_run_status = run_status.casefold()

    if normalized_run_status == "partial_success":
        st.warning(
            "The assessment was generated with limited question availability. "
            "Only quality-safe questions were used."
        )

    if user_messages:
        st.markdown("#### Agent 2 messages")
        for message in user_messages:
            st.warning(message)

    selected_questions = int(
        summary.get(
            "selected_questions",
            len(questions),
        )
        or 0
    )

    selected_marks = summary.get(
        "selected_marks",
        0 if not assessment_generated else "N/A",
    )

    topics_covered = summary.get(
        "selected_distinct_agent1_topics",
        summary.get(
            "selected_distinct_topics",
            summary.get(
                "selected_distinct_official_references",
                0 if not assessment_generated else "N/A",
            ),
        ),
    )

    evidence_used = summary.get(
        "actual_agent1_chunk_evidence_used",
        summary.get(
            "all_topics_use_actual_chunk_evidence",
            False,
        ),
    )

    metrics = st.columns(4)
    metrics[0].metric("Questions", selected_questions)
    metrics[1].metric("Total marks", selected_marks)
    metrics[2].metric("Topics covered", topics_covered)
    metrics[3].metric(
        "Actual chunk evidence",
        "Yes" if evidence_used else "No",
    )

    requested_paper = summary.get(
        "requested_paper_label",
        summary.get(
            "requested_paper_code",
            package.get("request", {}).get("paper_code"),
        ),
    )
    resolved_paper = summary.get(
        "resolved_kb_paper_code",
    )

    if requested_paper is not None or resolved_paper is not None:
        st.caption(
            "Paper filter — requested: "
            f"{requested_paper if requested_paper is not None else 'Both'}"
            + (
                f" | knowledge-base code: {resolved_paper}"
                if resolved_paper is not None
                else ""
            )
        )

    if not assessment_generated:
        st.warning(
            "No assessment was generated for the current filters. "
            "No incompatible, wrong-paper, or weak question was substituted. "
            "Change the paper/filter settings or approved topics and run again."
        )

        return

    feedback_flash = st.session_state.pop(
        "agent2_retrieval_feedback_flash",
        None,
    )
    if feedback_flash:
        st.success(feedback_flash)

    if assessment_generated:
        reviewed_count = sum(
            1
            for item in questions
            if isinstance(item, dict)
            and str(item.get("question_id") or "") in retrieval_feedback_lookup
        )
        historical_match_count = sum(
            1
            for value in retrieval_feedback_lookup.values()
            if isinstance(value, dict) and bool(value.get("historical_context_reused"))
        )
        suppressed_count = int(summary.get("phase4_exact_context_suppressed_count", 0) or 0)

        st.caption(
            f"Retrieval feedback: {reviewed_count}/{len(questions)} selected question(s) reviewed. "
            "Saved feedback is reused only when the future lesson context is compatible."
        )

        if historical_match_count:
            st.info(
                f"Reused {historical_match_count} previous feedback decision(s) from the same lesson context."
            )

        if suppressed_count:
            st.info(
                f"{suppressed_count} question(s) previously marked Not Relevant for this same lesson "
                "were filtered out before final selection."
            )

        if retrieval_feedback_error:
            st.warning(
                "Feedback storage is currently unavailable, but the assessment result is unaffected: "
                f"{retrieval_feedback_error}"
            )

    for position, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            continue

        question = item.get("question", {}) or {}
        topic = item.get("topic", {}) or {}
        retrieval = item.get("retrieval", {}) or {}
        mark_scheme = item.get("mark_scheme", {}) or {}

        marks = question.get("marks", "N/A")
        title = (
            f"Question {position} — {marks} mark"
            f"{'s' if marks != 1 else ''} — "
            f"{topic.get('detected_topic', 'Topic')}"
        )

        with st.expander(
            title,
            expanded=(position == 1),
        ):
            metadata_columns = st.columns(3)
            metadata_columns[0].metric(
                "Official reference",
                topic.get("official_reference", "N/A"),
            )
            metadata_columns[1].metric(
                "Role",
                topic.get("role", "N/A"),
            )
            metadata_columns[2].metric(
                "Paper",
                question.get("paper_code", "N/A"),
            )

            st.markdown("#### Question")
            st.write(question.get("text", ""))

            context_text = str(
                question.get("context", "")
                or ""
            ).strip()

            if context_text:
                st.markdown("**Context**")
                st.write(context_text)

            image_paths = (
                question.get("rendered_page_images", [])
                or []
            )

            if image_paths:
                st.markdown(
                    "#### Original question page image(s)"
                )

                for relative_path in image_paths:
                    image_path = Path(str(relative_path))

                    if not image_path.is_file():
                        image_path = output_dir / str(
                            relative_path
                        )

                    if image_path.is_file():
                        st.image(
                            str(image_path),
                            use_container_width=True,
                        )
                    else:
                        st.warning(
                            "Rendered image file not found: "
                            f"{relative_path}"
                        )

            raw_tab, structured_tab = st.tabs(
                [
                    "Mark scheme",
                    "Structured mark scheme",
                ]
            )

            with raw_tab:
                st.text_area(
                    "Raw marking guidance",
                    str(
                        mark_scheme.get(
                            "raw_marking_guidance",
                            mark_scheme.get(
                                "marking_guidance",
                                "",
                            ),
                        )
                        or ""
                    ),
                    height=320,
                    disabled=True,
                    key=(
                        "agent2_raw_mark_scheme_"
                        f"{Path(run_dir).name}_{position}"
                    ),
                )

            with structured_tab:
                _render_phase3_structured_mark_scheme(
                    mark_scheme.get(
                        "phase3_structured",
                        {},
                    )
                    or {}
                )

            _render_agent2_retrieval_learning_effect(item=item)

            _render_agent2_retrieval_feedback(
                run_dir=run_dir,
                package=package,
                item=item,
                position=position,
                existing_feedback=retrieval_feedback_lookup.get(
                    str(item.get("question_id") or "")
                ),
            )

    downloads = _agent2_current_download_files(
        package_path=package_path,
        package=package,
        manifest=manifest,
        output_dir=output_dir,
    )

    # Production UI: expose only the student-facing assessment PDF here.
    shortfall_manifest = _agent2_quiz_manifest(run_dir, "fill_shortfall")
    shortfall_output_files = (
        shortfall_manifest.get("output_files")
        if isinstance(shortfall_manifest, dict)
        else {}
    ) or {}
    shortfall_final_pdf = _resolve_agent2_file(
        shortfall_output_files.get("questions_and_marking_schemes_pdf"),
        output_dir=_agent2_quiz_output_dir(run_dir, "fill_shortfall"),
    )

    pdf_downloads = [
        Path(path)
        for path in downloads
        if Path(path).suffix.casefold() == ".pdf"
    ]
    if shortfall_final_pdf is not None:
        pdf_downloads = []

    if pdf_downloads:
        st.markdown("#### Download assessment")
        for path in pdf_downloads:
            download_file(path, f"Download {path.name}")


def _quiz_model_config_path(
    agent2_root_value: str | Path,
) -> Path:
    return (
        Path(
            agent2_root_value
        ).expanduser().resolve()
        / "config"
        / "quiz_model_config.json"
    )


def _load_quiz_model_config_for_ui(
    agent2_root_value: str | Path,
) -> dict[str, Any]:
    path = _quiz_model_config_path(
        agent2_root_value
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Quiz model config was not found: "
            f"{path}. Copy quiz_model_config.json into Agent2/config/."
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "quiz_model_config.json must contain a JSON object."
        )

    models = payload.get(
        "models"
    )

    if not isinstance(
        models,
        dict,
    ) or not models:
        raise ValueError(
            "quiz_model_config.json must contain a non-empty 'models' object."
        )

    return payload


def _quiz_model_selection_path(
    run_dir: Path,
) -> Path:
    path = (
        Path(
            run_dir
        )
        / "output"
        / "integration"
        / "quiz_model_selection.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _load_saved_quiz_model_key(
    run_dir: Path,
) -> str:
    path = _quiz_model_selection_path(
        run_dir
    )

    if not path.is_file():
        return ""

    payload = load_json(
        path
    )

    return str(
        payload.get(
            "model_key",
            "",
        )
        or ""
    ).strip()


def _save_quiz_model_selection(
    *,
    run_dir: Path,
    model_key: str,
    model_config: dict[str, Any],
) -> Path:
    models = model_config.get(
        "models"
    ) or {}

    selected = models.get(
        model_key
    )

    if not isinstance(
        selected,
        dict,
    ):
        raise ValueError(
            f"Unknown quiz model key: {model_key}"
        )

    path = _quiz_model_selection_path(
        run_dir
    )

    payload = {
        "schema_version":
            "agent2-quiz-model-selection-v1.0.0",
        "updated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "model_key":
            model_key,
        "display_name":
            selected.get(
                "display_name"
            ),
        "provider":
            selected.get(
                "provider"
            ),
        "model_id":
            selected.get(
                "model_id"
            ),
        "context_window_tokens":
            selected.get(
                "context_window_tokens"
            ),
        "hard_max_output_tokens":
            selected.get(
                "hard_max_output_tokens"
            ),
        "source":
            "streamlit_quiz_model_selector",
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


def _quiz_model_display_name(
    model_config: dict[str, Any],
    model_key: str,
) -> str:
    models = model_config.get(
        "models"
    ) or {}

    item = models.get(
        model_key
    ) or {}

    return str(
        item.get(
            "display_name",
            model_key,
        )
        or model_key
    ).strip()


def _agent2_quiz_output_dir(run_dir: Path, quiz_mode: str) -> Path:
    return Path(run_dir) / "output" / "agent2_quiz" / str(quiz_mode)


def _agent2_quiz_manifest(run_dir: Path, quiz_mode: str) -> dict[str, Any]:
    """Use MCP/Notebook 08 final output once the post-MCP PDF is ready."""
    output_dir = _agent2_quiz_output_dir(
        run_dir,
        quiz_mode,
    )

    mcp_manifest = (
        output_dir
        / "mcp_visuals"
        / "final_quiz_manifest_with_mcp_visuals.json"
    )

    if mcp_manifest.is_file():
        payload = load_json(mcp_manifest)
        mcp_pdf = payload.get("mcp_final_pdf") or {}
        if (
            isinstance(mcp_pdf, dict)
            and str(mcp_pdf.get("status") or "").strip().upper() == "SAVED"
        ):
            return payload

    return load_json(
        output_dir
        / "final_quiz_manifest.json"
    )


AGENT2_ASSESSMENT_PATTERN_OPTIONS = [
    "apply_or_predict",
    "explain_or_reason",
    "analyse_or_interpret",
    "compare_or_select",
    "diagnose_or_correct",
    "construct_or_complete",
    "scenario_application",
    "predict_consequence",
    "evaluate_or_justify",
    "adapt_or_modify",
]

AGENT2_QUESTION_ACTION_PLACEHOLDER = "__SELECT_ACTION__"

AGENT2_QUESTION_ACTION_LABELS = {
    "approve": "Approve this question",
    "edit_question": "Edit question text",
    "edit_marking_guidance": "Edit marking guidance",
    "change_pattern": "Change assessment pattern",
    "remove_visual": "Remove visual",
    "regenerate": "Regenerate only this question",
    "reject": "Reject this question",
}

AGENT2_QUESTION_REVIEW_STATES = {
    # Normal review states
    "AWAITING_QUESTION_LEVEL_REVIEW",
    "AWAITING_HUMAN_REVIEW",
    "BLOCKED_HUMAN_QUESTION_REJECTION",

    # Recoverable candidate states: keep question-level HITL visible so the
    # reviewer can fix metadata/content locally instead of losing the review UI.
    "BLOCKED_STRUCTURAL_VALIDATION",
    "BLOCKED_SEMANTIC_FAIL",
    "BLOCKED_HITL_ACTION_ERROR",
    "REGENERATION_FAILED_STRUCTURAL",
    "REGENERATION_FAILED_SEMANTIC",
    "REGENERATION_BUDGET_EXHAUSTED",
}


def _agent2_question_review_queue(
    *,
    run_dir: Path,
    quiz_mode: str,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load the Notebook 06 question-level HITL queue."""

    section = manifest.get("question_level_human_review") or {}
    if not isinstance(section, dict):
        section = {}

    queue = section.get("queue") or []
    if isinstance(queue, list):
        rows = [item for item in queue if isinstance(item, dict)]
        if rows:
            return rows

    path_values = [
        section.get("queue_path"),
        (manifest.get("source_artifacts") or {}).get("human_review_queue")
        if isinstance(manifest.get("source_artifacts") or {}, dict)
        else None,
        _agent2_quiz_output_dir(run_dir, quiz_mode)
        / "generated_human_review_queue.json",
    ]

    output_dir = _agent2_quiz_output_dir(run_dir, quiz_mode)
    for raw_path in path_values:
        path = _resolve_agent2_file(raw_path, output_dir=output_dir)
        if path is None:
            continue
        payload = load_json(path)
        raw_questions = payload.get("questions", payload.get("question_review_queue", []))
        if isinstance(raw_questions, list):
            rows = [item for item in raw_questions if isinstance(item, dict)]
            if rows:
                return rows

    generated_review = manifest.get("generated_human_review") or {}
    if isinstance(generated_review, dict):
        raw_questions = generated_review.get("question_review_queue") or []
        if isinstance(raw_questions, list):
            return [item for item in raw_questions if isinstance(item, dict)]

    return []


def _agent2_generated_human_review_payload(
    *,
    run_dir: Path,
    quiz_mode: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    value = manifest.get("generated_human_review") or {}
    if isinstance(value, dict) and value:
        return value

    return load_json(
        _agent2_quiz_output_dir(run_dir, quiz_mode)
        / "generated_quality_human_review.json"
    )


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
    path = _agent2_question_actions_path(run_dir, quiz_mode)
    payload = {
        "schema_version": "agent2-streamlit-question-actions-v1.0.0",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quiz_mode": quiz_mode,
        "actions": actions,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _persist_agent2_question_approval_fallback(
    *,
    run_dir: Path,
    quiz_mode: str,
    action: dict[str, Any],
) -> dict[str, Any]:
    """
    Persist an approve-only question decision when an older controller bridge
    does not yet accept the Notebook 06 v2.32 `pending` review decision.

    Approvals are audit history only and are deliberately NOT Qdrant memory.
    Corrective actions still go through Notebook 06 so they are revalidated and
    can become eligible generation memory.
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

    manifest = _agent2_quiz_manifest(run_dir, quiz_mode)
    candidate_questions = manifest.get("candidate_questions") or []
    if not isinstance(candidate_questions, list):
        candidate_questions = []

    question_id = str(action.get("question_id") or "").strip()
    try:
        plan_index = int(action.get("plan_index") or 0)
    except (TypeError, ValueError):
        plan_index = 0

    question = next(
        (
            item
            for item in candidate_questions
            if isinstance(item, dict)
            and (
                (
                    question_id
                    and str(
                        item.get("generated_question_id")
                        or item.get("question_id")
                        or ""
                    ).strip()
                    == question_id
                )
                or (
                    plan_index > 0
                    and int(item.get("plan_index") or 0) == plan_index
                )
            )
        ),
        {},
    )
    if not isinstance(question, dict):
        question = {}

    recorded_at = datetime.now(timezone.utc).isoformat()
    event = {
        "generation_request_fingerprint": str(
            manifest.get("generation_request_fingerprint") or ""
        ),
        "final_payload_fingerprint": str(
            (manifest.get("generated_human_review") or {}).get("payload_fingerprint")
            if isinstance(manifest.get("generated_human_review") or {}, dict)
            else ""
        ),
        "question_id": question_id,
        "plan_index": plan_index,
        "topic": str(question.get("topic") or ""),
        "official_reference": str(question.get("official_reference") or ""),
        "role": str(question.get("role") or ""),
        "assigned_pattern": str(question.get("assessment_pattern") or ""),
        "action": "approve",
        "reason": str(action.get("reason") or ""),
        "before_question": question,
        "after_question": question,
        "pipeline_version": str(
            (manifest.get("generated_human_review") or {}).get("pipeline_version")
            if isinstance(manifest.get("generated_human_review") or {}, dict)
            else ""
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
                "payload_fingerprint": event["final_payload_fingerprint"],
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
                "model_key": str(manifest.get("selected_model_key") or ""),
                "model_id": str(manifest.get("generation_model") or ""),
                "pipeline_version": event["pipeline_version"],
                "recorded_at_utc": recorded_at,
            },
        )
        persisted = max(0, int(result.rowcount or 0))

    # Keep the same local audit filename as Notebook 06 v2.32.
    feedback_path = (
        _agent2_quiz_output_dir(run_dir, quiz_mode)
        / "generated_human_review_feedback.jsonl"
    )
    existing_ids: set[str] = set()
    if feedback_path.is_file():
        for line in feedback_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except Exception:
                continue
            existing_id = str(payload.get("event_id") or "")
            if existing_id:
                existing_ids.add(existing_id)
    if event_id not in existing_ids:
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    return {
        "status": "OK",
        "persisted": persisted,
        "event_id": event_id,
        "qdrant_promoted": False,
    }


def _call_agent2_quiz_review_bridge(
    *,
    run_dir: Path,
    quiz_mode: str,
    decision: str,
    reason: str,
    question_actions: list[dict[str, Any]] | None = None,
) -> Any:
    """
    Submit Notebook 06 HITL through the existing LangGraph/MCP bridge.

    Notebook 06 reads question actions either directly from the bridge or from
    AGENT2_QUIZ_REVIEW_ACTIONS_PATH / JSON. The environment fallback keeps this
    app compatible while the controller-side adapter is updated independently.
    """

    actions = [
        item
        for item in (question_actions or [])
        if isinstance(item, dict)
    ]

    action_path: Path | None = None
    if actions:
        action_path = _write_agent2_question_actions(
            run_dir=run_dir,
            quiz_mode=quiz_mode,
            actions=actions,
        )

    # Notebook 06 v2.32 uses AGENT2_DATABASE_URL for question-feedback
    # persistence. Default it to the existing application DB when needed.
    if not str(os.getenv("AGENT2_DATABASE_URL", "") or "").strip():
        database_url = str(os.getenv("DATABASE_URL", "") or "").strip()
        if database_url:
            os.environ["AGENT2_DATABASE_URL"] = database_url

    previous_actions_path = os.environ.get("AGENT2_QUIZ_REVIEW_ACTIONS_PATH")
    previous_actions_json = os.environ.get("AGENT2_QUIZ_REVIEW_ACTIONS_JSON")

    if actions and action_path is not None:
        os.environ["AGENT2_QUIZ_REVIEW_ACTIONS_PATH"] = str(action_path.resolve())
        os.environ["AGENT2_QUIZ_REVIEW_ACTIONS_JSON"] = json.dumps(
            actions,
            ensure_ascii=False,
        )
    else:
        os.environ.pop("AGENT2_QUIZ_REVIEW_ACTIONS_PATH", None)
        os.environ.pop("AGENT2_QUIZ_REVIEW_ACTIONS_JSON", None)

    signature = inspect.signature(submit_human_agent2_quiz_review)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    kwargs: dict[str, Any] = {
        "frontend_root": PROJECT_ROOT,
        "run_id": Path(run_dir).name,
        "quiz_mode": quiz_mode,
        "decision": decision,
        "reason": reason,
        "reviewed_by": "streamlit",
    }

    def add_optional(names: list[str], value: Any) -> bool:
        for name in names:
            if name in parameters or accepts_var_kwargs:
                kwargs[name] = value
                return True
        return False

    if actions:
        add_optional(
            [
                "question_actions",
                "question_level_actions",
                "review_actions",
                "actions",
            ],
            actions,
        )
        if action_path is not None:
            add_optional(
                [
                    "question_actions_path",
                    "question_level_actions_path",
                    "review_actions_path",
                ],
                str(action_path.resolve()),
            )

    # Only pass names supported by a strict non-**kwargs bridge signature.
    if not accepts_var_kwargs:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }

    try:
        return submit_human_agent2_quiz_review(**kwargs)
    except Exception as exc:
        if (
            actions
            and decision == "pending"
            and all(
                str(item.get("action") or "").strip().casefold() == "approve"
                for item in actions
            )
        ):
            # Older bridge versions may not expose a no-op/pending quiz-review
            # decision. Approve-only question actions do not change content, so
            # persist them directly as audit history rather than accidentally
            # approving/regenerating the entire quiz.
            persisted = [
                _persist_agent2_question_approval_fallback(
                    run_dir=run_dir,
                    quiz_mode=quiz_mode,
                    action=item,
                )
                for item in actions
            ]
            return {
                "status": "QUESTION_APPROVAL_PERSISTED",
                "bridge_fallback": True,
                "details": persisted,
                "original_bridge_error": str(exc),
            }
        if actions and decision == "pending":
            raise RuntimeError(
                "Question-level Agent 2 HITL could not be submitted through "
                "the current LangGraph/MCP bridge. Notebook 06 HITL is ready, "
                "but submit_human_agent2_quiz_review must allow the 'pending' "
                "decision and forward the question-action JSON/path. Original "
                f"error: {exc}"
            ) from exc
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


def _agent2_question_priority_style(priority: str) -> tuple[str, str]:
    normalized = str(priority or "NORMAL").strip().upper()
    if normalized == "HIGH":
        return "🔴", "High-priority review"
    if normalized == "REVIEW":
        return "🟡", "Review suggested"
    return "🟢", "No actionable validator issue"



def _agent2_question_review_fingerprint(question: dict[str, Any]) -> str:
    """Fingerprint only the learner-visible / structural fields relevant to HITL."""
    payload = {
        "question_id": str(
            question.get("generated_question_id")
            or question.get("question_id")
            or ""
        ).strip(),
        "plan_index": int(question.get("plan_index") or 0),
        "topic": str(question.get("topic") or "").strip(),
        "official_reference": str(
            question.get("official_reference") or ""
        ).strip(),
        "role": str(question.get("role") or "").strip(),
        "marks": int(question.get("marks") or 0),
        "assessment_pattern": str(
            question.get("assessment_pattern") or ""
        ).strip(),
        "visual_requirement": str(
            question.get("visual_requirement") or "none"
        ).strip(),
        "question_text": str(
            question.get("question_text")
            or _generated_question_text(question)
            or ""
        ).strip(),
        "marking_guidance": question.get("marking_guidance") or [],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _agent2_latest_current_version_actions(
    *,
    manifest: dict[str, Any],
    candidate_questions: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """
    Return the latest PostgreSQL HITL action that applies to each CURRENT
    candidate question version.

    This prevents an approval for an older question version from silently
    satisfying the final gate after an edit/regeneration.
    """
    database_url = str(
        os.getenv("AGENT2_DATABASE_URL", "")
        or os.getenv("DATABASE_URL", "")
        or ""
    ).strip()
    if not database_url:
        return {}

    request_fp = str(
        manifest.get("generation_request_fingerprint") or ""
    ).strip()
    if not request_fp:
        return {}

    current_by_plan: dict[int, dict[str, Any]] = {}
    for question in candidate_questions:
        if not isinstance(question, dict):
            continue
        try:
            plan_index = int(question.get("plan_index") or 0)
        except (TypeError, ValueError):
            continue
        if plan_index > 0:
            current_by_plan[plan_index] = question

    if not current_by_plan:
        return {}

    try:
        engine = create_engine(
            _normalize_database_url(database_url),
            pool_pre_ping=True,
            future=True,
        )
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        id,
                        question_id,
                        plan_index,
                        action,
                        reason,
                        after_question,
                        recorded_at_utc
                    FROM agent2_question_feedback
                    WHERE generation_request_fingerprint = :request_fp
                    ORDER BY id DESC
                    """
                ),
                {"request_fp": request_fp},
            ).mappings().all()
    except Exception:
        # HITL UI remains usable even if the optional DB status lookup fails.
        return {}

    latest: dict[int, dict[str, Any]] = {}

    for row in rows:
        try:
            plan_index = int(row.get("plan_index") or 0)
        except (TypeError, ValueError):
            continue
        if plan_index <= 0 or plan_index in latest:
            continue

        current_question = current_by_plan.get(plan_index)
        if not isinstance(current_question, dict):
            continue

        after_question = row.get("after_question")
        if isinstance(after_question, str):
            try:
                after_question = json.loads(after_question)
            except Exception:
                after_question = {}
        if not isinstance(after_question, dict):
            after_question = {}

        if (
            _agent2_question_review_fingerprint(after_question)
            != _agent2_question_review_fingerprint(current_question)
        ):
            # Stale action belonging to an older question version.
            continue

        latest[plan_index] = dict(row)

    return latest


def _agent2_question_review_resolution_summary(
    *,
    manifest: dict[str, Any],
    queue: list[dict[str, Any]],
    candidate_questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    REVIEW/HIGH items need a current-version human approval unless a corrective
    action causes the rebuilt queue to become NORMAL.
    """
    latest_actions = _agent2_latest_current_version_actions(
        manifest=manifest,
        candidate_questions=candidate_questions,
    )

    unresolved: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []

    for item in queue:
        if not isinstance(item, dict):
            continue
        priority = str(item.get("priority") or "NORMAL").strip().upper()
        if priority not in {"REVIEW", "HIGH"}:
            continue

        try:
            plan_index = int(item.get("plan_index") or 0)
        except (TypeError, ValueError):
            plan_index = 0

        latest = latest_actions.get(plan_index) or {}
        latest_action = str(latest.get("action") or "").strip().casefold()

        if latest_action == "approve":
            resolved.append(
                {
                    "plan_index": plan_index,
                    "question_id": item.get("question_id"),
                    "priority": priority,
                    "action": latest_action,
                }
            )
        else:
            unresolved.append(
                {
                    "plan_index": plan_index,
                    "question_id": item.get("question_id"),
                    "priority": priority,
                    "latest_current_version_action": latest_action or None,
                }
            )

    return {
        "required_review_count": len(unresolved) + len(resolved),
        "resolved_review_count": len(resolved),
        "unresolved_review_count": len(unresolved),
        "resolved": resolved,
        "unresolved": unresolved,
        "latest_current_version_actions": latest_actions,
    }


def _agent2_candidate_question_lookup(
    questions: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_plan: dict[int, dict[str, Any]] = {}

    for question in questions:
        if not isinstance(question, dict):
            continue
        question_id = str(
            question.get("generated_question_id")
            or question.get("question_id")
            or ""
        ).strip()
        if question_id:
            by_id[question_id] = question

        try:
            plan_index = int(question.get("plan_index") or 0)
        except (TypeError, ValueError):
            plan_index = 0
        if plan_index > 0:
            by_plan[plan_index] = question

    return by_id, by_plan


def _render_agent2_review_visual(
    *,
    question: dict[str, Any],
    output_dir: Path,
) -> None:
    for key in (
        "visual_asset_path",
        "rendered_visual_path",
    ):
        path = _resolve_agent2_file(
            question.get(key),
            output_dir=output_dir,
        )
        if path is not None and path.suffix.casefold() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            st.image(str(path), use_container_width=True)
            return


def _render_agent2_question_level_hitl(
    *,
    run_dir: Path,
    quiz_mode: str,
    manifest: dict[str, Any],
    candidate_questions: list[dict[str, Any]],
    human_state: str,
) -> None:
    """Render Notebook 06 per-question HITL controls."""

    queue = _agent2_question_review_queue(
        run_dir=run_dir,
        quiz_mode=quiz_mode,
        manifest=manifest,
    )
    if not queue:
        if human_state == "AWAITING_QUESTION_LEVEL_REVIEW":
            st.warning(
                "Notebook 06 requested question-level review, but the review "
                "queue could not be loaded. Check generated_human_review_queue.json."
            )
        return

    counts = {"NORMAL": 0, "REVIEW": 0, "HIGH": 0}
    for item in queue:
        priority = str(item.get("priority") or "NORMAL").strip().upper()
        counts[priority if priority in counts else "REVIEW"] += 1

    st.markdown("#### Question-level Human Review")
    st.caption(
        "Review each generated question independently. Approve needs no reason; "
        "edits, pattern changes, visual removal, regeneration, and rejection "
        "require a written reason. Notebook 06 stores the action in PostgreSQL; "
        "eligible corrective feedback is promoted to Qdrant and can guide future "
        "similar targeted regenerations."
    )

    resolution_summary = _agent2_question_review_resolution_summary(
        manifest=manifest,
        queue=queue,
        candidate_questions=candidate_questions,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Questions", len(queue))
    m2.metric("Normal", counts["NORMAL"])
    m3.metric("Review", counts["REVIEW"])
    m4.metric("High", counts["HIGH"])
    m5.metric(
        "Needs decision",
        int(resolution_summary.get("unresolved_review_count") or 0),
    )

    unresolved_count = int(
        resolution_summary.get("unresolved_review_count") or 0
    )
    if unresolved_count:
        st.warning(
            f"{unresolved_count} REVIEW/HIGH question(s) still need an explicit "
            "human decision before final quiz approval."
        )
    elif int(resolution_summary.get("required_review_count") or 0) > 0:
        st.success(
            "All REVIEW/HIGH question recommendations have been explicitly resolved."
        )

    review_payload = _agent2_generated_human_review_payload(
        run_dir=run_dir,
        quiz_mode=quiz_mode,
        manifest=manifest,
    )
    if isinstance(review_payload, dict):
        db_status = review_payload.get("feedback_db_persistence") or {}
        qdrant_status = review_payload.get("feedback_qdrant_memory") or {}
        if isinstance(db_status, dict) or isinstance(qdrant_status, dict):
            with st.expander("HITL feedback persistence status", expanded=False):
                st.json(
                    {
                        "PostgreSQL": db_status,
                        "Qdrant memory": qdrant_status,
                        "human_targeted_regeneration_attempts_used": review_payload.get(
                            "human_targeted_regeneration_attempts_used"
                        ),
                        "max_human_targeted_regeneration_attempts": review_payload.get(
                            "max_human_targeted_regeneration_attempts"
                        ),
                    }
                )

    candidate_by_id, candidate_by_plan = _agent2_candidate_question_lookup(
        candidate_questions
    )
    output_dir = _agent2_quiz_output_dir(run_dir, quiz_mode)

    for item in queue:
        question_id = str(item.get("question_id") or "").strip()
        try:
            plan_index = int(item.get("plan_index") or 0)
        except (TypeError, ValueError):
            plan_index = 0

        question = (
            candidate_by_id.get(question_id)
            or candidate_by_plan.get(plan_index)
            or {}
        )

        priority = str(item.get("priority") or "NORMAL").strip().upper()
        icon, priority_label = _agent2_question_priority_style(priority)
        topic = str(item.get("topic") or question.get("topic") or "Topic").strip()
        marks = item.get("marks", question.get("marks", "N/A"))
        title = (
            f"{icon} Q{plan_index or '?'} — {topic} — {marks} mark"
            f"{'s' if marks != 1 else ''} — {priority_label}"
        )

        with st.expander(title, expanded=(priority != "NORMAL")):
            meta = st.columns(4)
            meta[0].metric(
                "Role",
                str(item.get("role") or question.get("role") or "N/A"),
            )
            meta[1].metric(
                "AQA ref",
                str(
                    item.get("official_reference")
                    or question.get("official_reference")
                    or "N/A"
                ),
            )
            meta[2].metric(
                "Pattern",
                str(item.get("assigned_pattern") or "N/A"),
            )
            meta[3].metric(
                "Priority",
                priority,
            )

            question_text = str(
                item.get("question_text")
                or question.get("question_text")
                or _generated_question_text(question)
                or ""
            ).strip()
            st.markdown("**Question**")
            st.write(question_text or "Question text unavailable.")
            _render_agent2_review_visual(
                question=question,
                output_dir=output_dir,
            )

            guidance = item.get("marking_guidance") or question.get("marking_guidance") or []
            if isinstance(guidance, list) and guidance:
                with st.expander("Current AI marking guidance", expanded=False):
                    for criterion in guidance:
                        if isinstance(criterion, dict):
                            st.write(
                                f"• {str(criterion.get('criterion') or '').strip()} "
                                f"({int(criterion.get('marks') or 0)} mark(s))"
                            )

            issue_rows: list[str] = []
            for failure in item.get("hard_failures") or []:
                issue_rows.append("🔴 " + str(failure))
            for issue in item.get("command_marking_issues") or []:
                issue_rows.append("🟡 Mark scheme: " + str(issue))
            if str(item.get("pattern_alignment_status") or "").upper() == "REVIEW":
                issue_rows.append(
                    "🟡 Pattern: "
                    + str(item.get("pattern_alignment_reason") or "Review pattern alignment.")
                )
            if str(item.get("topic_grounding_status") or "").upper() in {"REVIEW", "FAIL"}:
                issue_rows.append(
                    "🟡 Topic grounding: "
                    + str(item.get("topic_grounding_reason") or "Review grounding.")
                )
            if str(item.get("cognitive_demand_status") or "").upper() == "REVIEW":
                issue_rows.append(
                    "🟡 Cognitive demand: "
                    + str(item.get("cognitive_demand_reason") or "Question may be too low-demand.")
                )

            visual_necessity = str(item.get("visual_necessity") or "").strip().upper()
            visual_status = str(item.get("visual_relevance_status") or "").strip().upper()
            if visual_necessity or visual_status:
                st.caption(
                    "Visual check: "
                    f"{visual_necessity or 'N/A'}"
                    + (f" · {visual_status}" if visual_status else "")
                    + (
                        " · " + str(item.get("visual_relevance_reason") or "")
                        if item.get("visual_relevance_reason")
                        else ""
                    )
                )

            if issue_rows:
                st.markdown("**Validator findings**")
                for row in issue_rows:
                    st.write(row)
            elif priority == "NORMAL":
                st.success("No actionable deterministic validator issue was found.")

            notes = item.get("notes") or []
            if isinstance(notes, list) and notes:
                with st.expander("Additional quality notes", expanded=False):
                    for note in notes:
                        st.write(f"• {note}")

            suggested_pattern = str(item.get("suggested_pattern") or "").strip()
            if suggested_pattern:
                st.caption(f"Suggested pattern: {suggested_pattern}")

            current_action = (
                resolution_summary.get("latest_current_version_actions", {})
                or {}
            ).get(plan_index) or {}
            current_action_name = str(
                current_action.get("action") or ""
            ).strip().casefold()
            if current_action_name:
                st.caption(
                    "Current-version human decision: "
                    + AGENT2_QUESTION_ACTION_LABELS.get(
                        current_action_name,
                        current_action_name,
                    )
                )

            available_actions = [
                str(value).strip().casefold()
                for value in (item.get("available_actions") or [])
                if str(value).strip().casefold() in AGENT2_QUESTION_ACTION_LABELS
            ]
            if not available_actions:
                available_actions = list(AGENT2_QUESTION_ACTION_LABELS)

            # Do not offer visual removal when no generated visual exists.
            has_visual = bool(
                str(question.get("visual_requirement") or "none").strip().casefold()
                != "none"
                or _resolve_agent2_file(
                    question.get("visual_asset_path"),
                    output_dir=output_dir,
                )
                is not None
            )
            if not has_visual and "remove_visual" in available_actions:
                available_actions.remove("remove_visual")

            selector_key = (
                f"agent2_q_action_{Path(run_dir).name}_{quiz_mode}_{plan_index}_{question_id}"
            )
            action_options = [
                AGENT2_QUESTION_ACTION_PLACEHOLDER,
                *available_actions,
            ]
            selected_action = st.selectbox(
                "Human action",
                options=action_options,
                index=0,
                format_func=lambda value: (
                    "— Select an action —"
                    if value == AGENT2_QUESTION_ACTION_PLACEHOLDER
                    else AGENT2_QUESTION_ACTION_LABELS.get(value, value)
                ),
                key=selector_key,
            )

            form_key = (
                f"agent2_q_action_form_{Path(run_dir).name}_{quiz_mode}_{plan_index}_{question_id}"
            )
            with st.form(form_key):
                edited_question_text = question_text
                edited_guidance_df: pd.DataFrame | None = None
                selected_pattern = str(
                    item.get("assigned_pattern")
                    or question.get("assessment_pattern")
                    or AGENT2_ASSESSMENT_PATTERN_OPTIONS[0]
                ).strip()

                if selected_action == "edit_question":
                    edited_question_text = st.text_area(
                        "Corrected question text",
                        value=question_text,
                        height=180,
                    ).strip()

                elif selected_action == "edit_marking_guidance":
                    rows = []
                    for criterion in guidance if isinstance(guidance, list) else []:
                        if isinstance(criterion, dict):
                            rows.append(
                                {
                                    "marks": int(criterion.get("marks") or 0),
                                    "criterion": str(criterion.get("criterion") or "").strip(),
                                }
                            )
                    if not rows:
                        rows = [{"marks": 1, "criterion": ""}]
                    edited_guidance_df = st.data_editor(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                        num_rows="dynamic",
                        column_config={
                            "marks": st.column_config.NumberColumn(
                                "Marks",
                                min_value=1,
                                step=1,
                                required=True,
                            ),
                            "criterion": st.column_config.TextColumn(
                                "Criterion",
                                required=True,
                            ),
                        },
                        key=(
                            f"agent2_q_marking_editor_{Path(run_dir).name}_{quiz_mode}_{plan_index}"
                        ),
                    )

                elif selected_action == "change_pattern":
                    options = list(AGENT2_ASSESSMENT_PATTERN_OPTIONS)
                    for value in (
                        str(item.get("assigned_pattern") or "").strip(),
                        suggested_pattern,
                    ):
                        if value and value not in options:
                            options.append(value)
                    default_pattern = suggested_pattern or selected_pattern
                    default_index = (
                        options.index(default_pattern)
                        if default_pattern in options
                        else 0
                    )
                    selected_pattern = st.selectbox(
                        "Correct assessment pattern",
                        options=options,
                        index=default_index,
                    )

                action_selected = (
                    selected_action
                    != AGENT2_QUESTION_ACTION_PLACEHOLDER
                )
                reason_required = bool(
                    action_selected
                    and selected_action != "approve"
                )
                reason = st.text_area(
                    "Reason" + (
                        " (required)"
                        if reason_required
                        else " (optional)"
                    ),
                    help=str(
                        item.get("reason_prompt")
                        or "Corrective feedback is stored for audit and future learning."
                    ),
                    height=90,
                ).strip()

                submit_action = st.form_submit_button(
                    "Apply this question action",
                    type=(
                        "primary"
                        if selected_action in {
                            "regenerate",
                            "edit_question",
                            "edit_marking_guidance",
                        }
                        else "secondary"
                    ),
                    use_container_width=True,
                    disabled=not action_selected,
                )

            if submit_action:
                if selected_action == AGENT2_QUESTION_ACTION_PLACEHOLDER:
                    st.error("Select a human action first.")
                    continue
                if selected_action != "approve" and not reason:
                    st.error("A written reason is required for this action.")
                    continue

                action_payload: dict[str, Any] = {
                    "question_id": question_id,
                    "plan_index": plan_index,
                    "action": selected_action,
                    "reason": reason,
                }

                if selected_action == "edit_question":
                    if not edited_question_text:
                        st.error("Corrected question text cannot be blank.")
                        continue
                    action_payload["question_text"] = edited_question_text

                elif selected_action == "edit_marking_guidance":
                    assert edited_guidance_df is not None
                    edited_rows: list[dict[str, Any]] = []
                    for _, row in edited_guidance_df.iterrows():
                        criterion = str(row.get("criterion") or "").strip()
                        try:
                            row_marks = int(row.get("marks") or 0)
                        except (TypeError, ValueError):
                            row_marks = 0
                        if criterion and row_marks > 0:
                            edited_rows.append(
                                {"marks": row_marks, "criterion": criterion}
                            )
                    if not edited_rows:
                        st.error(
                            "Add at least one valid marking-guidance row with positive marks."
                        )
                        continue
                    action_payload["marking_guidance"] = edited_rows

                elif selected_action == "change_pattern":
                    action_payload["assessment_pattern"] = selected_pattern

                with st.spinner(
                    "Applying question-level review through LangGraph/MCP..."
                ):
                    try:
                        result = _call_agent2_quiz_review_bridge(
                            run_dir=run_dir,
                            quiz_mode=quiz_mode,
                            decision=(
                                "pending"
                                if selected_action == "approve"
                                else "regenerate"
                            ),
                            reason=(
                                reason
                                or f"Question {question_id or plan_index} approved in Streamlit."
                            ),
                            question_actions=[action_payload],
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["langgraph_last_result"] = result

                        # Return the control to the neutral placeholder after a
                        # successful action. This prevents a previous
                        # "Regenerate only this question" selection from
                        # remaining armed on the next Streamlit rerun.
                        st.session_state.pop(selector_key, None)

                        st.success(
                            f"Question {question_id or plan_index}: '{selected_action}' applied."
                        )
                        st.rerun()

def _agent2_current_package(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    manifest = load_json(_agent2_result_manifest_path(run_dir))
    output_dir = Path(run_dir) / "output" / "agent2"
    package_path = _resolve_agent2_file(
        manifest.get("package_path"),
        output_dir=output_dir,
    )
    if package_path is None:
        current = load_json(output_dir / "agent2_current_run.json")
        package_path = _resolve_agent2_file(
            current.get("assessment_package_json"),
            output_dir=output_dir,
        )
    return package_path, (load_json(package_path) if package_path else {})


def _agent2_official_shortfall(package: dict[str, Any]) -> dict[str, Any]:
    request = package.get("assessment_request") or {}
    summary = package.get("retrieval_summary") or {}
    questions = package.get("questions") or []
    if not isinstance(request, dict):
        request = {}
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(questions, list):
        questions = []

    target_marks = int(request.get("target_total_marks") or 0)
    requested_questions = int(request.get("number_of_questions") or 0)
    selected_questions = int(summary.get("selected_questions") or len(questions))

    selected_marks = summary.get("selected_marks")
    if selected_marks is None:
        selected_marks = sum(
            int(
                question.get("marks_numeric")
                or question.get("marks_postgres")
                or question.get("marks_retrieval")
                or question.get("maximum_marks")
                or 0
            )
            for question in questions
            if isinstance(question, dict)
        )
    selected_marks = int(selected_marks or 0)

    requirements = [
        bool(summary.get("requested_question_count_met", selected_questions >= requested_questions)),
        bool(summary.get("coverage_requirements_met", True)),
        bool(summary.get("primary_requirement_met", True)),
        bool(summary.get("supporting_requirement_met", True)),
        bool(summary.get("distinct_reference_requirement_met", True)),
    ]
    if target_marks > 0:
        requirements.append(selected_marks >= target_marks)

    sufficient = bool(requirements and all(requirements))
    return {
        "sufficient": sufficient,
        "selected_questions": selected_questions,
        "requested_questions": requested_questions,
        "selected_marks": selected_marks,
        "target_marks": target_marks,
        "missing_questions": max(0, requested_questions - selected_questions),
        "missing_marks": max(0, target_marks - selected_marks),
    }


def _generated_question_marks(question: dict[str, Any]) -> int:
    for key in ["marks", "marks_numeric", "marks_postgres", "marks_retrieval", "maximum_marks"]:
        try:
            value = int(float(question.get(key) or 0))
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def _generated_question_text(question: dict[str, Any]) -> str:
    for key in ["question_text", "question_text_canonical", "question_text_postgres", "question_text_retrieval"]:
        value = str(question.get(key) or "").strip()
        if value:
            return value
    return "Question text unavailable."


def _quiz_question_paper_label(
    question: dict[str, Any],
) -> str:
    """Return Paper 1 / Paper 2 for generated or official quiz questions."""
    explicit_label = str(
        question.get("paper_label")
        or ""
    ).strip()

    if explicit_label:
        normalized = explicit_label.casefold()
        if normalized in {"1", "paper 1", "paper1", "p1"}:
            return "Paper 1"
        if normalized in {"2", "paper 2", "paper2", "p2"}:
            return "Paper 2"
        return explicit_label

    raw_paper = str(
        question.get("paper_code")
        or question.get("paper")
        or question.get("requested_paper_code")
        or ""
    ).strip().casefold()

    if raw_paper in {"1", "1a", "1b", "paper 1", "paper1", "p1"}:
        return "Paper 1"
    if raw_paper in {"2", "2a", "2b", "paper 2", "paper2", "p2"}:
        return "Paper 2"

    reference = str(
        question.get("official_reference")
        or question.get("agent1_official_reference")
        or question.get("official_reference_canonical")
        or ""
    ).strip()

    if not reference:
        return ""

    # Final fallback is PostgreSQL-backed through the existing cached
    # SyllabusStore topic options. No database lookup is performed per
    # question after the first cached syllabus load.
    try:
        syllabus_options = _official_aqa_topic_options()
    except Exception:
        return ""

    def paper_label(value: Any) -> str:
        normalized = str(value or "").strip().casefold()
        if normalized in {"1", "1a", "1b", "paper 1", "paper1", "p1"}:
            return "Paper 1"
        if normalized in {"2", "2a", "2b", "paper 2", "paper2", "p2"}:
            return "Paper 2"
        return ""

    reference_rows = [
        (
            str(item.get("official_reference") or "").strip(),
            paper_label(item.get("paper")),
        )
        for item in syllabus_options
        if isinstance(item, dict)
    ]
    reference_rows = [
        (stored_reference, label)
        for stored_reference, label in reference_rows
        if stored_reference and label
    ]

    # Exact official-reference match.
    exact_labels = {
        label
        for stored_reference, label in reference_rows
        if stored_reference == reference
    }
    if len(exact_labels) == 1:
        return next(iter(exact_labels))

    # More-specific input than a stored concept: use the longest matching
    # stored parent reference when its paper ownership is unambiguous.
    parent_matches = [
        (stored_reference, label)
        for stored_reference, label in reference_rows
        if reference.startswith(stored_reference + ".")
    ]
    if parent_matches:
        longest_length = max(
            len(stored_reference)
            for stored_reference, _ in parent_matches
        )
        parent_labels = {
            label
            for stored_reference, label in parent_matches
            if len(stored_reference) == longest_length
        }
        if len(parent_labels) == 1:
            return next(iter(parent_labels))

    # Broader input than stored concepts: resolve only when all matching
    # child references agree on the same paper.
    child_labels = {
        label
        for stored_reference, label in reference_rows
        if stored_reference.startswith(reference + ".")
    }
    if len(child_labels) == 1:
        return next(iter(child_labels))

    return ""


def _render_quiz_questions(questions: list[dict[str, Any]]) -> None:
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            continue
        generated = (
            str(question.get("source_type") or "").strip().casefold()
            == "ai_generated_aqa_aligned"
            or bool(str(question.get("generated_question_id") or "").strip())
        )
        label = (
            "AI-generated AQA-aligned practice question"
            if generated
            else "Official AQA question"
        )
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**Question {index} — {label}**")
            c2.metric("Marks", _generated_question_marks(question))
            topic = str(question.get("topic") or question.get("detected_topic") or "").strip()
            ref = str(
                question.get("official_reference")
                or question.get("agent1_official_reference")
                or question.get("official_reference_canonical")
                or ""
            ).strip()
            paper_label_for_display = _quiz_question_paper_label(
                question
            )
            metadata = [
                value
                for value in [
                    topic,
                    f"AQA {ref}" if ref else "",
                    paper_label_for_display,
                ]
                if value
            ]
            if metadata:
                st.caption(" | ".join(metadata))

            st.write(_generated_question_text(question))
            if generated:
                guidance = question.get("marking_guidance") or []
                if isinstance(guidance, list) and guidance:
                    with st.expander("AI-generated marking guidance"):
                        st.caption("This is AI-generated marking guidance, not an official AQA mark scheme.")
                        for item in guidance:
                            if isinstance(item, dict):
                                st.write(
                                    f"• {str(item.get('criterion') or '').strip()} "
                                    f"({int(item.get('marks') or 0)} mark(s))"
                                )



def _fill_shortfall_manifest_is_current_and_consistent(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
) -> tuple[bool, str]:
    """
    Validate that a saved fill_shortfall manifest still belongs to the current
    Notebook 05 retrieval and does not claim OFFICIAL_ONLY while a target gap
    still exists.

    This is a deterministic UI/state-consistency guard. It does not call an
    LLM and does not alter quiz-generation logic.
    """
    if not isinstance(
        manifest,
        dict,
    ) or not manifest:
        return False, "Quiz manifest is missing."

    # ----------------------------------------------------------
    # 1. Exact Notebook 05 lineage when available.
    # ----------------------------------------------------------
    current_package_path, current_package = _agent2_current_package(
        run_dir
    )

    source_artifacts = manifest.get(
        "source_artifacts"
    ) or {}

    if not isinstance(
        source_artifacts,
        dict,
    ):
        source_artifacts = {}

    manifest_package_raw = str(
        source_artifacts.get(
            "notebook05_package",
            "",
        )
        or ""
    ).strip()

    if (
        current_package_path is not None
        and manifest_package_raw
    ):
        try:
            manifest_package_path = Path(
                manifest_package_raw
            ).expanduser().resolve()

            if (
                manifest_package_path
                != Path(
                    current_package_path
                ).expanduser().resolve()
            ):
                return (
                    False,
                    "Saved AI-shortfall result belongs to an older "
                    "Notebook 05 retrieval.",
                )
        except OSError:
            pass

    # ----------------------------------------------------------
    # 2. Resolve request targets from current package first.
    # ----------------------------------------------------------
    current_request = (
        current_package.get(
            "assessment_request"
        )
        if isinstance(
            current_package,
            dict,
        )
        else {}
    ) or {}

    if not isinstance(
        current_request,
        dict,
    ):
        current_request = {}

    assessment_filters = manifest.get(
        "assessment_filters"
    ) or {}

    if not isinstance(
        assessment_filters,
        dict,
    ):
        assessment_filters = {}

    target_marks = int(
        current_request.get(
            "target_total_marks"
        )
        or manifest.get(
            "target_marks"
        )
        or assessment_filters.get(
            "target_total_marks"
        )
        or 0
    )

    target_questions = int(
        current_request.get(
            "number_of_questions"
        )
        or manifest.get(
            "target_question_count"
        )
        or assessment_filters.get(
            "number_of_questions"
        )
        or 0
    )

    # ----------------------------------------------------------
    # 3. Resolve current official retrieval metrics.
    # ----------------------------------------------------------
    official_marks = int(
        manifest.get(
            "official_total_marks"
        )
        or 0
    )

    official_questions = int(
        manifest.get(
            "official_question_count"
        )
        or 0
    )

    # Prefer current Notebook 05 package summary if present.
    if isinstance(
        current_package,
        dict,
    ):
        retrieval_summary = current_package.get(
            "retrieval_summary"
        ) or {}

        if isinstance(
            retrieval_summary,
            dict,
        ):
            if retrieval_summary.get(
                "selected_marks"
            ) is not None:
                official_marks = int(
                    retrieval_summary.get(
                        "selected_marks"
                    )
                    or 0
                )

            if retrieval_summary.get(
                "selected_questions"
            ) is not None:
                official_questions = int(
                    retrieval_summary.get(
                        "selected_questions"
                    )
                    or 0
                )

    assessment_type = str(
        manifest.get(
            "assessment_type",
            "",
        )
        or ""
    ).strip().upper()

    marks_short = bool(
        target_marks > 0
        and official_marks < target_marks
    )

    questions_short = bool(
        target_questions > 0
        and official_questions < target_questions
    )

    # An OFFICIAL_ONLY manifest is valid only when there is no remaining
    # target gap. This protects the UI from stale/old Notebook 06 results.
    if (
        assessment_type == "OFFICIAL_ONLY"
        and (
            marks_short
            or questions_short
        )
    ):
        return (
            False,
            "Saved OFFICIAL_ONLY result is inconsistent with the current "
            f"request: official={official_marks}/{target_marks} marks, "
            f"{official_questions}/{target_questions} questions.",
        )

    return True, ""



def _render_agent2_quiz_result(*, run_dir: Path, quiz_mode: str) -> None:
    """
    Render the current Notebook 06 quiz state.

    Candidate questions are always rendered from the manifest when available,
    regardless of whether they are accepted. Release remains fail-closed:
    only an AWAITING_HUMAN_REVIEW candidate receives Approve/Regenerate/Reject
    controls, and accepted/released metrics stay separate from candidate metrics.
    """
    manifest = _agent2_quiz_manifest(
        run_dir,
        quiz_mode,
    )
    if not manifest:
        return

    if quiz_mode == "fill_shortfall":
        (
            manifest_is_current,
            manifest_consistency_reason,
        ) = _fill_shortfall_manifest_is_current_and_consistent(
            run_dir=run_dir,
            manifest=manifest,
        )

        if not manifest_is_current:
            # Do not present a stale/internally impossible result as the
            # current assessment. The existing shortfall CTA above uses the
            # current Notebook 05 package and will offer the correct action.
            st.info(
                "Previous AI-shortfall result is stale and has been hidden. "
                "Use **Generate Missing Quiz Coverage** above for the current "
                "official retrieval."
            )

            if manifest_consistency_reason:
                st.caption(
                    manifest_consistency_reason
                )

            return

    assessment_type = str(
        manifest.get("assessment_type")
        or ""
    ).upper()

    if quiz_mode == "complete_quiz":
        title = "Complete AI Quiz"
    elif assessment_type == "HYBRID":
        title = "Official + AI Hybrid Quiz"
    elif assessment_type == "OFFICIAL_ONLY":
        title = "Official Assessment — No AI Fill Needed"
    elif assessment_type == "GENERATED_ONLY":
        title = "AI Quiz from Official-Retrieval Shortfall"
    else:
        title = "Quiz Completion Result"

    st.markdown("---")
    st.subheader(title)

    generator_name = str(
        manifest.get("generation_model_display_name")
        or manifest.get("generation_model")
        or "Configured generation model"
    ).strip()

    applied_special_instructions = str(
        manifest.get("special_instructions_applied")
        or (manifest.get("assessment_filters") or {}).get(
            "special_instructions"
        )
        or ""
    ).strip()

    if applied_special_instructions:
        with st.expander(
            "Special instructions applied",
            expanded=False,
        ):
            st.write(applied_special_instructions)

    human_state = str(
        manifest.get("generated_human_review_state")
        or ""
    ).strip().upper()

    awaiting_human_review = (
        human_state
        == "AWAITING_HUMAN_REVIEW"
    )
    awaiting_question_level_review = (
        human_state
        == "AWAITING_QUESTION_LEVEL_REVIEW"
    )
    unresolved_question_rejection = (
        human_state
        == "BLOCKED_HUMAN_QUESTION_REJECTION"
    )
    question_level_reviewable = (
        human_state in AGENT2_QUESTION_REVIEW_STATES
    )

    generated_quality_accepted = bool(
        manifest.get("generated_quality_accepted")
    )

    # Primary source: Notebook 06 already stores the enriched candidate in the
    # final manifest specifically for HITL display.
    manifest_candidate_questions = manifest.get(
        "candidate_questions"
    ) or []
    if not isinstance(manifest_candidate_questions, list):
        manifest_candidate_questions = []

    candidate_questions = [
        question
        for question in manifest_candidate_questions
        if isinstance(question, dict)
    ]

    # Backward-compatible fallback for older Notebook 06 manifests.
    if not candidate_questions:
        candidate_payload = load_json(
            _agent2_quiz_output_dir(
                run_dir,
                quiz_mode,
            )
            / "generated_questions.json"
        )
        fallback_questions = candidate_payload.get(
            "questions",
            [],
        )
        if isinstance(fallback_questions, list):
            candidate_questions = [
                question
                for question in fallback_questions
                if isinstance(question, dict)
            ]

    candidate_question_count = int(
        manifest.get("candidate_question_count")
        if manifest.get("candidate_question_count") is not None
        else len(candidate_questions)
    )
    candidate_total_marks = int(
        manifest.get("candidate_total_marks")
        if manifest.get("candidate_total_marks") is not None
        else sum(
            _generated_question_marks(question)
            for question in candidate_questions
        )
    )

    assessment_filters = manifest.get(
        "assessment_filters"
    ) or {}
    if not isinstance(assessment_filters, dict):
        assessment_filters = {}

    target_marks = int(
        manifest.get("target_marks")
        or assessment_filters.get("target_total_marks")
        or 0
    )
    target_question_count = int(
        manifest.get("target_question_count")
        or assessment_filters.get("number_of_questions")
        or 0
    )

    accepted_questions = manifest.get("questions") or []
    if not isinstance(accepted_questions, list):
        accepted_questions = []
    accepted_questions = [
        question
        for question in accepted_questions
        if isinstance(question, dict)
    ]

    # IMPORTANT: actual_* may intentionally mirror candidate metrics while HITL
    # is pending. Use the explicit accepted_* fields for accepted/released data.
    accepted_question_count = int(
        manifest.get("accepted_question_count")
        if manifest.get("accepted_question_count") is not None
        else len(accepted_questions)
    )
    accepted_total_marks = int(
        manifest.get("accepted_total_marks")
        if manifest.get("accepted_total_marks") is not None
        else sum(
            _generated_question_marks(question)
            for question in accepted_questions
        )
    )

    candidate_available = bool(
        candidate_questions
        or manifest.get("candidate_available")
    )
    show_unaccepted_candidate = bool(
        candidate_questions
        and not generated_quality_accepted
    )

    if generated_quality_accepted and accepted_questions:
        display_outcome = (
            assessment_type
            or "READY"
        )
        display_question_count = accepted_question_count
        display_total_marks = accepted_total_marks
    elif candidate_available:
        state_labels = {
            "AWAITING_HUMAN_REVIEW": "FINAL REVIEW",
            "AWAITING_QUESTION_LEVEL_REVIEW": "QUESTION REVIEW",
            "BLOCKED_HUMAN_QUESTION_REJECTION": "QUESTION REJECTED",
            "BLOCKED_HITL_ACTION_ERROR": "HITL ACTION ERROR",
            "BLOCKED_STRUCTURAL_VALIDATION": "STRUCTURAL FAIL",
            "BLOCKED_SEMANTIC_FAIL": "QUALITY FAIL",
            "REGENERATION_FAILED_STRUCTURAL": "REGEN FAILED",
            "REGENERATION_FAILED_SEMANTIC": "REGEN FAILED",
            "REGENERATION_BUDGET_EXHAUSTED": "REGEN LIMIT",
            "HUMAN_REJECTED": "REJECTED",
        }
        display_outcome = state_labels.get(
            human_state,
            "CANDIDATE",
        )
        display_question_count = candidate_question_count
        display_total_marks = candidate_total_marks
    else:
        display_outcome = (
            assessment_type
            or (
                "READY"
                if manifest.get("release_ready")
                else "—"
            )
        )
        display_question_count = accepted_question_count
        display_total_marks = accepted_total_marks

    # In fill_shortfall mode, the generated candidate represents only the
    # missing AI coverage. Keep official retrieval, AI shortfall, and the
    # projected/accepted combined quiz visibly separate.
    shortfall_ui: dict[str, Any] = {}

    if quiz_mode == "fill_shortfall":
        ui_shortfall_summary = manifest.get("ui_shortfall_summary") or {}
        if not isinstance(ui_shortfall_summary, dict):
            ui_shortfall_summary = {}

        official_summary = ui_shortfall_summary.get("official_retrieval") or {}
        ai_summary = ui_shortfall_summary.get("ai_shortfall") or {}
        combined_summary = ui_shortfall_summary.get("combined_after_approval") or {}

        if not isinstance(official_summary, dict):
            official_summary = {}
        if not isinstance(ai_summary, dict):
            ai_summary = {}
        if not isinstance(combined_summary, dict):
            combined_summary = {}

        official_marks = int(
            official_summary.get("marks")
            if official_summary.get("marks") is not None
            else manifest.get("official_total_marks") or 0
        )
        official_questions = int(
            official_summary.get("questions")
            if official_summary.get("questions") is not None
            else manifest.get("official_question_count") or 0
        )

        ai_marks = int(
            ai_summary.get("marks")
            if ai_summary.get("marks") is not None
            else (
                manifest.get("ai_shortfall_accepted_marks")
                if generated_quality_accepted
                else manifest.get("ai_shortfall_candidate_marks")
            )
            or 0
        )
        ai_questions = int(
            ai_summary.get("questions")
            if ai_summary.get("questions") is not None
            else (
                manifest.get("ai_shortfall_accepted_question_count")
                if generated_quality_accepted
                else manifest.get("ai_shortfall_candidate_question_count")
            )
            or 0
        )

        combined_marks = int(
            combined_summary.get("marks")
            if combined_summary.get("marks") is not None
            else manifest.get("combined_after_approval_total_marks")
            or (official_marks + ai_marks)
        )
        combined_questions = int(
            combined_summary.get("questions")
            if combined_summary.get("questions") is not None
            else manifest.get("combined_after_approval_question_count")
            or (official_questions + ai_questions)
        )

        shortfall_ui = {
            "official_marks": official_marks,
            "official_questions": official_questions,
            "ai_marks": ai_marks,
            "ai_questions": ai_questions,
            "combined_marks": combined_marks,
            "combined_questions": combined_questions,
        }

        # Top metrics represent the whole projected/accepted hybrid quiz, not
        # the AI-only candidate. Release readiness still remains fail-closed.
        display_total_marks = combined_marks
        display_question_count = combined_questions

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outcome", display_outcome)

    if quiz_mode == "fill_shortfall" and shortfall_ui:
        c2.metric(
            "Combined marks",
            (
                f"{display_total_marks}/{target_marks}"
                if target_marks
                else str(display_total_marks)
            ),
        )
        c3.metric(
            "Combined questions",
            str(display_question_count),
        )
    else:
        c2.metric(
            "Marks",
            (
                f"{display_total_marks}/{target_marks}"
                if target_marks
                else str(display_total_marks)
            ),
        )
        c3.metric(
            "Questions",
            (
                f"{display_question_count}/{target_question_count}"
                if target_question_count
                else str(display_question_count)
            ),
        )

    c4.metric(
        "Release ready",
        "Yes" if manifest.get("release_ready") else "No",
    )

    if quiz_mode == "fill_shortfall" and shortfall_ui:
        ai_state_label = (
            "AI shortfall accepted"
            if generated_quality_accepted
            else "AI shortfall candidate"
        )
        target_marks_text = str(target_marks) if target_marks else "—"
        target_questions_text = (
            str(target_question_count)
            if target_question_count
            else "—"
        )

        st.info(
            "**Official retrieval:** "
            f"{shortfall_ui['official_marks']}/{target_marks_text} marks · "
            f"{shortfall_ui['official_questions']}/{target_questions_text} questions  \n"
            f"**{ai_state_label}:** "
            f"+{shortfall_ui['ai_marks']} marks · "
            f"+{shortfall_ui['ai_questions']} question(s)  \n"
            "**Combined after approval:** "
            f"{shortfall_ui['combined_marks']}/{target_marks_text} marks · "
            f"{shortfall_ui['combined_questions']} questions"
        )

        if (
            target_question_count
            and shortfall_ui["combined_questions"] > target_question_count
        ):
            st.caption(
                "The requested question count is already met by official "
                "retrieval. An extra AI question is added only to satisfy the "
                "remaining mark target."
            )

    token_summary = manifest.get(
        "model_token_usage"
    ) or {}
    if isinstance(
        token_summary,
        dict,
    ) and token_summary:
        st.caption(
            "Generation model: "
            f"{token_summary.get('display_name') or generator_name}"
        )

        token_cols = st.columns(6)
        token_cols[0].metric(
            "API hits",
            int(token_summary.get("api_hits") or 0),
        )
        token_cols[1].metric(
            "Input tokens",
            int(token_summary.get("actual_input_tokens") or 0),
        )
        token_cols[2].metric(
            "Output tokens",
            int(token_summary.get("actual_output_tokens") or 0),
        )
        token_cols[3].metric(
            "Reasoning",
            int(token_summary.get("actual_reasoning_tokens") or 0),
        )
        token_cols[4].metric(
            "Total tokens",
            int(token_summary.get("actual_total_tokens") or 0),
        )
        token_cols[5].metric(
            "Context window",
            int(token_summary.get("context_window_tokens") or 0),
        )

        max_context_pct = float(
            token_summary.get("max_context_utilization_pct") or 0.0
        )
        st.caption(
            "Max actual context used by one completed response: "
            f"{max_context_pct:.2f}% | "
            "Completed API responses: "
            f"{int(token_summary.get('completed_api_responses') or 0)} | "
            "Preflight blocks: "
            f"{int(token_summary.get('preflight_blocks') or 0)}"
        )

        if int(token_summary.get("context_window_blocks") or 0) > 0:
            blocked_context_rows = [
                row
                for row in (manifest.get("model_call_usage") or [])
                if isinstance(row, dict)
                and str(row.get("status") or "") == "BLOCKED_CONTEXT_WINDOW"
            ]
            latest_block = blocked_context_rows[-1] if blocked_context_rows else {}
            st.error(
                "Quiz generation was blocked before the API call because the "
                "selected model context window would be exceeded. "
                f"Estimated input: "
                f"{latest_block.get('conservative_estimated_input_tokens', 'N/A')} tokens | "
                f"Reserved output: "
                f"{latest_block.get('requested_max_output_tokens', 'N/A')} tokens | "
                f"Estimated total: "
                f"{latest_block.get('conservative_total_reserved_tokens', 'N/A')} tokens | "
                f"Context limit: "
                f"{latest_block.get('context_window_tokens', token_summary.get('context_window_tokens', 'N/A'))} tokens. "
                "No provider API request was sent for this blocked call."
            )

    model_call_usage = manifest.get(
        "model_call_usage"
    ) or []
    if isinstance(
        model_call_usage,
        list,
    ) and model_call_usage:
        with st.expander(
            "Model call token usage",
            expanded=False,
        ):
            usage_rows = [
                row
                for row in model_call_usage
                if isinstance(
                    row,
                    dict,
                )
            ]
            if usage_rows:
                st.dataframe(
                    pd.DataFrame(
                        usage_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    question_review_queue_for_gate = _agent2_question_review_queue(
        run_dir=run_dir,
        quiz_mode=quiz_mode,
        manifest=manifest,
    )
    question_resolution_summary = (
        _agent2_question_review_resolution_summary(
            manifest=manifest,
            queue=question_review_queue_for_gate,
            candidate_questions=candidate_questions,
        )
        if candidate_questions
        else {
            "required_review_count": 0,
            "resolved_review_count": 0,
            "unresolved_review_count": 0,
            "unresolved": [],
        }
    )
    unresolved_required_reviews = int(
        question_resolution_summary.get("unresolved_review_count") or 0
    )

    if human_state:
        st.caption(
            f"Generated candidate state: {human_state}"
        )

    if awaiting_human_review and candidate_questions:
        if unresolved_required_reviews:
            st.warning(
                "No hard validation blocker remains, but "
                f"{unresolved_required_reviews} REVIEW/HIGH question(s) still "
                "need an explicit human decision before final approval."
            )
        else:
            st.caption(
                "The candidate quiz is generated and shown below. "
                "All required question-level reviews are resolved; final whole-quiz "
                "approval is still required before release."
            )

    if assessment_type == "GENERATED_ONLY":
        st.success(
            f"Complete quiz generated directly by {generator_name}. "
            "Notebook 05 was not used."
        )
    elif assessment_type == "HYBRID":
        st.success(
            "Hybrid quiz: official AQA questions + "
            "AI-generated missing coverage."
        )
    elif assessment_type == "OFFICIAL_ONLY":
        st.info(
            "Official retrieval already satisfied the request; "
            "no AI shortfall question was needed."
        )

    # Always surface the actual unaccepted candidate if Notebook 06 produced it.
    # v2.32 adds a question-level HITL queue; repairable semantic failures are
    # reviewed here rather than being treated as terminal pipeline failures.
    if show_unaccepted_candidate:
        if awaiting_question_level_review:
            st.warning(
                "One or more generated questions need question-level review. "
                "Use the controls below to approve, edit, relabel, remove a visual, "
                "regenerate only the affected question, or reject it."
            )
            st.markdown("#### Candidate — question-level review required")
        elif unresolved_question_rejection:
            st.error(
                "At least one question is currently rejected. Resolve it with an "
                "edit or targeted regeneration before the quiz can return to final review."
            )
            st.markdown("#### Candidate — rejected question requires resolution")
        elif awaiting_human_review:
            st.warning(
                "The candidate passed the current local gates and is awaiting final "
                "human approval. You can still review or edit individual questions below."
            )
            st.markdown("#### Candidate awaiting final review")
        elif human_state == "HUMAN_REJECTED":
            st.warning(
                "This generated candidate was rejected and is not releaseable."
            )
            st.markdown("#### Rejected candidate")
        else:
            st.warning(
                "A generated candidate exists and is blocked from final release "
                f"because its validation/HITL state is {human_state or 'unknown'}. "
                "Question-level HITL remains available below so reviewer-fixable "
                "issues can still be corrected without regenerating the whole quiz."
            )
            st.markdown(
                "#### Generated candidate — blocked from release, "
                "question-level HITL still available"
            )

        # Render the richer question-level HITL whenever Notebook 06 produced
        # a real review queue for this candidate. This is intentionally not
        # limited to only AWAITING_* states: reviewer-fixable structural or
        # semantic blocks must keep the HITL controls visible.
        candidate_review_queue = _agent2_question_review_queue(
            run_dir=run_dir,
            quiz_mode=quiz_mode,
            manifest=manifest,
        )
        candidate_has_hitl_queue = bool(candidate_review_queue)

        if question_level_reviewable or candidate_has_hitl_queue:
            _render_agent2_question_level_hitl(
                run_dir=run_dir,
                quiz_mode=quiz_mode,
                manifest=manifest,
                candidate_questions=candidate_questions,
                human_state=human_state,
            )
        else:
            _render_quiz_questions(candidate_questions)

        semantic = manifest.get("semantic_quality_validation") or {}
        reasons = semantic.get("reasons") if isinstance(semantic, dict) else []
        if isinstance(reasons, list) and reasons:
            with st.expander("Candidate validation details", expanded=False):
                for reason in reasons:
                    st.write(f"• {reason}")

    # Final whole-quiz approval is intentionally separate from question-level
    # corrections. Notebook 06 resets this gate whenever any question changes.
    if awaiting_human_review and candidate_questions:
        st.markdown("#### Final whole-quiz review")
        if unresolved_required_reviews:
            st.warning(
                f"Final approval is locked: {unresolved_required_reviews} "
                "REVIEW/HIGH question(s) still need an explicit decision."
            )
        else:
            st.caption(
                "All required question-level reviews are resolved. Approve only "
                "after you are satisfied with the complete revised quiz."
            )
        st.caption(
            "Use question-level targeted regeneration whenever possible; "
            "whole-quiz regeneration is more expensive."
        )
        with st.form(
            key=f"quiz_review_{Path(run_dir).name}_{quiz_mode}"
        ):
            review_reason = st.text_area(
                "Final review reason",
                help="Required for final Approve, whole-quiz Regenerate, or Reject.",
            )
            r1, r2, r3 = st.columns(3)
            approve = r1.form_submit_button(
                "Approve final quiz",
                use_container_width=True,
                disabled=unresolved_required_reviews > 0,
            )
            regenerate = r2.form_submit_button(
                "Regenerate whole quiz",
                use_container_width=True,
                disabled=(
                    (
                        int(
                            manifest.get(
                                "automatic_corrective_regeneration_attempts_used",
                                manifest.get("regeneration_attempts_used", 0),
                            )
                            or 0
                        )
                        + int(
                            manifest.get(
                                "whole_quiz_human_regeneration_attempts_used",
                                0,
                            )
                            or 0
                        )
                    )
                    >= int(manifest.get("max_regeneration_attempts") or 1)
                ),
            )
            reject = r3.form_submit_button(
                "Reject final quiz",
                use_container_width=True,
            )

        decision = (
            "approve"
            if approve
            else "regenerate"
            if regenerate
            else "reject"
            if reject
            else None
        )

        if decision:
            if not review_reason.strip():
                st.error("A written final review reason is required.")
            else:
                try:
                    result = _call_agent2_quiz_review_bridge(
                        run_dir=run_dir,
                        quiz_mode=quiz_mode,
                        decision=decision,
                        reason=review_reason.strip(),
                        question_actions=None,
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.session_state["langgraph_last_result"] = result
                    st.success(
                        f"Final quiz review '{decision}' applied."
                    )
                    st.rerun()

    if accepted_questions:
        st.markdown("#### Accepted quiz")
        _render_quiz_questions(accepted_questions)

    if not manifest.get("release_ready"):
        blockers = manifest.get("official_release_failures") or []
        if blockers:
            st.warning(
                "The quiz content exists, but at least one official question "
                "has a rendering/release blocker."
            )

    output_dir = _agent2_quiz_output_dir(
        run_dir,
        quiz_mode,
    )
    manifest_output_files = (
        manifest.get("output_files")
        or {}
    )
    if not isinstance(
        manifest_output_files,
        dict,
    ):
        manifest_output_files = {}

    registered_pdf = _resolve_agent2_file(
        manifest_output_files.get(
            "questions_and_marking_schemes_pdf"
        ),
        output_dir=output_dir,
    )

    if quiz_mode == "fill_shortfall":
        st.markdown("#### Final Hybrid Quiz PDF")
        st.caption(
            "One student-facing PDF is produced. It starts with the final hybrid "
            "summary, then the **Official AQA Questions** section, followed by the "
            "**AI-Generated Missing Coverage** section."
        )

        mcp_final_pdf = manifest.get("mcp_final_pdf") or {}
        if (
            isinstance(mcp_final_pdf, dict)
            and bool(mcp_final_pdf.get("uses_notebook08_assets"))
        ):
            st.caption(
                "Visual source: LangGraph → MCP → Notebook 08 "
                "(SchemDraw / Kroki / local structured renderer)."
            )
        if registered_pdf is not None:
            download_file(
                registered_pdf,
                "Download Final Official + AI Hybrid Quiz PDF",
            )
        else:
            st.info("The final hybrid PDF is not available yet.")

    registered_usage = _resolve_agent2_file(
        (
            manifest.get("source_artifacts")
            or {}
        ).get(
            "model_call_usage"
        )
        if isinstance(
            manifest.get("source_artifacts")
            or {},
            dict,
        )
        else None,
        output_dir=output_dir,
    )

    if quiz_mode == "fill_shortfall":
        # The single final hybrid PDF is displayed above. Keep only non-PDF
        # technical artifacts here so source/temporary PDFs are never exposed as
        # separate student-facing downloads.
        download_candidates = [
            registered_usage,
            output_dir / "model_call_usage.json",
            output_dir / "final_quiz_manifest.json",
            output_dir / "quiz_generation_report.txt",
            output_dir / "generation_request.json",
            output_dir / "semantic_quality_review.json",
            output_dir / "generated_quality_human_review.json",
            output_dir / "generated_human_review_queue.json",
            output_dir / "generated_human_review_feedback.jsonl",
        ]
    else:
        download_candidates = [
            registered_pdf,
            output_dir / "Agent2_Quiz_Output_Questions_and_Marking_Schemes.pdf",
            registered_usage,
            output_dir / "model_call_usage.json",
            output_dir / "final_quiz_manifest.json",
            output_dir / "quiz_generation_report.txt",
            output_dir / "generation_request.json",
            output_dir / "semantic_quality_review.json",
            output_dir / "generated_quality_human_review.json",
            output_dir / "generated_human_review_queue.json",
            output_dir / "generated_human_review_feedback.jsonl",
        ]
    files = []
    for path in download_candidates:
        if (
            path is not None
            and Path(path).is_file()
            and Path(path) not in files
        ):
            files.append(
                Path(path)
            )
    if files:
        with st.expander(
            (
                "Supporting quiz-generation artifacts"
                if quiz_mode == "fill_shortfall"
                else "Quiz generation files"
            ),
            expanded=False,
        ):
            for path in files:
                download_file(
                    path,
                    f"Download {path.name}",
                )


def _official_package_is_newer_than_complete_quiz(run_dir: Path, package_path: Path) -> bool:
    """Hide a stale old official shortfall after a newer Complete Quiz action.

    If the user retrieves official questions again, the new Notebook 05 package
    becomes newer and the shortfall CTA is shown again.
    """
    complete_manifest = (
        _agent2_quiz_output_dir(run_dir, "complete_quiz")
        / "final_quiz_manifest.json"
    )
    if not complete_manifest.is_file():
        return True
    try:
        return package_path.stat().st_mtime >= complete_manifest.stat().st_mtime
    except OSError:
        return True


def _render_missing_quiz_cta(
    *,
    run_dir: Path,
    agent2_root_value: str,
    quiz_model_config: dict[str, Any],
    selected_model_key: str,
    model_selected: bool,
) -> None:
    package_path, package = _agent2_current_package(run_dir)
    if package_path is None or not package:
        return
    if not _official_package_is_newer_than_complete_quiz(run_dir, package_path):
        return
    shortfall = _agent2_official_shortfall(package)
    if shortfall["sufficient"]:
        st.success("Official retrieval satisfies the current assessment request. AI shortfall generation is not needed.")
        return

    st.warning(
        "Official retrieval is short for the current filters. "
        f"Missing: {shortfall['missing_marks']} mark(s), "
        f"{shortfall['missing_questions']} question(s)."
    )

    if not model_selected:
        st.info(
            "Select a Notebook 06 quiz generation model above before generating "
            "the missing coverage."
        )
        return

    selected_model_display = _quiz_model_display_name(
        quiz_model_config,
        selected_model_key,
    )

    if st.button(
        f"Generate Missing Quiz Coverage with {selected_model_display}",
        type="primary",
        use_container_width=True,
        key=f"generate_missing_quiz_{Path(run_dir).name}",
    ):
        try:
            _save_quiz_model_selection(
                run_dir=run_dir,
                model_key=selected_model_key,
                model_config=quiz_model_config,
            )
        except Exception as exc:
            st.error(
                f"Could not save quiz model selection: {exc}"
            )
            return

        graph_request = build_missing_quiz_request_text()
        status = st.status("LangGraph is generating only the missing quiz coverage", expanded=True)
        try:
            result = run_langgraph_request(
                frontend_root=PROJECT_ROOT,
                run_id=Path(run_dir).name,
                user_request=graph_request,
                agent2_action="missing_quiz",
                # Notebook 06 -> visual_plan -> up to 3 MCP visual tools.
                max_steps=10,
                mode="start",
                on_update=_step4_on_graph_update,
                agent2_project_root=agent2_root_value,
            )
        except Exception as exc:
            status.update(label="Missing quiz generation failed", state="error", expanded=True)
            st.error(str(exc))
        else:
            st.session_state["langgraph_last_result"] = result
            status.update(label="Missing quiz coverage workflow completed", state="complete", expanded=False)
            st.rerun()

def render_agent2_assessment_stage(*, run_dir: Path, transcript_name: str) -> None:
    """Final Agent 2 UI: official retrieval OR direct complete quiz generation."""
    _, approved_topics_path = _agent2_handoff_paths(run_dir)
    st.subheader("Agent 2 — Assessment / Quiz Filters")
    st.caption(
        "The same filters drive both actions. Retrieve Official Assessment uses Notebook 05. "
        "Generate Complete Quiz skips Notebook 05 and runs the single config-driven Notebook 06 with the model selected below."
    )
    if not approved_topics_path.is_file():
        st.info("Approve the Agent 1 topics in the Topic Approval tab first.")
        return

    approved_payload = load_json(approved_topics_path)
    approved_topics = approved_payload.get("topics", [])
    if not isinstance(approved_topics, list) or not approved_topics:
        st.warning("The approved topic handoff contains no topics.")
        return

    primary_topic_count = sum(
        str(topic.get("role", "")).casefold() == "primary"
        for topic in approved_topics if isinstance(topic, dict)
    )
    supporting_topic_count = sum(
        str(topic.get("role", "")).casefold() == "supporting"
        for topic in approved_topics if isinstance(topic, dict)
    )
    evidence_ready = bool(approved_payload.get("actual_chunk_evidence_available", False))

    top_columns = st.columns(4)
    top_columns[0].metric("Approved topics", len(approved_topics))
    top_columns[1].metric("Primary topics", primary_topic_count)
    top_columns[2].metric("Supporting topics", supporting_topic_count)
    top_columns[3].metric("Actual chunk evidence", "Available" if evidence_ready else "Missing")

    with st.expander("Agent 2 connection", expanded=False):
        agent2_root_value = st.text_input(
            "Agent 2 project folder",
            value=_default_agent2_project_root(),
            key=f"agent2_project_root_input_{Path(run_dir).name}",
        )
        notebook_value = st.text_input(
            "Notebook 05 path (optional)",
            value="",
            key=f"agent2_notebook_path_input_{Path(run_dir).name}",
        )
        try:
            resolved_root = resolve_agent2_project_root(PROJECT_ROOT, explicit_path=agent2_root_value)
            st.success(f"Agent 2 root resolved: {resolved_root}")
            resolved_notebook = resolve_agent2_notebook(
                agent2_project_root=resolved_root,
                frontend_project_root=PROJECT_ROOT,
                explicit_path=notebook_value or None,
            )
            st.success(f"Notebook 05 resolved: {resolved_notebook}")
            quiz_candidates = [
                resolved_root / "Notebooks" / "06_quiz_generation.ipynb",
                resolved_root / "notebooks" / "06_quiz_generation.ipynb",
            ]
            quiz_notebook = next((path for path in quiz_candidates if path.is_file()), None)
            if quiz_notebook:
                st.success(f"Notebook 06 resolved: {quiz_notebook}")
            else:
                st.warning("06_quiz_generation.ipynb was not found under Agent2/Notebooks yet.")
        except Exception as exc:
            st.warning(str(exc))

    try:
        quiz_model_config = _load_quiz_model_config_for_ui(
            agent2_root_value
        )
    except Exception as exc:
        st.error(
            f"Quiz model configuration is unavailable: {exc}"
        )
        return

    quiz_models = quiz_model_config.get(
        "models",
        {},
    )
    model_keys = list(
        quiz_models
    )

    # Model choice is intentionally NOT defaulted.
    # Notebook 06 must never run until the user explicitly chooses a model.
    model_placeholder = "__SELECT_QUIZ_MODEL__"
    model_select_options = [
        model_placeholder,
        *model_keys,
    ]

    st.markdown("#### Choose the model for Notebook 06")
    selected_model_key = st.selectbox(
        "Quiz generation model",
        options=model_select_options,
        index=0,
        format_func=lambda key: (
            "— Select a model —"
            if key == model_placeholder
            else _quiz_model_display_name(
                quiz_model_config,
                key,
            )
        ),
        help=(
            "Required for Notebook 06 quiz generation (complete quiz or missing coverage). "
            "The selected model is saved for this run and the same Notebook 06 "
            "routes to that provider."
        ),
        key=f"agent2_quiz_model_selector_{Path(run_dir).name}",
    )

    model_selected = (
        selected_model_key != model_placeholder
        and selected_model_key in quiz_models
    )

    if model_selected:
        selected_model = quiz_models[
            selected_model_key
        ]
        selected_model_display = _quiz_model_display_name(
            quiz_model_config,
            selected_model_key,
        )

        st.success(
            f"Notebook 06 model selected: {selected_model_display}"
        )
        provider_tpm_limit = selected_model.get(
            "provider_tpm_limit_tokens"
        )
        provider_tpm_text = (
            f"{provider_tpm_limit} tokens/min"
            if provider_tpm_limit not in {None, "", 0, "0"}
            else "not configured"
        )

        st.caption(
            "Selected model limits — context: "
            f"{selected_model.get('context_window_tokens', 'N/A')} tokens | "
            "hard max output: "
            f"{selected_model.get('hard_max_output_tokens', 'N/A')} tokens | "
            "provider/service-tier TPM: "
            f"{provider_tpm_text}."
        )

        st.info(
            "Notebook 06 checks the selected model context window before every "
            "generation request. If the estimated request would exceed that "
            "context window, generation stops locally and no API hit is spent."
        )

        if provider_tpm_limit not in {None, "", 0, "0"}:
            st.caption(
                "A separate provider/service-tier TPM ceiling is also checked. "
                "TPM-only oversize batches may be split/re-budgeted locally before API."
            )
    else:
        selected_model = {}
        selected_model_display = ""
        st.info(
            "Select Gemini 3.5 Flash, GPT-OSS 120B / Groq, GPT-5 mini, or GPT-5.4 mini "
            "before running Notebook 06 quiz generation."
        )

    default_question_count = max(5, len(approved_topics))
    with st.form(key=f"agent2_assessment_filter_form_{Path(run_dir).name}"):
        first_row = st.columns(3)
        number_of_questions = first_row[0].number_input("Number of questions", 1, 30, default_question_count, 1)
        paper_label = first_row[1].selectbox("Paper filter", ["Both papers", "Paper 1", "Paper 2"])
        target_total_marks = first_row[2].number_input("Target total marks", 1, 500, 20, 1)

        second_row = st.columns(4)
        minimum_marks = second_row[0].number_input("Minimum marks per question", 1, 20, 1, 1)
        maximum_marks = second_row[1].number_input("Maximum marks per question", 1, 30, 12, 1)
        minimum_primary = second_row[2].number_input(
            "Minimum primary questions", 0, 30, min(3, default_question_count), 1
        )
        minimum_supporting = second_row[3].number_input(
            "Minimum supporting questions", 0, 30, 1 if supporting_topic_count else 0, 1
        )

        third_row = st.columns(3)
        programming_language_label = third_row[0].selectbox("Programming language", ["Automatic", "Python"])
        include_code = third_row[1].checkbox("Include code questions", value=True)
        include_visual = third_row[2].checkbox("Include visual questions", value=True)
        cover_all_topics = st.checkbox(
            "Cover all approved topics",
            value=True,
        )

        special_instructions = st.text_area(
            "Special quiz instructions (optional)",
            placeholder=(
                "e.g. Use more scenario-based questions, make the wording concise, "
                "or give extra emphasis to one of the approved topics."
            ),
            help=(
                "Notebook 06 treats these as mandatory user requirements according to "
                "their natural-language meaning whenever they are feasible and remain inside "
                "the approved AQA scope. The same generation call interprets and applies them; "
                "Python still enforces objective constraints such as totals, paper routing, "
                "approved topics, and impossible conflicts."
            ),
            height=110,
            key=(
                "agent2_special_quiz_instructions_"
                f"{Path(run_dir).name}"
            ),
        ).strip()

        st.caption(
            "Retrieve Official Assessment → Notebook 05.   |   "
            "Generate Complete Quiz → config-selected model through Notebook 06 directly "
            "(no Notebook 05)."
        )
        b1, b2 = st.columns(2)
        retrieve_official_clicked = b1.form_submit_button(
            "Retrieve Official Assessment",
            use_container_width=True,
        )
        generate_complete_clicked = b2.form_submit_button(
            "Generate Complete Quiz",
            type="primary",
            use_container_width=True,
        )

    if retrieve_official_clicked or generate_complete_clicked:
        if generate_complete_clicked:
            special_instruction_path = (
                Path(run_dir)
                / "output"
                / "integration"
                / "quiz_special_instructions.json"
            )
            special_instruction_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            special_instruction_path.write_text(
                json.dumps(
                    {
                        "schema_version":
                            "agent2-quiz-special-instructions-v1.0.0",
                        "updated_at_utc":
                            datetime.now(timezone.utc).isoformat(),
                        "special_instructions":
                            special_instructions,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        errors = []
        if maximum_marks < minimum_marks:
            errors.append("Maximum marks must be at least minimum marks.")
        if minimum_primary > number_of_questions:
            errors.append("Minimum primary questions exceed total questions.")
        if minimum_supporting > number_of_questions:
            errors.append("Minimum supporting questions exceed total questions.")
        if minimum_primary + minimum_supporting > number_of_questions:
            errors.append("Primary and supporting minimums together exceed total questions.")
        if cover_all_topics and number_of_questions < len(approved_topics):
            errors.append(f"Request at least {len(approved_topics)} questions to cover all approved topics.")
        if minimum_supporting > 0 and supporting_topic_count == 0:
            errors.append("No approved supporting topic is available.")
        if generate_complete_clicked:
            if not model_selected:
                errors.append(
                    "Choose a quiz generation model before running Notebook 06."
                )
            if number_of_questions * minimum_marks > target_total_marks:
                errors.append("Complete quiz cannot meet the target marks: minimum marks × questions exceeds target total marks.")
            if number_of_questions * maximum_marks < target_total_marks:
                errors.append("Complete quiz cannot meet the target marks: maximum marks × questions is below target total marks.")

        if errors:
            for error in errors:
                st.error(error)
        else:
            # Only Notebook 06 needs a generation-model selection.
            # Official Notebook 05 retrieval remains provider-independent.
            if generate_complete_clicked:
                try:
                    model_selection_path = _save_quiz_model_selection(
                        run_dir=run_dir,
                        model_key=selected_model_key,
                        model_config=quiz_model_config,
                    )
                except Exception as exc:
                    st.error(
                        f"Could not save quiz model selection: {exc}"
                    )
                    return

                st.session_state[
                    f"agent2_quiz_model_key_{Path(run_dir).name}"
                ] = selected_model_key

            paper = {"Both papers": "Any", "Paper 1": "Paper 1", "Paper 2": "Paper 2"}[paper_label]
            common = dict(
                paper=paper,
                number_of_questions=int(number_of_questions),
                target_total_marks=int(target_total_marks),
                minimum_question_marks=int(minimum_marks),
                maximum_question_marks=int(maximum_marks),
                minimum_primary_questions=int(minimum_primary),
                minimum_supporting_questions=int(minimum_supporting),
                cover_all_approved_topics=bool(cover_all_topics),
                include_code_questions=bool(include_code),
                include_visual_questions=bool(include_visual),
                programming_language=programming_language_label,
            )
            if retrieve_official_clicked:
                graph_request = build_assessment_request_text(**common)
                agent2_action = "retrieve_official"
                status_label = "LangGraph is retrieving official AQA questions through MCP"
            else:
                graph_request = build_complete_quiz_request_text(
                    **common
                )
                if special_instructions:
                    graph_request += (
                        "\nSpecial quiz instructions "
                        "(mandatory user requirements unless impossible or conflicting "
                        "with approved AQA scope / fixed deterministic constraints): "
                        + special_instructions
                    )

                agent2_action = "complete_quiz"
                status_label = (
                    "LangGraph is generating the complete quiz "
                    f"with {selected_model_display} through MCP"
                )

            status = st.status(status_label, expanded=True)
            status.write(graph_request)
            try:
                result = run_langgraph_request(
                    frontend_root=PROJECT_ROOT,
                    run_id=Path(run_dir).name,
                    user_request=graph_request,
                    agent2_action=agent2_action,
                    # Complete quiz may continue through MCP visual rendering.
                    # Official retrieval keeps the original short budget.
                    max_steps=(10 if agent2_action == "complete_quiz" else 3),
                    mode="start",
                    on_update=_step4_on_graph_update,
                    agent2_project_root=agent2_root_value,
                    agent2_notebook_path=notebook_value or None,
                )
            except Exception as exc:
                status.update(label="Agent 2 action failed", state="error", expanded=True)
                st.error(str(exc))
            else:
                st.session_state["langgraph_last_result"] = result
                _step4_refresh_tracker(Path(run_dir).name)
                if generate_complete_clicked:
                    status.update(label="Complete quiz workflow finished", state="complete", expanded=False)
                else:
                    final_state = str(result.get("final_state") or "")
                    if final_state == "ASSESSMENT_READY":
                        status.update(label="Official assessment ready", state="complete", expanded=False)
                    elif final_state == "NO_SAFE_ASSESSMENT":
                        status.update(label="No safe official assessment found", state="complete", expanded=False)
                    else:
                        status.update(label=f"Official retrieval stopped at {final_state}", state="complete", expanded=True)
                st.rerun()

    # Existing official result UI stays unchanged.
    render_agent2_assessment_results(run_dir=run_dir)

    # Only official retrieval can expose the shortfall-generation CTA.
    _render_missing_quiz_cta(
        run_dir=run_dir,
        agent2_root_value=agent2_root_value,
        quiz_model_config=quiz_model_config,
        selected_model_key=selected_model_key,
        model_selected=model_selected,
    )

    # Both quiz modes have independent artifacts and can be inspected safely.
    _render_agent2_quiz_result(run_dir=run_dir, quiz_mode="complete_quiz")
    _render_agent2_quiz_result(run_dir=run_dir, quiz_mode="fill_shortfall")




def _final_topic_hitl_audit_data(
    *,
    edit_memory_runtime: dict[str, Any],
    merged_topics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a reviewer-friendly view of the deterministic final-topic HITL audit."""

    applied = [
        item
        for item in (edit_memory_runtime.get("applied") or [])
        if isinstance(item, dict)
    ]
    skipped = [
        item
        for item in (edit_memory_runtime.get("skipped") or [])
        if isinstance(item, dict)
    ]
    diagnostics = [
        str(value)
        for value in (edit_memory_runtime.get("retrieval_diagnostics") or [])
        if str(value).strip()
    ]

    topic_label_by_concept = {
        str(topic.get("concept_id") or "").strip(): str(topic.get("topic") or "").strip()
        for topic in merged_topics
        if isinstance(topic, dict) and str(topic.get("concept_id") or "").strip()
    }

    diagnostic_by_memory: dict[int, str] = {}
    for line in diagnostics:
        match = re.match(r"^memory\s+(\d+):\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            diagnostic_by_memory[int(match.group(1))] = match.group(2).strip()

    rows: list[dict[str, Any]] = []

    for item in applied:
        memory_id = item.get("memory_id")
        try:
            memory_id_int = int(memory_id) if memory_id is not None else None
        except (TypeError, ValueError):
            memory_id_int = None

        source_concept_id = str(item.get("source_concept_id") or "").strip()
        target_concept_id = str(item.get("target_concept_id") or "").strip()
        concept_label = (
            topic_label_by_concept.get(source_concept_id)
            or topic_label_by_concept.get(target_concept_id)
            or source_concept_id
            or target_concept_id
            or "Add-topic memory"
        )
        rows.append(
            {
                "Memory ID": memory_id_int,
                "Action": item.get("action"),
                "Topic / concept": concept_label,
                "Outcome": "Automatically applied",
                "Reviewer action": "None required",
                "Reason": item.get("explanation") or "Deterministic strong context match.",
                "Context diagnostic": diagnostic_by_memory.get(memory_id_int, "")
                if memory_id_int is not None
                else "",
                "_source_concept_id": source_concept_id,
                "_target_concept_id": target_concept_id,
            }
        )

    for item in skipped:
        memory_id = item.get("memory_id")
        try:
            memory_id_int = int(memory_id) if memory_id is not None else None
        except (TypeError, ValueError):
            memory_id_int = None

        reason = str(item.get("reason") or "").strip()
        reason_lower = reason.casefold()
        source_concept_id = str(item.get("source_concept_id") or "").strip()
        target_concept_id = str(item.get("target_concept_id") or "").strip()
        concept_label = (
            topic_label_by_concept.get(source_concept_id)
            or topic_label_by_concept.get(target_concept_id)
            or source_concept_id
            or target_concept_id
            or "Add-topic memory"
        )

        if "already matches the fresh module 3 role" in reason_lower:
            outcome = "Historical outcome already satisfied"
            reviewer_action = "None required"
        elif "uncertain" in reason_lower:
            outcome = "Ambiguous — human review required"
            reviewer_action = "Review the fresh result; edit only if correction is needed"
        elif "multiple edit-memory candidates" in reason_lower or "conflict" in reason_lower:
            outcome = "Conflicting memory — human review required"
            reviewer_action = "Resolve the conflicting historical edits"
        elif "incompatible" in reason_lower:
            outcome = "Old memory rejected"
            reviewer_action = "None required"
        else:
            outcome = "Not auto-applied"
            reviewer_action = "Review only if needed"

        rows.append(
            {
                "Memory ID": memory_id_int,
                "Action": item.get("action"),
                "Topic / concept": concept_label,
                "Outcome": outcome,
                "Reviewer action": reviewer_action,
                "Reason": reason,
                "Context diagnostic": diagnostic_by_memory.get(memory_id_int, "")
                if memory_id_int is not None
                else "",
                "_source_concept_id": source_concept_id,
                "_target_concept_id": target_concept_id,
            }
        )

    human_review_rows = [
        row
        for row in rows
        if "human review required" in str(row.get("Outcome") or "").casefold()
    ]
    rejected_rows = [
        row
        for row in rows
        if str(row.get("Outcome") or "") == "Old memory rejected"
    ]
    satisfied_rows = [
        row
        for row in rows
        if str(row.get("Outcome") or "") == "Historical outcome already satisfied"
    ]
    applied_rows = [
        row
        for row in rows
        if str(row.get("Outcome") or "") == "Automatically applied"
    ]

    return {
        "rows": rows,
        "human_review_rows": human_review_rows,
        "rejected_rows": rejected_rows,
        "satisfied_rows": satisfied_rows,
        "applied_rows": applied_rows,
        "diagnostics": diagnostics,
    }


def render_final_topic_hitl_audit(
    *,
    edit_memory_runtime: dict[str, Any],
    merged_topics: list[dict[str, Any]],
    raw_module3_result: dict[str, Any],
    run_dir: Path,
    transcript_name: str,
) -> dict[str, Any]:
    """Render and resolve the locked deterministic final-topic HITL gate."""

    audit = _final_topic_hitl_audit_data(
        edit_memory_runtime=edit_memory_runtime,
        merged_topics=merged_topics,
    )

    st.markdown("#### HITL memory review")
    flash_message = st.session_state.pop("final_topic_hitl_flash", None)
    if flash_message:
        st.success(flash_message)

    st.caption(
        "Historical reviewer-approved edits are reused automatically only when the "
        "context is a strong deterministic match. Strong mismatches are rejected. "
        "Ambiguous or conflicting memories stop here for an explicit human decision. "
        "Any historical edit that was auto-applied can also be reviewed and overridden "
        "by the human reviewer. Groq is not used in this automatic final-topic reuse gate."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Auto-applied", len(audit["applied_rows"]))
    m2.metric("Already consistent", len(audit["satisfied_rows"]))
    m3.metric("Auto-rejected", len(audit["rejected_rows"]))
    m4.metric("Needs your decision", len(audit["human_review_rows"]))
    m5.metric("HITL Groq calls", 0)

    store = None
    store_error: str | None = None
    try:
        store = _detected_topic_reuse_feedback_store()
    except Exception as exc:
        store_error = f"{type(exc).__name__}: {exc}"

    if store_error:
        st.error(
            "The reuse-feedback database helper could not be loaded, so ambiguous "
            f"memory decisions are read-only in this session. {store_error}"
        )

    if audit["human_review_rows"]:
        st.warning(
            "These historical memories are not safe to reuse automatically. "
            "For each item, choose whether the historical edit should apply to this "
            "lesson or whether the fresh Module 3 result should be kept. A reason is "
            "required and the decision is stored in PostgreSQL for this exact context."
        )

        summary_columns = [
            "Memory ID",
            "Action",
            "Topic / concept",
            "Outcome",
            "Context diagnostic",
        ]
        st.dataframe(
            pd.DataFrame(audit["human_review_rows"])[summary_columns],
            use_container_width=True,
            hide_index=True,
        )

        for position, row in enumerate(audit["human_review_rows"], start=1):
            memory_id = row.get("Memory ID")
            source_concept_id = str(row.get("_source_concept_id") or "").strip()
            topic_label = str(row.get("Topic / concept") or "Historical memory")
            is_conflict = "conflicting" in str(row.get("Outcome") or "").casefold()

            memories: list[dict[str, Any]] = []
            if store is not None:
                try:
                    if memory_id is not None:
                        snapshot = store.memory_snapshot(int(memory_id))
                        if snapshot:
                            memories = [snapshot]
                    elif is_conflict and source_concept_id:
                        memories = store.reusable_memories_for_source(
                            source_concept_id=source_concept_id,
                            spec_version=_current_aqa_spec_version(),
                        )
                except Exception as exc:
                    st.error(
                        f"Could not load historical memory details for {topic_label}: {exc}"
                    )

            with st.expander(
                f"{position}. {topic_label} — decision required",
                expanded=True,
            ):
                left, right = st.columns([1.1, 1.4])
                with left:
                    st.markdown("**Current lesson result**")
                    current_topic = next(
                        (
                            topic
                            for topic in merged_topics
                            if isinstance(topic, dict)
                            and str(topic.get("concept_id") or "").strip()
                            == source_concept_id
                        ),
                        None,
                    )
                    if current_topic:
                        st.write(
                            f"{current_topic.get('topic')} — "
                            f"**{current_topic.get('topic_role')}**"
                        )
                        st.caption(
                            f"AQA {current_topic.get('official_reference')} · "
                            f"chunks {current_topic.get('source_chunk_ids') or []}"
                        )
                    else:
                        st.write("The historical edit is an add-topic memory; the topic is currently absent.")

                    st.markdown("**Why automatic reuse stopped**")
                    st.write(row.get("Outcome"))
                    if row.get("Context diagnostic"):
                        st.caption(str(row.get("Context diagnostic")))

                with right:
                    st.markdown("**Historical human memory**")
                    if not memories:
                        st.info(
                            "Historical memory details could not be loaded. The fresh Module 3 "
                            "result remains unchanged."
                        )
                    else:
                        for memory in memories:
                            st.write(
                                f"**Memory {memory.get('memory_id')}** — "
                                f"{_final_topic_memory_outcome_label(memory)}"
                            )
                            reviewer_reason = str(memory.get("reviewer_reason") or "").strip()
                            if reviewer_reason:
                                st.caption("Original reviewer reason: " + reviewer_reason)

                        with st.expander("Show stored historical evidence", expanded=False):
                            for memory in memories:
                                st.markdown(f"**Memory {memory.get('memory_id')}**")
                                st.write(memory.get("stored_evidence") or "No stored evidence.")

                if not memories or store is None:
                    continue

                action_key = (
                    f"final_topic_hitl_{Path(run_dir).name}_{position}_"
                    + "_".join(str(item.get("memory_id")) for item in memories)
                )

                selected_memory: dict[str, Any] | None = None
                if is_conflict and len(memories) > 1:
                    memory_ids = [int(item["memory_id"]) for item in memories]
                    selected_memory_id = st.radio(
                        "Which historical outcome is correct for this lesson?",
                        options=memory_ids,
                        format_func=lambda value: next(
                            _final_topic_memory_outcome_label(item)
                            for item in memories
                            if int(item["memory_id"]) == int(value)
                        ),
                        key=f"{action_key}_choice",
                    )
                    selected_memory = next(
                        item
                        for item in memories
                        if int(item["memory_id"]) == int(selected_memory_id)
                    )
                else:
                    selected_memory = memories[0]

                review_reason = st.text_area(
                    "Reason for your decision (required)",
                    key=f"{action_key}_reason",
                    height=90,
                    placeholder=(
                        "Explain why the historical edit applies here, or why the fresh "
                        "Module 3 result should be kept."
                    ),
                ).strip()

                use_col, keep_col = st.columns(2)
                use_label = (
                    "Use selected historical outcome"
                    if is_conflict
                    else "Apply historical edit"
                )
                use_clicked = use_col.button(
                    use_label,
                    key=f"{action_key}_approve",
                    type="primary",
                    use_container_width=True,
                )
                keep_clicked = keep_col.button(
                    "Keep fresh Module 3 result",
                    key=f"{action_key}_reject",
                    use_container_width=True,
                )

                if use_clicked or keep_clicked:
                    if not review_reason:
                        st.error("A written reason is required before saving this HITL decision.")
                        continue

                    if selected_memory is None:
                        st.error("No historical memory outcome is available to review.")
                        continue

                    action = str(selected_memory.get("edit_action") or "")
                    evidence = (
                        _final_topic_addition_current_evidence(raw_module3_result)
                        if action == "add_topic"
                        else _final_topic_existing_current_evidence(
                            raw_module3_result=raw_module3_result,
                            source_concept_id=selected_memory.get("source_concept_id"),
                        )
                    )
                    if not evidence:
                        st.error(
                            "Could not reconstruct the exact current evidence used by the "
                            "HITL comparator. No database decision was written."
                        )
                        continue

                    try:
                        if use_clicked:
                            # The chosen memory is explicitly approved for this exact context.
                            store.record(
                                memory_id=int(selected_memory["memory_id"]),
                                current_evidence=evidence,
                                decision="approve_reuse",
                                reviewer_reason=review_reason,
                                spec_version=_current_aqa_spec_version(),
                                pipeline_run_id=Path(run_dir).name,
                                source_transcript=transcript_name,
                                source_concept_id=selected_memory.get("source_concept_id"),
                                reviewed_by="streamlit",
                            )

                            # For a conflict, explicitly reject the competing historical
                            # outcomes for the same exact context so the next render can
                            # deterministically use only the human-selected outcome.
                            for other in memories:
                                if int(other["memory_id"]) == int(selected_memory["memory_id"]):
                                    continue
                                other_action = str(other.get("edit_action") or "")
                                other_evidence = (
                                    _final_topic_addition_current_evidence(raw_module3_result)
                                    if other_action == "add_topic"
                                    else _final_topic_existing_current_evidence(
                                        raw_module3_result=raw_module3_result,
                                        source_concept_id=other.get("source_concept_id"),
                                    )
                                )
                                if other_evidence:
                                    store.record(
                                        memory_id=int(other["memory_id"]),
                                        current_evidence=other_evidence,
                                        decision="reject_reuse",
                                        reviewer_reason=(
                                            "Competing historical outcome rejected while resolving "
                                            "the same context. Reviewer reason: " + review_reason
                                        ),
                                        spec_version=_current_aqa_spec_version(),
                                        pipeline_run_id=Path(run_dir).name,
                                        source_transcript=transcript_name,
                                        source_concept_id=other.get("source_concept_id"),
                                        reviewed_by="streamlit",
                                    )

                            st.session_state["final_topic_hitl_flash"] = (
                                f"Historical memory {selected_memory['memory_id']} approved for "
                                "this exact lesson context. The final-topic overlay will re-run "
                                "deterministically."
                            )
                        else:
                            # Keep-fresh rejects every candidate represented by this row for
                            # the exact current evidence. Historical memories remain intact for
                            # other lesson contexts.
                            for memory in memories:
                                memory_action = str(memory.get("edit_action") or "")
                                memory_evidence = (
                                    _final_topic_addition_current_evidence(raw_module3_result)
                                    if memory_action == "add_topic"
                                    else _final_topic_existing_current_evidence(
                                        raw_module3_result=raw_module3_result,
                                        source_concept_id=memory.get("source_concept_id"),
                                    )
                                )
                                if not memory_evidence:
                                    continue
                                store.record(
                                    memory_id=int(memory["memory_id"]),
                                    current_evidence=memory_evidence,
                                    decision="reject_reuse",
                                    reviewer_reason=review_reason,
                                    spec_version=_current_aqa_spec_version(),
                                    pipeline_run_id=Path(run_dir).name,
                                    source_transcript=transcript_name,
                                    source_concept_id=memory.get("source_concept_id"),
                                    reviewed_by="streamlit",
                                )

                            st.session_state["final_topic_hitl_flash"] = (
                                "Fresh Module 3 result kept. The historical edit memory was "
                                "rejected only for this exact lesson context and the decision "
                                "was saved to PostgreSQL."
                            )
                    except Exception as exc:
                        st.error(f"Could not save the HITL reuse decision: {exc}")
                    else:
                        st.rerun()
    else:
        st.success(
            "No ambiguous/conflicting final-topic memory decision needs human review."
        )

    # Human remains authoritative even after a deterministic strong-match reuse.
    # Auto-applied edits are therefore reviewable/overridable without changing
    # the comparator, retrieval thresholds, or the historical memory itself.
    if audit["applied_rows"]:
        st.markdown("##### Auto-applied historical edits — optional review / override")
        st.caption(
            "These edits were applied because the deterministic context gate considered "
            "the historical memory safe to reuse. You do not need to confirm them. If an "
            "applied edit is wrong for this lesson, undo it here: the fresh Module 3 result "
            "is restored and the rejection is saved only for this exact lesson evidence."
        )

        raw_fresh_topics = [
            item
            for item in (raw_module3_result.get("merged_topics") or [])
            if isinstance(item, dict)
        ]

        for position, row in enumerate(audit["applied_rows"], start=1):
            memory_id = row.get("Memory ID")
            if memory_id is None:
                continue

            memory: dict[str, Any] | None = None
            if store is not None:
                try:
                    memory = store.memory_snapshot(int(memory_id))
                except Exception as exc:
                    st.error(
                        f"Could not load auto-applied historical memory {memory_id}: {exc}"
                    )

            topic_label = str(row.get("Topic / concept") or "Historical memory")
            card_title = f"Auto-applied {position}. {topic_label}"

            with st.expander(card_title, expanded=False):
                if not memory:
                    st.info(
                        "Historical memory details could not be loaded. The applied result "
                        "is left unchanged."
                    )
                    continue

                action = str(memory.get("edit_action") or "").strip()
                source_concept_id = str(memory.get("source_concept_id") or "").strip()
                target_concept_id = str(memory.get("target_concept_id") or "").strip()

                fresh_topic = next(
                    (
                        item
                        for item in raw_fresh_topics
                        if str(item.get("concept_id") or "").strip() == source_concept_id
                    ),
                    None,
                )
                effective_source_topic = next(
                    (
                        item
                        for item in merged_topics
                        if isinstance(item, dict)
                        and str(item.get("concept_id") or "").strip() == source_concept_id
                    ),
                    None,
                )
                effective_target_topic = next(
                    (
                        item
                        for item in merged_topics
                        if isinstance(item, dict)
                        and str(item.get("concept_id") or "").strip() == target_concept_id
                    ),
                    None,
                )

                left, right = st.columns([1.1, 1.4])
                with left:
                    st.markdown("**Fresh Module 3 result (before memory reuse)**")
                    if fresh_topic:
                        st.write(
                            f"{fresh_topic.get('topic')} — "
                            f"**{fresh_topic.get('topic_role')}**"
                        )
                        st.caption(
                            f"AQA {fresh_topic.get('official_reference')} · "
                            f"chunks {fresh_topic.get('source_chunk_ids') or []}"
                        )
                    elif action == "add_topic":
                        st.write("Topic absent from the fresh Module 3 result.")
                    else:
                        st.write("Fresh source topic could not be displayed.")

                    st.markdown("**Effective result after historical edit**")
                    if action == "remove_topic":
                        st.write("Topic removed from the effective final list.")
                    elif action == "replace_topic":
                        if effective_target_topic:
                            st.write(
                                f"{effective_target_topic.get('topic')} — "
                                f"**{effective_target_topic.get('topic_role')}**"
                            )
                        else:
                            st.write("Replacement applied; target topic is not displayable here.")
                    elif action == "add_topic":
                        if effective_target_topic:
                            st.write(
                                f"{effective_target_topic.get('topic')} — "
                                f"**{effective_target_topic.get('topic_role')}**"
                            )
                        else:
                            st.write(_final_topic_memory_outcome_label(memory))
                    elif effective_source_topic:
                        st.write(
                            f"{effective_source_topic.get('topic')} — "
                            f"**{effective_source_topic.get('topic_role')}**"
                        )
                    else:
                        st.write(_final_topic_memory_outcome_label(memory))

                    if row.get("Context diagnostic"):
                        st.markdown("**Why it was auto-applied**")
                        st.caption(str(row.get("Context diagnostic")))

                with right:
                    st.markdown("**Historical human memory used**")
                    st.write(
                        f"**Memory {memory.get('memory_id')}** — "
                        f"{_final_topic_memory_outcome_label(memory)}"
                    )
                    original_reason = str(memory.get("reviewer_reason") or "").strip()
                    if original_reason:
                        st.caption("Original reviewer reason: " + original_reason)
                    with st.expander("Show stored historical evidence", expanded=False):
                        st.write(memory.get("stored_evidence") or "No stored evidence.")

                st.info(
                    "No action is required if this auto-applied edit is correct. "
                    "Use the controls below only when you want to explicitly confirm "
                    "it or override it for this exact lesson context."
                )

                review_reason = st.text_area(
                    "Reason for confirm / override (required only if you click a button)",
                    key=(
                        f"auto_applied_hitl_reason_{Path(run_dir).name}_"
                        f"{int(memory_id)}"
                    ),
                    height=90,
                    placeholder=(
                        "Explain why the applied historical edit is correct, or why the "
                        "fresh Module 3 result should be restored."
                    ),
                ).strip()

                confirm_col, undo_col = st.columns(2)
                confirm_clicked = confirm_col.button(
                    "Confirm applied edit",
                    key=(
                        f"auto_applied_hitl_confirm_{Path(run_dir).name}_"
                        f"{int(memory_id)}"
                    ),
                    use_container_width=True,
                )
                undo_clicked = undo_col.button(
                    "Undo edit — keep fresh Module 3 result",
                    key=(
                        f"auto_applied_hitl_undo_{Path(run_dir).name}_"
                        f"{int(memory_id)}"
                    ),
                    type="primary",
                    use_container_width=True,
                )

                if confirm_clicked or undo_clicked:
                    if not review_reason:
                        st.error(
                            "A written reason is required before saving this HITL decision."
                        )
                        continue

                    current_evidence = (
                        _final_topic_addition_current_evidence(raw_module3_result)
                        if action == "add_topic"
                        else _final_topic_existing_current_evidence(
                            raw_module3_result=raw_module3_result,
                            source_concept_id=source_concept_id,
                        )
                    )
                    if not current_evidence:
                        st.error(
                            "Could not reconstruct the exact fresh Module 3 evidence used "
                            "for this historical-memory decision. Nothing was written to "
                            "PostgreSQL."
                        )
                        continue

                    try:
                        store.record(
                            memory_id=int(memory_id),
                            current_evidence=current_evidence,
                            decision=(
                                "approve_reuse" if confirm_clicked else "reject_reuse"
                            ),
                            reviewer_reason=review_reason,
                            spec_version=_current_aqa_spec_version(),
                            pipeline_run_id=Path(run_dir).name,
                            source_transcript=transcript_name,
                            source_concept_id=source_concept_id or None,
                            reviewed_by="streamlit",
                        )
                    except Exception as exc:
                        st.error(f"Could not save the auto-applied HITL review: {exc}")
                    else:
                        if confirm_clicked:
                            st.session_state["final_topic_hitl_flash"] = (
                                f"Historical memory {memory_id} explicitly confirmed for "
                                "this exact lesson context."
                            )
                        else:
                            st.session_state["final_topic_hitl_flash"] = (
                                f"Historical memory {memory_id} rejected for this exact "
                                "lesson context. The fresh Module 3 result has been restored."
                            )
                        st.rerun()

    other_rows = [
        row
        for row in audit["rows"]
        if row not in audit["human_review_rows"]
        and row not in audit["applied_rows"]
    ]
    if other_rows:
        with st.expander("Automatic final-topic memory decisions", expanded=False):
            visible_columns = [
                "Memory ID",
                "Action",
                "Topic / concept",
                "Outcome",
                "Reason",
                "Context diagnostic",
            ]
            st.dataframe(
                pd.DataFrame(other_rows)[visible_columns],
                use_container_width=True,
                hide_index=True,
            )

    if store is not None:
        try:
            feedback_rows = store.feedback_for_run(Path(run_dir).name)
        except Exception as exc:
            feedback_rows = []
            st.caption(f"Could not load saved HITL decision history: {exc}")

        if feedback_rows:
            with st.expander("Human HITL decisions saved for this run", expanded=False):
                display_rows = []
                for item in feedback_rows:
                    display_rows.append(
                        {
                            "Memory ID": item.get("memory_id"),
                            "Decision": (
                                "Apply historical edit"
                                if item.get("decision") == "approve_reuse"
                                else "Keep fresh Module 3 result"
                            ),
                            "Topic / concept": item.get("source_concept_id") or "Add-topic memory",
                            "Reason": item.get("reviewer_reason"),
                            "Reviewed by": item.get("reviewed_by"),
                            "Reviewed at": item.get("reviewed_at"),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(display_rows),
                    use_container_width=True,
                    hide_index=True,
                )

    return audit

def render_results(run_dir: Path) -> None:
    manifest = load_json(run_dir / "pipeline_manifest.json")
    transcript_name = manifest.get("transcript_name", "transcript")
    output_folder = run_dir / "output" / transcript_name

    module1_json = load_json(output_folder / "01_preprocessing.json")
    module2_json = load_json(output_folder / "02_chunking.json")
    module3_json = load_json(output_folder / "03_topic_mapping.json")

    cleaned_text = (
        (output_folder / "01_cleaned_transcript.txt").read_text(encoding="utf-8")
        if (output_folder / "01_cleaned_transcript.txt").is_file()
        else ""
    )
    chunks = module2_json.get("chunks", [])
    raw_module3_result = module3_json.get("module3_result", {})

    # V8: exact human corrections are keyed to a stable cleaned-transcript
    # lesson context.  First migrate safe pre-V8 manual add memories whose
    # human-selected stored evidence occurs verbatim in this exact lesson, then
    # bridge stable decisions onto the backend's current evidence representation.
    # This keeps same-transcript reruns deterministic without broadening reuse
    # to different lessons.
    legacy_add_bootstrap = _bootstrap_legacy_manual_add_memory_for_same_lesson(
        run_dir=run_dir,
        transcript_name=transcript_name,
    )
    stable_lesson_feedback_bridge = _bridge_stable_lesson_feedback_to_backend_evidence(
        run_dir=run_dir,
        transcript_name=transcript_name,
        raw_module3_result=raw_module3_result,
    )

    # Current-run human HITL decisions are authoritative. Normalize any saved
    # decisions (including rows created by earlier UI versions) onto the exact
    # fresh Module 3 evidence representation used by the backend gate BEFORE
    # the runtime overlay executes.
    hitl_feedback_canonicalization = _canonicalize_current_run_hitl_feedback(
        run_dir=run_dir,
        transcript_name=transcript_name,
        raw_module3_result=raw_module3_result,
    )

    edit_memory_runtime = apply_detected_topic_edit_runtime(
        module3_result_payload=raw_module3_result,
        module3_json=module3_json,
        run_dir=run_dir,
        transcript_name=transcript_name,
        frontend_project_root=PROJECT_ROOT,
    )

    edit_memory_runtime["legacy_add_memory_bootstrap"] = legacy_add_bootstrap
    edit_memory_runtime["stable_lesson_feedback_bridge"] = (
        stable_lesson_feedback_bridge
    )
    edit_memory_runtime["current_run_feedback_canonicalization"] = (
        hitl_feedback_canonicalization
    )

    module3_result = edit_memory_runtime["module3_result"]
    merged_topics = list(module3_result.get("merged_topics", []))

    # Exact-human-approved add_topic memories may represent concepts that the
    # fresh Module 3 run missed completely.  The backend deliberately defers
    # those additions rather than inventing Module 3 scores or falling back the
    # whole overlay.  Materialize them here using the same safe representation
    # already used by manual human additions.
    deferred_add_materialization = _materialize_exact_approved_add_topic_memories(
        run_dir=run_dir,
        transcript_name=transcript_name,
        raw_module3_result=raw_module3_result,
        edit_memory_runtime=edit_memory_runtime,
        merged_topics=merged_topics,
    )
    edit_memory_runtime["deferred_add_topic_materialization"] = (
        deferred_add_materialization
    )

    # Human ADD operations are stored outside the raw Module3Result so we do
    # not invent or overwrite Module 3 ranking/confidence metrics. They are
    # merged into the effective final list only after the deterministic
    # final-topic edit-memory overlay has derived the current transcript's safe
    # topic list. Ambiguous memory reuse is left for human review.
    saved_human_additions = module3_json.get("topic_output_additions", [])
    if isinstance(saved_human_additions, list):
        present_concept_ids = {
            str(topic.get("concept_id") or "").strip()
            for topic in merged_topics
            if isinstance(topic, dict)
        }
        for addition in saved_human_additions:
            if not isinstance(addition, dict):
                continue
            concept_id = str(addition.get("concept_id") or "").strip()
            if concept_id and concept_id not in present_concept_ids:
                merged_topics.append(dict(addition))
                present_concept_ids.add(concept_id)

    module3_result["merged_topics"] = merged_topics

    # In-memory only: saved 03_topic_mapping.json remains the untouched
    # fresh Module 3 output. UI / Topic Approval / Agent 2 receive the
    # validated effective topics.
    module3_json["module3_result"] = module3_result
    module3_json["detected_topic_edit_runtime"] = {
        key: value
        for key, value in edit_memory_runtime.items()
        if key != "module3_result"
    }
    llm_results = module3_json.get("llm_results", [])
    unmapped_inputs = module3_json.get("unmapped_inputs", [])
    topic_review_items = module3_json.get("topic_review_items", [])
    review_items = extract_review_items(module1_json)

    technical_review_payload = deep_get(
        module1_json,
        "technical_normalisation_result",
        default={},
    ) or {}
    review_data_present = (
        "review_items" in technical_review_payload
        or "review_items" in module1_json
    )
    review_message = str(
        technical_review_payload.get(
            "review_message",
            module1_json.get(
                "review_message",
                "No technical words to correct.",
            ),
        )
    )

    st.success(f"Pipeline completed for: {transcript_name}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Cleaned words", len(cleaned_text.split()))
    col2.metric("Semantic chunks", len(chunks))
    col3.metric("Official topics", len(merged_topics))
    col4.metric("LLM fallback items", len(unmapped_inputs))
    pending_review_count = sum(
        review_item_status(
            item,
            run_dir=run_dir,
            record_id=review_record_id(item),
        )
        == "pending"
        for item in review_items
    )
    col5.metric("Human review", pending_review_count)

    (
        overview_tab,
        module1_tab,
        module2_tab,
        module3_tab,
        topic_approval_tab,
        agent2_assessment_tab,
        files_tab,
        logs_tab,
    ) = st.tabs(
        [
            "Overview",
            "Module 1 — Cleaned Transcript",
            "Module 2 — Chunks",
            "Module 3 — Topics",
            "Topic Approval",
            "Agent 2 Assessment",
            "Generated Files",
            "Execution Logs",
        ]
    )

    with overview_tab:
        st.subheader("Pipeline status")
        rows = manifest.get("modules", [])
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.json(
            {
                "job_id": manifest.get("job_id"),
                "transcript": transcript_name,
                "status": manifest.get("status"),
                "output_folder": str(output_folder),
            }
        )

    with module1_tab:
        st.subheader("Final cleaned transcript")
        st.text_area(
            "Cleaned text",
            cleaned_text,
            height=420,
            label_visibility="collapsed",
        )

        stats = deep_get(module1_json, "preprocessing_result", "stats", default={}) or {}
        technical_stats = deep_get(
            module1_json,
            "technical_normalisation_result",
            "stats",
            default={},
        ) or {}

        left, right = st.columns(2)
        with left:
            st.markdown("#### Deterministic cleaning")
            st.json(stats)
        with right:
            st.markdown("#### Technical normalisation")
            st.json(technical_stats)

        unresolved = deep_get(
            module1_json,
            "technical_normalisation_result",
            "unresolved_issues",
            default=[],
        ) or []
        if unresolved:
            st.warning(f"{len(unresolved)} unresolved technical issue(s)")
            st.dataframe(pd.DataFrame(unresolved), use_container_width=True)

    with module2_tab:
        st.subheader("Semantic chunking summary")
        summary = {
            key: value
            for key, value in module2_json.items()
            if key != "chunks"
        }
        st.json(summary)

        if not chunks:
            st.warning("No chunks were found.")
        for chunk in chunks:
            title = (
                f"Chunk {chunk.get('chunk_id')} — "
                f"{chunk.get('word_count')} words — "
                f"{chunk.get('boundary_reason')}"
            )
            with st.expander(title, expanded=len(chunks) <= 3):
                st.write(chunk.get("text", ""))
                metadata = {
                    key: value
                    for key, value in chunk.items()
                    if key != "text"
                }
                st.json(metadata)

    with module3_tab:
        st.subheader("Module 3 — Official Topics & Human Review")
        st.info(
            "Recommended order: **1) resolve HITL memory decisions → 2) inspect the "
            "effective final topics → 3) make any manual correction if needed → "
            "4) approve the final topics for Agent 2 in the Topic Approval tab.**"
        )

        edit_runtime_status = edit_memory_runtime.get("status")
        has_missing_topic_label = any(
            isinstance(item, dict)
            and str(item.get("review_status") or "").casefold()
            == "needs_topic_label"
            for item in llm_results
        )

        final_topics_tab, hitl_memory_tab, mapping_review_tab, diagnostics_tab = st.tabs(
            [
                "1 — Final Topics",
                "2 — HITL Memory Review",
                "3 — Mapping / Corrections",
                "Advanced Audit",
            ]
        )

        with final_topics_tab:
            if edit_runtime_status == "applied":
                st.success(
                    "Reviewer-approved final-topic memory applied: "
                    f"{edit_memory_runtime.get('applied_count', 0)} edit(s)."
                )
            elif edit_runtime_status == "fallback":
                st.warning(
                    "Final-topic edit memory could not be applied safely. "
                    "The fresh Module 3 result is being used."
                )

            primary = [t for t in merged_topics if t.get("topic_role") == "primary"]
            supporting = [t for t in merged_topics if t.get("topic_role") == "supporting"]
            final_topic_diff = _final_topic_diff_summary(
                raw_module3_result=raw_module3_result,
                effective_topics=merged_topics,
            )
            final_topic_feedback = _final_topic_feedback_summary(
                run_dir=run_dir,
                edit_memory_runtime=edit_memory_runtime,
                merged_topics=merged_topics,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Effective topics", len(merged_topics))
            c2.metric("Primary", len(primary))
            c3.metric("Supporting", len(supporting))
            c4.metric(
                "Memory / human affected",
                final_topic_feedback.get("affected_count", 0),
            )

            st.caption(
                "Changes vs fresh Module 3: "
                f"{len(final_topic_diff.get('removed_keys', set()))} removed · "
                f"{len(final_topic_diff.get('added_keys', set()))} added · "
                f"{len(final_topic_diff.get('role_changed_keys', set()))} role updated. "
                f"Saved human HITL decisions for this run: "
                f"{final_topic_feedback.get('human_decision_count', 0)}."
            )

            if merged_topics:
                topic_rows = []
                for topic in merged_topics:
                    topic_rows.append(
                        {
                            "Topic": topic.get("topic"),
                            "Role": topic.get("topic_role"),
                            "Official reference": topic.get("official_reference"),
                            "Confidence": topic.get("confidence"),
                            "Ranking score": topic.get("ranking_score"),
                            "Source chunks": topic.get("source_chunk_ids"),
                            "HITL status": _final_topic_status_label(
                                topic,
                                diff=final_topic_diff,
                                feedback_summary=final_topic_feedback,
                            ),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(topic_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                pcol, scol = st.columns(2)
                with pcol:
                    st.markdown("#### Primary topics")
                    for topic in primary:
                        st.write(f"• {topic.get('topic')}")
                with scol:
                    st.markdown("#### Supporting topics")
                    for topic in supporting:
                        st.write(f"• {topic.get('topic')}")
            else:
                st.warning("No official topics were retained.")

            st.caption(
                "This is the effective topic list after safe deterministic memory reuse "
                "and any explicit human reuse decisions. Agent 2 receives only the "
                "separately approved handoff from the Topic Approval tab."
            )

        with hitl_memory_tab:
            render_live_effective_topic_list(
                merged_topics=merged_topics,
                raw_module3_result=raw_module3_result,
                edit_memory_runtime=edit_memory_runtime,
                run_dir=run_dir,
            )

            st.divider()
            render_final_topic_hitl_audit(
                edit_memory_runtime=edit_memory_runtime,
                merged_topics=merged_topics,
                raw_module3_result=raw_module3_result,
                run_dir=run_dir,
                transcript_name=transcript_name,
            )

            st.divider()
            st.markdown("#### Manual final-topic correction")
            st.caption(
                "Use this only when the final topic list is still wrong after resolving "
                "historical-memory decisions. Manual changes require a reason and continue "
                "through the existing reviewer-approved edit-memory path."
            )

            if has_missing_topic_label:
                st.info(
                    "Resolve the Missing Rough Topic Label item in the Mapping / Corrections "
                    "tab before manually editing the final detected-topic list."
                )
            elif merged_topics:
                render_detected_topic_editor(
                    merged_topics=merged_topics,
                    run_dir=run_dir,
                    transcript_name=transcript_name,
                )

        with mapping_review_tab:
            st.markdown("#### Rough-topic → official AQA resolution")
            st.caption(
                "This section is separate from final-topic edit-memory reuse. It shows "
                "how unresolved rough topics were mapped through PostgreSQL memory, Qdrant, "
                "and the existing Module 3 fallback path."
            )

            if llm_results:
                resolution_rows = []
                for result in llm_results:
                    resolution_rows.append(
                        {
                            "Detected rough topic": result.get("rough_topic"),
                            "Mapped topic": result.get("mapped_topic"),
                            "Official reference": result.get("official_reference"),
                            "Decision": result.get("decision"),
                            "Resolution source": _topic_resolution_source_label(
                                result.get("resolution_source")
                            ),
                            "Review status": result.get("review_status"),
                            "Confidence": result.get("confidence"),
                            "Source chunks": result.get("source_chunk_ids"),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(resolution_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No unmapped rough topic required resolution.")

            render_needs_topic_label_review(
                llm_results=llm_results,
                chunks=chunks,
                run_dir=run_dir,
                transcript_name=transcript_name,
            )

            flash_message = st.session_state.pop("topic_review_flash", None)
            if flash_message:
                st.success(flash_message)

            render_topic_mapping_review(
                topic_review_items,
                run_dir=run_dir,
            )

            render_human_review(
                review_items,
                run_dir=run_dir,
                review_message=review_message,
                review_data_present=review_data_present,
            )

        with diagnostics_tab:
            st.markdown("#### Final-topic memory runtime audit")
            st.json(
                {
                    key: value
                    for key, value in edit_memory_runtime.items()
                    if key != "module3_result"
                }
            )

            st.markdown("#### Complete Module 3 JSON")
            st.json(module3_json)

    with topic_approval_tab:
        render_agent2_topic_approval(
            run_dir=run_dir,
            transcript_name=transcript_name,
            merged_topics=merged_topics,
            chunks=chunks,
        )

    with agent2_assessment_tab:
        render_agent2_assessment_stage(
            run_dir=run_dir,
            transcript_name=transcript_name,
        )

    with files_tab:
        st.subheader("Generated output files")
        files = [
            "01_cleaned_transcript.txt",
            "01_preprocessing.json",
            "01_preprocessing.pdf",
            "02_chunking.json",
            "02_chunking.pdf",
            "03_topic_mapping.json",
            "03_topics_readable.pdf",
            "04_llm_mapping.pdf",
            "05_final_topic_summary.pdf",
        ]
        columns = st.columns(3)
        for index, filename in enumerate(files):
            with columns[index % 3]:
                path = output_folder / filename
                download_file(path, f"Download {filename}")

        st.markdown("#### Preview final summary PDF")
        display_pdf(output_folder / "05_final_topic_summary.pdf")

    with logs_tab:
        logs_dir = run_dir / "logs"
        log_paths = sorted(logs_dir.glob("*.log")) if logs_dir.is_dir() else []
        if not log_paths:
            st.info("No execution logs found.")
        for log_path in log_paths:
            with st.expander(log_path.name):
                st.code(log_path.read_text(encoding="utf-8", errors="replace"))


def _agent1_core_outputs_complete(run_dir: Path) -> bool:
    """Return True when all three Agent 1 notebook outputs exist.

    This keeps the UI stable if an older frontend build incorrectly marked
    pipeline_manifest.json as failed after a downstream Agent 2 / quiz error.
    """
    run_dir = Path(run_dir)
    manifest = load_json(
        run_dir
        / "pipeline_manifest.json"
    )

    transcript_name = str(
        manifest.get(
            "transcript_name",
            "",
        )
        or ""
    ).strip()

    if not transcript_name:
        input_dir = run_dir / "input"
        input_files = (
            [
                path
                for path in input_dir.glob("*")
                if path.is_file()
            ]
            if input_dir.is_dir()
            else []
        )
        if len(input_files) == 1:
            transcript_name = input_files[0].stem

    if not transcript_name:
        return False

    output_dir = (
        run_dir
        / "output"
        / transcript_name
    )

    required = [
        output_dir
        / "01_preprocessing.json",
        output_dir
        / "01_cleaned_transcript.txt",
        output_dir
        / "02_chunking.json",
        output_dir
        / "03_topic_mapping.json",
    ]

    try:
        return all(
            path.is_file()
            and path.stat().st_size > 0
            for path in required
        )
    except OSError:
        return False


st.title("Agent 1 + Agent 2 — LangGraph + MCP Transcript to Assessment / Quiz")
st.caption(
    "LangGraph now owns workflow orchestration and PostgreSQL-checkpointed HITL pauses. "
    "MCP remains the standardized tool boundary; the existing Agent 1 / Agent 2 logic is unchanged."
)

# Restore a recent completed Agent 1 run so the UI/tracker survives a Streamlit restart.
run_dir_value = st.session_state.get("agent1_run_dir")
if not run_dir_value:
    completed_run_candidates = sorted(
        PROJECT_ROOT.glob("runs/job_*/pipeline_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_candidate in completed_run_candidates:
        candidate_manifest = load_json(manifest_candidate)
        if (
            candidate_manifest.get("status") == "completed"
            or _agent1_core_outputs_complete(
                manifest_candidate.parent
            )
        ):
            run_dir_value = str(
                manifest_candidate.parent
            )
            st.session_state[
                "agent1_run_dir"
            ] = run_dir_value
            break

active_run_id = Path(run_dir_value).name if run_dir_value else ""
try:
    initial_tracker = tracker_from_snapshot(
        langgraph_snapshot(frontend_root=PROJECT_ROOT, run_id=active_run_id),
        run_id=active_run_id,
    ) if active_run_id else empty_tracker()
except Exception:
    initial_tracker = st.session_state.get("langgraph_tracker") or empty_tracker(run_id=active_run_id)
st.session_state["langgraph_tracker"] = initial_tracker

main_col, tracker_col = st.columns([3.2, 1.25], gap="large")
with tracker_col:
    LANGGRAPH_TRACKER_PLACEHOLDER = st.empty()
    render_tracker(LANGGRAPH_TRACKER_PLACEHOLDER, initial_tracker)

with st.sidebar:
    st.header("Environment checks")
    st.write("Project root:")
    st.code(str(PROJECT_ROOT))
    st.write("Orchestration: LangGraph StateGraph")
    st.write("Tool interface: MCP")
    st.write("HITL checkpointing: PostgreSQL")
    st.write("Qdrant must be running before Module 3 / Agent 2 retrieval.")
    st.write(
        "Groq may still be used by existing Module 1 / Module 3 semantic "
        "fallbacks. Final-topic HITL automatic memory reuse is deterministic "
        "and never calls Groq; ambiguous cases require human review."
    )

# Human-only MCP writes happen in the existing review UI. On the following
# Streamlit rerun, resume the exact persisted LangGraph thread and re-check DB state.
pending_resume_id = str(st.session_state.pop("langgraph_resume_pending_run_id", "") or "").strip()
if pending_resume_id:
    with main_col:
        resume_status = st.status("Resuming LangGraph after human action", expanded=True)
        try:
            resume_result = run_langgraph_request(
                frontend_root=PROJECT_ROOT,
                run_id=pending_resume_id,
                user_request="Continue safely after the human-reviewed database state changed.",
                max_steps=8,
                mode="resume",
                on_update=_step4_on_graph_update,
            )
        except Exception as exc:
            resume_status.update(label="LangGraph resume failed", state="error", expanded=True)
            st.error(str(exc))
        else:
            st.session_state["langgraph_last_result"] = resume_result
            run_dir_value = str(PROJECT_ROOT / "runs" / pending_resume_id)
            st.session_state["agent1_run_dir"] = run_dir_value
            active_run_id = pending_resume_id
            final_state = str(resume_result.get("final_state") or "")
            if resume_result.get("interrupt_count"):
                resume_status.update(
                    label=f"LangGraph paused at {final_state}", state="complete", expanded=False
                )
            else:
                resume_status.update(
                    label=f"LangGraph resumed to {final_state}", state="complete", expanded=False
                )
            _step4_refresh_tracker(pending_resume_id)

with main_col:
    uploaded = st.file_uploader(
        "Upload a transcript",
        type=["pdf", "docx", "txt"],
        help="PDF, DOCX, and TXT are supported by the existing Module 1.",
    )
    run_clicked = st.button(
        "Process Transcript with LangGraph",
        type="primary",
        disabled=uploaded is None,
        use_container_width=True,
    )

    if run_clicked and uploaded is not None:
        run_info = create_langgraph_run(
            frontend_root=PROJECT_ROOT,
            filename=uploaded.name,
            content=uploaded.getvalue(),
        )
        run_dir = Path(run_info["run_dir"])
        run_dir_value = str(run_dir)
        active_run_id = run_info["run_id"]
        st.session_state["agent1_run_dir"] = run_dir_value
        fresh_tracker = tracker_from_snapshot(
            {"state": "RAW_TRANSCRIPT_READY", "human_gate": "NONE"},
            run_id=active_run_id,
        )
        st.session_state["langgraph_tracker"] = fresh_tracker
        _step4_render_tracker_state(fresh_tracker)

        status = st.status("LangGraph is processing Agent 1 through MCP", expanded=True)
        status.write(f"Created run {active_run_id}")
        try:
            graph_result = run_langgraph_request(
                frontend_root=PROJECT_ROOT,
                run_id=active_run_id,
                user_request=(
                    "Process this transcript through Agent 1 using the valid next MCP tool for each state. "
                    "Stop immediately at any mandatory human gate."
                ),
                max_steps=8,
                mode="start",
                on_update=_step4_on_graph_update,
            )
        except Exception as exc:
            status.update(label="LangGraph pipeline failed", state="error", expanded=True)
            st.error(str(exc))
            st.info(f"Run folder preserved for debugging: {run_dir}")
        else:
            st.session_state["langgraph_last_result"] = graph_result
            final_state = str(graph_result.get("final_state") or "")
            human_required = bool(graph_result.get("human_action_required"))
            _step4_refresh_tracker(active_run_id)
            if human_required:
                status.update(
                    label=f"LangGraph paused for human input — {final_state}",
                    state="complete",
                    expanded=False,
                )
                st.info(
                    "A native LangGraph interrupt is checkpointed in PostgreSQL. "
                    "Use the human review/topic approval UI below; LangGraph will resume afterwards."
                )
            else:
                status.update(
                    label=f"LangGraph stopped at {final_state}",
                    state="complete",
                    expanded=False,
                )

if run_dir_value:
    run_dir = Path(run_dir_value)
    if (run_dir / "pipeline_manifest.json").is_file():
        manifest = load_json(run_dir / "pipeline_manifest.json")
        if (
            manifest.get("status") == "completed"
            or _agent1_core_outputs_complete(
                run_dir
            )
        ):
            st.divider()
            try:
                snapshot = langgraph_snapshot(
                    frontend_root=PROJECT_ROOT,
                    run_id=run_dir.name,
                )
                st.caption(
                    "LangGraph state: "
                    f"{snapshot.get('state')} | "
                    f"Human gate: {snapshot.get('human_gate')}"
                )
            except Exception:
                pass

            downstream_error = str(
                manifest.get(
                    "downstream_last_error",
                    "",
                )
                or ""
            ).strip()

            if downstream_error:
                st.warning(
                    "The Agent 1 run is still complete. "
                    "The last Agent 2 / quiz action failed: "
                    + downstream_error
                )

            render_results(
                run_dir
            )
