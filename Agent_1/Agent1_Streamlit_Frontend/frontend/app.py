from __future__ import annotations

import base64
import inspect
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from pipeline_runner import create_pipeline_run, run_pipeline
from agent2_runner import (
    agent2_project_candidates,
    resolve_agent2_notebook,
    resolve_agent2_project_root,
    run_agent2_notebook,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


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
) -> int:
    """Keep the current run JSON aligned with PostgreSQL review status."""

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
) -> dict[str, Any]:
    """
    Approve or reject one Module 3 topic proposal.

    The notebook installs a PostgreSQL trigger on ``topic_human_review``.
    Setting status to ``approved`` therefore immediately upserts the validated
    decision into ``topic_mapping_memory``.
    """

    normalised_status = str(status).strip().casefold()
    if normalised_status not in {"approved", "rejected"}:
        raise ValueError("Status must be 'approved' or 'rejected'.")

    statement = text(
        """
        UPDATE topic_human_review
        SET
            status = :status,
            reviewed_by = :reviewed_by,
            reviewed_at = NOW(),
            updated_at = NOW()
        WHERE id = :record_id
        RETURNING
            id,
            cache_key,
            original_topic,
            proposed_decision,
            proposed_mapped_concept_id,
            confidence,
            status,
            reviewed_at
        """
    )

    engine = _topic_review_engine()

    with engine.begin() as connection:
        row = connection.execute(
            statement,
            {
                "record_id": int(record_id),
                "status": normalised_status,
                "reviewed_by": reviewed_by,
            },
        ).mappings().one_or_none()

    if row is None:
        raise KeyError(f"Topic review record {record_id} was not found.")

    updated_json_records = _persist_topic_review_status_in_json(
        run_dir=run_dir,
        record_id=int(record_id),
        status=normalised_status,
    )

    return {
        **dict(row),
        "updated_json_records": updated_json_records,
        "promoted_to_mapping_memory": (
            normalised_status == "approved"
        ),
    }



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
    """Map backend status names to the three statuses shown in the UI."""

    status = str(value or "pending").strip().casefold()
    if status in {"candidate", "pending", "awaiting_review", "needs_review"}:
        return "pending"
    if status in {"approved", "accept", "accepted"}:
        return "approved"
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
    return {
        "Review ID": item.get("id"),
        "Detected topic": item.get("rough_topic"),
        "Proposed topic": item.get("mapped_topic"),
        "Official reference": item.get("official_reference"),
        "Decision": item.get("decision"),
        "Confidence": item.get("confidence"),
        "Status": item.get("status"),
    }


def render_topic_mapping_review(
    review_items: list[dict[str, Any]],
    *,
    run_dir: Path,
) -> None:
    """Render Module 3 topic proposals and persist approval decisions."""

    st.markdown("---")
    st.subheader("Human Review — Topic Mapping")
    st.caption(
        "A new Groq mapping is not reusable memory until it is approved. "
        "Approve writes it to topic_mapping_memory through PostgreSQL's "
        "self-improving trigger."
    )

    grouped: dict[str, list[dict[str, Any]]] = {
        "pending": [],
        "approved": [],
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
        grouped[status].append(
            {
                **raw_item,
                "status": status,
            }
        )

    pending = grouped["pending"]
    approved = grouped["approved"]
    rejected = grouped["rejected"]

    pcol, acol, rcol = st.columns(3)
    pcol.metric("Pending topic mappings", len(pending))
    acol.metric("Approved topic mappings", len(approved))
    rcol.metric("Rejected topic mappings", len(rejected))

    pending_tab, approved_tab, rejected_tab = st.tabs(
        [
            f"Pending ({len(pending)})",
            f"Approved ({len(approved)})",
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
            official_reference = (
                item.get("official_reference") or "None"
            )
            decision = item.get("decision") or "needs_review"
            confidence = item.get("confidence")
            reason = item.get("reason") or ""
            source_chunks = item.get("source_chunk_ids") or []

            title = f"Topic proposal {index}: {detected_topic}"
            if record_id is not None:
                title += f" — review {record_id}"

            with st.expander(title, expanded=True):
                metadata = [
                    f"Decision: {decision}",
                    "Source: Groq LLM",
                ]
                if confidence is not None:
                    try:
                        metadata.append(
                            f"Confidence: {float(confidence):.2f}"
                        )
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
                    st.caption(
                        f"Official reference: {official_reference}"
                    )

                if source_chunks:
                    st.markdown("**Source chunks**")
                    st.write(source_chunks)

                if reason:
                    st.markdown("**Reason**")
                    st.write(reason)

                can_approve = (
                    decision == "out_of_syllabus"
                    or item.get("mapped_concept_id") is not None
                )

                if record_id is None:
                    st.error(
                        "This topic proposal has no review ID and cannot be "
                        "updated."
                    )
                    continue

                if not can_approve:
                    st.warning(
                        "This proposal has no official mapped concept. It can "
                        "be rejected, but it cannot be approved until a valid "
                        "mapping is selected."
                    )

                approve_col, reject_col = st.columns(2)
                approve_clicked = approve_col.button(
                    "Approve and add to memory",
                    key=(
                        f"approve_topic_review_"
                        f"{run_dir.name}_{record_id}"
                    ),
                    type="primary",
                    use_container_width=True,
                    disabled=not can_approve,
                )
                reject_clicked = reject_col.button(
                    "Reject topic mapping",
                    key=(
                        f"reject_topic_review_"
                        f"{run_dir.name}_{record_id}"
                    ),
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
                        result = topic_review_set_status(
                            record_id=int(record_id),
                            status=selected_status,
                            run_dir=run_dir,
                        )
                    except Exception as exc:
                        st.error(
                            f"Could not set topic review {record_id} to "
                            f"{selected_status}: {exc}"
                        )
                    else:
                        if selected_status == "approved":
                            st.session_state["topic_review_flash"] = (
                                f"Topic review {record_id} approved and "
                                "promoted to PostgreSQL mapping memory."
                            )
                        else:
                            st.session_state["topic_review_flash"] = (
                                f"Topic review {record_id} rejected."
                            )
                        st.rerun()

    with approved_tab:
        if approved:
            st.success(
                "Approved records are reusable PostgreSQL mapping memory."
            )
            st.dataframe(
                pd.DataFrame(
                    _topic_review_row(item)
                    for item in approved
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No approved topic mappings for this run.")

    with rejected_tab:
        if rejected:
            st.dataframe(
                pd.DataFrame(
                    _topic_review_row(item)
                    for item in rejected
                ),
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
    (
        all_topics_path,
        approved_topics_path,
    ) = _agent2_handoff_paths(run_dir)

    output_path = (
        approved_topics_path
        if approved_only
        else all_topics_path
    )

    payload = {
        "schema_version": (
            "agent1-agent2-topic-handoff-v1.0.0"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "job_id": Path(run_dir).name,
        "transcript": transcript_name,
        "source": {
            "module2_output": (
                "02_chunking.json"
            ),
            "module3_output": (
                "03_topic_mapping.json"
            ),
            "notebook_logic_changed": False,
            "handoff_built_by": (
                "streamlit_frontend"
            ),
        },
        "approved_only": bool(
            approved_only
        ),
        "topic_count": len(topics),
        "actual_chunk_evidence_available": bool(
            topics
            and all(
                topic.get(
                    "source_chunk_texts"
                )
                for topic in topics
            )
        ),
        "topics": topics,
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


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
        "No notebook processing logic is changed here. The frontend reads "
        "Module 2 chunk text and Module 3 retained topics, then prepares the "
        "JSON handoff required by Agent 2."
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

    approve_clicked = st.button(
        "Approve Topics and Continue to Agent 2",
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
                f"Saved {len(approved_topics)} approved topic(s) with actual "
                "Module 2 source chunk text."
            )

            st.info(
                "Next integration step: add the Agent 2 assessment-filter form "
                "and pass this JSON file to Notebook 05."
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
    paths = sorted(
        Path(folder).glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return paths[0] if paths else None


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


def render_agent2_assessment_results(*, run_dir: Path) -> None:
    manifest_path = _agent2_result_manifest_path(run_dir)
    if not manifest_path.is_file():
        return
    manifest = load_json(manifest_path)
    package_path = Path(manifest.get("package_path", ""))
    if not package_path.is_file():
        st.error(
            "The Agent 2 execution manifest exists, but the assessment package "
            "file could not be found."
        )
        return
    package = load_json(package_path)
    summary = package.get("retrieval_summary", {}) or {}
    questions = package.get("questions", []) or []
    st.markdown("---")
    st.subheader("Generated Agent 2 Assessment")
    release_status = summary.get(
        "final_release_status",
        summary.get("assessment_release_status", "unknown"),
    )
    blockers = summary.get("release_blockers", []) or []
    _release_status_banner(str(release_status), blockers)
    metrics = st.columns(5)
    metrics[0].metric("Questions", summary.get("selected_questions", len(questions)))
    metrics[1].metric("Selected marks", summary.get("selected_marks", "N/A"))
    metrics[2].metric("Target marks", summary.get("target_marks", "N/A"))
    metrics[3].metric(
        "Topics covered",
        summary.get("selected_distinct_official_references", "N/A"),
    )
    evidence_used = summary.get(
        "actual_agent1_chunk_evidence_used",
        summary.get("all_topics_use_actual_chunk_evidence", False),
    )
    metrics[4].metric("Actual chunk evidence", "Yes" if evidence_used else "No")
    output_dir = Path(manifest.get("output_dir", package_path.parent))

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
        with st.expander(title, expanded=(position == 1)):
            metadata_columns = st.columns(4)
            metadata_columns[0].metric(
                "Official reference", topic.get("official_reference", "N/A")
            )
            metadata_columns[1].metric("Role", topic.get("role", "N/A"))
            metadata_columns[2].metric("Paper", question.get("paper_code", "N/A"))
            semantic_score = retrieval.get("semantic_score")
            metadata_columns[3].metric(
                "Semantic score",
                f"{float(semantic_score):.4f}" if semantic_score is not None else "N/A",
            )
            st.markdown("#### Question")
            st.write(question.get("text", ""))
            context_text = str(question.get("context", "") or "").strip()
            if context_text:
                st.markdown("**Context**")
                st.write(context_text)
            image_paths = question.get("rendered_page_images", []) or []
            if image_paths:
                st.markdown("#### Original question page image(s)")
                for relative_path in image_paths:
                    image_path = output_dir / str(relative_path)
                    if image_path.is_file():
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.warning(f"Rendered image file not found: {relative_path}")
            raw_tab, structured_tab, evidence_tab = st.tabs(
                [
                    "Raw mark scheme",
                    "Phase 3 structured mark scheme",
                    "Retrieval evidence",
                ]
            )
            with raw_tab:
                st.text_area(
                    "Raw marking guidance",
                    str(
                        mark_scheme.get(
                            "raw_marking_guidance",
                            mark_scheme.get("marking_guidance", ""),
                        )
                        or ""
                    ),
                    height=320,
                    disabled=True,
                    key=f"agent2_raw_mark_scheme_{Path(run_dir).name}_{position}",
                )
            with structured_tab:
                _render_phase3_structured_mark_scheme(
                    mark_scheme.get("phase3_structured", {}) or {}
                )
            with evidence_tab:
                st.json(retrieval)

    st.markdown("#### Agent 2 generated files")
    download_candidates = [
        package_path,
        _latest_matching_file(output_dir, "agent2_assessment_package_*.md"),
        _latest_matching_file(output_dir, "agent2_assessment_evaluation_*.txt"),
        _latest_matching_file(output_dir, "agent2_selected_questions_*.csv"),
        _latest_matching_file(
            output_dir, "agent2_assessment_release_readiness_*.json"
        ),
    ]
    valid_downloads = [
        path for path in download_candidates if path is not None and path.is_file()
    ]
    columns = st.columns(min(3, max(1, len(valid_downloads))))
    for index, path in enumerate(valid_downloads):
        with columns[index % len(columns)]:
            download_file(path, f"Download {path.name}")
    with st.expander("Agent 2 execution integrity and log"):
        st.json(manifest)
        log_path = Path(manifest.get("log_path", ""))
        if log_path.is_file():
            st.code(
                log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
            )


def render_agent2_assessment_stage(*, run_dir: Path, transcript_name: str) -> None:
    _, approved_topics_path = _agent2_handoff_paths(run_dir)
    st.subheader("Agent 2 — Assessment Filters")
    st.caption(
        "The frontend passes approved Agent 1 topics and these filters to a "
        "temporary execution copy of the existing Notebook 05. The original "
        "notebook and its processing logic remain unchanged."
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
        for topic in approved_topics
        if isinstance(topic, dict)
    )
    supporting_topic_count = sum(
        str(topic.get("role", "")).casefold() == "supporting"
        for topic in approved_topics
        if isinstance(topic, dict)
    )
    unique_references = sorted(
        {
            str(topic.get("official_reference", "")).strip()
            for topic in approved_topics
            if isinstance(topic, dict)
            and str(topic.get("official_reference", "")).strip()
        }
    )
    evidence_ready = bool(
        approved_payload.get("actual_chunk_evidence_available", False)
    )
    top_columns = st.columns(4)
    top_columns[0].metric("Approved topics", len(approved_topics))
    top_columns[1].metric("Primary topics", primary_topic_count)
    top_columns[2].metric("Supporting topics", supporting_topic_count)
    top_columns[3].metric(
        "Actual chunk evidence", "Available" if evidence_ready else "Missing"
    )

    with st.expander("Agent 2 notebook connection", expanded=False):
        agent2_root_value = st.text_input(
            "Agent 2 project folder",
            value=_default_agent2_project_root(),
            help=(
                "Usually C:\\Users\\hp\\EDTECH\\Agent2. The folder should "
                "contain .env, cache and notebooks."
            ),
            key=f"agent2_project_root_input_{Path(run_dir).name}",
        )
        notebook_value = st.text_input(
            "Notebook 05 path (optional)",
            value="",
            help=(
                "Leave blank to use Notebook 05 in Agent2/notebooks. A "
                "byte-identical bundled fallback is also included."
            ),
            key=f"agent2_notebook_path_input_{Path(run_dir).name}",
        )
        try:
            resolved_root = resolve_agent2_project_root(
                PROJECT_ROOT, explicit_path=agent2_root_value
            )
            st.success(f"Agent 2 root resolved: {resolved_root}")
            resolved_notebook = resolve_agent2_notebook(
                agent2_project_root=resolved_root,
                frontend_project_root=PROJECT_ROOT,
                explicit_path=notebook_value or None,
            )
            st.success(f"Notebook 05 resolved: {resolved_notebook}")
        except Exception as exc:
            st.warning(str(exc))

    default_question_count = max(5, len(unique_references))
    with st.form(key=f"agent2_assessment_filter_form_{Path(run_dir).name}"):
        first_row = st.columns(3)
        number_of_questions = first_row[0].number_input(
            "Number of questions", 1, 30, default_question_count, 1
        )
        target_total_marks = first_row[1].number_input(
            "Target total marks", 1, 200, 20, 1
        )
        paper_label = first_row[2].selectbox(
            "Paper filter", ["Both papers", "Paper 1", "Paper 2"]
        )
        second_row = st.columns(4)
        minimum_marks = second_row[0].number_input(
            "Minimum marks per question", 1, 20, 1, 1
        )
        maximum_marks = second_row[1].number_input(
            "Maximum marks per question", 1, 30, 12, 1
        )
        minimum_primary = second_row[2].number_input(
            "Minimum primary questions", 1, 30, min(3, default_question_count), 1
        )
        minimum_supporting = second_row[3].number_input(
            "Minimum supporting questions",
            0,
            30,
            1 if supporting_topic_count else 0,
            1,
        )
        third_row = st.columns(3)
        programming_language_label = third_row[0].selectbox(
            "Programming language", ["Automatic", "Python"]
        )
        include_code = third_row[1].checkbox("Include code questions", value=True)
        include_visual = third_row[2].checkbox(
            "Include visual questions", value=True
        )
        cover_all_topics = st.checkbox("Cover all approved topics", value=True)
        run_agent2_clicked = st.form_submit_button(
            "Generate Assessment with Existing Notebook 05",
            type="primary",
            use_container_width=True,
        )

    if run_agent2_clicked:
        errors = []
        if maximum_marks < minimum_marks:
            errors.append("Maximum marks must be at least minimum marks.")
        if minimum_primary > number_of_questions:
            errors.append("Minimum primary questions exceed total questions.")
        if minimum_supporting > number_of_questions:
            errors.append("Minimum supporting questions exceed total questions.")
        if minimum_primary + minimum_supporting > number_of_questions:
            errors.append(
                "Primary and supporting minimums together exceed total questions."
            )
        if cover_all_topics and number_of_questions < len(unique_references):
            errors.append(
                f"Request at least {len(unique_references)} questions to cover all topics."
            )
        if minimum_supporting > 0 and supporting_topic_count == 0:
            errors.append("No approved supporting topic is available.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            paper_code = {
                "Both papers": None,
                "Paper 1": "1",
                "Paper 2": "2",
            }[paper_label]
            programming_language = (
                None
                if programming_language_label == "Automatic"
                else programming_language_label
            )
            request = {
                "number_of_questions": int(number_of_questions),
                "target_total_marks": int(target_total_marks),
                "minimum_question_marks": int(minimum_marks),
                "maximum_question_marks": int(maximum_marks),
                "minimum_primary_questions": int(minimum_primary),
                "minimum_supporting_questions": int(minimum_supporting),
                "minimum_distinct_official_references": (
                    len(unique_references) if cover_all_topics else 1
                ),
                "include_code_questions": bool(include_code),
                "include_visual_questions": bool(include_visual),
                "paper_code": paper_code,
                "programming_language": programming_language,
            }
            request_path = _write_agent2_assessment_request(
                run_dir=run_dir,
                transcript_name=transcript_name,
                assessment_request=request,
                approved_topics=approved_topics,
            )
            status = st.status("Running Agent 2 Notebook 05", expanded=True)
            progress = st.progress(0, text="Preparing Agent 2 execution")
            progress_steps = {
                "Preparing a temporary Notebook 05 execution copy": 20,
                "Running existing Agent 2 Notebook 05": 45,
                "Agent 2 assessment package generated": 100,
            }

            def agent2_progress(message: str) -> None:
                progress.progress(progress_steps.get(message, 60), text=message)
                status.write(message)

            try:
                resolved_root = resolve_agent2_project_root(
                    PROJECT_ROOT, explicit_path=agent2_root_value
                )
                resolved_notebook = resolve_agent2_notebook(
                    agent2_project_root=resolved_root,
                    frontend_project_root=PROJECT_ROOT,
                    explicit_path=notebook_value or None,
                )
                result = run_agent2_notebook(
                    frontend_project_root=PROJECT_ROOT,
                    run_dir=run_dir,
                    approved_topics_path=approved_topics_path,
                    assessment_request_path=request_path,
                    agent2_project_root=resolved_root,
                    source_notebook=resolved_notebook,
                    progress_callback=agent2_progress,
                )
            except Exception as exc:
                status.update(
                    label="Agent 2 assessment failed",
                    state="error",
                    expanded=True,
                )
                st.error(str(exc))
            else:
                progress.progress(100, text="Agent 2 assessment generated")
                status.update(
                    label="Agent 2 assessment generated",
                    state="complete",
                    expanded=False,
                )
                st.session_state["agent2_execution_manifest_path"] = str(
                    result.manifest_path
                )
                st.success(
                    "Notebook 05 completed using approved Agent 1 topics and "
                    "actual source chunk evidence."
                )
                st.rerun()

    render_agent2_assessment_results(run_dir=run_dir)


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
    module3_result = module3_json.get("module3_result", {})
    merged_topics = module3_result.get("merged_topics", [])
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
        st.subheader("Detected official topics")
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
                    }
                )
            st.dataframe(pd.DataFrame(topic_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No official topics were retained.")

        primary = [t for t in merged_topics if t.get("topic_role") == "primary"]
        supporting = [t for t in merged_topics if t.get("topic_role") == "supporting"]
        pcol, scol = st.columns(2)
        with pcol:
            st.markdown("#### Primary topics")
            for topic in primary:
                st.write(f"• {topic.get('topic')}")
        with scol:
            st.markdown("#### Supporting topics")
            for topic in supporting:
                st.write(f"• {topic.get('topic')}")

        st.markdown("#### Memory / Qdrant / Groq resolution")
        if llm_results:
            resolution_rows = []
            for result in llm_results:
                resolution_rows.append(
                    {
                        "Detected rough topic": result.get(
                            "rough_topic"
                        ),
                        "Mapped topic": result.get("mapped_topic"),
                        "Official reference": result.get(
                            "official_reference"
                        ),
                        "Decision": result.get("decision"),
                        "Resolution source": (
                            _topic_resolution_source_label(
                                result.get("resolution_source")
                            )
                        ),
                        "Review status": result.get(
                            "review_status"
                        ),
                        "Confidence": result.get("confidence"),
                        "Source chunks": result.get(
                            "source_chunk_ids"
                        ),
                    }
                )
            st.dataframe(
                pd.DataFrame(resolution_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No unmapped topic required resolution.")

        flash_message = st.session_state.pop(
            "topic_review_flash",
            None,
        )
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

        with st.expander("Complete Module 3 JSON"):
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


st.title("Agent 1 + Agent 2 — Transcript to Assessment")
st.caption(
    "Upload one transcript. The page runs the existing Agent 1 notebooks, lets "
    "you approve topics, then executes the existing Agent 2 Notebook 05 through "
    "a temporary parameterized copy. Original notebook logic is not changed."
)

with st.sidebar:
    st.header("Environment checks")
    st.write("Project root:")
    st.code(str(PROJECT_ROOT))
    st.write("Qdrant must be running before Module 3.")
    st.write("Groq is optional unless an unresolved topic requires fallback.")
    st.write("Agent 2 requires PostgreSQL, Qdrant, cached PMT PDFs, and Notebook 05.")

uploaded = st.file_uploader(
    "Upload a transcript",
    type=["pdf", "docx", "txt"],
    help="PDF, DOCX, and TXT are supported by Module 1.",
)

run_clicked = st.button(
    "Run Agent 1 Pipeline",
    type="primary",
    disabled=uploaded is None,
    use_container_width=True,
)

if run_clicked and uploaded is not None:
    run = create_pipeline_run(
        PROJECT_ROOT,
        uploaded.name,
        uploaded.getvalue(),
    )

    progress = st.progress(0, text="Preparing pipeline")
    status = st.status("Running notebooks", expanded=True)

    def update_progress(completed_modules: int, message: str) -> None:
        percentage = int((completed_modules / 3) * 100)
        progress.progress(min(percentage, 100), text=message)
        status.write(message)

    try:
        run_pipeline(
            project_root=PROJECT_ROOT,
            run=run,
            progress_callback=update_progress,
        )
    except Exception as exc:
        status.update(label="Pipeline failed", state="error", expanded=True)
        st.error(str(exc))
        st.info(f"Run folder preserved for debugging: {run.run_dir}")
        st.session_state["agent1_run_dir"] = str(run.run_dir)
    else:
        progress.progress(100, text="Pipeline completed")
        status.update(label="Pipeline completed", state="complete", expanded=False)
        st.session_state["agent1_run_dir"] = str(run.run_dir)

run_dir_value = st.session_state.get("agent1_run_dir")

# Resume the most recent completed run after a Streamlit restart. This lets
# the user continue from topic approval into Agent 2 without rerunning Agent 1.
if not run_dir_value:
    completed_run_candidates = sorted(
        PROJECT_ROOT.glob("runs/job_*/pipeline_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for manifest_candidate in completed_run_candidates:
        candidate_manifest = load_json(
            manifest_candidate
        )

        if candidate_manifest.get("status") == "completed":
            run_dir_value = str(
                manifest_candidate.parent
            )
            st.session_state[
                "agent1_run_dir"
            ] = run_dir_value
            break

if run_dir_value:
    run_dir = Path(run_dir_value)
    if (run_dir / "pipeline_manifest.json").is_file():
        manifest = load_json(run_dir / "pipeline_manifest.json")
        if manifest.get("status") == "completed":
            st.divider()
            render_results(run_dir)