from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


VALID_ROLES = frozenset({"primary", "supporting"})


@dataclass(frozen=True, slots=True)
class PlannedDetectedTopicEdit:
    edit_action: str
    source_concept_id: str | None
    source_topic: str | None
    source_role: str | None
    target_concept_id: str | None
    target_topic: str | None
    target_role: str | None
    evidence_text: str
    source_chunk_ids: tuple[int, ...]
    reviewer_reason: str
    topic_index: int | None
    description: str


@dataclass(frozen=True, slots=True)
class EditPlanResult:
    edits: tuple[PlannedDetectedTopicEdit, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _normalise_role(value: Any) -> str:
    role = _normalise_text(value)
    if role not in VALID_ROLES:
        raise ValueError("Topic role must be 'primary' or 'supporting'.")
    return role


def _clean_reason(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _int_list(values: Iterable[Any] | str | None) -> list[int]:
    if values is None:
        return []

    if isinstance(values, str):
        values = [
            part.strip()
            for part in values.split(",")
            if part.strip()
        ]

    output: list[int] = []

    for value in values:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue

        if integer not in output:
            output.append(integer)

    return output


def _topic_evidence(topic: dict[str, Any]) -> str:
    chunk_texts = [
        " ".join(str(value or "").strip().split())
        for value in (topic.get("source_chunk_texts") or [])
        if str(value or "").strip()
    ]

    if chunk_texts:
        return "\n\n".join(chunk_texts)

    evidence = [
        " ".join(str(value or "").strip().split())
        for value in (topic.get("evidence") or [])
        if str(value or "").strip()
    ]

    return "\n\n".join(evidence)


def _syllabus_store():
    from app.services.syllabus_store import get_syllabus_store
    return get_syllabus_store()


def _get_concept(concept_id: str):
    concept = _syllabus_store().get_concept(concept_id)
    if concept is None:
        raise KeyError(f"Unknown AQA concept_id: {concept_id}")
    return concept


def catalog_topic_options() -> list[dict[str, str]]:
    store = _syllabus_store()

    rows = [
        {
            "concept_id": concept.concept_id,
            "label": concept.label,
            "official_reference": concept.official_reference,
            "display": (
                f"{concept.official_reference} — {concept.label} "
                f"[{concept.concept_id}]"
            ),
        }
        for concept in store.get_all_concepts()
    ]

    return sorted(
        rows,
        key=lambda row: (
            row["official_reference"],
            row["label"].casefold(),
            row["concept_id"],
        ),
    )


def build_manual_addition(
    *,
    concept_id: str,
    role: str,
    source_chunk_ids: Sequence[int],
    chunks: Sequence[dict[str, Any]],
    reviewer_reason: str,
) -> dict[str, Any]:
    reason = _clean_reason(reviewer_reason)
    if not reason:
        raise ValueError("A reason is required for a manually added topic.")

    normalized_role = _normalise_role(role)
    selected_ids = _int_list(source_chunk_ids)

    if not selected_ids:
        raise ValueError(
            "Select at least one current transcript chunk as evidence."
        )

    chunk_text_by_id: dict[int, str] = {}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        try:
            chunk_id = int(chunk.get("chunk_id"))
        except (TypeError, ValueError):
            continue

        text = str(chunk.get("text") or "").strip()
        if text:
            chunk_text_by_id[chunk_id] = text

    source_chunk_texts = [
        chunk_text_by_id[chunk_id]
        for chunk_id in selected_ids
        if chunk_text_by_id.get(chunk_id)
    ]

    if not source_chunk_texts:
        raise ValueError(
            "The selected chunks do not contain transcript text."
        )

    concept = _get_concept(str(concept_id).strip())

    return {
        "topic_index": None,
        "concept_id": concept.concept_id,
        "topic": concept.label,
        "detected_topic": concept.label,
        "role": normalized_role,
        "topic_role": normalized_role,
        "official_reference": concept.official_reference,
        "official_title": concept.official_title,
        "chapter_reference": concept.chapter_reference,
        "domain": concept.domain,
        "paper": concept.paper,
        "confidence": None,
        "ranking_score": None,
        "source_chunks": selected_ids,
        "source_chunk_texts": source_chunk_texts,
        "source_chunk_text_count": len(source_chunk_texts),
        "source_chunk_count": len(selected_ids),
        "missing_source_chunk_ids": [],
        "evidence": source_chunk_texts,
        "human_added_topic": True,
        "human_change_reason": reason,
    }


def merge_manual_additions(
    *,
    topic_payload: Sequence[dict[str, Any]],
    manual_additions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = [dict(topic) for topic in topic_payload]

    existing_concept_ids = {
        str(topic.get("concept_id") or "").strip()
        for topic in output
    }

    next_index = (
        max(
            [
                int(topic.get("topic_index", 0))
                for topic in output
                if topic.get("topic_index") is not None
            ],
            default=0,
        )
        + 1
    )

    for raw in manual_additions:
        topic = dict(raw)
        concept_id = str(topic.get("concept_id") or "").strip()

        if not concept_id or concept_id in existing_concept_ids:
            continue

        topic["topic_index"] = next_index
        next_index += 1
        existing_concept_ids.add(concept_id)
        output.append(topic)

    return output


def _resolve_target_concept(
    *,
    source_concept_id: str,
    edited_topic: str,
    edited_reference: str,
):
    store = _syllabus_store()
    source = _get_concept(source_concept_id)

    topic_norm = _normalise_text(edited_topic)
    reference = str(edited_reference or "").strip()

    if (
        reference == source.official_reference
        and topic_norm
        in {
            _normalise_text(source.label),
            _normalise_text(source.official_title),
        }
    ):
        return source

    candidates = list(
        store.get_concepts_by_reference(reference)
    )

    exact_label = [
        concept
        for concept in candidates
        if _normalise_text(concept.label) == topic_norm
    ]

    if len(exact_label) == 1:
        return exact_label[0]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise ValueError(
            f"{edited_reference!r} is not an official AQA reference "
            "in the current catalogue."
        )

    available = "; ".join(
        f"{concept.label} [{concept.concept_id}]"
        for concept in candidates
    )

    raise ValueError(
        "The edited official reference maps to multiple catalogue "
        "concepts. Use the exact official topic label. Available: "
        + available
    )


def plan_detected_topic_edits(
    *,
    baseline_topics: Sequence[dict[str, Any]],
    edited_rows: Sequence[dict[str, Any]],
) -> EditPlanResult:
    baseline_by_index = {
        int(topic["topic_index"]): dict(topic)
        for topic in baseline_topics
        if topic.get("topic_index") is not None
    }

    edits: list[PlannedDetectedTopicEdit] = []
    errors: list[str] = []
    warnings: list[str] = []

    for row in edited_rows:
        raw_index = row.get("_topic_index")

        try:
            topic_index = int(raw_index) if raw_index is not None else None
        except (TypeError, ValueError):
            topic_index = None

        approved = bool(row.get("Approve", False))
        learn = bool(row.get("Learn correction", False))
        reason = _clean_reason(
            row.get("Reason for human change", "")
        )

        if bool(row.get("_human_added_topic", False)):
            if not approved:
                continue

            if not learn:
                errors.append(
                    "A manually added topic must keep "
                    "'Learn correction' enabled."
                )
                continue

            if not reason:
                errors.append(
                    f"Reason required for added topic {row.get('Topic')!r}."
                )
                continue

            concept_id = str(row.get("_concept_id") or "").strip()
            evidence_text = str(row.get("_evidence_text") or "").strip()
            source_chunk_ids = tuple(
                _int_list(row.get("_source_chunk_ids"))
            )

            try:
                target_role = _normalise_role(row.get("Role"))
                target = _get_concept(concept_id)
            except Exception as exc:
                errors.append(str(exc))
                continue

            if (
                _normalise_text(row.get("Topic")) != _normalise_text(target.label)
                or str(row.get("Official reference") or "").strip()
                != target.official_reference
            ):
                errors.append(
                    f"Added topic {target.label!r} was edited after selection. "
                    "Remove it and add the intended official concept again."
                )
                continue

            if not evidence_text or not source_chunk_ids:
                errors.append(
                    f"Added topic {target.label!r} is missing "
                    "current transcript evidence."
                )
                continue

            edits.append(
                PlannedDetectedTopicEdit(
                    edit_action="add_topic",
                    source_concept_id=None,
                    source_topic=None,
                    source_role=None,
                    target_concept_id=target.concept_id,
                    target_topic=target.label,
                    target_role=target_role,
                    evidence_text=evidence_text,
                    source_chunk_ids=source_chunk_ids,
                    reviewer_reason=reason,
                    topic_index=topic_index,
                    description=f"Add missing official topic {target.label!r}.",
                )
            )
            continue

        if topic_index is None:
            continue

        baseline = baseline_by_index.get(topic_index)
        if baseline is None:
            errors.append(
                f"Could not match editor topic index {topic_index} "
                "to the current Agent 1 topic list."
            )
            continue

        source_concept_id = str(
            baseline.get("concept_id") or ""
        ).strip()

        if not source_concept_id:
            errors.append(
                f"Topic index {topic_index} has no stable concept_id; "
                "its correction cannot be learned safely."
            )
            continue

        source_topic = str(baseline.get("topic") or "").strip()

        try:
            source_role = _normalise_role(baseline.get("role"))
            edited_role = _normalise_role(row.get("Role"))
        except ValueError as exc:
            errors.append(f"Topic {source_topic!r}: {exc}")
            continue

        edited_topic = str(row.get("Topic") or "").strip()
        edited_reference = str(
            row.get("Official reference") or ""
        ).strip()

        topic_changed = (
            _normalise_text(edited_topic)
            != _normalise_text(source_topic)
        )
        reference_changed = (
            edited_reference
            != str(baseline.get("official_reference") or "").strip()
        )
        role_changed = edited_role != source_role

        # Agent 2 deselection by itself is NOT training data.
        if not learn:
            continue

        if not reason:
            errors.append(
                f"Reason required for learned correction to "
                f"{source_topic!r}."
            )
            continue

        evidence_text = _topic_evidence(baseline)
        source_chunk_ids = tuple(
            _int_list(baseline.get("source_chunks"))
        )

        if not evidence_text:
            errors.append(
                f"Topic {source_topic!r} has no current transcript "
                "evidence; correction was not learned."
            )
            continue

        if not approved:
            edits.append(
                PlannedDetectedTopicEdit(
                    edit_action="remove_topic",
                    source_concept_id=source_concept_id,
                    source_topic=source_topic,
                    source_role=source_role,
                    target_concept_id=None,
                    target_topic=None,
                    target_role=None,
                    evidence_text=evidence_text,
                    source_chunk_ids=source_chunk_ids,
                    reviewer_reason=reason,
                    topic_index=topic_index,
                    description=(
                        f"Remove {source_topic!r} from final detected topics."
                    ),
                )
            )
            continue

        # Replacement takes precedence over a simultaneous role change.
        if topic_changed or reference_changed:
            try:
                target = _resolve_target_concept(
                    source_concept_id=source_concept_id,
                    edited_topic=edited_topic,
                    edited_reference=edited_reference,
                )
            except Exception as exc:
                errors.append(f"Topic {source_topic!r}: {exc}")
                continue

            if target.concept_id == source_concept_id:
                if role_changed:
                    edits.append(
                        PlannedDetectedTopicEdit(
                            edit_action="change_role",
                            source_concept_id=source_concept_id,
                            source_topic=source_topic,
                            source_role=source_role,
                            target_concept_id=source_concept_id,
                            target_topic=source_topic,
                            target_role=edited_role,
                            evidence_text=evidence_text,
                            source_chunk_ids=source_chunk_ids,
                            reviewer_reason=reason,
                            topic_index=topic_index,
                            description=(
                                f"Change {source_topic!r} role "
                                f"{source_role} -> {edited_role}."
                            ),
                        )
                    )
                else:
                    warnings.append(
                        f"{source_topic!r}: label/reference wording changed "
                        "but still resolves to the same official concept. "
                        "No reusable syllabus edit memory was created."
                    )
                continue

            edits.append(
                PlannedDetectedTopicEdit(
                    edit_action="replace_topic",
                    source_concept_id=source_concept_id,
                    source_topic=source_topic,
                    source_role=source_role,
                    target_concept_id=target.concept_id,
                    target_topic=target.label,
                    target_role=edited_role,
                    evidence_text=evidence_text,
                    source_chunk_ids=source_chunk_ids,
                    reviewer_reason=reason,
                    topic_index=topic_index,
                    description=(
                        f"Replace {source_topic!r} with {target.label!r}."
                    ),
                )
            )
            continue

        if role_changed:
            edits.append(
                PlannedDetectedTopicEdit(
                    edit_action="change_role",
                    source_concept_id=source_concept_id,
                    source_topic=source_topic,
                    source_role=source_role,
                    target_concept_id=source_concept_id,
                    target_topic=source_topic,
                    target_role=edited_role,
                    evidence_text=evidence_text,
                    source_chunk_ids=source_chunk_ids,
                    reviewer_reason=reason,
                    topic_index=topic_index,
                    description=(
                        f"Change {source_topic!r} role "
                        f"{source_role} -> {edited_role}."
                    ),
                )
            )
            continue

        warnings.append(
            f"{source_topic!r}: 'Learn correction' was checked but "
            "no detected-topic correction was made."
        )

    return EditPlanResult(
        edits=tuple(edits),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def persist_detected_topic_edits(
    *,
    plan: EditPlanResult,
    transcript_name: str,
    spec_version: str,
    reviewed_by: str = "streamlit",
) -> list[dict[str, Any]]:
    if plan.errors:
        raise ValueError(
            "Cannot persist an edit plan that contains validation errors."
        )

    if not plan.edits:
        return []

    from app.db.repositories.detected_topic_edit_memory_repository import (
        DetectedTopicEditMemoryRepository,
    )
    from app.db.session import session_scope
    from app.services.detected_topic_edit_memory_service import (
        DetectedTopicEdit,
        DetectedTopicEditMemoryService,
    )

    stored: list[dict[str, Any]] = []

    # session_scope commits all edits together or rolls all of them back.
    with session_scope() as session:
        repository = DetectedTopicEditMemoryRepository(session)
        service = DetectedTopicEditMemoryService(repository)

        for item in plan.edits:
            record = service.remember(
                DetectedTopicEdit(
                    edit_action=item.edit_action,
                    source_concept_id=item.source_concept_id,
                    source_topic=item.source_topic,
                    source_role=item.source_role,
                    target_concept_id=item.target_concept_id,
                    target_topic=item.target_topic,
                    target_role=item.target_role,
                    evidence_text=item.evidence_text,
                    source_chunk_ids=item.source_chunk_ids,
                    reviewer_reason=item.reviewer_reason,
                    source_transcript=transcript_name,
                    spec_version=spec_version,
                    reviewed_by=reviewed_by,
                )
            )

            stored.append(
                {
                    "memory_id": int(record.id),
                    "edit_action": record.edit_action,
                    "source_concept_id": record.source_concept_id,
                    "target_concept_id": record.target_concept_id,
                    "reviewer_reason": record.reviewer_reason,
                    "description": item.description,
                }
            )

    return stored



def persist_detected_topic_edit_memory(
    *,
    edit_action: str,
    source_concept_id: str | None,
    source_topic: str | None,
    source_role: str | None,
    target_concept_id: str | None,
    target_topic: str | None,
    target_role: str | None,
    evidence_text: str,
    source_chunk_ids: Sequence[int],
    reviewer_reason: str,
    source_transcript: str,
    spec_version: str,
    reviewed_by: str = "streamlit",
) -> dict[str, Any]:
    """
    Persist one explicit final-topic human correction using the existing
    detected-topic edit-memory service.

    This function only writes validated reviewer feedback. It does not change
    Module 3 detection/scoring and it does not decide whether the memory should
    be reused later. The future-run read side performs contextual comparison
    first and uses the existing Groq rationale validator only when ambiguous.
    """

    from app.db.repositories.detected_topic_edit_memory_repository import (
        DetectedTopicEditMemoryRepository,
    )
    from app.db.session import session_scope
    from app.services.detected_topic_edit_memory_service import (
        DetectedTopicEdit,
        DetectedTopicEditMemoryService,
    )

    with session_scope() as session:
        repository = DetectedTopicEditMemoryRepository(session)
        service = DetectedTopicEditMemoryService(repository)

        record = service.remember(
            DetectedTopicEdit(
                edit_action=edit_action,
                source_concept_id=source_concept_id,
                source_topic=source_topic,
                source_role=source_role,
                target_concept_id=target_concept_id,
                target_topic=target_topic,
                target_role=target_role,
                evidence_text=str(evidence_text or "").strip(),
                source_chunk_ids=tuple(int(value) for value in source_chunk_ids),
                reviewer_reason=str(reviewer_reason or "").strip(),
                source_transcript=str(source_transcript or "").strip() or None,
                spec_version=str(spec_version or "").strip(),
                reviewed_by=str(reviewed_by or "").strip() or None,
            )
        )

        return {
            "memory_id": int(record.id),
            "edit_action": record.edit_action,
            "source_concept_id": record.source_concept_id,
            "source_topic": record.source_topic,
            "source_role": record.source_role,
            "target_concept_id": record.target_concept_id,
            "target_topic": record.target_topic,
            "target_role": record.target_role,
            "reviewer_reason": record.reviewer_reason,
            "spec_version": record.spec_version,
            "reviewer_approved": bool(record.reviewer_approved),
            "validation_status": record.validation_status,
        }
