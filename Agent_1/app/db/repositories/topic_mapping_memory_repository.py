from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models.topic_mapping_memory import (
    TopicMappingDecision,
    TopicMappingMemory,
    TopicMappingValidationStatus,
)


class TopicMappingMemoryRepository:
    """Repository for safe, reviewer-approved topic-mapping memory."""

    REUSABLE_STATUSES = {"validated", "human_corrected"}
    CACHEABLE_DECISIONS = {
        "mapped",
        "out_of_syllabus",
        "resolved_by_module3",
    }

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(self, record_id: int) -> TopicMappingMemory | None:
        return self.session.get(TopicMappingMemory, record_id)

    def get_by_cache_key(
        self,
        cache_key: str,
        *,
        spec_version: str,
        reusable_only: bool = True,
    ) -> TopicMappingMemory | None:
        """
        Find an exact memory record for the active specification.

        reusable_only=True enforces the updated safety contract:
        reviewer-approved + active spec version + reusable status.
        """

        statement = select(TopicMappingMemory).where(
            TopicMappingMemory.cache_key == cache_key,
            TopicMappingMemory.spec_version == spec_version,
        )

        if reusable_only:
            statement = statement.where(
                TopicMappingMemory.reviewer_approved.is_(True),
                TopicMappingMemory.validation_status.in_(self.REUSABLE_STATUSES),
            )

        return self.session.execute(statement).scalar_one_or_none()

    def find_reusable_candidates(
        self,
        *,
        normalized_topic: str,
        spec_version: str,
        limit: int = 20,
    ) -> list[TopicMappingMemory]:
        """
        Return safe historical candidates for evidence/context comparison.

        This method deliberately does not decide whether a memory is a strong
        match. The evidence/reviewer-reason similarity gate belongs in the
        memory compatibility service added in the next implementation step.
        """

        if limit < 1:
            raise ValueError("limit must be at least 1.")

        statement = (
            select(TopicMappingMemory)
            .where(
                TopicMappingMemory.normalized_topic == normalized_topic.strip(),
                TopicMappingMemory.spec_version == spec_version.strip(),
                TopicMappingMemory.reviewer_approved.is_(True),
                TopicMappingMemory.validation_status.in_(self.REUSABLE_STATUSES),
            )
            .order_by(TopicMappingMemory.updated_at.desc())
            .limit(limit)
        )

        return list(self.session.execute(statement).scalars())

    def mark_used(self, record_id: int) -> TopicMappingMemory:
        """Increment usage statistics only after a strong memory match."""

        statement = (
            update(TopicMappingMemory)
            .where(TopicMappingMemory.id == record_id)
            .values(
                hit_count=TopicMappingMemory.hit_count + 1,
                last_used_at=func.now(),
                updated_at=func.now(),
            )
        )

        result = self.session.execute(statement)
        if result.rowcount == 0:
            raise KeyError(f"No topic mapping memory record with id {record_id}.")

        self.session.flush()
        record = self.session.get(TopicMappingMemory, record_id)
        if record is None:
            raise RuntimeError("Memory usage update succeeded but reload failed.")
        return record

    def get_and_mark_used(
        self,
        cache_key: str,
        *,
        spec_version: str,
    ) -> TopicMappingMemory | None:
        """Exact safe-hit helper; use only after compatibility is established."""

        record = self.get_by_cache_key(
            cache_key,
            spec_version=spec_version,
            reusable_only=True,
        )
        if record is None:
            return None
        return self.mark_used(record.id)

    def upsert_mapping(
        self,
        *,
        cache_key: str,
        normalized_topic: str,
        original_topic: str,
        evidence_hash: str,
        evidence_text: str,
        candidate_concept_ids: Sequence[str],
        module3_concept_ids: Sequence[str],
        decision: TopicMappingDecision,
        mapped_concept_id: str | None,
        confidence: float,
        reason: str,
        model_name: str,
        prompt_version: str,
        spec_version: str,
        reviewer_approved: bool,
        reviewer_reason: str | None,
        reviewed_by: str,
        reviewed_at: datetime | None = None,
        source_transcript: str | None = None,
        source_chunk_ids: Sequence[int] = (),
        validation_status: TopicMappingValidationStatus = "validated",
    ) -> TopicMappingMemory:
        """
        Promote an approved/corrected review into reusable memory.

        Fresh automated/provisional mappings must never call this method with
        reviewer_approved=False. They belong in topic_human_review instead.
        """

        self._validate_mapping_input(
            cache_key=cache_key,
            evidence_hash=evidence_hash,
            decision=decision,
            mapped_concept_id=mapped_concept_id,
            confidence=confidence,
            validation_status=validation_status,
            spec_version=spec_version,
            reviewer_approved=reviewer_approved,
            reviewer_reason=reviewer_reason,
            reviewed_by=reviewed_by,
        )

        reviewed_at = reviewed_at or datetime.now(timezone.utc)

        values = {
            "cache_key": cache_key,
            "normalized_topic": normalized_topic.strip(),
            "original_topic": original_topic.strip(),
            "evidence_hash": evidence_hash,
            "evidence_text": evidence_text.strip(),
            "candidate_concept_ids": list(candidate_concept_ids),
            "module3_concept_ids": list(module3_concept_ids),
            "decision": decision,
            "mapped_concept_id": mapped_concept_id,
            "confidence": float(confidence),
            "reason": reason.strip(),
            "model_name": model_name.strip(),
            "prompt_version": prompt_version.strip(),
            "validation_status": validation_status,
            "source_transcript": (
                source_transcript.strip() if source_transcript else None
            ),
            "source_chunk_ids": list(source_chunk_ids),
            "spec_version": spec_version.strip(),
            "reviewer_approved": reviewer_approved,
            "reviewer_reason": (
                reviewer_reason.strip() if reviewer_reason else None
            ),
            "reviewed_by": reviewed_by.strip(),
            "reviewed_at": reviewed_at,
        }

        insert_statement = insert(TopicMappingMemory).values(**values)
        upsert_statement = (
            insert_statement.on_conflict_do_update(
                index_elements=[TopicMappingMemory.cache_key],
                set_={
                    "normalized_topic": insert_statement.excluded.normalized_topic,
                    "original_topic": insert_statement.excluded.original_topic,
                    "evidence_hash": insert_statement.excluded.evidence_hash,
                    "evidence_text": insert_statement.excluded.evidence_text,
                    "candidate_concept_ids": (
                        insert_statement.excluded.candidate_concept_ids
                    ),
                    "module3_concept_ids": (
                        insert_statement.excluded.module3_concept_ids
                    ),
                    "decision": insert_statement.excluded.decision,
                    "mapped_concept_id": insert_statement.excluded.mapped_concept_id,
                    "confidence": insert_statement.excluded.confidence,
                    "reason": insert_statement.excluded.reason,
                    "model_name": insert_statement.excluded.model_name,
                    "prompt_version": insert_statement.excluded.prompt_version,
                    "validation_status": (
                        insert_statement.excluded.validation_status
                    ),
                    "source_transcript": (
                        insert_statement.excluded.source_transcript
                    ),
                    "source_chunk_ids": insert_statement.excluded.source_chunk_ids,
                    "spec_version": insert_statement.excluded.spec_version,
                    "reviewer_approved": (
                        insert_statement.excluded.reviewer_approved
                    ),
                    "reviewer_reason": insert_statement.excluded.reviewer_reason,
                    "reviewed_by": insert_statement.excluded.reviewed_by,
                    "reviewed_at": insert_statement.excluded.reviewed_at,
                    "updated_at": func.now(),
                },
            ).returning(TopicMappingMemory.id)
        )

        record_id = self.session.execute(upsert_statement).scalar_one()
        self.session.flush()

        record = self.session.get(TopicMappingMemory, record_id)
        if record is None:
            raise RuntimeError(
                "Topic mapping memory upsert succeeded, but reload failed."
            )
        return record

    def mark_human_corrected(
        self,
        record_id: int,
        *,
        decision: TopicMappingDecision,
        mapped_concept_id: str | None,
        confidence: float,
        correction_reason: str,
        reviewed_by: str,
        spec_version: str,
    ) -> TopicMappingMemory:
        """Correct an existing memory row and retain the human reason."""

        self._validate_decision_mapping(
            decision=decision,
            mapped_concept_id=mapped_concept_id,
        )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not correction_reason.strip():
            raise ValueError("correction_reason is required.")
        if not reviewed_by.strip():
            raise ValueError("reviewed_by is required.")
        if not spec_version.strip():
            raise ValueError("spec_version is required.")

        statement = (
            update(TopicMappingMemory)
            .where(TopicMappingMemory.id == record_id)
            .values(
                decision=decision,
                mapped_concept_id=mapped_concept_id,
                confidence=float(confidence),
                reason=correction_reason.strip(),
                reviewer_reason=correction_reason.strip(),
                reviewer_approved=True,
                reviewed_by=reviewed_by.strip(),
                reviewed_at=func.now(),
                spec_version=spec_version.strip(),
                validation_status="human_corrected",
                updated_at=func.now(),
            )
        )

        result = self.session.execute(statement)
        if result.rowcount == 0:
            raise KeyError(f"No topic mapping memory record with id {record_id}.")

        self.session.flush()
        record = self.session.get(TopicMappingMemory, record_id)
        if record is None:
            raise RuntimeError("Human correction succeeded but reload failed.")
        return record

    def set_validation_status(
        self,
        record_id: int,
        status: TopicMappingValidationStatus,
    ) -> TopicMappingMemory:
        """Enable/disable a stored mapping without deleting its history."""

        if status not in {"validated", "human_corrected", "disabled"}:
            raise ValueError("Invalid validation_status.")

        values: dict[str, object] = {
            "validation_status": status,
            "updated_at": func.now(),
        }
        if status == "disabled":
            values["reviewer_approved"] = False

        statement = (
            update(TopicMappingMemory)
            .where(TopicMappingMemory.id == record_id)
            .values(**values)
        )

        result = self.session.execute(statement)
        if result.rowcount == 0:
            raise KeyError(f"No topic mapping memory record with id {record_id}.")

        self.session.flush()
        record = self.session.get(TopicMappingMemory, record_id)
        if record is None:
            raise RuntimeError("Status update succeeded but reload failed.")
        return record

    def list_records(
        self,
        *,
        decision: TopicMappingDecision | None = None,
        validation_status: TopicMappingValidationStatus | None = None,
        mapped_concept_id: str | None = None,
        normalized_topic: str | None = None,
        spec_version: str | None = None,
        reviewer_approved: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TopicMappingMemory]:
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if offset < 0:
            raise ValueError("offset cannot be negative.")

        statement = select(TopicMappingMemory)

        if decision is not None:
            statement = statement.where(TopicMappingMemory.decision == decision)
        if validation_status is not None:
            statement = statement.where(
                TopicMappingMemory.validation_status == validation_status
            )
        if mapped_concept_id is not None:
            statement = statement.where(
                TopicMappingMemory.mapped_concept_id == mapped_concept_id
            )
        if normalized_topic is not None:
            statement = statement.where(
                TopicMappingMemory.normalized_topic == normalized_topic
            )
        if spec_version is not None:
            statement = statement.where(
                TopicMappingMemory.spec_version == spec_version
            )
        if reviewer_approved is not None:
            statement = statement.where(
                TopicMappingMemory.reviewer_approved.is_(reviewer_approved)
            )

        statement = (
            statement.order_by(TopicMappingMemory.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.execute(statement).scalars())

    def delete_mapping(self, record_id: int) -> None:
        statement = delete(TopicMappingMemory).where(
            TopicMappingMemory.id == record_id
        )
        result = self.session.execute(statement)
        if result.rowcount == 0:
            raise KeyError(f"No topic mapping memory record with id {record_id}.")
        self.session.flush()

    @classmethod
    def _validate_mapping_input(
        cls,
        *,
        cache_key: str,
        evidence_hash: str,
        decision: TopicMappingDecision,
        mapped_concept_id: str | None,
        confidence: float,
        validation_status: TopicMappingValidationStatus,
        spec_version: str,
        reviewer_approved: bool,
        reviewer_reason: str | None,
        reviewed_by: str,
    ) -> None:
        if len(cache_key) != 64:
            raise ValueError("cache_key must be a 64-character SHA-256 string.")
        if len(evidence_hash) != 64:
            raise ValueError("evidence_hash must be a 64-character SHA-256 string.")
        if decision not in cls.CACHEABLE_DECISIONS:
            raise ValueError(f"Decision '{decision}' cannot be cached.")
        if validation_status not in {
            "validated",
            "human_corrected",
            "disabled",
        }:
            raise ValueError("Invalid validation_status.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not spec_version.strip():
            raise ValueError("spec_version is required.")
        if not reviewer_approved:
            raise ValueError(
                "Only reviewer-approved decisions may be promoted to memory."
            )
        if validation_status == "disabled":
            raise ValueError("A disabled mapping cannot be promoted as approved.")
        if not reviewed_by.strip():
            raise ValueError("reviewed_by is required for reusable memory.")
        if validation_status == "human_corrected" and not (
            reviewer_reason and reviewer_reason.strip()
        ):
            raise ValueError(
                "reviewer_reason is required for a human-corrected mapping."
            )

        cls._validate_decision_mapping(
            decision=decision,
            mapped_concept_id=mapped_concept_id,
        )

    @staticmethod
    def _validate_decision_mapping(
        *,
        decision: TopicMappingDecision,
        mapped_concept_id: str | None,
    ) -> None:
        if decision in {"mapped", "resolved_by_module3"} and not mapped_concept_id:
            raise ValueError(f"{decision} requires mapped_concept_id.")
        if decision == "out_of_syllabus" and mapped_concept_id is not None:
            raise ValueError("out_of_syllabus cannot have mapped_concept_id.")
