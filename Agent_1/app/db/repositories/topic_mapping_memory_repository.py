from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import (
    delete,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models.topic_mapping_memory import (
    TopicMappingDecision,
    TopicMappingMemory,
    TopicMappingValidationStatus,
)


class TopicMappingMemoryRepository:
    """
    PostgreSQL repository for reusable Module 4 mapping decisions.

    Transaction ownership remains outside this repository. Methods flush
    changes but do not commit them.
    """

    REUSABLE_STATUSES = {
        "validated",
        "human_corrected",
    }

    CACHEABLE_DECISIONS = {
        "mapped",
        "out_of_syllabus",
        "resolved_by_module3",
    }

    def __init__(
        self,
        session: Session,
    ) -> None:
        self.session = session

    def get_by_id(
        self,
        record_id: int,
    ) -> TopicMappingMemory | None:
        """
        Return one memory record by primary key.
        """

        return self.session.get(
            TopicMappingMemory,
            record_id,
        )

    def get_by_cache_key(
        self,
        cache_key: str,
        *,
        reusable_only: bool = True,
    ) -> TopicMappingMemory | None:
        """
        Find an exact mapping-memory record.

        When reusable_only is True, disabled records are ignored.
        """

        statement = select(
            TopicMappingMemory
        ).where(
            TopicMappingMemory.cache_key
            == cache_key
        )

        if reusable_only:
            statement = statement.where(
                TopicMappingMemory.validation_status.in_(
                    self.REUSABLE_STATUSES
                )
            )

        return self.session.execute(
            statement
        ).scalar_one_or_none()

    def get_and_mark_used(
        self,
        cache_key: str,
    ) -> TopicMappingMemory | None:
        """
        Load a reusable memory record and increment its usage statistics.

        This should be used for real cache hits in Module 4.
        """

        record = self.get_by_cache_key(
            cache_key,
            reusable_only=True,
        )

        if record is None:
            return None

        statement = (
            update(TopicMappingMemory)
            .where(
                TopicMappingMemory.id
                == record.id
            )
            .values(
                hit_count=(
                    TopicMappingMemory.hit_count
                    + 1
                ),
                last_used_at=func.now(),
                updated_at=func.now(),
            )
        )

        self.session.execute(
            statement
        )

        self.session.flush()

        self.session.refresh(
            record
        )

        return record

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
        source_transcript: str | None = None,
        source_chunk_ids: Sequence[int] = (),
        validation_status: TopicMappingValidationStatus = (
            "validated"
        ),
    ) -> TopicMappingMemory:
        """
        Insert or update one validated mapping decision.

        Existing hit_count and created_at values are preserved when the same
        cache key is updated.

        API errors and unresolved needs-review outcomes must not be passed to
        this method.
        """

        self._validate_mapping_input(
            cache_key=cache_key,
            evidence_hash=evidence_hash,
            decision=decision,
            mapped_concept_id=mapped_concept_id,
            confidence=confidence,
            validation_status=validation_status,
        )

        values = {
            "cache_key": cache_key,
            "normalized_topic": (
                normalized_topic.strip()
            ),
            "original_topic": (
                original_topic.strip()
            ),
            "evidence_hash": evidence_hash,
            "evidence_text": evidence_text.strip(),
            "candidate_concept_ids": list(
                candidate_concept_ids
            ),
            "module3_concept_ids": list(
                module3_concept_ids
            ),
            "decision": decision,
            "mapped_concept_id": mapped_concept_id,
            "confidence": float(confidence),
            "reason": reason.strip(),
            "model_name": model_name.strip(),
            "prompt_version": (
                prompt_version.strip()
            ),
            "validation_status": (
                validation_status
            ),
            "source_transcript": (
                source_transcript.strip()
                if source_transcript
                else None
            ),
            "source_chunk_ids": list(
                source_chunk_ids
            ),
        }

        insert_statement = insert(
            TopicMappingMemory
        ).values(
            **values
        )

        upsert_statement = (
            insert_statement
            .on_conflict_do_update(
                index_elements=[
                    TopicMappingMemory.cache_key
                ],
                set_={
                    "normalized_topic": (
                        insert_statement
                        .excluded
                        .normalized_topic
                    ),
                    "original_topic": (
                        insert_statement
                        .excluded
                        .original_topic
                    ),
                    "evidence_hash": (
                        insert_statement
                        .excluded
                        .evidence_hash
                    ),
                    "evidence_text": (
                        insert_statement
                        .excluded
                        .evidence_text
                    ),
                    "candidate_concept_ids": (
                        insert_statement
                        .excluded
                        .candidate_concept_ids
                    ),
                    "module3_concept_ids": (
                        insert_statement
                        .excluded
                        .module3_concept_ids
                    ),
                    "decision": (
                        insert_statement
                        .excluded
                        .decision
                    ),
                    "mapped_concept_id": (
                        insert_statement
                        .excluded
                        .mapped_concept_id
                    ),
                    "confidence": (
                        insert_statement
                        .excluded
                        .confidence
                    ),
                    "reason": (
                        insert_statement
                        .excluded
                        .reason
                    ),
                    "model_name": (
                        insert_statement
                        .excluded
                        .model_name
                    ),
                    "prompt_version": (
                        insert_statement
                        .excluded
                        .prompt_version
                    ),
                    "validation_status": (
                        insert_statement
                        .excluded
                        .validation_status
                    ),
                    "source_transcript": (
                        insert_statement
                        .excluded
                        .source_transcript
                    ),
                    "source_chunk_ids": (
                        insert_statement
                        .excluded
                        .source_chunk_ids
                    ),
                    "updated_at": func.now(),
                },
            )
            .returning(
                TopicMappingMemory.id
            )
        )

        record_id = self.session.execute(
            upsert_statement
        ).scalar_one()

        self.session.flush()

        record = self.session.get(
            TopicMappingMemory,
            record_id,
        )

        if record is None:
            raise RuntimeError(
                "Topic mapping memory upsert succeeded, "
                "but the record could not be reloaded."
            )

        return record

    def mark_human_corrected(
        self,
        record_id: int,
        *,
        decision: TopicMappingDecision,
        mapped_concept_id: str | None,
        confidence: float,
        reason: str,
    ) -> TopicMappingMemory:
        """
        Replace an automated result with a human-reviewed decision.
        """

        self._validate_decision_mapping(
            decision=decision,
            mapped_concept_id=mapped_concept_id,
        )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        statement = (
            update(TopicMappingMemory)
            .where(
                TopicMappingMemory.id
                == record_id
            )
            .values(
                decision=decision,
                mapped_concept_id=(
                    mapped_concept_id
                ),
                confidence=float(confidence),
                reason=reason.strip(),
                validation_status=(
                    "human_corrected"
                ),
                updated_at=func.now(),
            )
        )

        result = self.session.execute(
            statement
        )

        if result.rowcount == 0:
            raise KeyError(
                "No topic mapping memory record "
                f"with id {record_id}."
            )

        self.session.flush()

        record = self.session.get(
            TopicMappingMemory,
            record_id,
        )

        if record is None:
            raise RuntimeError(
                "Human correction succeeded, but the "
                "record could not be reloaded."
            )

        return record

    def set_validation_status(
        self,
        record_id: int,
        status: TopicMappingValidationStatus,
    ) -> TopicMappingMemory:
        """
        Change whether a stored decision may be reused.
        """

        statement = (
            update(TopicMappingMemory)
            .where(
                TopicMappingMemory.id
                == record_id
            )
            .values(
                validation_status=status,
                updated_at=func.now(),
            )
        )

        result = self.session.execute(
            statement
        )

        if result.rowcount == 0:
            raise KeyError(
                "No topic mapping memory record "
                f"with id {record_id}."
            )

        self.session.flush()

        record = self.session.get(
            TopicMappingMemory,
            record_id,
        )

        if record is None:
            raise RuntimeError(
                "Status update succeeded, but the "
                "record could not be reloaded."
            )

        return record

    def list_records(
        self,
        *,
        decision: TopicMappingDecision | None = None,
        validation_status: (
            TopicMappingValidationStatus | None
        ) = None,
        mapped_concept_id: str | None = None,
        normalized_topic: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TopicMappingMemory]:
        """
        List mapping-memory records with optional filters.
        """

        if limit < 1:
            raise ValueError(
                "limit must be at least 1."
            )

        if offset < 0:
            raise ValueError(
                "offset cannot be negative."
            )

        statement = select(
            TopicMappingMemory
        )

        if decision is not None:
            statement = statement.where(
                TopicMappingMemory.decision
                == decision
            )

        if validation_status is not None:
            statement = statement.where(
                TopicMappingMemory
                .validation_status
                == validation_status
            )

        if mapped_concept_id is not None:
            statement = statement.where(
                TopicMappingMemory
                .mapped_concept_id
                == mapped_concept_id
            )

        if normalized_topic is not None:
            statement = statement.where(
                TopicMappingMemory
                .normalized_topic
                == normalized_topic
            )

        statement = (
            statement
            .order_by(
                TopicMappingMemory
                .updated_at
                .desc()
            )
            .limit(limit)
            .offset(offset)
        )

        return list(
            self.session.execute(
                statement
            ).scalars()
        )

    def delete_mapping(
        self,
        record_id: int,
    ) -> None:
        """
        Permanently delete one mapping-memory record.
        """

        statement = delete(
            TopicMappingMemory
        ).where(
            TopicMappingMemory.id
            == record_id
        )

        result = self.session.execute(
            statement
        )

        if result.rowcount == 0:
            raise KeyError(
                "No topic mapping memory record "
                f"with id {record_id}."
            )

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
        validation_status: (
            TopicMappingValidationStatus
        ),
    ) -> None:
        if len(cache_key) != 64:
            raise ValueError(
                "cache_key must be a 64-character "
                "SHA-256 hexadecimal string."
            )

        if len(evidence_hash) != 64:
            raise ValueError(
                "evidence_hash must be a 64-character "
                "SHA-256 hexadecimal string."
            )

        if decision not in cls.CACHEABLE_DECISIONS:
            raise ValueError(
                f"Decision '{decision}' cannot be cached."
            )

        if validation_status not in {
            "validated",
            "human_corrected",
            "disabled",
        }:
            raise ValueError(
                "Invalid validation_status."
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
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
        if decision in {
            "mapped",
            "resolved_by_module3",
        }:
            if not mapped_concept_id:
                raise ValueError(
                    f"{decision} requires "
                    "mapped_concept_id."
                )

        elif decision == "out_of_syllabus":
            if mapped_concept_id is not None:
                raise ValueError(
                    "out_of_syllabus cannot have "
                    "mapped_concept_id."
                )