from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


TopicMappingDecision = Literal[
    "mapped",
    "out_of_syllabus",
    "resolved_by_module3",
]

TopicMappingValidationStatus = Literal[
    "validated",
    "human_corrected",
    "disabled",
]


class TopicMappingMemory(Base):
    """
    Stores reusable, validated topic-mapping decisions.

    Qdrant remains responsible for semantic syllabus retrieval.

    This table provides deterministic PostgreSQL memory for decisions that
    have already been validated by Python, the LLM, or a human reviewer.

    Unsafe outcomes such as API errors and unresolved needs-review cases
    should not be stored as reusable memory.
    """

    __tablename__ = "topic_mapping_memory"

    __table_args__ = (
        CheckConstraint(
            """
            decision IN (
                'mapped',
                'out_of_syllabus',
                'resolved_by_module3'
            )
            """,
            name="ck_topic_mapping_memory_decision",
        ),
        CheckConstraint(
            """
            validation_status IN (
                'validated',
                'human_corrected',
                'disabled'
            )
            """,
            name="ck_topic_mapping_memory_validation_status",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_topic_mapping_memory_confidence",
        ),
        CheckConstraint(
            "hit_count >= 0",
            name="ck_topic_mapping_memory_hit_count",
        ),
        CheckConstraint(
            """
            (
                decision IN (
                    'mapped',
                    'resolved_by_module3'
                )
                AND mapped_concept_id IS NOT NULL
            )
            OR
            (
                decision = 'out_of_syllabus'
                AND mapped_concept_id IS NULL
            )
            """,
            name="ck_topic_mapping_memory_mapped_concept",
        ),
        CheckConstraint(
            "jsonb_typeof(candidate_concept_ids) = 'array'",
            name="ck_topic_mapping_memory_candidate_ids_array",
        ),
        CheckConstraint(
            "jsonb_typeof(module3_concept_ids) = 'array'",
            name="ck_topic_mapping_memory_module3_ids_array",
        ),
        CheckConstraint(
            "jsonb_typeof(source_chunk_ids) = 'array'",
            name="ck_topic_mapping_memory_chunk_ids_array",
        ),
        Index(
            "ix_topic_mapping_memory_normalized_topic",
            "normalized_topic",
        ),
        Index(
            "ix_topic_mapping_memory_mapped_concept",
            "mapped_concept_id",
        ),
        Index(
            "ix_topic_mapping_memory_validation",
            "validation_status",
            "decision",
        ),
        Index(
            "ix_topic_mapping_memory_model_prompt",
            "model_name",
            "prompt_version",
        ),
        Index(
            "ix_topic_mapping_memory_last_used",
            "last_used_at",
        ),
        Index(
            "ix_topic_mapping_memory_candidates_gin",
            "candidate_concept_ids",
            postgresql_using="gin",
        ),
        Index(
            "ix_topic_mapping_memory_module3_concepts_gin",
            "module3_concept_ids",
            postgresql_using="gin",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    cache_key: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )

    normalized_topic: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    original_topic: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    evidence_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    evidence_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    candidate_concept_ids: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    module3_concept_ids: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    decision: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    mapped_concept_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    prompt_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    validation_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="validated",
        server_default="validated",
    )

    source_transcript: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    source_chunk_ids: Mapped[list[int]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    hit_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def is_reusable(self) -> bool:
        """
        Return whether the record may safely be reused by Module 4.
        """

        return self.validation_status in {
            "validated",
            "human_corrected",
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Return a serializable representation for notebook/report usage.
        """

        return {
            "id": self.id,
            "cache_key": self.cache_key,
            "normalized_topic": self.normalized_topic,
            "original_topic": self.original_topic,
            "evidence_hash": self.evidence_hash,
            "evidence_text": self.evidence_text,
            "candidate_concept_ids": list(
                self.candidate_concept_ids
            ),
            "module3_concept_ids": list(
                self.module3_concept_ids
            ),
            "decision": self.decision,
            "mapped_concept_id": self.mapped_concept_id,
            "confidence": self.confidence,
            "reason": self.reason,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "validation_status": self.validation_status,
            "source_transcript": self.source_transcript,
            "source_chunk_ids": list(
                self.source_chunk_ids
            ),
            "hit_count": self.hit_count,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
            "updated_at": (
                self.updated_at.isoformat()
                if self.updated_at
                else None
            ),
            "last_used_at": (
                self.last_used_at.isoformat()
                if self.last_used_at
                else None
            ),
        }