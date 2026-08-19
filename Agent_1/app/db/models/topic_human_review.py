from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


TopicHumanReviewStatus = Literal[
    "pending",
    "approved",
    "corrected",
    "rejected",
]


class TopicHumanReview(Base):
    """Provisional topic mappings waiting for human review."""

    __tablename__ = "topic_human_review"

    __table_args__ = (
        CheckConstraint(
            """
            status <> 'corrected'
            OR (
                corrected_decision IS NOT NULL
                AND correction_reason IS NOT NULL
                AND LENGTH(TRIM(correction_reason)) > 0
            )
            """,
            name="ck_topic_human_review_correction",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_topic_human_review_confidence",
        ),
        CheckConstraint(
            "jsonb_typeof(candidate_concept_ids) = 'array'",
            name="ck_topic_human_review_candidate_ids_array",
        ),
        CheckConstraint(
            "jsonb_typeof(qdrant_candidates) = 'array'",
            name="ck_topic_human_review_qdrant_candidates_array",
        ),
        CheckConstraint(
            "jsonb_typeof(source_chunk_ids) = 'array'",
            name="ck_topic_human_review_chunk_ids_array",
        ),
        Index(
            "uq_topic_human_review_transcript_topic",
            "source_transcript",
            "normalized_topic",
            unique=True,
            postgresql_where=text("source_transcript IS NOT NULL"),
        ),
        Index("ix_topic_human_review_status", "status"),
        Index(
            "ix_topic_human_review_confidence",
            "confidence_band",
            "confidence",
        ),
        Index(
            "ix_topic_human_review_spec_status",
            "spec_version",
            "status",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    cache_key: Mapped[str] = mapped_column(String(64), nullable=False)
    normalized_topic: Mapped[str] = mapped_column(Text, nullable=False)
    original_topic: Mapped[str] = mapped_column(Text, nullable=False)

    evidence_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)

    source_transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_chunk_ids: Mapped[list[int]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    memory_lookup_result: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        default="miss",
        server_default="miss",
    )

    memory_source_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("topic_mapping_memory.id", ondelete="SET NULL"),
        nullable=True,
    )

    candidate_concept_ids: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    qdrant_candidates: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    proposed_decision: Mapped[str] = mapped_column(String(40), nullable=False)
    proposed_mapped_concept_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    confidence_band: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="unknown",
        server_default="unknown",
    )

    reason: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
        server_default="pending",
    )

    corrected_decision: Mapped[str | None] = mapped_column(
        String(40),
        nullable=True,
    )
    corrected_mapped_concept_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    correction_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    spec_version: Mapped[str] = mapped_column(String(80), nullable=False)

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cache_key": self.cache_key,
            "normalized_topic": self.normalized_topic,
            "original_topic": self.original_topic,
            "evidence_hash": self.evidence_hash,
            "evidence_text": self.evidence_text,
            "source_transcript": self.source_transcript,
            "source_chunk_ids": list(self.source_chunk_ids),
            "memory_lookup_result": self.memory_lookup_result,
            "memory_source_id": self.memory_source_id,
            "candidate_concept_ids": list(self.candidate_concept_ids),
            "qdrant_candidates": list(self.qdrant_candidates),
            "proposed_decision": self.proposed_decision,
            "proposed_mapped_concept_id": self.proposed_mapped_concept_id,
            "confidence": self.confidence,
            "confidence_band": self.confidence_band,
            "reason": self.reason,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "status": self.status,
            "corrected_decision": self.corrected_decision,
            "corrected_mapped_concept_id": self.corrected_mapped_concept_id,
            "correction_reason": self.correction_reason,
            "review_notes": self.review_notes,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": (
                self.reviewed_at.isoformat() if self.reviewed_at else None
            ),
            "spec_version": self.spec_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
