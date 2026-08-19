from __future__ import annotations

from datetime import datetime
from typing import Any

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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class TopicMappingDecisionLog(Base):
    """Append-only audit trail for mapping, review, and memory reuse."""

    __tablename__ = "topic_mapping_decision_log"

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_topic_decision_log_confidence",
        ),
        CheckConstraint(
            "jsonb_typeof(source_chunk_ids) = 'array'",
            name="ck_topic_decision_log_chunk_ids_array",
        ),
        CheckConstraint(
            "jsonb_typeof(details) = 'object'",
            name="ck_topic_decision_log_details_object",
        ),
        Index("ix_topic_decision_log_review", "review_id"),
        Index("ix_topic_decision_log_memory", "memory_id"),
        Index("ix_topic_decision_log_topic", "normalized_topic"),
        Index("ix_topic_decision_log_stage", "decision_stage", "action"),
        Index("ix_topic_decision_log_created", "created_at"),
        Index("ix_topic_decision_log_source_memory", "source_memory_id"),
        Index("ix_topic_decision_log_spec_version", "spec_version"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    review_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("topic_human_review.id", ondelete="SET NULL"),
        nullable=True,
    )

    # memory_id identifies the memory row created/updated by this decision.
    memory_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("topic_mapping_memory.id", ondelete="SET NULL"),
        nullable=True,
    )

    # source_memory_id identifies the memory row reused/considered as input.
    source_memory_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("topic_mapping_memory.id", ondelete="SET NULL"),
        nullable=True,
    )

    pipeline_run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    cache_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    normalized_topic: Mapped[str] = mapped_column(Text, nullable=False)
    source_transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_chunk_ids: Mapped[list[int]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    decision_stage: Mapped[str] = mapped_column(String(40), nullable=False)
    actor_type: Mapped[str] = mapped_column(String(20), nullable=False)
    action: Mapped[str] = mapped_column(String(40), nullable=False)

    decision: Mapped[str | None] = mapped_column(String(40), nullable=True)
    mapped_concept_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_by: Mapped[str | None] = mapped_column(Text, nullable=True)

    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    spec_version: Mapped[str] = mapped_column(String(80), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "review_id": self.review_id,
            "memory_id": self.memory_id,
            "source_memory_id": self.source_memory_id,
            "pipeline_run_id": self.pipeline_run_id,
            "cache_key": self.cache_key,
            "normalized_topic": self.normalized_topic,
            "source_transcript": self.source_transcript,
            "source_chunk_ids": list(self.source_chunk_ids),
            "decision_stage": self.decision_stage,
            "actor_type": self.actor_type,
            "action": self.action,
            "decision": self.decision,
            "mapped_concept_id": self.mapped_concept_id,
            "confidence": self.confidence,
            "reason": self.reason,
            "decided_by": self.decided_by,
            "details": dict(self.details),
            "spec_version": self.spec_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
