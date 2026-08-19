from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DetectedTopicEditMemory(Base):
    """
    Reviewer-approved memory for edits made to Module 3's final topic list.

    This table is intentionally separate from ``topic_mapping_memory``.

    It stores contextual final-topic edits such as:
    - remove_topic
    - add_topic
    - replace_topic
    - change_role

    No edit is applied automatically merely because a topic name matches.
    Future reuse must still pass the separate contextual compatibility layer.
    """

    __tablename__ = "detected_topic_edit_memory"

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

    edit_action: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    source_concept_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    source_topic: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    source_role: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    target_concept_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    target_topic: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    target_role: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    evidence_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    evidence_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source_chunk_ids: Mapped[list[int]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )

    reviewer_reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source_transcript: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    spec_version: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
    )

    reviewed_by: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    validation_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="human_validated",
        server_default=text("'human_validated'"),
    )

    reviewer_approved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("TRUE"),
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("TRUE"),
    )

    hit_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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

    __table_args__ = (
        CheckConstraint(
            "edit_action IN "
            "('remove_topic', 'add_topic', 'replace_topic', 'change_role')",
            name="ck_detected_topic_edit_action",
        ),
        CheckConstraint(
            "source_role IS NULL "
            "OR source_role IN ('primary', 'supporting')",
            name="ck_detected_topic_source_role",
        ),
        CheckConstraint(
            "target_role IS NULL "
            "OR target_role IN ('primary', 'supporting')",
            name="ck_detected_topic_target_role",
        ),
        Index(
            "ix_detected_topic_edit_memory_source_concept",
            "source_concept_id",
        ),
        Index(
            "ix_detected_topic_edit_memory_target_concept",
            "target_concept_id",
        ),
        Index(
            "ix_detected_topic_edit_memory_spec_version",
            "spec_version",
        ),
        Index(
            "ix_detected_topic_edit_memory_reusable",
            "reviewer_approved",
            "is_active",
            "spec_version",
        ),
        Index(
            "ix_detected_topic_edit_memory_action",
            "edit_action",
        ),
    )

    def __repr__(self) -> str:
        return (
            "DetectedTopicEditMemory("
            f"id={self.id!r}, "
            f"edit_action={self.edit_action!r}, "
            f"source_concept_id={self.source_concept_id!r}, "
            f"target_concept_id={self.target_concept_id!r}, "
            f"reviewer_approved={self.reviewer_approved!r}, "
            f"is_active={self.is_active!r}"
            ")"
        )
