from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class TechnicalCorrection(Base):
    """
    Context-aware correction memory for technical ASR/spelling errors.
    """

    __tablename__ = "technical_corrections"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    original_phrase: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    original_key: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    corrected_phrase: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    corrected_key: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    subject: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="Computer Science",
        server_default=text("'Computer Science'"),
    )

    context_keywords: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )

    context_signature: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="candidate",
        server_default=text("'candidate'"),
    )

    source_model: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    times_seen: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
    )

    times_applied: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
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
            "status IN "
            "('candidate', 'approved', 'rejected', 'disabled')",
            name="ck_technical_corrections_status",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_technical_corrections_confidence",
        ),
        UniqueConstraint(
            "original_key",
            "corrected_key",
            "context_signature",
            name="uq_technical_correction_phrase_context",
        ),
        Index(
            "ix_technical_corrections_lookup",
            "original_key",
            "status",
        ),
    )

    def __repr__(self) -> str:
        return (
            "TechnicalCorrection("
            f"id={self.id!r}, "
            f"original_phrase={self.original_phrase!r}, "
            f"corrected_phrase={self.corrected_phrase!r}, "
            f"status={self.status!r}"
            ")"
        )