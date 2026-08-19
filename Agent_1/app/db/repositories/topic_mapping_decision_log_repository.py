from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.topic_mapping_decision_log import TopicMappingDecisionLog


class TopicMappingDecisionLogRepository:
    """Append-only audit-log repository. No update/delete API is exposed."""

    VALID_ACTORS = {"system", "human"}
    VALID_ACTIONS = {
        "memory_hit",
        "memory_miss",
        "proposed",
        "sent_for_review",
        "approve",
        "correct",
        "reject",
        "reuse",
        "promoted",
    }

    def __init__(self, session: Session) -> None:
        self.session = session

    def log(
        self,
        *,
        normalized_topic: str,
        decision_stage: str,
        actor_type: str,
        action: str,
        spec_version: str,
        review_id: int | None = None,
        memory_id: int | None = None,
        source_memory_id: int | None = None,
        pipeline_run_id: str | None = None,
        cache_key: str | None = None,
        source_transcript: str | None = None,
        source_chunk_ids: Sequence[int] = (),
        decision: str | None = None,
        mapped_concept_id: str | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        decided_by: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> TopicMappingDecisionLog:
        """Append one decision row with source memory/spec/timestamp context."""

        actor_type = actor_type.strip().lower()
        action = action.strip().lower()

        if actor_type not in self.VALID_ACTORS:
            raise ValueError(f"Invalid actor_type: {actor_type}")
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action: {action}")
        if not normalized_topic.strip():
            raise ValueError("normalized_topic is required.")
        if not decision_stage.strip():
            raise ValueError("decision_stage is required.")
        if not spec_version.strip():
            raise ValueError("spec_version is required.")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        record = TopicMappingDecisionLog(
            review_id=review_id,
            memory_id=memory_id,
            source_memory_id=source_memory_id,
            pipeline_run_id=pipeline_run_id,
            cache_key=cache_key,
            normalized_topic=normalized_topic.strip(),
            source_transcript=(
                source_transcript.strip() if source_transcript else None
            ),
            source_chunk_ids=list(source_chunk_ids),
            decision_stage=decision_stage.strip(),
            actor_type=actor_type,
            action=action,
            decision=decision,
            mapped_concept_id=mapped_concept_id,
            confidence=float(confidence) if confidence is not None else None,
            reason=reason.strip() if reason else None,
            decided_by=decided_by.strip() if decided_by else None,
            details=details or {},
            spec_version=spec_version.strip(),
        )

        self.session.add(record)
        self.session.flush()
        self.session.refresh(record)
        return record

    def list_for_review(
        self,
        review_id: int,
    ) -> list[TopicMappingDecisionLog]:
        statement = (
            select(TopicMappingDecisionLog)
            .where(TopicMappingDecisionLog.review_id == review_id)
            .order_by(TopicMappingDecisionLog.created_at.asc())
        )
        return list(self.session.execute(statement).scalars())

    def list_for_topic(
        self,
        normalized_topic: str,
        *,
        limit: int = 100,
    ) -> list[TopicMappingDecisionLog]:
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        statement = (
            select(TopicMappingDecisionLog)
            .where(
                TopicMappingDecisionLog.normalized_topic
                == normalized_topic.strip()
            )
            .order_by(TopicMappingDecisionLog.created_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(statement).scalars())
