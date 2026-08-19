from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.topic_human_review import TopicHumanReview


class TopicHumanReviewRepository:
    """Repository for provisional mappings and reviewer decisions."""

    VALID_STATUSES = {"pending", "approved", "corrected", "rejected"}

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_id(self, review_id: int) -> TopicHumanReview | None:
        return self.session.get(TopicHumanReview, review_id)

    def create_or_refresh_pending(
        self,
        *,
        cache_key: str,
        normalized_topic: str,
        original_topic: str,
        evidence_hash: str,
        evidence_text: str,
        candidate_concept_ids: Sequence[str],
        qdrant_candidates: Sequence[dict[str, Any]],
        proposed_decision: str,
        proposed_mapped_concept_id: str | None,
        confidence: float,
        confidence_band: str,
        reason: str,
        spec_version: str,
        source_transcript: str | None = None,
        source_chunk_ids: Sequence[int] = (),
        memory_lookup_result: str = "miss",
        memory_source_id: int | None = None,
        model_name: str | None = None,
        prompt_version: str | None = None,
    ) -> TopicHumanReview:
        """
        Create a review row, or refresh the existing transcript/topic row.

        The current database has a unique index on
        (source_transcript, normalized_topic), so repeated pipeline runs reuse
        that review row while the append-only decision log preserves history.
        """

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not spec_version.strip():
            raise ValueError("spec_version is required.")

        existing: TopicHumanReview | None = None
        if source_transcript:
            statement = select(TopicHumanReview).where(
                TopicHumanReview.source_transcript == source_transcript.strip(),
                TopicHumanReview.normalized_topic == normalized_topic.strip(),
            )
            existing = self.session.execute(statement).scalar_one_or_none()

        if existing is None:
            record = TopicHumanReview(
                cache_key=cache_key,
                normalized_topic=normalized_topic.strip(),
                original_topic=original_topic.strip(),
                evidence_hash=evidence_hash,
                evidence_text=evidence_text.strip(),
                source_transcript=(
                    source_transcript.strip() if source_transcript else None
                ),
                source_chunk_ids=list(source_chunk_ids),
                memory_lookup_result=memory_lookup_result,
                memory_source_id=memory_source_id,
                candidate_concept_ids=list(candidate_concept_ids),
                qdrant_candidates=list(qdrant_candidates),
                proposed_decision=proposed_decision,
                proposed_mapped_concept_id=proposed_mapped_concept_id,
                confidence=float(confidence),
                confidence_band=confidence_band.strip(),
                reason=reason.strip(),
                model_name=model_name.strip() if model_name else None,
                prompt_version=prompt_version.strip() if prompt_version else None,
                status="pending",
                spec_version=spec_version.strip(),
            )
            self.session.add(record)
        else:
            record = existing
            record.cache_key = cache_key
            record.original_topic = original_topic.strip()
            record.evidence_hash = evidence_hash
            record.evidence_text = evidence_text.strip()
            record.source_chunk_ids = list(source_chunk_ids)
            record.memory_lookup_result = memory_lookup_result
            record.memory_source_id = memory_source_id
            record.candidate_concept_ids = list(candidate_concept_ids)
            record.qdrant_candidates = list(qdrant_candidates)
            record.proposed_decision = proposed_decision
            record.proposed_mapped_concept_id = proposed_mapped_concept_id
            record.confidence = float(confidence)
            record.confidence_band = confidence_band.strip()
            record.reason = reason.strip()
            record.model_name = model_name.strip() if model_name else None
            record.prompt_version = (
                prompt_version.strip() if prompt_version else None
            )
            record.status = "pending"
            record.corrected_decision = None
            record.corrected_mapped_concept_id = None
            record.correction_reason = None
            record.review_notes = None
            record.reviewed_by = None
            record.reviewed_at = None
            record.spec_version = spec_version.strip()

        self.session.flush()
        self.session.refresh(record)
        return record

    def list_pending(
        self,
        *,
        spec_version: str | None = None,
        limit: int = 100,
    ) -> list[TopicHumanReview]:
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        statement = select(TopicHumanReview).where(
            TopicHumanReview.status == "pending"
        )
        if spec_version is not None:
            statement = statement.where(
                TopicHumanReview.spec_version == spec_version
            )
        statement = statement.order_by(TopicHumanReview.created_at.asc()).limit(limit)
        return list(self.session.execute(statement).scalars())

    def approve(
        self,
        review_id: int,
        *,
        reviewed_by: str,
        review_notes: str | None = None,
    ) -> TopicHumanReview:
        record = self._require_pending(review_id)
        if not reviewed_by.strip():
            raise ValueError("reviewed_by is required.")

        record.status = "approved"
        record.reviewed_by = reviewed_by.strip()
        record.review_notes = review_notes.strip() if review_notes else None
        record.corrected_decision = None
        record.corrected_mapped_concept_id = None
        record.correction_reason = None
        record.reviewed_at = self._database_now()

        self.session.flush()
        self.session.refresh(record)
        return record

    def correct(
        self,
        review_id: int,
        *,
        corrected_decision: str,
        corrected_mapped_concept_id: str | None,
        correction_reason: str,
        reviewed_by: str,
        review_notes: str | None = None,
    ) -> TopicHumanReview:
        record = self._require_pending(review_id)

        if not corrected_decision.strip():
            raise ValueError("corrected_decision is required.")
        if not correction_reason.strip():
            raise ValueError("correction_reason is required for Correct.")
        if not reviewed_by.strip():
            raise ValueError("reviewed_by is required.")

        if corrected_decision in {"mapped", "resolved_by_module3"}:
            if not corrected_mapped_concept_id:
                raise ValueError(
                    f"{corrected_decision} requires corrected_mapped_concept_id."
                )
        elif corrected_decision == "out_of_syllabus":
            if corrected_mapped_concept_id is not None:
                raise ValueError(
                    "out_of_syllabus cannot have corrected_mapped_concept_id."
                )

        record.status = "corrected"
        record.corrected_decision = corrected_decision.strip()
        record.corrected_mapped_concept_id = corrected_mapped_concept_id
        record.correction_reason = correction_reason.strip()
        record.review_notes = review_notes.strip() if review_notes else None
        record.reviewed_by = reviewed_by.strip()
        record.reviewed_at = self._database_now()

        self.session.flush()
        self.session.refresh(record)
        return record

    def reject(
        self,
        review_id: int,
        *,
        rejection_reason: str,
        reviewed_by: str,
    ) -> TopicHumanReview:
        record = self._require_pending(review_id)
        if not rejection_reason.strip():
            raise ValueError("rejection_reason is required for Reject.")
        if not reviewed_by.strip():
            raise ValueError("reviewed_by is required.")

        record.status = "rejected"
        record.review_notes = rejection_reason.strip()
        record.corrected_decision = None
        record.corrected_mapped_concept_id = None
        record.correction_reason = None
        record.reviewed_by = reviewed_by.strip()
        record.reviewed_at = self._database_now()

        self.session.flush()
        self.session.refresh(record)
        return record

    def _require_pending(self, review_id: int) -> TopicHumanReview:
        record = self.get_by_id(review_id)
        if record is None:
            raise KeyError(f"No topic human review record with id {review_id}.")
        if record.status != "pending":
            raise ValueError(
                f"Review {review_id} has already been '{record.status}'."
            )
        return record

    def _database_now(self):
        from sqlalchemy import func, select

        return self.session.execute(select(func.now())).scalar_one()
