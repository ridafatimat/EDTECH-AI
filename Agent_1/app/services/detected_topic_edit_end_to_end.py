from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

import numpy as np

from app.db.models.detected_topic_edit_memory import (
    DetectedTopicEditMemory,
)
from app.db.repositories.detected_topic_edit_memory_repository import (
    DetectedTopicEditMemoryRepository,
)
from app.schemas.topic import (
    MergedTopic,
    Module3Result,
    TopicCandidate,
)
from app.services.syllabus_store import get_syllabus_store
from app.services.detected_topic_edit_embedding_adapter import (
    Agent1EditMemoryEmbeddingAdapter,
)
from app.services.detected_topic_edit_contextual_comparator import (
    DetectedTopicEditContextualComparator,
)
from app.services.detected_topic_edit_groq_reason_provider import (
    GroqReviewerReasonProvider,
)
from app.services.detected_topic_edit_overlay import (
    DetectedTopicEditOverlay,
    EditMemoryCandidate,
    OverlayResult,
    OverlayTopic,
    ReasonValidation,
    SkippedOverlayEdit,
)
from app.services.detected_topic_edit_reason_validator import (
    ReviewerReasonContextValidator,
    ReviewerReasonValidationRequest,
)
from app.services.detected_topic_edit_reuse_feedback_store import (
    DetectedTopicEditReuseFeedbackStore,
)
from app.services.module3_topic_overlay_adapter import (
    ActualModule3TopicOverlayAdapter,
    AddedTopicMaterialization,
    OfficialConceptMetadata,
)


def get_concept(concept_id: str):
    """Preserve the old HITL lookup contract while sourcing data from PostgreSQL."""
    concept = get_syllabus_store().get_concept(concept_id)

    if concept is None:
        raise KeyError(
            f"Unknown AQA concept_id: {concept_id}"
        )

    return concept


# -------------------------------------------------------------------------
# Retrieval-only settings
# -------------------------------------------------------------------------
#
# These are NOT automatic-edit decision thresholds.
#
# Step 4.5 proved that MiniLM similarity is not safe as a final decision
# mechanism. Here embeddings are used only to rank plausible reviewer-approved
# memories after hard filters such as spec_version + source concept have
# already been applied.
# -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    max_existing_records: int = 100
    max_add_records: int = 100
    max_add_target_concepts: int = 8


@dataclass(frozen=True, slots=True)
class RankedMemory:
    record: DetectedTopicEditMemory
    similarity: float


@dataclass(frozen=True, slots=True)
class EndToEndEditMemoryResult:
    """
    Controlled result object.

    No persistence happens here. The caller receives:
    - a NEW Module3Result;
    - the neutral overlay audit;
    - retrieval diagnostics.

    Hit counts are intentionally NOT updated in Step 5.
    """

    module3_result: Module3Result
    overlay_result: OverlayResult
    retrieval_diagnostics: tuple[str, ...]


class EmbeddingRankedPostgresCandidateProvider:
    """
    Real PostgreSQL + real MiniLM candidate retrieval.

    IMPORTANT:
    MiniLM ranks candidates only.

    It does NOT decide that an edit is safe.

    Existing-topic lookup:
        same spec + same source concept + approved/active/human-validated
        -> rank evidence
        -> if all records represent one outcome, return the strongest record
        -> if multiple distinct human outcomes exist, return one representative
           per outcome so the Step 4.8 overlay fails closed as a conflict.

    Add-topic lookup:
        same spec + approved/active/human-validated add memories
        -> rank by current evidence
        -> retain one strongest record per target outcome
        -> conflicting roles/outcomes for the same target are intentionally
           kept so the overlay can abstain.
    """

    def __init__(
        self,
        *,
        repository: DetectedTopicEditMemoryRepository,
        embedder: Agent1EditMemoryEmbeddingAdapter,
        config: RetrievalConfig | None = None,
        reuse_feedback_store: DetectedTopicEditReuseFeedbackStore | None = None,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.reuse_feedback_store = reuse_feedback_store

        self.diagnostics: list[str] = []

    @staticmethod
    def _cosine(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        a = np.asarray(left, dtype=np.float32)
        b = np.asarray(right, dtype=np.float32)

        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))

        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0

        return float(
            np.dot(a, b) / (a_norm * b_norm)
        )

    def _rank(
        self,
        *,
        current_evidence: str,
        records: Sequence[DetectedTopicEditMemory],
    ) -> list[RankedMemory]:
        if not records:
            return []

        texts = [
            str(current_evidence).strip(),
            *[
                str(record.evidence_text).strip()
                for record in records
            ],
        ]

        vectors = self.embedder.embed_texts(texts)
        query = vectors[0]

        ranked = [
            RankedMemory(
                record=record,
                similarity=self._cosine(
                    query,
                    vector,
                ),
            )
            for record, vector in zip(
                records,
                vectors[1:],
            )
        ]

        ranked.sort(
            key=lambda item: (
                item.similarity,
                int(item.record.id),
            ),
            reverse=True,
        )

        return ranked

    @staticmethod
    def _outcome_key(
        record: DetectedTopicEditMemory,
    ) -> tuple[
        str,
        str | None,
        str | None,
    ]:
        return (
            record.edit_action,
            record.target_concept_id,
            record.target_role,
        )

    @staticmethod
    def _to_candidate(
        record: DetectedTopicEditMemory,
    ) -> EditMemoryCandidate:
        return EditMemoryCandidate(
            memory_id=int(record.id),
            spec_version=record.spec_version,
            edit_action=record.edit_action,
            source_concept_id=record.source_concept_id,
            source_topic=record.source_topic,
            source_role=record.source_role,
            target_concept_id=record.target_concept_id,
            target_topic=record.target_topic,
            target_role=record.target_role,
            target_official_reference=None,
            reviewer_reason=record.reviewer_reason,
            stored_evidence=record.evidence_text,
            current_source_chunk_ids=tuple(
                int(value)
                for value in (
                    record.source_chunk_ids or []
                )
            ),
        )

    def _feedback_filtered_ranked(
        self,
        *,
        ranked: Sequence[RankedMemory],
        current_evidence: str,
        spec_version: str,
    ) -> list[RankedMemory]:
        """Apply exact-context human reuse feedback before outcome selection.

        Explicit reject_reuse still suppresses only that historical memory.
        Explicit approve_reuse is moved ahead of unreviewed memories so, when
        several historical records describe the same outcome, the human-chosen
        record is the representative that reaches the overlay/reason gate.

        No similarity threshold, comparator rule, or edit outcome is changed.
        """
        if self.reuse_feedback_store is None:
            return list(ranked)

        approved: list[RankedMemory] = []
        undecided: list[RankedMemory] = []

        for item in ranked:
            try:
                feedback = self.reuse_feedback_store.get_decision(
                    memory_id=int(item.record.id),
                    current_evidence=current_evidence,
                    spec_version=spec_version,
                )
            except Exception as exc:
                self.diagnostics.append(
                    f"memory {int(item.record.id)}: reuse-feedback lookup failed "
                    f"({type(exc).__name__}: {exc}); normal deterministic gate retained."
                )
                undecided.append(item)
                continue

            if feedback is not None and feedback.decision == "reject_reuse":
                self.diagnostics.append(
                    f"memory {int(item.record.id)}: explicit human reuse rejection "
                    "matches the exact current evidence; candidate suppressed."
                )
                continue

            if feedback is not None and feedback.decision == "approve_reuse":
                self.diagnostics.append(
                    f"memory {int(item.record.id)}: explicit human reuse approval "
                    "matches the exact current evidence; candidate prioritized."
                )
                approved.append(item)
                continue

            undecided.append(item)

        # Preserve embedding rank within each group. The only priority change is
        # that an exact-context human approval outranks an unreviewed duplicate
        # outcome, ensuring human authority is not lost during best-by-outcome
        # deduplication.
        return [*approved, *undecided]

    def candidates_for_existing_topic(
        self,
        *,
        spec_version: str,
        topic: OverlayTopic,
        current_evidence: str,
    ) -> Sequence[EditMemoryCandidate]:
        records = self.repository.list_reusable(
            spec_version=spec_version,
            source_concept_id=topic.concept_id,
            edit_actions=(
                "remove_topic",
                "replace_topic",
                "change_role",
            ),
            limit=self.config.max_existing_records,
        )

        ranked = self._rank(
            current_evidence=current_evidence,
            records=records,
        )
        ranked = self._feedback_filtered_ranked(
            ranked=ranked,
            current_evidence=current_evidence,
            spec_version=spec_version,
        )

        if not ranked:
            self.diagnostics.append(
                f"{topic.concept_id}: no reusable edit memory."
            )
            return []

        best_by_outcome: dict[
            tuple[str, str | None, str | None],
            RankedMemory,
        ] = {}

        for item in ranked:
            key = self._outcome_key(
                item.record
            )

            if key not in best_by_outcome:
                best_by_outcome[key] = item

        selected = sorted(
            best_by_outcome.values(),
            key=lambda item: item.similarity,
            reverse=True,
        )

        self.diagnostics.append(
            (
                f"{topic.concept_id}: "
                f"{len(records)} reusable record(s), "
                f"{len(selected)} distinct outcome(s), "
                f"best similarity={selected[0].similarity:.4f}."
            )
        )

        # If >1 distinct outcome exists, returning >1 candidates is
        # intentional: the overlay treats this as a conflict and abstains.
        return [
            self._to_candidate(
                item.record
            )
            for item in selected
        ]

    def candidates_for_additions(
        self,
        *,
        spec_version: str,
        current_chunk_evidence: Sequence[str],
        already_present_concept_ids: Sequence[str],
    ) -> Sequence[EditMemoryCandidate]:
        records = self.repository.list_reusable_additions(
            spec_version=spec_version,
            limit=self.config.max_add_records,
        )

        if not records:
            self.diagnostics.append(
                "add_topic: no reusable add memories."
            )
            return []

        current_text = "\n".join(
            str(text).strip()
            for text in current_chunk_evidence
            if str(text).strip()
        )

        if not current_text:
            self.diagnostics.append(
                "add_topic: no current evidence supplied."
            )
            return []

        present = {
            str(value)
            for value in already_present_concept_ids
        }

        records = [
            record
            for record in records
            if (
                record.target_concept_id
                and record.target_concept_id not in present
            )
        ]

        ranked = self._rank(
            current_evidence=current_text,
            records=records,
        )
        ranked = self._feedback_filtered_ranked(
            ranked=ranked,
            current_evidence=current_text,
            spec_version=spec_version,
        )

        # One strongest memory per distinct target outcome.
        best_by_outcome: dict[
            tuple[str, str | None, str | None],
            RankedMemory,
        ] = {}

        for item in ranked:
            key = self._outcome_key(
                item.record
            )

            if key not in best_by_outcome:
                best_by_outcome[key] = item

        selected = sorted(
            best_by_outcome.values(),
            key=lambda item: item.similarity,
            reverse=True,
        )[
            : self.config.max_add_target_concepts
        ]

        if selected:
            self.diagnostics.append(
                (
                    "add_topic: "
                    f"{len(records)} reusable record(s), "
                    f"{len(selected)} ranked target outcome(s), "
                    f"best similarity={selected[0].similarity:.4f}."
                )
            )

        return [
            self._to_candidate(
                item.record
            )
            for item in selected
        ]


class GroqReasonValidationBridge:
    """
    Bridge between Step 4.6/4.7 and Step 4.8.

    Real Groq performs rationale validation; the fail-closed Step 4.6 policy
    decides `safe_for_automatic_reuse`.
    """

    def __init__(
        self,
        *,
        validator: ReviewerReasonContextValidator,
    ) -> None:
        self.validator = validator

    def validate(
        self,
        *,
        candidate: EditMemoryCandidate,
        current_evidence: str,
    ) -> ReasonValidation:
        result = self.validator.validate(
            ReviewerReasonValidationRequest(
                edit_action=candidate.edit_action,
                source_topic=candidate.source_topic,
                source_role=candidate.source_role,
                target_topic=candidate.target_topic,
                target_role=candidate.target_role,
                reviewer_reason=candidate.reviewer_reason,
                stored_evidence=candidate.stored_evidence,
                current_evidence=current_evidence,
            )
        )

        return ReasonValidation(
            decision=result.decision,
            confidence=result.confidence,
            safe_for_automatic_reuse=(
                result.safe_for_automatic_reuse
            ),
            explanation=result.explanation,
        )


class ContextFirstReasonValidationBridge:
    """
    Final-topic memory gate for the locked Agent 1 HITL architecture:

        stored evidence + reviewer reason + current evidence
            -> deterministic contextual comparison
            -> strong match: reuse the reviewer-approved edit
            -> strong mismatch: reject the old edit
            -> ambiguous/error: abstain and require human review

    Groq is deliberately NOT used for automatic final-topic edit-memory reuse.
    Ambiguous evidence always fails closed so the fresh Module 3 result remains
    unchanged until a human reviewer decides what to do.
    """

    def __init__(
        self,
        *,
        comparator: DetectedTopicEditContextualComparator,
        groq_provider: GroqReviewerReasonProvider | None = None,
        reuse_feedback_store: DetectedTopicEditReuseFeedbackStore | None = None,
    ) -> None:
        self.comparator = comparator
        self.reuse_feedback_store = reuse_feedback_store

        # Kept only for constructor/backward compatibility with existing callers.
        # The locked final-topic HITL path never invokes this provider.
        self.groq_provider = groq_provider
        self._llm_validator: ReviewerReasonContextValidator | None = None
        self.diagnostics: list[str] = []

    def _validator(self) -> ReviewerReasonContextValidator:
        """Legacy helper retained for API compatibility; not used by this gate."""
        if self._llm_validator is None:
            provider = self.groq_provider or GroqReviewerReasonProvider()
            self.groq_provider = provider
            self._llm_validator = ReviewerReasonContextValidator(
                provider=provider
            )
        return self._llm_validator

    @staticmethod
    def _metric(value: float | None) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    def validate(
        self,
        *,
        candidate: EditMemoryCandidate,
        current_evidence: str,
    ) -> ReasonValidation:
        # Explicit human resolution of a previous ambiguous/conflicting reuse
        # decision is authoritative only for this exact current evidence.
        if self.reuse_feedback_store is not None:
            try:
                feedback = self.reuse_feedback_store.get_decision(
                    memory_id=candidate.memory_id,
                    current_evidence=current_evidence,
                    spec_version=candidate.spec_version,
                )
            except Exception as exc:
                self.diagnostics.append(
                    f"memory {candidate.memory_id}: reuse-feedback lookup failed "
                    f"({type(exc).__name__}: {exc}); continuing with the "
                    "normal deterministic comparator."
                )
            else:
                if feedback is not None and feedback.decision == "approve_reuse":
                    self.diagnostics.append(
                        f"memory {candidate.memory_id}: explicit human reuse approval "
                        "matches the exact current evidence; old edit authorized. "
                        "Groq not called."
                    )
                    return ReasonValidation(
                        decision="compatible",
                        confidence=1.0,
                        safe_for_automatic_reuse=True,
                        explanation=(
                            "Explicit human reuse approval for this exact current "
                            "evidence: " + feedback.reviewer_reason
                        ),
                    )
                if feedback is not None and feedback.decision == "reject_reuse":
                    self.diagnostics.append(
                        f"memory {candidate.memory_id}: explicit human reuse rejection "
                        "matches the exact current evidence; old edit rejected. "
                        "Groq not called."
                    )
                    return ReasonValidation(
                        decision="incompatible",
                        confidence=1.0,
                        safe_for_automatic_reuse=False,
                        explanation=(
                            "Explicit human reuse rejection for this exact current "
                            "evidence: " + feedback.reviewer_reason
                        ),
                    )

        try:
            comparison = self.comparator.compare(
                edit_action=candidate.edit_action,
                stored_evidence=candidate.stored_evidence,
                reviewer_reason=candidate.reviewer_reason,
                current_evidence=current_evidence,
            )
        except Exception as exc:
            # Comparator failure is treated exactly like an ambiguous case:
            # fail closed, preserve the fresh Module 3 result, and require a
            # human decision. No LLM/Groq fallback is invoked.
            self.diagnostics.append(
                f"memory {candidate.memory_id}: contextual comparison error "
                f"({type(exc).__name__}: {exc}); human review required; "
                "fresh Module 3 result kept. Groq not called."
            )
            return ReasonValidation(
                decision="uncertain",
                confidence=0.0,
                safe_for_automatic_reuse=False,
                explanation=(
                    "Context comparison could not safely determine whether the "
                    "old reviewer-approved edit still applies. Human review is "
                    "required and the fresh Module 3 output was preserved."
                ),
            )

        metric_text = (
            f"evidence={self._metric(comparison.evidence_similarity)}, "
            f"reason={self._metric(comparison.reviewer_reason_similarity)}, "
            f"combined={self._metric(comparison.combined_similarity)}, "
            f"token={self._metric(comparison.token_containment)}"
        )

        if comparison.is_strong_match:
            self.diagnostics.append(
                f"memory {candidate.memory_id}: deterministic context hit "
                f"({metric_text}); reviewer-approved edit may be reused "
                "automatically. Groq not called."
            )
            return ReasonValidation(
                decision="compatible",
                confidence=float(comparison.confidence),
                safe_for_automatic_reuse=True,
                explanation=comparison.explanation,
            )

        if comparison.is_strong_mismatch:
            self.diagnostics.append(
                f"memory {candidate.memory_id}: deterministic context miss "
                f"({metric_text}); old edit rejected. Groq not called."
            )
            return ReasonValidation(
                decision="incompatible",
                confidence=float(comparison.confidence),
                safe_for_automatic_reuse=False,
                explanation=comparison.explanation,
            )

        # Locked HITL policy: ambiguity is resolved by a human, not by an LLM.
        self.diagnostics.append(
            f"memory {candidate.memory_id}: context ambiguous "
            f"({metric_text}); human review required; fresh Module 3 result "
            "kept. Groq not called."
        )
        return ReasonValidation(
            decision="uncertain",
            confidence=float(comparison.confidence),
            safe_for_automatic_reuse=False,
            explanation=(
                "Stored edit memory is neither a strong deterministic match nor "
                "a strong deterministic mismatch for the current transcript. "
                "Automatic reuse abstained; human review is required and the "
                "fresh Module 3 output was preserved."
            ),
        )


class Module3AddTopicEvidenceMaterializer:
    """
    Materialize human-added topics ONLY from real current Module 3 candidate
    evidence.

    No confidence/ranking/evidence value is invented.

    The aggregation mirrors the current Module 3 TopicMerger defaults:
    - confidence = max cs relevance + non-adjacent support bonus
    - semantic/keyword/salience = means
    - coverage = supporting chunks / topic-bearing chunks
    - ranking = 0.20 confidence + 0.20 semantic + 0.20 salience
                + 0.40 coverage

    Both retained and rejected candidates are inspected because a human
    add_topic correction typically represents a candidate that Module 3 saw
    but did not retain in the final merged list.

    If no current candidate evidence exists, materialization fails closed.
    """

    NON_ADJACENT_SPAN_BONUS = 0.02
    MAXIMUM_SUPPORT_BONUS = 0.06
    MAXIMUM_MERGED_CONFIDENCE = 0.95

    RANKING_CONFIDENCE_WEIGHT = 0.20
    RANKING_SEMANTIC_WEIGHT = 0.20
    RANKING_SALIENCE_WEIGHT = 0.20
    RANKING_COVERAGE_WEIGHT = 0.40

    MAX_EVIDENCE_PER_TOPIC = 5

    @staticmethod
    def _count_spans(
        chunk_ids: Sequence[int],
    ) -> int:
        ids = sorted(
            {
                int(value)
                for value in chunk_ids
            }
        )

        if not ids:
            return 0

        spans = 1

        for previous, current in zip(
            ids,
            ids[1:],
        ):
            if current > previous + 1:
                spans += 1

        return spans

    @staticmethod
    def _unique_strings(
        values: Sequence[str],
    ) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = " ".join(
                str(value).strip().split()
            )
            normalized = cleaned.casefold()

            if not cleaned or normalized in seen:
                continue

            seen.add(normalized)
            output.append(cleaned)

        return output

    @classmethod
    def build(
        cls,
        *,
        module3_result: Module3Result,
        memory_id: int,
        target_concept_id: str,
        target_role: str,
    ) -> AddedTopicMaterialization:
        occurrences: list[
            tuple[int, TopicCandidate]
        ] = []

        for chunk in module3_result.chunk_results:
            candidates = [
                *chunk.topic_candidates,
                *chunk.rejected_candidates,
            ]

            for candidate in candidates:
                if (
                    candidate.concept_id
                    == target_concept_id
                ):
                    occurrences.append(
                        (
                            int(chunk.chunk_id),
                            candidate,
                        )
                    )

        if not occurrences:
            raise ValueError(
                "No current Module 3 candidate evidence exists for "
                f"human-added concept {target_concept_id!r}."
            )

        first = occurrences[0][1]

        # Verify official identity against the current AQA catalogue.
        concept = get_concept(
            target_concept_id
        )

        if (
            first.official_reference
            != concept.official_reference
        ):
            raise ValueError(
                "Current candidate reference does not match AQA catalogue."
            )

        source_chunk_ids = sorted(
            {
                chunk_id
                for chunk_id, _ in occurrences
            }
        )

        support_span_count = cls._count_spans(
            source_chunk_ids
        )

        base_confidence = max(
            candidate.cs_relevance_score
            for _, candidate in occurrences
        )

        support_bonus = min(
            cls.MAXIMUM_SUPPORT_BONUS,
            cls.NON_ADJACENT_SPAN_BONUS
            * max(
                0,
                support_span_count - 1,
            ),
        )

        confidence = min(
            cls.MAXIMUM_MERGED_CONFIDENCE,
            base_confidence + support_bonus,
        )

        mean_semantic_score = fmean(
            candidate.semantic_score
            for _, candidate in occurrences
        )
        mean_keyword_score = fmean(
            candidate.keyword_score
            for _, candidate in occurrences
        )
        mean_salience_score = fmean(
            candidate.salience_score
            for _, candidate in occurrences
        )

        topic_bearing_chunk_count = max(
            1,
            sum(
                1
                for chunk in module3_result.chunk_results
                if chunk.topic_candidates
            ),
        )

        coverage_score = min(
            1.0,
            len(source_chunk_ids)
            / topic_bearing_chunk_count,
        )

        semantic_component = max(
            0.0,
            min(
                1.0,
                mean_semantic_score,
            ),
        )

        ranking_score = (
            cls.RANKING_CONFIDENCE_WEIGHT
            * confidence
            + cls.RANKING_SEMANTIC_WEIGHT
            * semantic_component
            + cls.RANKING_SALIENCE_WEIGHT
            * mean_salience_score
            + cls.RANKING_COVERAGE_WEIGHT
            * coverage_score
        )

        ranking_score = max(
            0.0,
            min(
                1.0,
                ranking_score,
            ),
        )

        evidence: list[str] = []

        for _, candidate in occurrences:
            evidence.extend(
                candidate.evidence
            )

        metadata = OfficialConceptMetadata(
            concept_id=concept.concept_id,
            topic=concept.label,
            domain=concept.domain,
            official_reference=(
                concept.official_reference
            ),
            chapter_reference=(
                concept.chapter_reference
            ),
            official_title=concept.official_title,
            paper=concept.paper,
            source_pages=tuple(
                concept.source_pages
            ),
        )

        return AddedTopicMaterialization(
            memory_id=int(memory_id),
            metadata=metadata,
            confidence=round(
                confidence,
                4,
            ),
            ranking_score=round(
                ranking_score,
                4,
            ),
            source_chunk_ids=tuple(
                source_chunk_ids
            ),
            support_span_count=(
                support_span_count
            ),
            mean_semantic_score=round(
                mean_semantic_score,
                4,
            ),
            mean_keyword_score=round(
                mean_keyword_score,
                4,
            ),
            mean_salience_score=round(
                mean_salience_score,
                4,
            ),
            coverage_score=round(
                coverage_score,
                4,
            ),
            evidence=tuple(
                cls._unique_strings(
                    evidence
                )[
                    : cls.MAX_EVIDENCE_PER_TOPIC
                ]
            ),
            supporting_candidate_count=len(
                occurrences
            ),
        )


class DetectedTopicEditEndToEndService:
    """
    Final-topic self-improving chain:

    PostgreSQL reviewer-approved memory
        -> MiniLM ranking (retrieval only)
        -> context-first comparison of current evidence against
           stored evidence + reviewer reason
        -> deterministic strong hit/miss when safe
        -> ambiguous/conflicting context abstains for human review
        -> Step 4.8 fail-closed overlay
        -> Step 4.9 real Module3Result adapter

    No database writes are performed by this service.
    In particular, hit_count / last_used_at are NOT updated here.
    """

    def __init__(
        self,
        *,
        repository: DetectedTopicEditMemoryRepository,
        embedder: Agent1EditMemoryEmbeddingAdapter | None = None,
        groq_provider: GroqReviewerReasonProvider | None = None,
    ) -> None:
        self.repository = repository

        self.embedder = (
            embedder
            or Agent1EditMemoryEmbeddingAdapter()
        )

        # Kept for backward-compatible construction only. The locked final-topic
        # HITL auto-reuse path never invokes Groq; ambiguous cases abstain for
        # explicit human review.
        self.groq_provider = groq_provider
        self.reuse_feedback_store = DetectedTopicEditReuseFeedbackStore()

        self.candidate_provider = (
            EmbeddingRankedPostgresCandidateProvider(
                repository=self.repository,
                embedder=self.embedder,
                reuse_feedback_store=self.reuse_feedback_store,
            )
        )

        self.context_comparator = (
            DetectedTopicEditContextualComparator(
                embedder=self.embedder
            )
        )

        self.reason_bridge = (
            ContextFirstReasonValidationBridge(
                comparator=self.context_comparator,
                groq_provider=self.groq_provider,
                reuse_feedback_store=self.reuse_feedback_store,
            )
        )

        self.overlay = DetectedTopicEditOverlay(
            candidate_provider=(
                self.candidate_provider
            ),
            reason_validator=self.reason_bridge,
        )

    @staticmethod
    def _evidence_by_concept(
        module3_result: Module3Result,
    ) -> dict[str, str]:
        return {
            topic.concept_id: "\n".join(
                str(value).strip()
                for value in topic.evidence
                if str(value).strip()
            )
            for topic in module3_result.merged_topics
        }

    @staticmethod
    def _addition_current_evidence(
        module3_result: Module3Result,
    ) -> list[str]:
        evidence: list[str] = []

        for chunk in module3_result.chunk_results:
            for candidate in [
                *chunk.topic_candidates,
                *chunk.rejected_candidates,
            ]:
                evidence.extend(
                    str(value).strip()
                    for value in candidate.evidence
                    if str(value).strip()
                )

        return evidence

    @staticmethod
    def _official_metadata_for_replacements(
        overlay_result: OverlayResult,
    ) -> dict[
        str,
        OfficialConceptMetadata,
    ]:
        output: dict[
            str,
            OfficialConceptMetadata,
        ] = {}

        for edit in overlay_result.applied:
            if (
                edit.action != "replace_topic"
                or not edit.target_concept_id
            ):
                continue

            concept = get_concept(
                edit.target_concept_id
            )

            output[
                concept.concept_id
            ] = OfficialConceptMetadata(
                concept_id=concept.concept_id,
                topic=concept.label,
                domain=concept.domain,
                official_reference=(
                    concept.official_reference
                ),
                chapter_reference=(
                    concept.chapter_reference
                ),
                official_title=(
                    concept.official_title
                ),
                paper=concept.paper,
                source_pages=tuple(
                    concept.source_pages
                ),
            )

        return output

    @staticmethod
    def _current_candidate_concept_ids(
        module3_result: Module3Result,
    ) -> set[str]:
        """Return concepts that have real current-run Module 3 candidate metrics."""
        output: set[str] = set()
        for chunk in module3_result.chunk_results:
            for candidate in [
                *chunk.topic_candidates,
                *chunk.rejected_candidates,
            ]:
                concept_id = str(candidate.concept_id or "").strip()
                if concept_id:
                    output.add(concept_id)
        return output

    def _prepare_add_topic_materialization_overlays(
        self,
        *,
        module3_result: Module3Result,
        overlay_result: OverlayResult,
        spec_version: str,
    ) -> tuple[OverlayResult, OverlayResult]:
        """Separate strict Module3 materialization from safe effective-list additions.

        Existing Module3Result materialization intentionally refuses to invent
        ranking/confidence metrics for an add_topic when the current run never
        produced that official concept as either a kept or rejected candidate.

        That strict rule is correct, but it must not cause the *whole* HITL
        overlay to fall back after a human has explicitly approved a missed
        topic for this exact current evidence.  In that case we defer only the
        add_topic to the frontend's existing human-addition representation
        (which already supports null Module3 scores), while still materializing
        remove/change_role/replace edits normally.

        Automatic add_topic reuse without a real current candidate remains
        fail-closed unless an exact-context human approve_reuse exists.
        """
        candidate_ids = self._current_candidate_concept_ids(module3_result)
        current_add_evidence = "\n".join(
            self._addition_current_evidence(module3_result)
        )

        deferred_ids: set[int] = set()
        unsupported_ids: set[int] = set()
        extra_skipped: list[SkippedOverlayEdit] = []

        for edit in overlay_result.applied:
            if edit.action != "add_topic" or not edit.target_concept_id:
                continue

            if edit.target_concept_id in candidate_ids:
                continue

            feedback = None
            try:
                feedback = self.reuse_feedback_store.get_decision(
                    memory_id=int(edit.memory_id),
                    current_evidence=current_add_evidence,
                    spec_version=spec_version,
                )
            except Exception as exc:
                self.candidate_provider.diagnostics.append(
                    f"memory {int(edit.memory_id)}: exact reuse-feedback lookup "
                    f"failed while checking add_topic materialization "
                    f"({type(exc).__name__}: {exc})."
                )

            if feedback is not None and feedback.decision == "approve_reuse":
                deferred_ids.add(int(edit.memory_id))
                self.candidate_provider.diagnostics.append(
                    f"memory {int(edit.memory_id)}: exact human-approved "
                    "add_topic has no current Module 3 candidate metrics; "
                    "deferred to effective-list human-addition materialization."
                )
                continue

            unsupported_ids.add(int(edit.memory_id))
            extra_skipped.append(
                SkippedOverlayEdit(
                    memory_id=int(edit.memory_id),
                    action="add_topic",
                    source_concept_id=None,
                    reason=(
                        "add_topic had no current Module 3 candidate metrics and "
                        "no exact-context human approval; kept fail-closed."
                    ),
                )
            )
            self.candidate_provider.diagnostics.append(
                f"memory {int(edit.memory_id)}: add_topic suppressed because "
                "the current run has no candidate metrics and there is no "
                "exact-context human approval."
            )

        non_materialized_ids = deferred_ids | unsupported_ids

        # Adapter overlay: only edits that can be represented as a real
        # Module3Result are allowed through the strict schema materializer.
        materialization_overlay = OverlayResult(
            topics=tuple(
                topic
                for topic in overlay_result.topics
                if not (
                    topic.memory_applied
                    and topic.memory_action == "add_topic"
                    and topic.memory_id is not None
                    and int(topic.memory_id) in non_materialized_ids
                )
            ),
            applied=tuple(
                edit
                for edit in overlay_result.applied
                if int(edit.memory_id) not in non_materialized_ids
            ),
            skipped=tuple(overlay_result.skipped),
        )

        # Audit overlay: exact-approved deferred additions still count as
        # applied HITL decisions; unsupported automatic additions are shown as
        # skipped instead.
        audit_overlay = OverlayResult(
            topics=tuple(
                topic
                for topic in overlay_result.topics
                if not (
                    topic.memory_applied
                    and topic.memory_action == "add_topic"
                    and topic.memory_id is not None
                    and int(topic.memory_id) in unsupported_ids
                )
            ),
            applied=tuple(
                edit
                for edit in overlay_result.applied
                if int(edit.memory_id) not in unsupported_ids
            ),
            skipped=tuple([*overlay_result.skipped, *extra_skipped]),
        )

        return materialization_overlay, audit_overlay

    @staticmethod
    def _add_materializations(
        *,
        module3_result: Module3Result,
        overlay_result: OverlayResult,
    ) -> dict[
        int,
        AddedTopicMaterialization,
    ]:
        output: dict[
            int,
            AddedTopicMaterialization,
        ] = {}

        overlay_by_memory = {
            int(topic.memory_id): topic
            for topic in overlay_result.topics
            if (
                topic.memory_applied
                and topic.memory_action == "add_topic"
                and topic.memory_id is not None
            )
        }

        for edit in overlay_result.applied:
            if (
                edit.action != "add_topic"
                or not edit.target_concept_id
            ):
                continue

            overlay_topic = overlay_by_memory.get(
                int(edit.memory_id)
            )

            if overlay_topic is None:
                raise ValueError(
                    "Applied add_topic is missing from overlay topics."
                )

            output[
                int(edit.memory_id)
            ] = (
                Module3AddTopicEvidenceMaterializer
                .build(
                    module3_result=module3_result,
                    memory_id=edit.memory_id,
                    target_concept_id=(
                        edit.target_concept_id
                    ),
                    target_role=(
                        overlay_topic.role
                    ),
                )
            )

        return output

    def apply(
        self,
        *,
        module3_result: Module3Result,
        spec_version: str,
    ) -> EndToEndEditMemoryResult:
        original = module3_result.model_copy(
            deep=True
        )

        overlay_result = self.overlay.apply(
            topics=(
                ActualModule3TopicOverlayAdapter
                .to_overlay_topics(
                    original.merged_topics
                )
            ),
            spec_version=spec_version,
            evidence_by_concept_id=(
                self._evidence_by_concept(
                    original
                )
            ),
            current_chunk_evidence=(
                self._addition_current_evidence(
                    original
                )
            ),
        )

        (
            materialization_overlay,
            audit_overlay,
        ) = self._prepare_add_topic_materialization_overlays(
            module3_result=original,
            overlay_result=overlay_result,
            spec_version=spec_version,
        )

        official_metadata = (
            self._official_metadata_for_replacements(
                materialization_overlay
            )
        )

        added_topics = (
            self._add_materializations(
                module3_result=original,
                overlay_result=materialization_overlay,
            )
        )

        updated = (
            ActualModule3TopicOverlayAdapter
            .materialize_module3_result(
                original_result=original,
                overlay_result=materialization_overlay,
                official_metadata=official_metadata,
                added_topics=added_topics,
                sort_result=True,
            )
        )

        return EndToEndEditMemoryResult(
            module3_result=updated,
            overlay_result=audit_overlay,
            retrieval_diagnostics=tuple(
                [
                    *self.candidate_provider.diagnostics,
                    *self.reason_bridge.diagnostics,
                ]
            ),
        )
