from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np

from app.db.models.topic_mapping_memory import TopicMappingMemory
from app.db.repositories.topic_mapping_decision_log_repository import (
    TopicMappingDecisionLogRepository,
)
from app.db.repositories.topic_mapping_memory_repository import (
    TopicMappingMemoryRepository,
)
DEFAULT_TOPIC_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _default_embedding_function(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    # Import lazily so lightweight tests can inject a fake embedder without
    # loading sentence-transformers. Production still uses the existing
    # Agent 1 embedding service.
    from app.services.embedding_service import embed_texts

    return embed_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
    )


MemoryLookupStatus = Literal["hit", "miss", "ambiguous"]
EmbeddingFunction = Callable[[Sequence[str], str, int], np.ndarray]


@dataclass(frozen=True)
class TopicMappingMemoryConfig:
    """
    Conservative thresholds for reviewer-approved memory reuse.

    The memory layer should prefer a false MISS (fall back to Qdrant) over a
    questionable HIT. Every value can be overridden through environment
    variables without changing Module 3 logic.
    """

    embedding_model: str = DEFAULT_TOPIC_EMBEDDING_MODEL
    evidence_similarity_threshold: float = 0.82
    reviewer_reason_similarity_threshold: float = 0.60
    combined_similarity_threshold: float = 0.80
    # A separate, deliberately scoped path for reviewer-corrected memories.
    # These lower semantic thresholds NEVER apply to ordinary approved memory.
    # The path is allowed only when the same normalized rough topic + same spec
    # already matched in PostgreSQL and BOTH historical evidence and the human
    # correction reason semantically agree with the new evidence.
    human_corrected_evidence_threshold: float = 0.60
    human_corrected_reason_threshold: float = 0.60
    human_corrected_combined_threshold: float = 0.60
    # Near-identical evidence should be allowed to reuse a reviewer correction
    # even when the short correction reason is lexically/semantically very
    # different from the full transcript chunk. This threshold is deliberately
    # high and only applies to human-corrected memory.
    human_corrected_exact_evidence_threshold: float = 0.95
    reviewer_reason_weight: float = 0.25
    minimum_score_margin: float = 0.04
    max_candidates: int = 20
    minimum_evidence_characters: int = 40
    batch_size: int = 32

    @classmethod
    def from_environment(cls) -> "TopicMappingMemoryConfig":
        return cls(
            embedding_model=os.getenv(
                "TOPIC_EMBEDDING_MODEL",
                DEFAULT_TOPIC_EMBEDDING_MODEL,
            ).strip(),
            evidence_similarity_threshold=float(
                os.getenv("MEMORY_EVIDENCE_SIMILARITY_THRESHOLD", "0.82")
            ),
            reviewer_reason_similarity_threshold=float(
                os.getenv("MEMORY_REASON_SIMILARITY_THRESHOLD", "0.60")
            ),
            combined_similarity_threshold=float(
                os.getenv("MEMORY_COMBINED_SIMILARITY_THRESHOLD", "0.80")
            ),
            human_corrected_evidence_threshold=float(
                os.getenv("MEMORY_HUMAN_CORRECTED_EVIDENCE_THRESHOLD", "0.60")
            ),
            human_corrected_reason_threshold=float(
                os.getenv("MEMORY_HUMAN_CORRECTED_REASON_THRESHOLD", "0.60")
            ),
            human_corrected_combined_threshold=float(
                os.getenv("MEMORY_HUMAN_CORRECTED_COMBINED_THRESHOLD", "0.60")
            ),
            human_corrected_exact_evidence_threshold=float(
                os.getenv("MEMORY_HUMAN_CORRECTED_EXACT_EVIDENCE_THRESHOLD", "0.95")
            ),
            reviewer_reason_weight=float(
                os.getenv("MEMORY_REVIEWER_REASON_WEIGHT", "0.25")
            ),
            minimum_score_margin=float(
                os.getenv("MEMORY_MINIMUM_SCORE_MARGIN", "0.04")
            ),
            max_candidates=int(os.getenv("MEMORY_MAX_CANDIDATES", "20")),
            minimum_evidence_characters=int(
                os.getenv("MEMORY_MINIMUM_EVIDENCE_CHARACTERS", "40")
            ),
            batch_size=int(os.getenv("MEMORY_EMBEDDING_BATCH_SIZE", "32")),
        )

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_similarity_threshold", self.evidence_similarity_threshold),
            (
                "reviewer_reason_similarity_threshold",
                self.reviewer_reason_similarity_threshold,
            ),
            ("combined_similarity_threshold", self.combined_similarity_threshold),
            (
                "human_corrected_evidence_threshold",
                self.human_corrected_evidence_threshold,
            ),
            (
                "human_corrected_reason_threshold",
                self.human_corrected_reason_threshold,
            ),
            (
                "human_corrected_combined_threshold",
                self.human_corrected_combined_threshold,
            ),
            (
                "human_corrected_exact_evidence_threshold",
                self.human_corrected_exact_evidence_threshold,
            ),
            ("minimum_score_margin", self.minimum_score_margin),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if not 0.0 <= self.reviewer_reason_weight <= 1.0:
            raise ValueError("reviewer_reason_weight must be between 0 and 1.")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1.")
        if self.minimum_evidence_characters < 1:
            raise ValueError("minimum_evidence_characters must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if not self.embedding_model:
            raise ValueError("embedding_model cannot be empty.")


@dataclass(frozen=True)
class MemoryCandidateScore:
    memory: TopicMappingMemory
    evidence_similarity: float
    reviewer_reason_similarity: float | None
    combined_similarity: float
    passes_evidence_gate: bool
    passes_reason_gate: bool
    passes_combined_gate: bool
    passes_human_corrected_exact_evidence_gate: bool
    passes_human_corrected_paraphrase_gate: bool

    @property
    def is_strong(self) -> bool:
        strict_match = (
            self.passes_evidence_gate
            and self.passes_reason_gate
            and self.passes_combined_gate
        )
        return (
            strict_match
            or self.passes_human_corrected_exact_evidence_gate
            or self.passes_human_corrected_paraphrase_gate
        )

    @property
    def match_mode(self) -> str | None:
        if (
            self.passes_evidence_gate
            and self.passes_reason_gate
            and self.passes_combined_gate
        ):
            return "strict"
        if self.passes_human_corrected_exact_evidence_gate:
            return "human_corrected_exact_evidence"
        if self.passes_human_corrected_paraphrase_gate:
            return "human_corrected_paraphrase"
        return None

    @property
    def outcome_key(self) -> tuple[str, str | None]:
        return (self.memory.decision, self.memory.mapped_concept_id)


@dataclass(frozen=True)
class MemoryLookupResult:
    status: MemoryLookupStatus
    matched_memory: TopicMappingMemory | None
    evidence_similarity: float | None
    reviewer_reason_similarity: float | None
    combined_similarity: float | None
    candidates_evaluated: int
    reason: str

    @property
    def is_hit(self) -> bool:
        return self.status == "hit" and self.matched_memory is not None

    @property
    def source_memory_id(self) -> int | None:
        if self.matched_memory is None:
            return None
        return int(self.matched_memory.id)


class TopicMappingMemoryService:
    """
    Decide whether a reviewer-approved historical mapping is safe to reuse.

    Safety contract:
      1. PostgreSQL repository only returns reviewer-approved rows.
      2. spec_version must equal the active specification version.
      3. rough/normalized topic must match.
      4. new evidence must strongly match stored evidence.
      5. ordinary approved memories keep the existing strict evidence +
         combined thresholds.
      6. human-corrected memories have a near-identical evidence path: when
         evidence similarity is >= 0.95, the prior reviewer correction can be
         reused even if the short correction reason is dissimilar to the full
         transcript chunk. Same normalized topic + same spec are already
         required by lookup.
      7. human-corrected memories additionally support a conservative
         paraphrase path: BOTH evidence and the human correction reason must
         independently reach 0.60 semantic similarity, with combined
         similarity >= 0.60. This path does not weaken ordinary memory.
      8. competing strong memories with different outcomes must have a clear
         score winner; otherwise the result is ambiguous and falls back.

    This service does NOT replace Qdrant. A miss/ambiguous result means the
    caller should continue through the existing Qdrant + review flow.
    """

    def __init__(
        self,
        memory_repository: TopicMappingMemoryRepository,
        decision_log_repository: TopicMappingDecisionLogRepository | None = None,
        *,
        config: TopicMappingMemoryConfig | None = None,
        embedding_function: EmbeddingFunction = _default_embedding_function,
    ) -> None:
        self.memory_repository = memory_repository
        self.decision_log_repository = decision_log_repository
        self.config = config or TopicMappingMemoryConfig.from_environment()
        self.embedding_function = embedding_function

    def evaluate(
        self,
        *,
        normalized_topic: str,
        new_evidence: str,
        spec_version: str,
    ) -> MemoryLookupResult:
        """Evaluate memory compatibility without mutating the database."""

        normalized_topic = normalized_topic.strip()
        new_evidence = new_evidence.strip()
        spec_version = spec_version.strip()

        if not normalized_topic:
            raise ValueError("normalized_topic is required.")
        if not spec_version:
            raise ValueError("spec_version is required.")

        if len(new_evidence) < self.config.minimum_evidence_characters:
            return MemoryLookupResult(
                status="miss",
                matched_memory=None,
                evidence_similarity=None,
                reviewer_reason_similarity=None,
                combined_similarity=None,
                candidates_evaluated=0,
                reason=(
                    "Evidence is too short for safe memory reuse; continue to "
                    "Qdrant."
                ),
            )

        candidates = self.memory_repository.find_reusable_candidates(
            normalized_topic=normalized_topic,
            spec_version=spec_version,
            limit=self.config.max_candidates,
        )

        if not candidates:
            return MemoryLookupResult(
                status="miss",
                matched_memory=None,
                evidence_similarity=None,
                reviewer_reason_similarity=None,
                combined_similarity=None,
                candidates_evaluated=0,
                reason=(
                    "No reviewer-approved memory exists for this normalized "
                    "topic and specification version."
                ),
            )

        scored = self._score_candidates(
            new_evidence=new_evidence,
            candidates=candidates,
        )
        strong = [candidate for candidate in scored if candidate.is_strong]

        if not strong:
            best = max(scored, key=lambda item: item.combined_similarity)
            return MemoryLookupResult(
                status="miss",
                matched_memory=None,
                evidence_similarity=best.evidence_similarity,
                reviewer_reason_similarity=best.reviewer_reason_similarity,
                combined_similarity=best.combined_similarity,
                candidates_evaluated=len(scored),
                reason=(
                    "Reviewer-approved memory candidates were found, but none "
                    "passed the evidence and combined compatibility gates."
                ),
            )

        strong.sort(key=lambda item: item.combined_similarity, reverse=True)
        best = strong[0]

        # If another strong candidate recommends a different outcome and its
        # score is too close, do not silently choose one. Send the case back to
        # Qdrant/human review instead.
        conflicting_runner_up = next(
            (
                candidate
                for candidate in strong[1:]
                if candidate.outcome_key != best.outcome_key
            ),
            None,
        )

        if conflicting_runner_up is not None:
            margin = best.combined_similarity - conflicting_runner_up.combined_similarity
            if margin < self.config.minimum_score_margin:
                return MemoryLookupResult(
                    status="ambiguous",
                    matched_memory=None,
                    evidence_similarity=best.evidence_similarity,
                    reviewer_reason_similarity=best.reviewer_reason_similarity,
                    combined_similarity=best.combined_similarity,
                    candidates_evaluated=len(scored),
                    reason=(
                        "Two reviewer-approved memories with different outcomes "
                        "are too close in compatibility score; continue to "
                        "Qdrant/human review."
                    ),
                )

        hit_reason = (
            "Reviewer-approved memory passed specification, evidence, and "
            "combined compatibility checks."
        )
        if best.match_mode == "human_corrected_exact_evidence":
            hit_reason = (
                "Human-corrected memory passed the near-identical evidence "
                "reuse path: same normalized topic/spec plus evidence "
                "similarity above the exact-evidence safety threshold."
            )
        elif best.match_mode == "human_corrected_paraphrase":
            hit_reason = (
                "Human-corrected memory passed the conservative paraphrase "
                "reuse path: same normalized topic/spec plus aligned new "
                "evidence and correction reason."
            )

        return MemoryLookupResult(
            status="hit",
            matched_memory=best.memory,
            evidence_similarity=best.evidence_similarity,
            reviewer_reason_similarity=best.reviewer_reason_similarity,
            combined_similarity=best.combined_similarity,
            candidates_evaluated=len(scored),
            reason=hit_reason,
        )

    def evaluate_and_record(
        self,
        *,
        normalized_topic: str,
        new_evidence: str,
        spec_version: str,
        pipeline_run_id: str | None = None,
        cache_key: str | None = None,
        source_transcript: str | None = None,
        source_chunk_ids: Sequence[int] = (),
    ) -> MemoryLookupResult:
        """
        Evaluate memory and write the lookup/reuse decision to the audit log.

        On a HIT this method also increments hit_count/last_used_at. On a MISS
        or ambiguous result it does not modify any memory row.
        """

        result = self.evaluate(
            normalized_topic=normalized_topic,
            new_evidence=new_evidence,
            spec_version=spec_version,
        )

        if result.is_hit:
            memory = result.matched_memory
            assert memory is not None

            # Count usage only after all strong-match gates have passed.
            memory = self.memory_repository.mark_used(int(memory.id))

            if self.decision_log_repository is not None:
                reuse_confidence = min(
                    float(memory.confidence),
                    float(result.combined_similarity or 0.0),
                )
                self.decision_log_repository.log(
                    normalized_topic=normalized_topic,
                    decision_stage="memory_reuse",
                    actor_type="system",
                    action="reuse",
                    spec_version=spec_version,
                    memory_id=int(memory.id),
                    source_memory_id=int(memory.id),
                    pipeline_run_id=pipeline_run_id,
                    cache_key=cache_key,
                    source_transcript=source_transcript,
                    source_chunk_ids=source_chunk_ids,
                    decision=memory.decision,
                    mapped_concept_id=memory.mapped_concept_id,
                    confidence=reuse_confidence,
                    reason=result.reason,
                    details={
                        "evidence_similarity": result.evidence_similarity,
                        "reviewer_reason_similarity": (
                            result.reviewer_reason_similarity
                        ),
                        "combined_similarity": result.combined_similarity,
                        "stored_mapping_confidence": float(memory.confidence),
                        "validation_status": memory.validation_status,
                        "reviewer_reason": memory.reviewer_reason,
                        "candidates_evaluated": result.candidates_evaluated,
                    },
                )

            # Return the refreshed memory object after usage counters changed.
            return MemoryLookupResult(
                status="hit",
                matched_memory=memory,
                evidence_similarity=result.evidence_similarity,
                reviewer_reason_similarity=result.reviewer_reason_similarity,
                combined_similarity=result.combined_similarity,
                candidates_evaluated=result.candidates_evaluated,
                reason=result.reason,
            )

        if self.decision_log_repository is not None:
            self.decision_log_repository.log(
                normalized_topic=normalized_topic,
                decision_stage="memory_lookup",
                actor_type="system",
                action="memory_miss",
                spec_version=spec_version,
                pipeline_run_id=pipeline_run_id,
                cache_key=cache_key,
                source_transcript=source_transcript,
                source_chunk_ids=source_chunk_ids,
                confidence=result.combined_similarity,
                reason=result.reason,
                details={
                    "lookup_status": result.status,
                    "evidence_similarity": result.evidence_similarity,
                    "reviewer_reason_similarity": (
                        result.reviewer_reason_similarity
                    ),
                    "combined_similarity": result.combined_similarity,
                    "candidates_evaluated": result.candidates_evaluated,
                },
            )

        return result

    def _score_candidates(
        self,
        *,
        new_evidence: str,
        candidates: Sequence[TopicMappingMemory],
    ) -> list[MemoryCandidateScore]:
        texts: list[str] = [new_evidence]
        positions: list[tuple[int, int | None]] = []

        for memory in candidates:
            evidence_position = len(texts)
            texts.append(memory.evidence_text.strip())

            reason_position: int | None = None
            reviewer_reason = (memory.reviewer_reason or "").strip()
            if reviewer_reason:
                reason_position = len(texts)
                texts.append(reviewer_reason)

            positions.append((evidence_position, reason_position))

        embeddings = self.embedding_function(
            texts,
            self.config.embedding_model,
            self.config.batch_size,
        )

        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise RuntimeError(
                "Embedding function returned an unexpected shape for memory "
                "compatibility scoring."
            )

        query_vector = embeddings[0]
        scored: list[MemoryCandidateScore] = []

        for memory, (evidence_position, reason_position) in zip(
            candidates,
            positions,
            strict=True,
        ):
            evidence_similarity = self._cosine_from_normalized_vectors(
                query_vector,
                embeddings[evidence_position],
            )

            reviewer_reason_similarity: float | None = None
            # Only human-corrected memories have a reviewer reason that should
            # act as a reuse gate. A normal approval keeps the system reason
            # for audit/explanation, but must not treat that system-generated
            # text as if it were a human correction rule.
            if (
                memory.validation_status == "human_corrected"
                and reason_position is not None
            ):
                reviewer_reason_similarity = self._cosine_from_normalized_vectors(
                    query_vector,
                    embeddings[reason_position],
                )

            combined_similarity = self._combined_similarity(
                evidence_similarity=evidence_similarity,
                reviewer_reason_similarity=reviewer_reason_similarity,
            )

            # Normal strict path: reviewer reason is not a separate hard
            # gate; it contributes through combined_similarity. This preserves
            # exact/near-exact reuse even when a short human explanation has
            # only moderate cosine similarity to a long transcript chunk.
            passes_reason_gate = True

            # Near-identical evidence path ONLY for human-corrected memories.
            # A full transcript chunk can be semantically identical to the
            # historical evidence while being very dissimilar to a short human
            # correction reason (for example, a reason that says content is
            # beyond GCSE). In that case the reason must not veto a virtually
            # identical reviewed case. The 0.95 threshold keeps this narrow.
            passes_human_corrected_exact_evidence_gate = (
                memory.validation_status == "human_corrected"
                and evidence_similarity
                >= self.config.human_corrected_exact_evidence_threshold
            )

            # Separate paraphrase path ONLY for human-corrected memories.
            # It is intentionally not available to ordinary approved memory.
            # Same normalized_topic + same spec_version are already enforced by
            # repository lookup. We then require BOTH the historical evidence
            # and the human correction reason to semantically align with the
            # new evidence. This supports genuine paraphrases without globally
            # lowering the normal 0.82/0.80 safety gates.
            passes_human_corrected_paraphrase_gate = (
                memory.validation_status == "human_corrected"
                and reviewer_reason_similarity is not None
                and evidence_similarity
                >= self.config.human_corrected_evidence_threshold
                and reviewer_reason_similarity
                >= self.config.human_corrected_reason_threshold
                and combined_similarity
                >= self.config.human_corrected_combined_threshold
            )

            scored.append(
                MemoryCandidateScore(
                    memory=memory,
                    evidence_similarity=evidence_similarity,
                    reviewer_reason_similarity=reviewer_reason_similarity,
                    combined_similarity=combined_similarity,
                    passes_evidence_gate=(
                        evidence_similarity
                        >= self.config.evidence_similarity_threshold
                    ),
                    passes_reason_gate=passes_reason_gate,
                    passes_combined_gate=(
                        combined_similarity
                        >= self.config.combined_similarity_threshold
                    ),
                    passes_human_corrected_exact_evidence_gate=(
                        passes_human_corrected_exact_evidence_gate
                    ),
                    passes_human_corrected_paraphrase_gate=(
                        passes_human_corrected_paraphrase_gate
                    ),
                )
            )

        return scored

    def _combined_similarity(
        self,
        *,
        evidence_similarity: float,
        reviewer_reason_similarity: float | None,
    ) -> float:
        if reviewer_reason_similarity is None:
            return evidence_similarity

        reason_weight = self.config.reviewer_reason_weight
        evidence_weight = 1.0 - reason_weight
        return (
            evidence_weight * evidence_similarity
            + reason_weight * reviewer_reason_similarity
        )

    @staticmethod
    def _cosine_from_normalized_vectors(
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        # Existing embed_texts() returns normalized vectors. Clamp tiny
        # floating-point drift so logged values stay in the expected range.
        return float(np.clip(np.dot(left, right), -1.0, 1.0))
