from __future__ import annotations

from dataclasses import dataclass

from app.schemas.topic import ChunkTopicResult, RawTopicCandidate, TopicCandidate


@dataclass(frozen=True)
class CSRelevanceConfig:
    candidate_keep_threshold: float = 0.46
    uncertain_candidate_floor: float = 0.34
    llm_fallback_min_words: int = 80
    max_rejected_candidates: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.candidate_keep_threshold <= 1.0:
            raise ValueError("candidate_keep_threshold must be between 0 and 1.")
        if not 0.0 <= self.uncertain_candidate_floor <= 1.0:
            raise ValueError("uncertain_candidate_floor must be between 0 and 1.")
        if self.uncertain_candidate_floor > self.candidate_keep_threshold:
            raise ValueError(
                "uncertain_candidate_floor cannot exceed candidate_keep_threshold."
            )


class CSRelevanceFilter:
    def __init__(self, config: CSRelevanceConfig | None = None) -> None:
        self.config = config or CSRelevanceConfig()

    def filter(
        self,
        chunk_id: int,
        source_word_count: int,
        candidates: list[RawTopicCandidate],
    ) -> ChunkTopicResult:
        relevant: list[TopicCandidate] = []
        rejected: list[TopicCandidate] = []

        for candidate in candidates:
            relevance_score = candidate.confidence
            is_relevant = relevance_score >= self.config.candidate_keep_threshold
            filtered_candidate = TopicCandidate(
                **candidate.model_dump(),
                cs_relevance_score=round(relevance_score, 4),
                cs_relevant=is_relevant,
            )
            (relevant if is_relevant else rejected).append(filtered_candidate)

        relevant.sort(
            key=lambda candidate: (
                candidate.cs_relevance_score,
                candidate.salience_score,
            ),
            reverse=True,
        )
        rejected.sort(
            key=lambda candidate: candidate.cs_relevance_score,
            reverse=True,
        )

        if relevant:
            best_score = max(c.cs_relevance_score for c in relevant)
            support_bonus = min(0.06, 0.02 * max(0, len(relevant) - 1))
            chunk_relevance_score = min(0.95, best_score + support_bonus)
        elif rejected:
            best_score = max(c.cs_relevance_score for c in rejected)
            chunk_relevance_score = best_score
        else:
            best_score = 0.0
            chunk_relevance_score = 0.0

        requires_llm_fallback = (
            not relevant
            and source_word_count >= self.config.llm_fallback_min_words
            and best_score >= self.config.uncertain_candidate_floor
        )

        notes: list[str] = []
        if not candidates:
            notes.append("No official AQA topic candidate was detected.")
        elif not relevant:
            notes.append("Only low-confidence official AQA candidates were detected.")
        if requires_llm_fallback:
            notes.append("Chunk is borderline and may require GPT-OSS fallback.")

        return ChunkTopicResult(
            chunk_id=chunk_id,
            source_word_count=source_word_count,
            classification="official_aqa_topic" if relevant else "no_topic",
            is_cs_relevant=bool(relevant),
            creates_new_topic=bool(relevant),
            cs_relevance_score=round(chunk_relevance_score, 4),
            topic_candidates=relevant,
            rejected_candidates=rejected[: self.config.max_rejected_candidates],
            requires_llm_fallback=requires_llm_fallback,
            notes=notes,
        )