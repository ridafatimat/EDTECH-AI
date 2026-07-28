from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean

from app.schemas.topic import ChunkTopicResult, MergedTopic, TopicCandidate


@dataclass(frozen=True)
class TopicMergeConfig:
    """
    Merge repeated official topics and rank lesson topics without allowing
    repeated low-semantic keyword matches to dominate the final order.
    """

    max_evidence_per_topic: int = 5

    # Only a new non-adjacent support span adds confidence.
    non_adjacent_span_bonus: float = 0.02
    maximum_support_bonus: float = 0.06
    maximum_merged_confidence: float = 0.95

    # Ranking combines independent signals. These weights do not change the
    # candidate acceptance threshold; they only order already-retained topics.
    ranking_confidence_weight: float = 0.30
    ranking_semantic_weight: float = 0.30
    ranking_salience_weight: float = 0.25
    ranking_coverage_weight: float = 0.15

    primary_min_semantic_score: float = 0.32
    primary_min_salience_score: float = 0.45
    primary_min_ranking_score: float = 0.48
    primary_min_coverage_score: float = 0.25


class TopicMerger:
    """
    Merge the same official concept across chunks.

    Consecutive chunk IDs are one support span because they may come from one
    long discussion split by Module 2's size guardrail.
    """

    def __init__(self, config: TopicMergeConfig | None = None) -> None:
        self.config = config or TopicMergeConfig()

    def merge(
        self,
        chunk_results: list[ChunkTopicResult],
    ) -> list[MergedTopic]:
        grouped: dict[
            str,
            list[tuple[int, TopicCandidate]],
        ] = defaultdict(list)

        topic_bearing_chunk_count = max(
            1,
            sum(
                1
                for chunk_result in chunk_results
                if chunk_result.topic_candidates
            ),
        )

        for chunk_result in chunk_results:
            for candidate in chunk_result.topic_candidates:
                grouped[candidate.concept_id].append(
                    (chunk_result.chunk_id, candidate)
                )

        merged_topics: list[MergedTopic] = []

        for concept_id, occurrences in grouped.items():
            first_candidate = occurrences[0][1]
            source_chunk_ids = sorted(
                {chunk_id for chunk_id, _ in occurrences}
            )

            support_span_count = self._count_contiguous_spans(
                source_chunk_ids
            )

            base_confidence = max(
                candidate.cs_relevance_score
                for _, candidate in occurrences
            )

            support_bonus = min(
                self.config.maximum_support_bonus,
                self.config.non_adjacent_span_bonus
                * max(0, support_span_count - 1),
            )

            confidence = min(
                self.config.maximum_merged_confidence,
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

            coverage_score = min(
                1.0,
                len(source_chunk_ids) / topic_bearing_chunk_count,
            )

            ranking_score = self._ranking_score(
                confidence=confidence,
                mean_semantic_score=mean_semantic_score,
                mean_salience_score=mean_salience_score,
                coverage_score=coverage_score,
            )

            topic_role = self._topic_role(
                ranking_score=ranking_score,
                mean_semantic_score=mean_semantic_score,
                mean_salience_score=mean_salience_score,
                coverage_score=coverage_score,
                support_span_count=support_span_count,
                supporting_candidate_count=len(occurrences),
            )

            evidence: list[str] = []
            for _, candidate in occurrences:
                evidence.extend(candidate.evidence)

            merged_topics.append(
                MergedTopic(
                    concept_id=concept_id,
                    topic=first_candidate.topic,
                    domain=first_candidate.domain,
                    official_reference=first_candidate.official_reference,
                    chapter_reference=first_candidate.chapter_reference,
                    official_title=first_candidate.official_title,
                    paper=first_candidate.paper,
                    source_pages=first_candidate.source_pages,
                    confidence=round(confidence, 4),
                    ranking_score=round(ranking_score, 4),
                    topic_role=topic_role,
                    source_chunk_ids=source_chunk_ids,
                    support_span_count=support_span_count,
                    mean_semantic_score=round(mean_semantic_score, 4),
                    mean_keyword_score=round(mean_keyword_score, 4),
                    mean_salience_score=round(mean_salience_score, 4),
                    coverage_score=round(coverage_score, 4),
                    evidence=self._unique_strings(evidence)[
                        : self.config.max_evidence_per_topic
                    ],
                    supporting_candidate_count=len(occurrences),
                )
            )

        role_priority = {
            "primary": 1,
            "supporting": 0,
        }

        merged_topics.sort(
            key=lambda topic: (
                role_priority[topic.topic_role],
                topic.ranking_score,
                topic.mean_semantic_score,
                topic.confidence,
            ),
            reverse=True,
        )

        return merged_topics

    def _ranking_score(
        self,
        confidence: float,
        mean_semantic_score: float,
        mean_salience_score: float,
        coverage_score: float,
    ) -> float:
        semantic_component = max(
            0.0,
            min(1.0, mean_semantic_score),
        )

        score = (
            self.config.ranking_confidence_weight * confidence
            + self.config.ranking_semantic_weight * semantic_component
            + self.config.ranking_salience_weight * mean_salience_score
            + self.config.ranking_coverage_weight * coverage_score
        )

        return max(0.0, min(1.0, score))

    def _topic_role(
        self,
        ranking_score: float,
        mean_semantic_score: float,
        mean_salience_score: float,
        coverage_score: float,
        support_span_count: int,
        supporting_candidate_count: int,
    ) -> str:
        has_broad_lesson_support = any(
            (
                coverage_score >= self.config.primary_min_coverage_score,
                support_span_count >= 2,
                supporting_candidate_count >= 3,
            )
        )

        if (
            ranking_score >= self.config.primary_min_ranking_score
            and mean_semantic_score >= self.config.primary_min_semantic_score
            and mean_salience_score >= self.config.primary_min_salience_score
            and has_broad_lesson_support
        ):
            return "primary"

        return "supporting"

    @staticmethod
    def _count_contiguous_spans(chunk_ids: list[int]) -> int:
        if not chunk_ids:
            return 0

        spans = 1
        for previous, current in zip(chunk_ids, chunk_ids[1:]):
            if current > previous + 1:
                spans += 1
        return spans

    @staticmethod
    def _unique_strings(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = " ".join(value.lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(value.strip())

        return unique