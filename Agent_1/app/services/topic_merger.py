from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean

from app.schemas.topic import ChunkTopicResult, MergedTopic, TopicCandidate


@dataclass(frozen=True)
class TopicMergeConfig:
    """
    Merge repeated official topics and rank their importance in the lesson.

    Score responsibilities are deliberately separate:

    confidence:
        certainty that a topic exists;

    coverage:
        breadth of the lesson containing the topic;

    ranking:
        estimated lesson importance;

    topic role:
        relative primary/supporting classification based mainly on coverage
        and ranking, never confidence alone.
    """

    max_evidence_per_topic: int = 5

    # Only a new non-adjacent support span adds confidence.
    non_adjacent_span_bonus: float = 0.02
    maximum_support_bonus: float = 0.06
    maximum_merged_confidence: float = 0.95

    # Coverage is weighted most strongly because primary/supporting is about
    # lesson importance, not merely certainty that a topic was mentioned.
    ranking_confidence_weight: float = 0.20
    ranking_semantic_weight: float = 0.20
    ranking_salience_weight: float = 0.20
    ranking_coverage_weight: float = 0.40

    # Relative promotion rules.
    minimum_primary_coverage: float = 0.35
    dominant_coverage_ratio: float = 0.75
    primary_ranking_margin: float = 0.12
    minimum_primary_ranking_score: float = 0.42

    def __post_init__(self) -> None:
        weights = (
            self.ranking_confidence_weight,
            self.ranking_semantic_weight,
            self.ranking_salience_weight,
            self.ranking_coverage_weight,
        )

        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError(
                "Ranking weights must sum to 1.0."
            )

        probability_fields = (
            "non_adjacent_span_bonus",
            "maximum_support_bonus",
            "maximum_merged_confidence",
            "minimum_primary_coverage",
            "dominant_coverage_ratio",
            "primary_ranking_margin",
            "minimum_primary_ranking_score",
        )

        for field_name in probability_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )


class TopicMerger:
    """
    Merge the same official concept across chunks.

    Consecutive chunk IDs are one support span because they may come from one
    long discussion split by Module 2's size guardrail.

    Roles are assigned only after every merged topic has been scored, allowing
    primary/supporting decisions to be lesson-relative rather than based on
    independent absolute thresholds.
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

        provisional_topics: list[MergedTopic] = []

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

            evidence: list[str] = []
            for _, candidate in occurrences:
                evidence.extend(candidate.evidence)

            provisional_topics.append(
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
                    topic_role="supporting",
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

        merged_topics = self._assign_relative_roles(
            provisional_topics
        )

        role_priority = {
            "primary": 1,
            "supporting": 0,
        }

        merged_topics.sort(
            key=lambda topic: (
                role_priority[topic.topic_role],
                topic.ranking_score,
                topic.coverage_score,
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

    def _assign_relative_roles(
        self,
        topics: list[MergedTopic],
    ) -> list[MergedTopic]:
        if not topics:
            return []

        max_ranking = max(
            topic.ranking_score
            for topic in topics
        )
        max_coverage = max(
            topic.coverage_score
            for topic in topics
        )

        dominant_coverage_threshold = max(
            self.config.minimum_primary_coverage,
            max_coverage
            * self.config.dominant_coverage_ratio,
        )

        ranking_threshold = max(
            self.config.minimum_primary_ranking_score,
            max_ranking
            - self.config.primary_ranking_margin,
        )

        promoted: list[MergedTopic] = []

        for topic in topics:
            is_primary = (
                topic.coverage_score
                >= dominant_coverage_threshold
                and topic.ranking_score
                >= ranking_threshold
            )

            promoted.append(
                topic.model_copy(
                    update={
                        "topic_role": (
                            "primary"
                            if is_primary
                            else "supporting"
                        )
                    }
                )
            )

        # A lesson with valid official topics should always have at least one
        # primary topic. This fallback chooses the strongest lesson-relative
        # topic, preferring ranking and coverage over raw confidence.
        if not any(
            topic.topic_role == "primary"
            for topic in promoted
        ):
            best_topic = max(
                promoted,
                key=lambda topic: (
                    topic.ranking_score,
                    topic.coverage_score,
                    topic.mean_salience_score,
                    topic.mean_semantic_score,
                ),
            )

            promoted = [
                (
                    topic.model_copy(
                        update={
                            "topic_role": "primary"
                        }
                    )
                    if topic.concept_id == best_topic.concept_id
                    else topic
                )
                for topic in promoted
            ]

        return promoted

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