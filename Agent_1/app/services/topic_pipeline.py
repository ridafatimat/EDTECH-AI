from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.schemas.topic import (
    ChunkTopicResult,
    Module3Result,
    UnmappedCSSignal,
)
from app.services.cs_relevance_filter import CSRelevanceFilter
from app.services.cs_unmapped_detector import CSUnmappedDetector
from app.services.topic_candidate_extractor import TopicCandidateExtractor
from app.services.topic_merger import TopicMerger


@dataclass(frozen=True)
class Module3PipelineConfig:
    """
    Decision rules for LLM fallback and continuation handling.

    Strong, specific lexical evidence can be retained directly. Semantic-only
    or ambiguous evidence is escalated because it requires interpretation.
    """

    strong_unmapped_lexical_score: float = 0.70
    strong_unmapped_semantic_score: float = 0.82
    ambiguous_signal_margin: float = 0.04

    def __post_init__(self) -> None:
        for field_name in (
            "strong_unmapped_lexical_score",
            "strong_unmapped_semantic_score",
            "ambiguous_signal_margin",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )


class Module3TopicPipeline:
    """
    Complete Module 3 pipeline:

    Module 2 chunks
        → official AQA candidate extraction
        → salience-aware filtering
        → continuation/no-new-topic handling
        → unmapped CS detection
        → official topic merging
    """

    def __init__(
        self,
        extractor: TopicCandidateExtractor | None = None,
        relevance_filter: CSRelevanceFilter | None = None,
        unmapped_detector: CSUnmappedDetector | None = None,
        merger: TopicMerger | None = None,
        config: Module3PipelineConfig | None = None,
    ) -> None:
        self.extractor = extractor or TopicCandidateExtractor()
        self.relevance_filter = relevance_filter or CSRelevanceFilter()
        self.unmapped_detector = unmapped_detector or CSUnmappedDetector()
        self.merger = merger or TopicMerger()
        self.config = config or Module3PipelineConfig()

    def process_chunks(self, chunks: Sequence[Any]) -> Module3Result:
        chunk_results: list[ChunkTopicResult] = []

        for raw_chunk in chunks:
            chunk_id = int(self._get_value(raw_chunk, "chunk_id"))
            text = str(self._get_value(raw_chunk, "text")).strip()

            word_count_value = self._get_optional_value(
                raw_chunk,
                "word_count",
            )
            word_count = (
                int(word_count_value)
                if word_count_value is not None
                else len(re.findall(r"\S+", text))
            )

            overlap_word_count = int(
                self._get_optional_value(
                    raw_chunk,
                    "overlap_word_count",
                )
                or 0
            )

            raw_candidates = self.extractor.extract(
                chunk_id=chunk_id,
                text=text,
            )

            base_result = self.relevance_filter.filter(
                chunk_id=chunk_id,
                source_word_count=word_count,
                candidates=raw_candidates,
            )

            previous_result = chunk_results[-1] if chunk_results else None

            # A chunk created with overlap immediately after a retained CS
            # chunk may simply conclude the same question. It should not be
            # forced into a new topic or sent to the unmapped detector.
            current_topic_ids = {
                candidate.concept_id
                for candidate in base_result.topic_candidates
            }
            previous_topic_ids = self._effective_previous_topic_ids(
                previous_result=previous_result,
                completed_results=chunk_results,
            )

            continuation_only = (
                overlap_word_count > 0
                and previous_result is not None
                and bool(previous_topic_ids)
                and (
                    not current_topic_ids
                    or current_topic_ids.issubset(previous_topic_ids)
                )
            )

            if continuation_only:
                final_result = base_result.model_copy(
                    update={
                        "classification": "continuation_no_new_topic",
                        "is_cs_relevant": False,
                        "creates_new_topic": False,
                        "cs_relevance_score": 0.0,
                        "topic_candidates": [],
                        "has_unmapped_cs_content": False,
                        "unmapped_cs_signals": [],
                        "continuation_of_chunk_id": (
                            previous_result.continuation_of_chunk_id
                            or previous_result.chunk_id
                        ),
                        "requires_llm_fallback": False,
                        "notes": [
                            "No new standalone topic was detected; the chunk "
                            "continues the previous overlapped discussion."
                        ],
                    }
                )
                chunk_results.append(final_result)
                continue

            unmapped_signals = self.unmapped_detector.detect(
                text=text,
                official_candidates=base_result.topic_candidates,
            )
            has_unmapped = bool(unmapped_signals)
            has_official = bool(base_result.topic_candidates)

            if has_official and has_unmapped:
                classification = "mixed_official_and_unmapped"
            elif has_official:
                classification = "official_aqa_topic"
            elif has_unmapped:
                classification = "cs_related_unmapped"
            else:
                classification = "no_topic"

            unmapped_requires_llm = (
                self._unmapped_requires_llm_fallback(unmapped_signals)
            )

            requires_llm_fallback = (
                unmapped_requires_llm
                or (
                    base_result.requires_llm_fallback
                    and not has_unmapped
                )
            )

            notes = list(base_result.notes)
            if has_unmapped and unmapped_requires_llm:
                notes.append(
                    "Unmapped CS evidence is semantic-only, borderline or "
                    "ambiguous, so later fallback should refine the rough "
                    "topic without inventing an AQA label."
                )
            elif has_unmapped:
                notes.append(
                    "Strong specific unmapped-CS evidence was retained "
                    "directly; no LLM fallback is required."
                )

            final_score = base_result.cs_relevance_score
            if has_unmapped and not has_official:
                final_score = max(
                    signal.score for signal in unmapped_signals
                )

            final_result = base_result.model_copy(
                update={
                    "classification": classification,
                    "is_cs_relevant": has_official or has_unmapped,
                    "creates_new_topic": has_official or has_unmapped,
                    "cs_relevance_score": round(final_score, 4),
                    "has_unmapped_cs_content": has_unmapped,
                    "unmapped_cs_signals": unmapped_signals,
                    "requires_llm_fallback": requires_llm_fallback,
                    "notes": notes,
                }
            )

            chunk_results.append(final_result)

        merged_topics = self.merger.merge(chunk_results)

        classification_counts = {
            "official_aqa_topic": 0,
            "mixed_official_and_unmapped": 0,
            "cs_related_unmapped": 0,
            "continuation_no_new_topic": 0,
            "no_topic": 0,
        }

        for result in chunk_results:
            classification_counts[result.classification] += 1

        cs_relevant_chunks = sum(
            1 for result in chunk_results if result.is_cs_relevant
        )

        llm_fallback_ids = [
            result.chunk_id
            for result in chunk_results
            if result.requires_llm_fallback
        ]

        return Module3Result(
            chunk_results=chunk_results,
            merged_topics=merged_topics,
            total_chunks=len(chunk_results),
            cs_relevant_chunks=cs_relevant_chunks,
            non_cs_chunks=len(chunk_results) - cs_relevant_chunks,
            official_topic_chunks=classification_counts[
                "official_aqa_topic"
            ],
            mixed_official_unmapped_chunks=classification_counts[
                "mixed_official_and_unmapped"
            ],
            unmapped_cs_chunks=classification_counts[
                "cs_related_unmapped"
            ],
            continuation_chunks=classification_counts[
                "continuation_no_new_topic"
            ],
            no_topic_chunks=classification_counts["no_topic"],
            llm_fallback_chunk_ids=llm_fallback_ids,
            embedding_model=self.extractor.config.embedding_model,
            candidate_keep_threshold=(
                self.relevance_filter.config.candidate_keep_threshold
            ),
        )

    @staticmethod
    def _effective_previous_topic_ids(
        previous_result: ChunkTopicResult | None,
        completed_results: list[ChunkTopicResult],
    ) -> set[str]:
        if previous_result is None:
            return set()

        direct_ids = {
            candidate.concept_id
            for candidate in previous_result.topic_candidates
        }
        if direct_ids:
            return direct_ids

        source_id = previous_result.continuation_of_chunk_id
        if source_id is None:
            return set()

        for result in reversed(completed_results):
            if result.chunk_id != source_id:
                continue

            return {
                candidate.concept_id
                for candidate in result.topic_candidates
            }

        return set()

    def _unmapped_requires_llm_fallback(
        self,
        signals: list[UnmappedCSSignal],
    ) -> bool:
        """
        Escalate only evidence that genuinely needs interpretation.

        Directly accept strong lexical or lexical-semantic rough topics.
        Escalate semantic-only, borderline, or same-evidence ambiguous signals.
        """

        if not signals:
            return False

        for signal in signals:
            if signal.detection_method == "semantic":
                if (
                    signal.score
                    < self.config.strong_unmapped_semantic_score
                ):
                    return True
            elif (
                signal.score
                < self.config.strong_unmapped_lexical_score
            ):
                return True

        evidence_groups: dict[str, list[UnmappedCSSignal]] = {}

        for signal in signals:
            evidence_key = self._normalise_for_decision(signal.evidence)
            evidence_groups.setdefault(evidence_key, []).append(signal)

        for grouped_signals in evidence_groups.values():
            distinct_topics = {
                self._normalise_for_decision(signal.rough_topic)
                for signal in grouped_signals
            }

            if len(distinct_topics) < 2:
                continue

            ordered_scores = sorted(
                (signal.score for signal in grouped_signals),
                reverse=True,
            )

            if (
                ordered_scores[0] - ordered_scores[1]
                <= self.config.ambiguous_signal_margin
            ):
                return True

        return False

    @staticmethod
    def _normalise_for_decision(text: str) -> str:
        text = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _get_value(item: Any, field_name: str) -> Any:
        if isinstance(item, dict):
            if field_name not in item:
                raise KeyError(f"Missing chunk field: {field_name}")
            return item[field_name]

        if not hasattr(item, field_name):
            raise AttributeError(f"Chunk has no field: {field_name}")
        return getattr(item, field_name)

    @staticmethod
    def _get_optional_value(item: Any, field_name: str) -> Any | None:
        if isinstance(item, dict):
            return item.get(field_name)
        return getattr(item, field_name, None)