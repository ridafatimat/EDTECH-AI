from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from app.schemas.chunk import (
    ChunkingResult,
    TranscriptChunk,
)
from app.services.embedding_service import (
    CHUNKING_EMBEDDING_MODEL,
    embed_texts,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticChunkingConfig:
    """
    Configuration for guarded semantic chunking.

    Design goals:
    - avoid over-fragmenting teacher/student discussion
    - respect explicit lesson transitions
    - prevent excessively large chunks
    - use overlap only when size forces a split
    - add deterministic logical segment metadata
    - use MiniLM-specific adaptive similarity thresholds
    """

    min_chunk_words: int = 150
    target_chunk_words: int = 325
    max_chunk_words: int = 550

    strong_transition_min_words: int = 80
    semantic_unit_words: int = 60

    boundary_percentile: float = 15.0
    threshold_floor: float = 0.10
    threshold_ceiling: float = 0.45

    soft_transition_margin: float = 0.10
    soft_transition_similarity_ceiling: float = 0.35

    size_penalty_weight: float = 0.12
    strong_transition_bonus: float = 0.10
    soft_transition_bonus: float = 0.04

    max_size_overlap_words: int = 45
    max_size_overlap_sentences: int = 2

    embedding_model: str = CHUNKING_EMBEDDING_MODEL

    def __post_init__(self) -> None:
        if not (
            0
            < self.strong_transition_min_words
            <= self.min_chunk_words
            <= self.target_chunk_words
            <= self.max_chunk_words
        ):
            raise ValueError(
                "Chunk sizes must satisfy: "
                "0 < strong_transition_min <= min <= target <= max."
            )

        if self.semantic_unit_words <= 0:
            raise ValueError(
                "semantic_unit_words must be positive."
            )

        if not 0 <= self.boundary_percentile <= 100:
            raise ValueError(
                "boundary_percentile must be between 0 and 100."
            )

        if not -1.0 <= self.threshold_floor <= 1.0:
            raise ValueError(
                "threshold_floor must be between -1 and 1."
            )

        if not -1.0 <= self.threshold_ceiling <= 1.0:
            raise ValueError(
                "threshold_ceiling must be between -1 and 1."
            )

        if self.threshold_floor > self.threshold_ceiling:
            raise ValueError(
                "threshold_floor cannot be greater than threshold_ceiling."
            )

        if self.soft_transition_margin < 0:
            raise ValueError(
                "soft_transition_margin cannot be negative."
            )

        if not -1.0 <= self.soft_transition_similarity_ceiling <= 1.0:
            raise ValueError(
                "soft_transition_similarity_ceiling must be between -1 and 1."
            )

        if self.max_size_overlap_words < 0:
            raise ValueError(
                "max_size_overlap_words cannot be negative."
            )

        if self.max_size_overlap_sentences < 0:
            raise ValueError(
                "max_size_overlap_sentences cannot be negative."
            )


# ---------------------------------------------------------------------
# Internal structures
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _SemanticUnit:
    text: str
    start_sentence: int
    end_sentence: int
    word_count: int

    # Transition at the START of this unit.
    transition_strength: str | None = None


@dataclass(frozen=True)
class _Boundary:
    """Candidate boundary occurring AFTER unit_index."""

    unit_index: int
    similarity: float
    reason: str
    transition_strength: str | None = None


@dataclass(frozen=True)
class _ChunkPlan:
    """
    Non-overlapping/core chunk boundaries.

    Overlap and logical segment metadata are added while materializing
    final TranscriptChunk objects.
    """

    start_unit: int
    end_unit: int
    reason: str
    similarity: float | None
    transition_strength: str | None = None


# ---------------------------------------------------------------------
# Semantic Chunker
# ---------------------------------------------------------------------


class SemanticChunker:
    """
    Module 2.

    Input:
        Final cleaned transcript from Module 1.

    Output:
        Meaningful physical chunks plus logical segment metadata.

    A logical segment may contain multiple physical chunks when a long,
    continuous discussion must be split only because of max_chunk_words.

    This module does NOT:
    - identify topics
    - map to the AQA syllabus
    - call an LLM
    - store transcript embeddings
    """

    STRONG_TRANSITION_PATTERNS = (
        r"\bnext (?:chapter|topic|concept|section|question)\b",
        r"\bthe next (?:chapter|topic|concept|section|question)\b",
        r"\bstart (?:the )?(?:next )?chapter\b",
        r"\bbegin (?:the )?(?:next )?chapter\b",
        r"\bchapter (?:number )?\d+\b",
        r"\bnew (?:chapter|topic|concept|section)\b",
        r"\bmove on to (?:the )?(?:next )?(?:chapter|topic|concept|section)\b",
        r"\blet'?s move on to (?:the )?next (?:thing|one|question)\b",
        r"\bmove on to (?:the )?next (?:thing|one|question)\b",
        r"\blet'?s look at (?:the )?next question\b",
        r"\blet'?s go to (?:the )?next question\b",
    )

    SOFT_TRANSITION_PATTERNS = (
        r"\blet'?s move on\b",
        r"\bmove on to\b",
        r"\bmoving on\b",
        r"\blet'?s talk about\b",
        r"\blet'?s discuss\b",
        r"\blet'?s look at\b",
        r"\bnext one\b",
        r"\bnext thing\b",
        r"\bnow (?:we can|we will|let'?s) move\b",
        r"\bnow (?:we can|we will|let'?s) (?:talk|discuss|look)\b",
    )

    def __init__(
        self,
        config: SemanticChunkingConfig | None = None,
    ) -> None:
        self.config = (
            config
            or SemanticChunkingConfig()
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def chunk(
        self,
        cleaned_transcript: str,
    ) -> ChunkingResult:
        """Convert a cleaned transcript into guarded semantic chunks."""

        if not isinstance(
            cleaned_transcript,
            str,
        ):
            raise TypeError(
                "cleaned_transcript must be a string."
            )

        cleaned_transcript = cleaned_transcript.strip()

        if not cleaned_transcript:
            raise ValueError(
                "Cannot chunk an empty transcript."
            )

        sentences = self._split_sentences(
            cleaned_transcript
        )

        if not sentences:
            raise ValueError(
                "No usable sentences found in transcript."
            )

        units = self._build_semantic_units(
            sentences
        )

        if not units:
            raise ValueError(
                "No semantic units could be created."
            )

        if len(units) == 1:
            unit = units[0]

            chunk = TranscriptChunk(
                chunk_id=1,
                text=unit.text,
                word_count=unit.word_count,
                sentence_count=(
                    unit.end_sentence
                    - unit.start_sentence
                    + 1
                ),
                start_sentence=unit.start_sentence,
                end_sentence=unit.end_sentence,
                core_start_sentence=unit.start_sentence,
                core_end_sentence=unit.end_sentence,
                boundary_reason="end_of_transcript",
                boundary_similarity=None,
                boundary_transition_strength=None,
                overlap_word_count=0,
                segment_id="segment_001",
                segment_root_chunk_id=1,
                segment_chunk_index=1,
                segment_chunk_count=1,
                segment_position="single",
                is_continuation=False,
                continuation_of_chunk_id=None,
                continuation_reason=None,
            )

            return self._build_result(
                chunks=[chunk],
                sentences=sentences,
                units=units,
                cleaned_transcript=cleaned_transcript,
                semantic_threshold=self.config.threshold_ceiling,
            )

        unit_embeddings = embed_texts(
            [
                unit.text
                for unit in units
            ],
            model_name=self.config.embedding_model,
        )

        similarities = self._neighbour_similarities(
            unit_embeddings
        )

        semantic_threshold = self._calculate_threshold(
            similarities
        )

        boundaries = self._detect_boundaries(
            units=units,
            similarities=similarities,
            semantic_threshold=semantic_threshold,
        )

        plans = self._build_chunk_plans(
            units=units,
            similarities=similarities,
            boundaries=boundaries,
        )

        chunks = self._materialize_chunks(
            plans=plans,
            units=units,
            sentences=sentences,
        )

        return self._build_result(
            chunks=chunks,
            sentences=sentences,
            units=units,
            cleaned_transcript=cleaned_transcript,
            semantic_threshold=semantic_threshold,
        )

    def _build_result(
        self,
        chunks: list[TranscriptChunk],
        sentences: list[str],
        units: list[_SemanticUnit],
        cleaned_transcript: str,
        semantic_threshold: float,
    ) -> ChunkingResult:
        segment_count = len(
            {
                chunk.segment_id
                for chunk in chunks
            }
        )

        return ChunkingResult(
            chunks=chunks,
            total_sentences=len(sentences),
            total_words=self._word_count(
                cleaned_transcript
            ),
            semantic_unit_count=len(units),
            segment_count=segment_count,
            embedding_model=self.config.embedding_model,
            semantic_threshold=round(
                semantic_threshold,
                4,
            ),
            min_chunk_words=self.config.min_chunk_words,
            target_chunk_words=self.config.target_chunk_words,
            max_chunk_words=self.config.max_chunk_words,
            max_size_overlap_words=(
                self.config.max_size_overlap_words
            ),
        )

    # -----------------------------------------------------------------
    # Sentence preparation
    # -----------------------------------------------------------------

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:
        """
        Lightweight sentence segmentation.

        A period inside `array.length` is not split because there is no
        whitespace after that period.
        """

        text = text.replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\s*\n+\s*",
            " ",
            text,
        )

        parts = re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    # -----------------------------------------------------------------
    # Transition detection
    # -----------------------------------------------------------------

    def _transition_strength(
        self,
        sentence: str,
    ) -> str | None:
        head = sentence[:220].lower()

        if any(
            re.search(pattern, head)
            for pattern
            in self.STRONG_TRANSITION_PATTERNS
        ):
            return "strong"

        if any(
            re.search(pattern, head)
            for pattern
            in self.SOFT_TRANSITION_PATTERNS
        ):
            return "soft"

        return None

    # -----------------------------------------------------------------
    # Semantic units
    # -----------------------------------------------------------------

    def _build_semantic_units(
        self,
        sentences: list[str],
    ) -> list[_SemanticUnit]:
        units: list[_SemanticUnit] = []

        buffer: list[str] = []
        buffer_words = 0
        start_sentence = 0
        buffer_transition_strength: str | None = None

        def flush_buffer(
            end_sentence: int,
        ) -> None:
            nonlocal buffer
            nonlocal buffer_words
            nonlocal start_sentence
            nonlocal buffer_transition_strength

            if not buffer:
                return

            units.append(
                _SemanticUnit(
                    text=" ".join(buffer),
                    start_sentence=start_sentence,
                    end_sentence=end_sentence,
                    word_count=buffer_words,
                    transition_strength=(
                        buffer_transition_strength
                    ),
                )
            )

            buffer = []
            buffer_words = 0
            buffer_transition_strength = None

        for sentence_index, sentence in enumerate(
            sentences
        ):
            transition_strength = self._transition_strength(
                sentence
            )

            if transition_strength and buffer:
                flush_buffer(
                    sentence_index - 1
                )

            if not buffer:
                start_sentence = sentence_index
                buffer_transition_strength = (
                    transition_strength
                )

            buffer.append(sentence)
            buffer_words += self._word_count(
                sentence
            )

            if (
                buffer_words
                >= self.config.semantic_unit_words
            ):
                flush_buffer(
                    sentence_index
                )

        if buffer:
            flush_buffer(
                len(sentences) - 1
            )

        if len(units) >= 2:
            minimum_tail = max(
                15,
                self.config.semantic_unit_words
                // 2,
            )

            last = units[-1]

            if (
                last.word_count < minimum_tail
                and last.transition_strength is None
            ):
                previous = units[-2]

                units[-2] = _SemanticUnit(
                    text=(
                        previous.text
                        + " "
                        + last.text
                    ),
                    start_sentence=(
                        previous.start_sentence
                    ),
                    end_sentence=(
                        last.end_sentence
                    ),
                    word_count=(
                        previous.word_count
                        + last.word_count
                    ),
                    transition_strength=(
                        previous.transition_strength
                    ),
                )

                units.pop()

        return units

    # -----------------------------------------------------------------
    # Similarity
    # -----------------------------------------------------------------

    @staticmethod
    def _neighbour_similarities(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        if len(embeddings) < 2:
            return np.array(
                [],
                dtype=np.float32,
            )

        similarities = np.sum(
            embeddings[:-1]
            * embeddings[1:],
            axis=1,
        )

        return similarities.astype(
            np.float32
        )

    def _calculate_threshold(
        self,
        similarities: np.ndarray,
    ) -> float:
        if len(similarities) == 0:
            return self.config.threshold_ceiling

        percentile_value = float(
            np.percentile(
                similarities,
                self.config.boundary_percentile,
            )
        )

        threshold = max(
            self.config.threshold_floor,
            min(
                self.config.threshold_ceiling,
                percentile_value,
            ),
        )

        return round(
            threshold,
            4,
        )

    def _soft_transition_threshold(
        self,
        semantic_threshold: float,
    ) -> float:
        return min(
            self.config.soft_transition_similarity_ceiling,
            semantic_threshold
            + self.config.soft_transition_margin,
        )

    # -----------------------------------------------------------------
    # Boundary detection
    # -----------------------------------------------------------------

    def _detect_boundaries(
        self,
        units: list[_SemanticUnit],
        similarities: np.ndarray,
        semantic_threshold: float,
    ) -> dict[int, _Boundary]:
        boundaries: dict[int, _Boundary] = {}

        soft_threshold = self._soft_transition_threshold(
            semantic_threshold
        )

        for index, similarity_value in enumerate(
            similarities
        ):
            similarity = float(
                similarity_value
            )

            semantic_shift = (
                similarity
                <= semantic_threshold
            )

            next_unit = units[index + 1]

            transition_strength = (
                next_unit.transition_strength
            )

            strong_transition = (
                transition_strength == "strong"
            )

            soft_transition = (
                transition_strength == "soft"
                and similarity <= soft_threshold
            )

            transition_boundary = (
                strong_transition
                or soft_transition
            )

            if not (
                semantic_shift
                or transition_boundary
            ):
                continue

            if (
                semantic_shift
                and transition_boundary
            ):
                reason = (
                    "semantic_shift+transition_phrase"
                )
            elif transition_boundary:
                reason = "transition_phrase"
            else:
                reason = "semantic_shift"

            boundaries[index] = _Boundary(
                unit_index=index,
                similarity=similarity,
                reason=reason,
                transition_strength=(
                    transition_strength
                    if transition_boundary
                    else None
                ),
            )

        return boundaries

    # -----------------------------------------------------------------
    # Chunk planning
    # -----------------------------------------------------------------

    def _build_chunk_plans(
        self,
        units: list[_SemanticUnit],
        similarities: np.ndarray,
        boundaries: dict[int, _Boundary],
    ) -> list[_ChunkPlan]:
        prefix_words = [0]

        for unit in units:
            prefix_words.append(
                prefix_words[-1]
                + unit.word_count
            )

        def words_between(
            start: int,
            end: int,
        ) -> int:
            return (
                prefix_words[end + 1]
                - prefix_words[start]
            )

        plans: list[_ChunkPlan] = []
        start = 0
        number_of_units = len(units)

        while start < number_of_units:
            remaining_words = words_between(
                start,
                number_of_units - 1,
            )

            candidates: list[
                tuple[int, _Boundary]
            ] = []

            for end_index in range(
                start,
                number_of_units - 1,
            ):
                current_words = words_between(
                    start,
                    end_index,
                )

                if (
                    current_words
                    > self.config.max_chunk_words
                ):
                    break

                boundary = boundaries.get(
                    end_index
                )

                if boundary is None:
                    continue

                required_min = (
                    self.config.strong_transition_min_words
                    if (
                        boundary.transition_strength
                        == "strong"
                    )
                    else self.config.min_chunk_words
                )

                if current_words < required_min:
                    continue

                words_after = words_between(
                    end_index + 1,
                    number_of_units - 1,
                )

                if (
                    0
                    < words_after
                    < self.config.min_chunk_words
                ):
                    continue

                candidates.append(
                    (
                        end_index,
                        boundary,
                    )
                )

            if candidates:
                def candidate_score(
                    candidate: tuple[
                        int,
                        _Boundary,
                    ],
                ) -> float:
                    end_index, boundary = candidate

                    current_words = words_between(
                        start,
                        end_index,
                    )

                    size_distance = abs(
                        current_words
                        - self.config.target_chunk_words
                    )

                    size_penalty = (
                        size_distance
                        / self.config.target_chunk_words
                    )

                    score = (
                        boundary.similarity
                        + (
                            self.config.size_penalty_weight
                            * size_penalty
                        )
                    )

                    if (
                        boundary.transition_strength
                        == "strong"
                    ):
                        score -= (
                            self.config.strong_transition_bonus
                        )
                    elif (
                        boundary.transition_strength
                        == "soft"
                    ):
                        score -= (
                            self.config.soft_transition_bonus
                        )

                    return score

                end, selected = min(
                    candidates,
                    key=candidate_score,
                )

                plans.append(
                    _ChunkPlan(
                        start_unit=start,
                        end_unit=end,
                        reason=selected.reason,
                        similarity=selected.similarity,
                        transition_strength=(
                            selected.transition_strength
                        ),
                    )
                )

                start = end + 1
                continue

            if (
                remaining_words
                <= self.config.max_chunk_words
            ):
                plans.append(
                    _ChunkPlan(
                        start_unit=start,
                        end_unit=(
                            number_of_units - 1
                        ),
                        reason="end_of_transcript",
                        similarity=None,
                        transition_strength=None,
                    )
                )
                break

            possible_ends: list[int] = []

            for end_index in range(
                start,
                number_of_units - 1,
            ):
                current_words = words_between(
                    start,
                    end_index,
                )

                if (
                    current_words
                    > self.config.max_chunk_words
                ):
                    break

                words_after = words_between(
                    end_index + 1,
                    number_of_units - 1,
                )

                if (
                    words_after
                    >= self.config.min_chunk_words
                ):
                    possible_ends.append(
                        end_index
                    )

            if possible_ends:
                end = possible_ends[-1]
            else:
                end = start

            if end < len(similarities):
                similarity: float | None = float(
                    similarities[end]
                )
            else:
                similarity = None

            plans.append(
                _ChunkPlan(
                    start_unit=start,
                    end_unit=end,
                    reason="max_size",
                    similarity=similarity,
                    transition_strength=None,
                )
            )

            start = end + 1

        return plans

    # -----------------------------------------------------------------
    # Materialize chunks + logical segment metadata
    # -----------------------------------------------------------------

    def _materialize_chunks(
        self,
        plans: list[_ChunkPlan],
        units: list[_SemanticUnit],
        sentences: list[str],
    ) -> list[TranscriptChunk]:
        chunks: list[TranscriptChunk] = []

        for index, plan in enumerate(
            plans
        ):
            core_start_sentence = (
                units[
                    plan.start_unit
                ].start_sentence
            )

            core_end_sentence = (
                units[
                    plan.end_unit
                ].end_sentence
            )

            text_start_sentence = (
                core_start_sentence
            )

            overlap_word_count = 0

            if (
                index > 0
                and plans[
                    index - 1
                ].reason == "max_size"
            ):
                (
                    text_start_sentence,
                    overlap_word_count,
                ) = self._find_overlap_start(
                    sentences=sentences,
                    core_start_sentence=(
                        core_start_sentence
                    ),
                )

            selected_sentences = sentences[
                text_start_sentence
                : core_end_sentence + 1
            ]

            text = " ".join(
                selected_sentences
            ).strip()

            chunks.append(
                TranscriptChunk(
                    chunk_id=index + 1,
                    text=text,
                    word_count=self._word_count(
                        text
                    ),
                    sentence_count=(
                        core_end_sentence
                        - text_start_sentence
                        + 1
                    ),
                    start_sentence=(
                        text_start_sentence
                    ),
                    end_sentence=(
                        core_end_sentence
                    ),
                    core_start_sentence=(
                        core_start_sentence
                    ),
                    core_end_sentence=(
                        core_end_sentence
                    ),
                    boundary_reason=(
                        plan.reason
                    ),
                    boundary_similarity=(
                        round(
                            plan.similarity,
                            4,
                        )
                        if (
                            plan.similarity
                            is not None
                        )
                        else None
                    ),
                    boundary_transition_strength=(
                        plan.transition_strength
                    ),
                    overlap_word_count=(
                        overlap_word_count
                    ),
                )
            )

        return self._assign_segment_metadata(
            chunks=chunks,
            plans=plans,
        )

    def _assign_segment_metadata(
        self,
        *,
        chunks: list[TranscriptChunk],
        plans: list[_ChunkPlan],
    ) -> list[TranscriptChunk]:
        """
        Group physical chunks into deterministic logical segments.

        Safe rule:
        - A chunk continues the same segment only when the previous
          chunk ended because of `max_size`.
        - Semantic and transition boundaries start a new segment.

        This avoids guessing topic identity inside Module 2 while still
        telling Module 3 which chunks are definitely continuations.
        """

        if not chunks:
            return []

        assignments: list[dict[str, object]] = []

        segment_number = 1
        segment_root_chunk_id = chunks[0].chunk_id
        segment_chunk_index = 1

        assignments.append(
            {
                "segment_number": segment_number,
                "segment_root_chunk_id": (
                    segment_root_chunk_id
                ),
                "segment_chunk_index": (
                    segment_chunk_index
                ),
                "is_continuation": False,
                "continuation_of_chunk_id": None,
                "continuation_reason": None,
            }
        )

        for index in range(
            1,
            len(chunks),
        ):
            previous_plan = plans[
                index - 1
            ]

            is_continuation = (
                previous_plan.reason
                == "max_size"
            )

            if is_continuation:
                segment_chunk_index += 1
            else:
                segment_number += 1
                segment_root_chunk_id = (
                    chunks[index].chunk_id
                )
                segment_chunk_index = 1

            assignments.append(
                {
                    "segment_number": segment_number,
                    "segment_root_chunk_id": (
                        segment_root_chunk_id
                    ),
                    "segment_chunk_index": (
                        segment_chunk_index
                    ),
                    "is_continuation": (
                        is_continuation
                    ),
                    "continuation_of_chunk_id": (
                        chunks[index - 1].chunk_id
                        if is_continuation
                        else None
                    ),
                    "continuation_reason": (
                        "max_size_split"
                        if is_continuation
                        else None
                    ),
                }
            )

        counts: dict[int, int] = defaultdict(int)

        for assignment in assignments:
            counts[
                int(
                    assignment[
                        "segment_number"
                    ]
                )
            ] += 1

        updated_chunks: list[
            TranscriptChunk
        ] = []

        for chunk, assignment in zip(
            chunks,
            assignments,
            strict=True,
        ):
            segment_number = int(
                assignment[
                    "segment_number"
                ]
            )

            segment_chunk_count = counts[
                segment_number
            ]

            segment_chunk_index = int(
                assignment[
                    "segment_chunk_index"
                ]
            )

            if segment_chunk_count == 1:
                segment_position = "single"
            elif segment_chunk_index == 1:
                segment_position = "start"
            elif (
                segment_chunk_index
                == segment_chunk_count
            ):
                segment_position = "end"
            else:
                segment_position = "middle"

            updates = {
                "segment_id": (
                    f"segment_{segment_number:03d}"
                ),
                "segment_root_chunk_id": int(
                    assignment[
                        "segment_root_chunk_id"
                    ]
                ),
                "segment_chunk_index": (
                    segment_chunk_index
                ),
                "segment_chunk_count": (
                    segment_chunk_count
                ),
                "segment_position": (
                    segment_position
                ),
                "is_continuation": bool(
                    assignment[
                        "is_continuation"
                    ]
                ),
                "continuation_of_chunk_id": (
                    assignment[
                        "continuation_of_chunk_id"
                    ]
                ),
                "continuation_reason": (
                    assignment[
                        "continuation_reason"
                    ]
                ),
            }

            if hasattr(
                chunk,
                "model_copy",
            ):
                updated = chunk.model_copy(
                    update=updates
                )
            else:
                updated = chunk.copy(
                    update=updates
                )

            updated_chunks.append(
                updated
            )

        return updated_chunks

    def _find_overlap_start(
        self,
        sentences: list[str],
        core_start_sentence: int,
    ) -> tuple[int, int]:
        if (
            self.config.max_size_overlap_words
            <= 0
            or self.config.max_size_overlap_sentences
            <= 0
            or core_start_sentence <= 0
        ):
            return (
                core_start_sentence,
                0,
            )

        overlap_start = (
            core_start_sentence
        )

        overlap_words = 0
        overlap_sentences = 0

        sentence_index = (
            core_start_sentence - 1
        )

        while (
            sentence_index >= 0
            and overlap_sentences
            < self.config.max_size_overlap_sentences
        ):
            sentence_words = self._word_count(
                sentences[
                    sentence_index
                ]
            )

            if (
                overlap_sentences > 0
                and (
                    overlap_words
                    + sentence_words
                )
                > self.config.max_size_overlap_words
            ):
                break

            overlap_start = sentence_index
            overlap_words += sentence_words
            overlap_sentences += 1
            sentence_index -= 1

            if (
                overlap_words
                >= self.config.max_size_overlap_words
            ):
                break

        return (
            overlap_start,
            overlap_words,
        )

    # -----------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------

    @staticmethod
    def _word_count(
        text: str,
    ) -> int:
        return len(
            re.findall(
                r"\S+",
                text,
            )
        )