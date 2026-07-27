from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


BoundaryReason = Literal[
    "semantic_shift",
    "transition_phrase",
    "semantic_shift+transition_phrase",
    "max_size",
    "end_of_transcript",
]

TransitionStrength = Literal[
    "strong",
    "soft",
]


class TranscriptChunk(BaseModel):
    """One meaningful section of a lesson transcript."""

    chunk_id: int = Field(ge=1)

    # Actual text passed downstream.
    # For a max-size split, this may include a small overlap from
    # the previous chunk for context preservation.
    text: str = Field(min_length=1)

    word_count: int = Field(ge=1)
    sentence_count: int = Field(ge=1)

    # Sentence range represented by `text`.
    # This may overlap the previous chunk only after a forced max-size split.
    start_sentence: int = Field(ge=0)
    end_sentence: int = Field(ge=0)

    # Non-overlapping/core sentence range belonging to this chunk.
    core_start_sentence: int = Field(ge=0)
    core_end_sentence: int = Field(ge=0)

    # Why this chunk ended.
    boundary_reason: BoundaryReason

    # Similarity between the semantic units around the ending boundary.
    # Lower similarity normally indicates a stronger semantic change.
    boundary_similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    # Whether a transition phrase supported the ending boundary.
    boundary_transition_strength: TransitionStrength | None = None

    # Context repeated from the previous chunk.
    # This is non-zero only when the PREVIOUS chunk ended because of max_size.
    overlap_word_count: int = Field(
        default=0,
        ge=0,
    )


class ChunkingResult(BaseModel):
    """Complete output of Module 2."""

    chunks: list[TranscriptChunk]

    total_sentences: int = Field(ge=0)
    total_words: int = Field(ge=0)
    semantic_unit_count: int = Field(ge=0)

    embedding_model: str

    # Actual semantic threshold calculated for this transcript.
    semantic_threshold: float

    # Effective configuration is included for reproducible testing.
    min_chunk_words: int = Field(ge=1)
    target_chunk_words: int = Field(ge=1)
    max_chunk_words: int = Field(ge=1)
    max_size_overlap_words: int = Field(ge=0)