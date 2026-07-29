from __future__ import annotations

from app.schemas.chunk import TranscriptChunk
from app.services.semantic_chunker import (
    SemanticChunker,
    _ChunkPlan,
)


def make_chunk(
    chunk_id: int,
) -> TranscriptChunk:
    return TranscriptChunk(
        chunk_id=chunk_id,
        text=f"Chunk {chunk_id}",
        word_count=2,
        sentence_count=1,
        start_sentence=chunk_id - 1,
        end_sentence=chunk_id - 1,
        core_start_sentence=chunk_id - 1,
        core_end_sentence=chunk_id - 1,
        boundary_reason="end_of_transcript",
        boundary_similarity=None,
        boundary_transition_strength=None,
        overlap_word_count=0,
    )


def main() -> None:
    chunker = SemanticChunker()

    chunks = [
        make_chunk(1),
        make_chunk(2),
        make_chunk(3),
        make_chunk(4),
        make_chunk(5),
    ]

    # Chunk 1 ends semantically: chunk 2 starts a new segment.
    # Chunk 2 ends by max size: chunk 3 continues chunk 2.
    # Chunk 3 ends by max size: chunk 4 continues the same segment.
    # Chunk 4 ends semantically: chunk 5 starts a new segment.
    plans = [
        _ChunkPlan(
            start_unit=0,
            end_unit=0,
            reason="semantic_shift",
            similarity=0.1,
        ),
        _ChunkPlan(
            start_unit=1,
            end_unit=1,
            reason="max_size",
            similarity=0.8,
        ),
        _ChunkPlan(
            start_unit=2,
            end_unit=2,
            reason="max_size",
            similarity=0.8,
        ),
        _ChunkPlan(
            start_unit=3,
            end_unit=3,
            reason="semantic_shift",
            similarity=0.1,
        ),
        _ChunkPlan(
            start_unit=4,
            end_unit=4,
            reason="end_of_transcript",
            similarity=None,
        ),
    ]

    updated = chunker._assign_segment_metadata(
        chunks=chunks,
        plans=plans,
    )

    assert updated[0].segment_id == "segment_001"
    assert updated[0].segment_position == "single"
    assert updated[0].is_continuation is False

    assert updated[1].segment_id == "segment_002"
    assert updated[1].segment_position == "start"
    assert updated[1].segment_chunk_count == 3

    assert updated[2].segment_id == "segment_002"
    assert updated[2].segment_position == "middle"
    assert updated[2].is_continuation is True
    assert updated[2].continuation_of_chunk_id == 2
    assert updated[2].segment_root_chunk_id == 2

    assert updated[3].segment_id == "segment_002"
    assert updated[3].segment_position == "end"
    assert updated[3].continuation_of_chunk_id == 3

    assert updated[4].segment_id == "segment_003"
    assert updated[4].segment_position == "single"
    assert updated[4].is_continuation is False

    print(
        "MODULE 2 SEGMENT METADATA TEST PASSED"
    )


if __name__ == "__main__":
    main()