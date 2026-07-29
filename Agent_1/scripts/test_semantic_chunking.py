from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.services.semantic_chunker import (
    SemanticChunker,
    SemanticChunkingConfig,
)


def save_json(
    result,
    output_path: Path,
) -> None:
    """Save the Pydantic chunking result as readable JSON."""

    if hasattr(
        result,
        "model_dump",
    ):
        data = result.model_dump()
    else:
        data = result.dict()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test Module 2 semantic chunking and "
            "logical continuation metadata."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to a cleaned transcript text file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError(
            "--repeat must be at least 1."
        )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    cleaned_text = args.input.read_text(
        encoding="utf-8"
    ).strip()

    if not cleaned_text:
        raise ValueError(
            "Cleaned transcript file is empty."
        )

    config = SemanticChunkingConfig(
        min_chunk_words=150,
        target_chunk_words=325,
        max_chunk_words=550,
        strong_transition_min_words=80,
        semantic_unit_words=60,
        boundary_percentile=15.0,
        threshold_floor=0.10,
        threshold_ceiling=0.45,
        soft_transition_margin=0.10,
        soft_transition_similarity_ceiling=0.35,
        max_size_overlap_words=45,
        max_size_overlap_sentences=2,
    )

    chunker = SemanticChunker(
        config=config
    )

    result = None
    timings: list[float] = []

    for run_number in range(
        1,
        args.repeat + 1,
    ):
        start = time.perf_counter()

        result = chunker.chunk(
            cleaned_text
        )

        timings.append(
            time.perf_counter()
            - start
        )

        print(
            f"Run {run_number}: "
            f"{timings[-1]:.3f} seconds"
        )

    if result is None:
        raise RuntimeError(
            "Chunking did not produce a result."
        )

    print()
    print("=" * 100)
    print(
        "MODULE 2 — CHUNKS + SEGMENT METADATA"
    )
    print("=" * 100)
    print(
        f"Embedding model: {result.embedding_model}"
    )
    print(
        f"Semantic threshold: {result.semantic_threshold}"
    )
    print(
        f"Physical chunks: {len(result.chunks)}"
    )
    print(
        f"Logical segments: {result.segment_count}"
    )

    for chunk in result.chunks:
        print()
        print("-" * 100)
        print(
            f"CHUNK {chunk.chunk_id}"
        )
        print(
            f"Segment: {chunk.segment_id}"
        )
        print(
            "Segment position: "
            f"{chunk.segment_position} "
            f"({chunk.segment_chunk_index}/"
            f"{chunk.segment_chunk_count})"
        )
        print(
            "Segment root chunk: "
            f"{chunk.segment_root_chunk_id}"
        )
        print(
            "Is continuation: "
            f"{chunk.is_continuation}"
        )
        print(
            "Continuation of chunk: "
            f"{chunk.continuation_of_chunk_id}"
        )
        print(
            "Continuation reason: "
            f"{chunk.continuation_reason}"
        )
        print(
            f"Boundary reason: {chunk.boundary_reason}"
        )
        print(
            f"Words: {chunk.word_count}"
        )
        print("-" * 100)
        print(chunk.text)

    output_path = (
        args.output
        if args.output
        else args.input.with_name(
            args.input.stem
            + "_chunks.json"
        )
    )

    save_json(
        result,
        output_path,
    )

    print()
    print(
        f"Chunk JSON saved to: {output_path}"
    )


if __name__ == "__main__":
    main()