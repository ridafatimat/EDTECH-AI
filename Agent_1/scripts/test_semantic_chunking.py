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
            "Test Module 2 guarded semantic chunking "
            "on a Module 1 cleaned transcript."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to a cleaned transcript .txt file."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path."
        ),
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Number of chunking runs. "
            "Use --repeat 2 to compare first-run vs warm-run timing."
        ),
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

    cleaned_text = (
        args.input.read_text(
            encoding="utf-8"
        ).strip()
    )

    if not cleaned_text:
        raise ValueError(
            "Cleaned transcript file is empty."
        )

    # MiniLM-tuned configuration.
    config = SemanticChunkingConfig(
        min_chunk_words=150,
        target_chunk_words=325,
        max_chunk_words=550,
        strong_transition_min_words=80,
        semantic_unit_words=60,

        # Adaptive MiniLM thresholding.
        boundary_percentile=15.0,
        threshold_floor=0.10,
        threshold_ceiling=0.45,

        # Soft transitions require semantic support.
        soft_transition_margin=0.10,
        soft_transition_similarity_ceiling=0.35,

        # Overlap only on forced max-size boundaries.
        max_size_overlap_words=45,
        max_size_overlap_sentences=2,
    )

    chunker = SemanticChunker(
        config=config
    )

    print()
    print("=" * 100)
    print(
        "MODULE 2 — GUARDED SEMANTIC CHUNKING TEST"
    )
    print("=" * 100)

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

        elapsed = (
            time.perf_counter()
            - start
        )

        timings.append(
            elapsed
        )

        print(
            f"\nRun {run_number}: "
            f"{elapsed:.3f} seconds"
        )

    if result is None:
        raise RuntimeError(
            "Chunking did not produce a result."
        )

    print()
    print("-" * 100)

    print(
        f"Embedding model: "
        f"{result.embedding_model}"
    )

    print(
        f"Semantic threshold: "
        f"{result.semantic_threshold}"
    )

    print(
        f"Chunk size config: "
        f"{result.min_chunk_words}"
        f" / {result.target_chunk_words}"
        f" / {result.max_chunk_words}"
        f" words "
        f"(min / target / max)"
    )

    print(
        f"Max-size overlap target: "
        f"{result.max_size_overlap_words} words"
    )

    print(
        f"Sentences: "
        f"{result.total_sentences}"
    )

    print(
        f"Words: "
        f"{result.total_words}"
    )

    print(
        f"Semantic units: "
        f"{result.semantic_unit_count}"
    )

    print(
        f"Final chunks: "
        f"{len(result.chunks)}"
    )

    if len(timings) > 1:
        print(
            f"First run: "
            f"{timings[0]:.3f} seconds"
        )

        print(
            f"Warm run: "
            f"{timings[-1]:.3f} seconds"
        )

    print()
    print("=" * 100)
    print("FINAL CHUNKS")
    print("=" * 100)

    for chunk in result.chunks:
        print()
        print("-" * 100)

        print(
            f"CHUNK {chunk.chunk_id}"
        )

        print(
            f"Words: "
            f"{chunk.word_count}"
        )

        print(
            f"Sentences: "
            f"{chunk.sentence_count}"
        )

        print(
            f"Text sentence range: "
            f"{chunk.start_sentence}"
            f" → "
            f"{chunk.end_sentence}"
        )

        print(
            f"Core sentence range: "
            f"{chunk.core_start_sentence}"
            f" → "
            f"{chunk.core_end_sentence}"
        )

        print(
            f"Overlap words: "
            f"{chunk.overlap_word_count}"
        )

        print(
            f"Boundary reason: "
            f"{chunk.boundary_reason}"
        )

        print(
            f"Boundary similarity: "
            f"{chunk.boundary_similarity}"
        )

        print(
            f"Boundary transition strength: "
            f"{chunk.boundary_transition_strength}"
        )

        print("-" * 100)

        print(
            chunk.text
        )

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
    print("=" * 100)

    print(
        f"Chunk JSON saved to: "
        f"{output_path}"
    )

    print(
        "MODULE 2 TEST COMPLETED"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()