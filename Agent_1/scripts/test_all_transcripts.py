from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

from docx import Document

from app.services.transcript_preprocessor import (
    preprocess_transcript,
)
from app.services.semantic_chunker import (
    SemanticChunker,
    SemanticChunkingConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_ROOT = PROJECT_ROOT / "test_outputs" / "module_1_2_batch"


def extract_text_from_docx(docx_path: Path) -> str:
    if not docx_path.exists():
        raise FileNotFoundError(
            f"Transcript not found: {docx_path}"
        )

    document = Document(docx_path)

    paragraphs: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    extracted_text = "\n".join(paragraphs).strip()

    if not extracted_text:
        raise ValueError(
            f"No transcript text found in: {docx_path.name}"
        )

    return extracted_text


def pydantic_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def save_chunks_readable(
    chunking_result,
    output_path: Path,
) -> None:
    lines: list[str] = []

    for chunk in chunking_result.chunks:
        lines.append("=" * 100)
        lines.append(f"CHUNK {chunk.chunk_id}")
        lines.append(
            f"Words: {chunk.word_count} | "
            f"Sentences: {chunk.sentence_count}"
        )
        lines.append(
            f"Core range: "
            f"{chunk.core_start_sentence}"
            f" -> "
            f"{chunk.core_end_sentence}"
        )
        lines.append(
            f"Boundary: {chunk.boundary_reason} | "
            f"Similarity: {chunk.boundary_similarity} | "
            f"Transition: {chunk.boundary_transition_strength}"
        )
        lines.append(
            f"Overlap words: {chunk.overlap_word_count}"
        )
        lines.append("-" * 100)
        lines.append(chunk.text)
        lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def discover_test_files() -> list[Path]:
    """
    Pick only the five new evaluation transcripts:
    - Transcript_Test_*.docx
    - Transcript_Raw_*.docx

    The earlier baseline Transcript 1.docx is intentionally excluded.
    """

    test_files = sorted(
        list(TEST_DATA_DIR.glob("Transcript_Test_*.docx"))
        + list(TEST_DATA_DIR.glob("Transcript_Raw_*.docx"))
    )

    # Remove duplicates defensively.
    unique: dict[str, Path] = {
        str(path.resolve()): path
        for path in test_files
    }

    return list(unique.values())


def build_chunker() -> SemanticChunker:
    """
    Current finalized Module 2 configuration.
    """

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

    return SemanticChunker(
        config=config
    )


def process_transcript(
    docx_path: Path,
    chunker: SemanticChunker,
) -> dict:
    print()
    print("=" * 110)
    print(f"TESTING: {docx_path.name}")
    print("=" * 110)

    transcript_output_dir = (
        OUTPUT_ROOT / docx_path.stem
    )

    transcript_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # MODULE 1
    # -----------------------------------------------------------------

    raw_text = extract_text_from_docx(
        docx_path
    )

    module1_start = time.perf_counter()

    preprocessing_result = (
        preprocess_transcript(
            raw_text=raw_text
        )
    )

    module1_seconds = (
        time.perf_counter()
        - module1_start
    )

    cleaned_text = (
        preprocessing_result.cleaned_text.strip()
    )

    if not cleaned_text:
        raise ValueError(
            f"Module 1 returned empty text for {docx_path.name}"
        )

    raw_output_path = (
        transcript_output_dir / "raw_extracted.txt"
    )

    cleaned_output_path = (
        transcript_output_dir / "cleaned.txt"
    )

    raw_output_path.write_text(
        raw_text,
        encoding="utf-8",
    )

    cleaned_output_path.write_text(
        cleaned_text,
        encoding="utf-8",
    )

    stats = preprocessing_result.stats

    print("\nMODULE 1")
    print("-" * 110)
    print(
        f"Original chars: {stats.original_characters}"
    )
    print(
        f"Cleaned chars:  {stats.cleaned_characters}"
    )
    print(
        f"Timestamps removed:        {stats.timestamps_removed}"
    )
    print(
        f"Speaker labels removed:    {stats.speaker_labels_removed}"
    )
    print(
        f"Fillers removed:           {stats.fillers_removed}"
    )
    print(
        f"Artefacts removed:         {stats.artefacts_removed}"
    )
    print(
        f"Uncertainty removed:       {stats.uncertainty_markers_removed}"
    )
    print(
        f"Repeated words removed:    {stats.repeated_words_removed}"
    )
    print(
        f"Repeated sentences removed:{stats.repeated_sentences_removed}"
    )
    print(
        f"Processing time:           {module1_seconds:.3f}s"
    )

    # -----------------------------------------------------------------
    # MODULE 2
    # -----------------------------------------------------------------

    module2_start = time.perf_counter()

    chunking_result = chunker.chunk(
        cleaned_text
    )

    module2_seconds = (
        time.perf_counter()
        - module2_start
    )

    chunks_json_path = (
        transcript_output_dir / "chunks.json"
    )

    chunks_readable_path = (
        transcript_output_dir
        / "chunks_readable.txt"
    )

    chunks_json_path.write_text(
        json.dumps(
            pydantic_to_dict(
                chunking_result
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    save_chunks_readable(
        chunking_result,
        chunks_readable_path,
    )

    boundary_counts = Counter(
        chunk.boundary_reason
        for chunk
        in chunking_result.chunks
    )

    overlap_chunks = sum(
        1
        for chunk
        in chunking_result.chunks
        if chunk.overlap_word_count > 0
    )

    print("\nMODULE 2")
    print("-" * 110)
    print(
        f"Embedding model:     "
        f"{chunking_result.embedding_model}"
    )
    print(
        f"Semantic threshold:  "
        f"{chunking_result.semantic_threshold}"
    )
    print(
        f"Sentences:           "
        f"{chunking_result.total_sentences}"
    )
    print(
        f"Words:               "
        f"{chunking_result.total_words}"
    )
    print(
        f"Semantic units:      "
        f"{chunking_result.semantic_unit_count}"
    )
    print(
        f"Final chunks:        "
        f"{len(chunking_result.chunks)}"
    )
    print(
        f"Overlap chunks:      "
        f"{overlap_chunks}"
    )
    print(
        f"Processing time:     "
        f"{module2_seconds:.3f}s"
    )
    print(
        "Boundary counts:     "
        f"{dict(boundary_counts)}"
    )

    warnings = list(
        preprocessing_result.warnings
    )

    return {
        "transcript": docx_path.name,

        # Module 1
        "module1_seconds": round(
            module1_seconds,
            4,
        ),
        "original_characters": (
            stats.original_characters
        ),
        "cleaned_characters": (
            stats.cleaned_characters
        ),
        "timestamps_removed": (
            stats.timestamps_removed
        ),
        "speaker_labels_removed": (
            stats.speaker_labels_removed
        ),
        "fillers_removed": (
            stats.fillers_removed
        ),
        "artefacts_removed": (
            stats.artefacts_removed
        ),
        "uncertainty_markers_removed": (
            stats.uncertainty_markers_removed
        ),
        "repeated_words_removed": (
            stats.repeated_words_removed
        ),
        "repeated_sentences_removed": (
            stats.repeated_sentences_removed
        ),
        "module1_warning_count": len(
            warnings
        ),
        "module1_warnings": warnings,

        # Module 2
        "module2_seconds": round(
            module2_seconds,
            4,
        ),
        "embedding_model": (
            chunking_result.embedding_model
        ),
        "semantic_threshold": (
            chunking_result.semantic_threshold
        ),
        "total_sentences": (
            chunking_result.total_sentences
        ),
        "total_words": (
            chunking_result.total_words
        ),
        "semantic_units": (
            chunking_result.semantic_unit_count
        ),
        "final_chunks": len(
            chunking_result.chunks
        ),
        "overlap_chunks": (
            overlap_chunks
        ),
        "semantic_shift_boundaries": (
            boundary_counts.get(
                "semantic_shift",
                0,
            )
        ),
        "transition_boundaries": (
            boundary_counts.get(
                "transition_phrase",
                0,
            )
        ),
        "semantic_transition_boundaries": (
            boundary_counts.get(
                "semantic_shift+transition_phrase",
                0,
            )
        ),
        "max_size_boundaries": (
            boundary_counts.get(
                "max_size",
                0,
            )
        ),
        "end_boundaries": (
            boundary_counts.get(
                "end_of_transcript",
                0,
            )
        ),
    }


def save_summary(
    results: list[dict],
) -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        OUTPUT_ROOT / "batch_summary.json"
    )

    csv_path = (
        OUTPUT_ROOT / "batch_summary.csv"
    )

    json_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if results:
        fieldnames = list(
            results[0].keys()
        )

        with csv_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                results
            )


def zip_outputs() -> Path:
    zip_path = (
        PROJECT_ROOT
        / "test_outputs"
        / "module_1_2_batch_results.zip"
    )

    zip_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in OUTPUT_ROOT.rglob("*"):
            if path.is_file():
                archive.write(
                    path,
                    arcname=path.relative_to(
                        OUTPUT_ROOT.parent
                    ),
                )

    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-test Module 1 preprocessing and "
            "Module 2 semantic chunking on the five "
            "evaluation DOCX transcripts."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of transcripts to test. "
            "Example: --limit 1"
        ),
    )

    args = parser.parse_args()

    files = discover_test_files()

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError(
                "--limit must be at least 1."
            )

        files = files[
            : args.limit
        ]

    if not files:
        raise FileNotFoundError(
            "No Transcript_Test_*.docx or "
            "Transcript_Raw_*.docx files were found in "
            f"{TEST_DATA_DIR}"
        )

    print("=" * 110)
    print("AGENT 1 — MODULE 1 + MODULE 2 BATCH EVALUATION")
    print("=" * 110)

    print("\nFiles found:")

    for index, path in enumerate(
        files,
        start=1,
    ):
        print(
            f"{index}. {path.name}"
        )

    # One chunker instance for the whole batch.
    # MiniLM loads once and is reused for all transcripts.
    chunker = build_chunker()

    results: list[dict] = []

    for docx_path in files:
        result = process_transcript(
            docx_path=docx_path,
            chunker=chunker,
        )

        results.append(
            result
        )

    save_summary(
        results
    )

    zip_path = zip_outputs()

    print()
    print("=" * 110)
    print("BATCH TEST COMPLETED")
    print("=" * 110)

    print(
        f"\nDetailed outputs: "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Summary CSV: "
        f"{OUTPUT_ROOT / 'batch_summary.csv'}"
    )

    print(
        f"Summary JSON: "
        f"{OUTPUT_ROOT / 'batch_summary.json'}"
    )

    print(
        f"ZIP for review: "
        f"{zip_path}"
    )


if __name__ == "__main__":
    main()
    