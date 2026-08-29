from __future__ import annotations

from pathlib import Path

from docx import Document

# Module 1
from app.services.transcript_preprocessor import (
    preprocess_transcript,
)

# Module 2
from app.services.semantic_chunker import (
    SemanticChunker,
    SemanticChunkingConfig,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DOCX_PATH = (
    PROJECT_ROOT
    / "test_data"
    / "Transcript 1.docx"
)

# Module 1 output
CLEANED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "test_data"
    / "transcript_1_cleaned.txt"
)

# Module 2 readable chunks output
CHUNKS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "test_data"
    / "transcript_1_chunks.txt"
)


# =========================================================
# DOCX TEXT EXTRACTION
# =========================================================

def extract_text_from_docx(
    docx_path: Path,
) -> str:
    """
    Extract transcript text from a DOCX file.
    """

    if not docx_path.exists():
        raise FileNotFoundError(
            f"Transcript not found: {docx_path}"
        )

    document = Document(
        docx_path
    )

    paragraphs: list[str] = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(
                text
            )

    extracted_text = "\n".join(
        paragraphs
    ).strip()

    if not extracted_text:
        raise ValueError(
            "No transcript text was found inside the DOCX."
        )

    return extracted_text


# =========================================================
# SAVE CLEANED TRANSCRIPT
# =========================================================

def save_cleaned_transcript(
    cleaned_text: str,
    output_path: Path,
) -> None:
    """
    Save Module 1 cleaned transcript.
    """

    cleaned_text = cleaned_text.strip()

    if not cleaned_text:
        raise ValueError(
            "Cannot save an empty cleaned transcript."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        cleaned_text,
        encoding="utf-8",
    )


# =========================================================
# SAVE READABLE CHUNKS
# =========================================================

def save_chunks_to_text(
    chunking_result,
    output_path: Path,
) -> None:
    """
    Save Module 2 chunks in a readable Notepad text file.

    The output includes:
    - chunk number
    - word count
    - sentence count
    - sentence ranges
    - boundary reason
    - similarity
    - transition strength
    - overlap
    - complete chunk text
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = []

    # Overall result summary
    lines.append(
        "=" * 100
    )

    lines.append(
        "TRANSCRIPT 1 — MODULE 2 CHUNKS"
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        f"Embedding model: "
        f"{chunking_result.embedding_model}"
    )

    lines.append(
        f"Semantic threshold: "
        f"{chunking_result.semantic_threshold}"
    )

    lines.append(
        f"Total transcript words: "
        f"{chunking_result.total_words}"
    )

    lines.append(
        f"Total transcript sentences: "
        f"{chunking_result.total_sentences}"
    )

    lines.append(
        f"Semantic units: "
        f"{chunking_result.semantic_unit_count}"
    )

    lines.append(
        f"Final chunks: "
        f"{len(chunking_result.chunks)}"
    )

    lines.append("")

    # Individual chunks
    for chunk in chunking_result.chunks:

        lines.append(
            "=" * 100
        )

        lines.append(
            f"CHUNK {chunk.chunk_id}"
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"Words: "
            f"{chunk.word_count}"
        )

        lines.append(
            f"Sentences: "
            f"{chunk.sentence_count}"
        )

        lines.append(
            f"Text sentence range: "
            f"{chunk.start_sentence}"
            f" -> "
            f"{chunk.end_sentence}"
        )

        lines.append(
            f"Core sentence range: "
            f"{chunk.core_start_sentence}"
            f" -> "
            f"{chunk.core_end_sentence}"
        )

        lines.append(
            f"Boundary reason: "
            f"{chunk.boundary_reason}"
        )

        lines.append(
            f"Boundary similarity: "
            f"{chunk.boundary_similarity}"
        )

        lines.append(
            f"Transition strength: "
            f"{chunk.boundary_transition_strength}"
        )

        lines.append(
            f"Overlap words: "
            f"{chunk.overlap_word_count}"
        )

        lines.append(
            "-" * 100
        )

        lines.append(
            chunk.text
        )

        lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =========================================================
# BUILD MODULE 2 CHUNKER
# =========================================================

def build_chunker() -> SemanticChunker:
    """
    Create Module 2 using the finalised MiniLM configuration.
    """

    config = SemanticChunkingConfig(

        # Final chunk sizes
        min_chunk_words=150,
        target_chunk_words=325,
        max_chunk_words=550,

        # Strong transition can create a smaller chunk
        strong_transition_min_words=80,

        # Temporary semantic unit size
        semantic_unit_words=60,

        # Adaptive threshold configuration
        boundary_percentile=15.0,
        threshold_floor=0.10,
        threshold_ceiling=0.45,

        # Soft transition rules
        soft_transition_margin=0.10,
        soft_transition_similarity_ceiling=0.35,

        # Overlap only after max-size split
        max_size_overlap_words=45,
        max_size_overlap_sentences=2,
    )

    return SemanticChunker(
        config=config
    )


# =========================================================
# MAIN TEST
# =========================================================

def main() -> None:

    print(
        "=" * 100
    )

    print(
        "AGENT 1 — TRANSCRIPT 1 "
        "PREPROCESSING + CHUNKING TEST"
    )

    print(
        "=" * 100
    )

    # =====================================================
    # STEP 1: EXTRACT DOCX TEXT
    # =====================================================

    raw_text = extract_text_from_docx(
        DOCX_PATH
    )

    print(
        "\nRAW TRANSCRIPT EXTRACTED"
    )

    print(
        f"Characters: {len(raw_text)}"
    )

    # =====================================================
    # STEP 2: MODULE 1 PREPROCESSING
    # =====================================================

    preprocessing_result = (
        preprocess_transcript(
            raw_text=raw_text
        )
    )

    cleaned_text = (
        preprocessing_result
        .cleaned_text
        .strip()
    )

    # =====================================================
    # STEP 3: SAVE CLEANED TRANSCRIPT
    # =====================================================

    save_cleaned_transcript(
        cleaned_text=cleaned_text,
        output_path=CLEANED_OUTPUT_PATH,
    )

    print(
        "\nMODULE 1 COMPLETED"
    )

    print(
        f"Cleaned transcript saved to:\n"
        f"{CLEANED_OUTPUT_PATH}"
    )

    # =====================================================
    # STEP 4: MODULE 1 STATS
    # =====================================================

    stats = preprocessing_result.stats

    print(
        "\nMODULE 1 STATISTICS"
    )

    print(
        "-" * 100
    )

    print(
        f"Original characters: "
        f"{stats.original_characters}"
    )

    print(
        f"Cleaned characters: "
        f"{stats.cleaned_characters}"
    )

    print(
        f"Timestamps removed: "
        f"{stats.timestamps_removed}"
    )

    print(
        f"Speaker labels removed: "
        f"{stats.speaker_labels_removed}"
    )

    print(
        f"Fillers removed: "
        f"{stats.fillers_removed}"
    )

    print(
        f"Artefacts removed: "
        f"{stats.artefacts_removed}"
    )

    print(
        f"Uncertainty markers removed: "
        f"{stats.uncertainty_markers_removed}"
    )

    print(
        f"Repeated words removed: "
        f"{stats.repeated_words_removed}"
    )

    print(
        f"Repeated sentences removed: "
        f"{stats.repeated_sentences_removed}"
    )

    # =====================================================
    # STEP 5: MODULE 2 CHUNKING
    # =====================================================

    chunker = build_chunker()

    chunking_result = chunker.chunk(
        cleaned_transcript=cleaned_text
    )

    # =====================================================
    # STEP 6: SAVE CHUNKS TO NOTEPAD FILE
    # =====================================================

    save_chunks_to_text(
        chunking_result=chunking_result,
        output_path=CHUNKS_OUTPUT_PATH,
    )

    print(
        "\nMODULE 2 COMPLETED"
    )

    print(
        f"Chunks saved to:\n"
        f"{CHUNKS_OUTPUT_PATH}"
    )

    # =====================================================
    # STEP 7: MODULE 2 SUMMARY
    # =====================================================

    print(
        "\nMODULE 2 SUMMARY"
    )

    print(
        "-" * 100
    )

    print(
        f"Embedding model: "
        f"{chunking_result.embedding_model}"
    )

    print(
        f"Semantic threshold: "
        f"{chunking_result.semantic_threshold}"
    )

    print(
        f"Transcript words: "
        f"{chunking_result.total_words}"
    )

    print(
        f"Transcript sentences: "
        f"{chunking_result.total_sentences}"
    )

    print(
        f"Semantic units: "
        f"{chunking_result.semantic_unit_count}"
    )

    print(
        f"Final chunks: "
        f"{len(chunking_result.chunks)}"
    )

    # Show compact chunk summary in terminal
    for chunk in chunking_result.chunks:

        print(
            "\n"
            f"Chunk {chunk.chunk_id}: "
            f"{chunk.word_count} words | "
            f"{chunk.sentence_count} sentences | "
            f"{chunk.boundary_reason}"
        )

    # =====================================================
    # WARNINGS
    # =====================================================

    print(
        "\nMODULE 1 WARNINGS"
    )

    if preprocessing_result.warnings:

        for warning in (
            preprocessing_result.warnings
        ):
            print(
                f"- {warning}"
            )

    else:
        print(
            "- None"
        )

    # =====================================================
    # COMPLETE
    # =====================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "TRANSCRIPT 1 PREPROCESSING "
        "AND CHUNKING TEST COMPLETED"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()