from __future__ import annotations

from pathlib import Path

from docx import Document

# IMPORTANT:
# Module 1 uses ONLY lightweight deterministic preprocessing.
# hybrid_preprocessor.py is NOT used here.
from app.services.transcript_preprocessor import (
    preprocess_transcript,
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

# Cleaned transcript produced by Module 1.
# Module 2 uses this file during independent testing.
CLEANED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "test_data"
    / "transcript_1_cleaned.txt"
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
        text = (
            paragraph.text.strip()
        )

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
    Save the Module 1 cleaned transcript so Module 2 can be
    tested independently.
    """

    cleaned_text = (
        cleaned_text.strip()
    )

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
# MAIN TEST
# =========================================================

def main() -> None:
    print("=" * 100)
    print(
        "AGENT 1 - TRANSCRIPT 1 "
        "LIGHTWEIGHT PREPROCESSING TEST"
    )
    print("=" * 100)

    # =====================================================
    # STEP 1: EXTRACT TEXT
    # =====================================================

    raw_text = extract_text_from_docx(
        DOCX_PATH
    )

    print(
        "\nRAW EXTRACTED TRANSCRIPT:"
    )
    print("-" * 100)
    print(
        raw_text
    )

    # =====================================================
    # STEP 2: RUN MODULE 1 PREPROCESSING
    # =====================================================

    result = preprocess_transcript(
        raw_text=raw_text
    )

    # =====================================================
    # STEP 3: SHOW CLEANED TRANSCRIPT
    # =====================================================

    print(
        "\n\nFINAL CLEANED TRANSCRIPT:"
    )
    print("-" * 100)

    print(
        result.cleaned_text
    )

    # =====================================================
    # STEP 4: SAVE CLEANED TRANSCRIPT
    # =====================================================

    save_cleaned_transcript(
        cleaned_text=(
            result.cleaned_text
        ),
        output_path=(
            CLEANED_OUTPUT_PATH
        ),
    )

    print(
        "\nCleaned transcript saved to:"
    )

    print(
        CLEANED_OUTPUT_PATH
    )

    # =====================================================
    # STEP 5: PREPROCESSING STATS
    # =====================================================

    print(
        "\n\nMODULE 1 RESULT:"
    )
    print("-" * 100)

    print(
        f"Original characters: "
        f"{result.stats.original_characters}"
    )

    print(
        f"Final characters: "
        f"{result.stats.cleaned_characters}"
    )

    print(
        f"Timestamps removed: "
        f"{result.stats.timestamps_removed}"
    )

    print(
        f"Speaker labels removed: "
        f"{result.stats.speaker_labels_removed}"
    )

    print(
        f"Fillers removed: "
        f"{result.stats.fillers_removed}"
    )

    print(
        f"Artefacts removed: "
        f"{result.stats.artefacts_removed}"
    )

    print(
        f"Uncertainty markers removed: "
        f"{result.stats.uncertainty_markers_removed}"
    )

    print(
        f"Repeated words removed: "
        f"{result.stats.repeated_words_removed}"
    )

    print(
        f"Repeated sentences removed: "
        f"{result.stats.repeated_sentences_removed}"
    )

    # =====================================================
    # STEP 6: WARNINGS
    # =====================================================

    print(
        "\nWARNINGS:"
    )

    if result.warnings:
        for warning in result.warnings:
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
        "TRANSCRIPT 1 PREPROCESSING TEST COMPLETED"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()