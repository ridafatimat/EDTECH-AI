from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from app.services.hybrid_preprocessor import (
    preprocess_transcript_hybrid,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PDF_FOLDER = (
    PROJECT_ROOT
    / "test_data"
    / "noisy_transcripts"
)


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(
    pdf_path: Path,
) -> str:
    """
    Extract transcript text from a PDF using PyMuPDF.
    """

    pages: list[str] = []

    with fitz.open(pdf_path) as document:

        for page in document:

            page_text = page.get_text(
                "text"
            )

            if page_text.strip():

                pages.append(
                    page_text
                )

    extracted_text = "\n".join(
        pages
    ).strip()

    if not extracted_text:

        raise ValueError(
            f"No text could be extracted from "
            f"{pdf_path.name}"
        )

    return extracted_text


# =========================================================
# TEST ONE PDF
# =========================================================

def test_pdf(
    pdf_path: Path,
) -> dict:
    """
    Run one PDF through the complete Module 1 pipeline.

    Returns a small summary dictionary which will later
    be used for the final comparison table.
    """

    print("\n")
    print("=" * 100)
    print(
        f"PDF: {pdf_path.name}"
    )
    print("=" * 100)

    # =====================================================
    # STEP 1:
    # PDF EXTRACTION
    # =====================================================

    try:

        raw_text = extract_text_from_pdf(
            pdf_path
        )

    except Exception as error:

        print(
            "\nPDF EXTRACTION FAILED"
        )

        print(
            f"Error: {error}"
        )

        return {
            "pdf": pdf_path.name,
            "status": "EXTRACTION FAILED",
        }

    print(
        "\nRAW EXTRACTED TRANSCRIPT:"
    )

    print(
        "-" * 100
    )

    print(
        raw_text
    )

    # =====================================================
    # STEP 2:
    # MODULE 1 PREPROCESSING
    # =====================================================

    try:

        result = (
            preprocess_transcript_hybrid(
                raw_text=raw_text
            )
        )

    except Exception as error:

        print(
            "\nPREPROCESSING FAILED"
        )

        print(
            f"Error: {error}"
        )

        return {
            "pdf": pdf_path.name,
            "status": "PREPROCESSING FAILED",
        }

    # =====================================================
    # STEP 3:
    # FINAL TRANSCRIPT
    # =====================================================

    print("\n")
    print(
        "FINAL CLEANED TRANSCRIPT:"
    )

    print(
        "-" * 100
    )

    print(
        result.cleaned_text
    )

    # =====================================================
    # STEP 4:
    # PIPELINE INFORMATION
    # =====================================================

    print("\n")
    print(
        "MODULE 1 RESULT:"
    )

    print(
        "-" * 100
    )

    print(
        f"LLM used: "
        f"{result.llm_used}"
    )

    print(
        f"LLM accepted: "
        f"{result.llm_accepted}"
    )

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

    # =====================================================
    # STEP 5:
    # GPT-OSS CHANGES
    # =====================================================

    print(
        "\nLLM CHANGES:"
    )

    if result.llm_changes:

        for change in result.llm_changes:

            print(
                f"- {change}"
            )

    else:

        print(
            "- None"
        )

    # =====================================================
    # RETURN SUMMARY
    # =====================================================

    return {
        "pdf": pdf_path.name,
        "status": "PASS",
        "llm_used": result.llm_used,
        "llm_accepted": result.llm_accepted,
        "timestamps": result.stats.timestamps_removed,
        "speakers": result.stats.speaker_labels_removed,
        "fillers": result.stats.fillers_removed,
        "artefacts": result.stats.artefacts_removed,
        "uncertainty": (
            result.stats.uncertainty_markers_removed
        ),
    }


# =========================================================
# SUMMARY TABLE
# =========================================================

def print_summary(
    results: list[dict],
) -> None:
    """
    Print a compact overview after all PDF tests finish.
    """

    print("\n")
    print("=" * 100)
    print(
        "MODULE 1 - PDF TEST SUMMARY"
    )
    print("=" * 100)

    header = (
        f"{'PDF':<38}"
        f"{'LLM':<8}"
        f"{'ACCEPTED':<12}"
        f"{'UNCERTAIN':<12}"
        f"{'STATUS':<15}"
    )

    print(
        header
    )

    print(
        "-" * 100
    )

    for result in results:

        pdf_name = result.get(
            "pdf",
            "-"
        )

        status = result.get(
            "status",
            "-"
        )

        llm_used = result.get(
            "llm_used",
            "-"
        )

        llm_accepted = result.get(
            "llm_accepted",
            "-"
        )

        uncertainty = result.get(
            "uncertainty",
            "-"
        )

        print(
            f"{pdf_name:<38}"
            f"{str(llm_used):<8}"
            f"{str(llm_accepted):<12}"
            f"{str(uncertainty):<12}"
            f"{status:<15}"
        )


# =========================================================
# RUN ALL PDF TESTS
# =========================================================

def main() -> None:

    print(
        "=" * 100
    )

    print(
        "AGENT 1 - MODULE 1 PDF PREPROCESSING TEST"
    )

    print(
        "=" * 100
    )

    # =====================================================
    # CHECK TEST FOLDER
    # =====================================================

    if not PDF_FOLDER.exists():

        raise FileNotFoundError(
            f"Test folder does not exist: "
            f"{PDF_FOLDER}"
        )

    # =====================================================
    # FIND PDFs
    # =====================================================

    pdf_files = sorted(
        PDF_FOLDER.glob(
            "*.pdf"
        )
    )

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF files found in: "
            f"{PDF_FOLDER}"
        )

    print(
        f"\nFound "
        f"{len(pdf_files)} "
        f"transcript PDF(s)."
    )

    # =====================================================
    # RUN TESTS
    # =====================================================

    results: list[dict] = []

    for pdf_path in pdf_files:

        result = test_pdf(
            pdf_path
        )

        results.append(
            result
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    print_summary(
        results
    )

    print("\n")
    print(
        "=" * 100
    )

    print(
        "ALL PDF TESTS COMPLETED"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()