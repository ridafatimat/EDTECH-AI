from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.db.repositories.technical_correction_repository import (
    PostgreSQLTechnicalCorrectionRepository,
)
from app.db.session import session_scope
from app.services.selective_technical_normalizer import (
    SelectiveTechnicalNormaliser,
)
from app.services.technical_correction_client import (
    GroqTechnicalCorrectionClient,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "test_outputs"
    / "module_1_2_batch"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "test_outputs"
    / "technical_normalisation_batch"
)

FINAL_TRANSCRIPT_FILENAME = "cleaned_transcript.txt"
AUDIT_FILENAME = "technical_normalisation_audit.json"


def _build_client(
    *,
    no_llm: bool,
) -> GroqTechnicalCorrectionClient | None:
    if no_llm:
        return None

    return GroqTechnicalCorrectionClient()


def _get_destination_directory(
    *,
    input_file: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Preserve the transcript folder name inside the output directory.

    Example:
    input:
      test_outputs/module_1_2_batch/
      Transcript_Raw_3_Algorithms_Programming/cleaned.txt

    output:
      test_outputs/technical_normalisation_batch/
      Transcript_Raw_3_Algorithms_Programming/
      cleaned_transcript.txt
    """

    resolved_input = input_file.resolve()
    resolved_input_dir = input_dir.resolve()

    try:
        relative_parent = (
            resolved_input.parent.relative_to(
                resolved_input_dir
            )
        )
    except ValueError:
        # The supplied file is outside --input-dir.
        # Use its parent-folder name to avoid overwriting
        # another transcript whose input file is also cleaned.txt.
        relative_parent = Path(
            resolved_input.parent.name
        )

    return output_dir / relative_parent


def process_one(
    *,
    input_file: Path,
    output_file: Path,
    audit_file: Path,
    correction_client: (
        GroqTechnicalCorrectionClient | None
    ),
) -> None:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input transcript not found: {input_file}"
        )

    if not input_file.is_file():
        raise ValueError(
            f"Input path is not a file: {input_file}"
        )

    cleaned_text = input_file.read_text(
        encoding="utf-8"
    )

    with session_scope() as session:
        repository = (
            PostgreSQLTechnicalCorrectionRepository(
                session
            )
        )

        normaliser = SelectiveTechnicalNormaliser(
            repository=repository,
            correction_client=correction_client,
        )

        result = normaliser.normalise(
            cleaned_text
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        result.normalised_text,
        encoding="utf-8",
    )

    audit_file.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nInput: {input_file}")
    print(
        "Cleaned transcript: "
        f"{output_file.resolve()}"
    )
    print(
        "Audit file: "
        f"{audit_file.resolve()}"
    )
    print(
        "Spoken-code corrections: "
        f"{result.stats.spoken_code_corrections}"
    )
    print(
        "Suspicious spans: "
        f"{result.stats.suspicious_spans_detected}"
    )
    print(
        f"Memory hits: {result.stats.memory_hits}"
    )
    print(
        f"LLM calls: {result.stats.llm_calls}"
    )
    print(
        "Accepted LLM corrections: "
        f"{result.stats.accepted_llm_corrections}"
    )
    print(
        "Unresolved issues: "
        f"{len(result.unresolved_issues)}"
    )

    for correction in result.corrections:
        print(
            f"- {correction.original!r} "
            f"-> {correction.replacement!r} "
            f"[{correction.source}]"
        )


def run_batch(
    *,
    input_dir: Path,
    output_dir: Path,
    correction_client: (
        GroqTechnicalCorrectionClient | None
    ),
    limit: int | None,
) -> None:
    cleaned_files = sorted(
        input_dir.rglob("cleaned.txt")
    )

    if limit is not None:
        cleaned_files = cleaned_files[:limit]

    if not cleaned_files:
        raise RuntimeError(
            f"No cleaned.txt files found under {input_dir}"
        )

    for input_file in cleaned_files:
        destination = _get_destination_directory(
            input_file=input_file,
            input_dir=input_dir,
            output_dir=output_dir,
        )

        process_one(
            input_file=input_file,
            output_file=(
                destination
                / FINAL_TRANSCRIPT_FILENAME
            ),
            audit_file=(
                destination
                / AUDIT_FILENAME
            ),
            correction_client=correction_client,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Produce the final cleaned transcript after "
            "selective technical terminology normalisation."
        )
    )

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "--file",
        type=Path,
        help=(
            "Process one cleaned.txt transcript."
        ),
    )

    source.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Process every cleaned.txt under --input-dir."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Use deterministic spoken-code rules and "
            "approved PostgreSQL memory only."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    client = _build_client(
        no_llm=args.no_llm
    )

    if args.file is not None:
        destination = _get_destination_directory(
            input_file=args.file,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )

        process_one(
            input_file=args.file,
            output_file=(
                destination
                / FINAL_TRANSCRIPT_FILENAME
            ),
            audit_file=(
                destination
                / AUDIT_FILENAME
            ),
            correction_client=client,
        )
        return

    run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        correction_client=client,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()