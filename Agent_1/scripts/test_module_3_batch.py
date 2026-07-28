from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.services.topic_pipeline import Module3TopicPipeline


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "test_outputs"
    / "module_1_2_batch"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "test_outputs"
    / "module_3_batch"
)


# =========================================================
# GENERIC MODEL SERIALISATION
# =========================================================

def model_to_dict(
    model: Any,
) -> dict[str, Any]:
    """
    Convert a Pydantic model to a plain dictionary.
    """

    if hasattr(
        model,
        "model_dump",
    ):
        return model.model_dump()

    if hasattr(
        model,
        "dict",
    ):
        return model.dict()

    raise TypeError(
        "Expected a Pydantic-style result model."
    )


# =========================================================
# INPUT DISCOVERY
# =========================================================

def is_module_2_chunk_file(
    json_path: Path,
) -> bool:
    """
    Return True only when the JSON file contains Module 2 chunks.

    This avoids processing:
    - batch summaries
    - Module 3 outputs
    - unrelated JSON files
    """

    lower_name = json_path.name.lower()

    excluded_name_parts = (
        "_topics",
        "module_3",
        "batch_summary",
        "summary",
    )

    if any(
        part in lower_name
        for part in excluded_name_parts
    ):
        return False

    try:
        data = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return False

    if isinstance(
        data,
        dict,
    ):
        chunks = data.get(
            "chunks"
        )

    elif isinstance(
        data,
        list,
    ):
        chunks = data

    else:
        return False

    if not isinstance(
        chunks,
        list,
    ):
        return False

    if not chunks:
        return False

    first_chunk = chunks[0]

    return (
        isinstance(
            first_chunk,
            dict,
        )
        and "chunk_id" in first_chunk
        and "text" in first_chunk
    )


def discover_chunk_files(
    input_dir: Path,
) -> list[Path]:
    """
    Recursively discover all valid Module 2 chunk JSON files.
    """

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Module 1 + 2 batch folder not found: {input_dir}"
        )

    discovered = [
        json_path
        for json_path in input_dir.rglob(
            "*.json"
        )
        if is_module_2_chunk_file(
            json_path
        )
    ]

    return sorted(
        discovered
    )


def load_chunks(
    input_path: Path,
) -> list[dict[str, Any]]:
    """
    Load chunks from one Module 2 JSON file.
    """

    data = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(
        data,
        dict,
    ):
        chunks = data[
            "chunks"
        ]
    else:
        chunks = data

    return chunks


# =========================================================
# READABLE OUTPUT
# =========================================================

def save_readable_result(
    result: Any,
    source_file: Path,
    output_path: Path,
) -> None:
    """
    Save a human-readable Module 3 result for one transcript.
    """

    lines: list[str] = []

    lines.append(
        "=" * 110
    )

    lines.append(
        "AGENT 1 — MODULE 3 BATCH TOPIC EXTRACTION"
    )

    lines.append(
        "=" * 110
    )

    lines.append(
        f"Source chunk file: {source_file}"
    )

    lines.append(
        f"Embedding model: {result.embedding_model}"
    )

    lines.append(
        f"Candidate keep threshold: "
        f"{result.candidate_keep_threshold}"
    )

    lines.append(
        f"Total chunks: {result.total_chunks}"
    )

    lines.append(
        f"CS-relevant chunks: "
        f"{result.cs_relevant_chunks}"
    )

    classifications = Counter(
        getattr(
            chunk_result,
            "classification",
            "unknown",
        )
        for chunk_result in result.chunk_results
    )

    lines.append(
        "Classifications: "
        + str(
            dict(
                classifications
            )
        )
    )

    lines.append(
        f"LLM fallback chunks: "
        f"{result.llm_fallback_chunk_ids}"
    )

    lines.append("")

    for chunk_result in result.chunk_results:
        lines.append(
            "=" * 110
        )

        lines.append(
            f"CHUNK {chunk_result.chunk_id}"
        )

        lines.append(
            "=" * 110
        )

        lines.append(
            f"Source words: "
            f"{chunk_result.source_word_count}"
        )

        lines.append(
            f"Classification: "
            f"{chunk_result.classification}"
        )

        lines.append(
            f"CS relevant: "
            f"{chunk_result.is_cs_relevant}"
        )

        lines.append(
            f"Creates new topic: "
            f"{chunk_result.creates_new_topic}"
        )

        lines.append(
            f"Chunk relevance score: "
            f"{chunk_result.cs_relevance_score}"
        )

        lines.append(
            f"Requires LLM fallback: "
            f"{chunk_result.requires_llm_fallback}"
        )

        if chunk_result.notes:
            lines.append(
                "Notes:"
            )

            for note in chunk_result.notes:
                lines.append(
                    f"- {note}"
                )

        lines.append(
            "\nRETAINED OFFICIAL AQA TOPICS"
        )

        lines.append(
            "-" * 110
        )

        if not chunk_result.topic_candidates:
            lines.append(
                "None"
            )

        for candidate in chunk_result.topic_candidates:
            lines.append(
                f"- {candidate.topic}"
            )

            lines.append(
                f"  concept_id: "
                f"{candidate.concept_id}"
            )

            lines.append(
                f"  official reference: "
                f"{getattr(candidate, 'official_reference', None)}"
            )

            lines.append(
                f"  confidence: "
                f"{candidate.confidence}"
            )

            lines.append(
                f"  salience: "
                f"{getattr(candidate, 'salience_score', None)}"
            )

            lines.append(
                f"  keyword score: "
                f"{candidate.keyword_score}"
            )

            lines.append(
                f"  semantic score: "
                f"{candidate.semantic_score}"
            )

            lines.append(
                f"  aliases: "
                f"{candidate.matched_aliases}"
            )

            if candidate.evidence:
                lines.append(
                    "  evidence:"
                )

                for evidence in candidate.evidence:
                    lines.append(
                        f"    • {evidence}"
                    )

        unmapped_signals = getattr(
            chunk_result,
            "unmapped_cs_signals",
            [],
        )

        if unmapped_signals:
            lines.append(
                "\nUNMAPPED CS SIGNALS"
            )

            lines.append(
                "-" * 110
            )

            for signal in unmapped_signals:
                lines.append(
                    f"- {getattr(signal, 'rough_topic', signal.domain)}"
                )

                lines.append(
                    f"  domain: {signal.domain}"
                )

                lines.append(
                    f"  score: {signal.score}"
                )

                lines.append(
                    f"  method: "
                    f"{getattr(signal, 'detection_method', None)}"
                )

                lines.append(
                    f"  aliases: "
                    f"{getattr(signal, 'matched_aliases', [])}"
                )

                lines.append(
                    f"  evidence: {signal.evidence}"
                )

        rejected = getattr(
            chunk_result,
            "rejected_candidates",
            [],
        )

        if rejected:
            lines.append(
                "\nREJECTED / LOW-CONFIDENCE CANDIDATES"
            )

            lines.append(
                "-" * 110
            )

            for candidate in rejected:
                lines.append(
                    f"- {candidate.topic}: "
                    f"{candidate.cs_relevance_score}"
                )

        lines.append("")

    lines.append(
        "=" * 110
    )

    lines.append(
        "MERGED LESSON TOPICS"
    )

    lines.append(
        "=" * 110
    )

    if not result.merged_topics:
        lines.append(
            "No merged official AQA topics."
        )

    for topic in result.merged_topics:
        lines.append(
            f"- {topic.topic}"
        )

        lines.append(
            f"  concept_id: {topic.concept_id}"
        )

        lines.append(
            f"  role: "
            f"{getattr(topic, 'topic_role', None)}"
        )

        lines.append(
            f"  confidence: {topic.confidence}"
        )

        lines.append(
            f"  ranking score: "
            f"{getattr(topic, 'ranking_score', None)}"
        )

        lines.append(
            f"  source chunks: "
            f"{topic.source_chunk_ids}"
        )

        lines.append(
            f"  support spans: "
            f"{getattr(topic, 'support_span_count', None)}"
        )

        lines.append(
            f"  mean semantic score: "
            f"{getattr(topic, 'mean_semantic_score', None)}"
        )

        lines.append(
            f"  mean salience score: "
            f"{getattr(topic, 'mean_salience_score', None)}"
        )

        lines.append(
            f"  coverage score: "
            f"{getattr(topic, 'coverage_score', None)}"
        )

        if topic.evidence:
            lines.append(
                "  evidence:"
            )

            for evidence in topic.evidence:
                lines.append(
                    f"    • {evidence}"
                )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# =========================================================
# SUMMARY CREATION
# =========================================================

def build_summary_row(
    source_file: Path,
    result: Any,
    processing_time: float,
) -> dict[str, Any]:
    classifications = Counter(
        getattr(
            chunk_result,
            "classification",
            "unknown",
        )
        for chunk_result in result.chunk_results
    )

    merged_primary = [
        topic.topic
        for topic in result.merged_topics
        if getattr(
            topic,
            "topic_role",
            None,
        ) == "primary"
    ]

    merged_supporting = [
        topic.topic
        for topic in result.merged_topics
        if getattr(
            topic,
            "topic_role",
            None,
        ) == "supporting"
    ]

    tracing_chunks = [
        chunk_result.chunk_id
        for chunk_result in result.chunk_results
        if any(
            candidate.concept_id
            == "aqa_3_1_1_algorithm_purpose_trace"
            for candidate in (
                chunk_result.topic_candidates
            )
        )
    ]

    unmapped_signal_count = sum(
        len(
            getattr(
                chunk_result,
                "unmapped_cs_signals",
                [],
            )
        )
        for chunk_result in result.chunk_results
    )

    return {
        "source_file": str(
            source_file
        ),
        "processing_time_seconds": round(
            processing_time,
            4,
        ),
        "total_chunks": result.total_chunks,
        "cs_relevant_chunks": (
            result.cs_relevant_chunks
        ),
        "official_topic_chunks": (
            classifications[
                "official_aqa_topic"
            ]
        ),
        "mixed_official_unmapped_chunks": (
            classifications[
                "mixed_official_and_unmapped"
            ]
        ),
        "unmapped_cs_chunks": (
            classifications[
                "cs_related_unmapped"
            ]
        ),
        "continuation_chunks": (
            classifications[
                "continuation_no_new_topic"
            ]
        ),
        "no_topic_chunks": (
            classifications[
                "no_topic"
            ]
        ),
        "llm_fallback_chunks": (
            ",".join(
                str(chunk_id)
                for chunk_id in (
                    result
                    .llm_fallback_chunk_ids
                )
            )
        ),
        "tracing_chunks": ",".join(
            str(chunk_id)
            for chunk_id in tracing_chunks
        ),
        "unmapped_signal_count": (
            unmapped_signal_count
        ),
        "merged_topic_count": len(
            result.merged_topics
        ),
        "primary_topics": " | ".join(
            merged_primary
        ),
        "supporting_topics": " | ".join(
            merged_supporting
        ),
    }


def save_summary_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Agent 1 Module 3 over every Module 2 "
            "chunk JSON in the batch output folder."
        )
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
        "--limit",
        type=int,
        default=None,
        help=(
            "Optionally test only the first N discovered transcripts."
        ),
    )

    args = parser.parse_args()

    chunk_files = discover_chunk_files(
        args.input_dir
    )

    if args.limit is not None:
        chunk_files = chunk_files[
            : args.limit
        ]

    if not chunk_files:
        raise RuntimeError(
            "No Module 2 chunk JSON files were discovered in "
            f"{args.input_dir}"
        )

    print(
        "=" * 110
    )

    print(
        "AGENT 1 — MODULE 3 BATCH EVALUATION"
    )

    print(
        "=" * 110
    )

    print(
        f"\nChunk files found: {len(chunk_files)}"
    )

    for index, chunk_file in enumerate(
        chunk_files,
        start=1,
    ):
        print(
            f"{index}. "
            f"{chunk_file.relative_to(args.input_dir)}"
        )

    pipeline = Module3TopicPipeline()

    summary_rows: list[
        dict[str, Any]
    ] = []

    for index, chunk_file in enumerate(
        chunk_files,
        start=1,
    ):
        print(
            "\n"
            + "=" * 110
        )

        print(
            f"TESTING {index}/{len(chunk_files)}: "
            f"{chunk_file.name}"
        )

        print(
            "=" * 110
        )

        chunks = load_chunks(
            chunk_file
        )

        started = time.perf_counter()

        result = pipeline.process_chunks(
            chunks
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        relative_parent = (
            chunk_file
            .parent
            .relative_to(
                args.input_dir
            )
        )

        transcript_output_dir = (
            args.output_dir
            / relative_parent
            / chunk_file.stem
        )

        json_output = (
            transcript_output_dir
            / "module_3_topics.json"
        )

        readable_output = (
            transcript_output_dir
            / "module_3_topics_readable.txt"
        )

        json_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_output.write_text(
            json.dumps(
                model_to_dict(
                    result
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        save_readable_result(
            result=result,
            source_file=chunk_file,
            output_path=readable_output,
        )

        summary_row = build_summary_row(
            source_file=chunk_file,
            result=result,
            processing_time=elapsed,
        )

        summary_rows.append(
            summary_row
        )

        print(
            f"Processing time: {elapsed:.3f}s"
        )

        print(
            f"Total chunks: {result.total_chunks}"
        )

        print(
            f"CS-relevant chunks: "
            f"{result.cs_relevant_chunks}"
        )

        print(
            f"Merged topics: "
            f"{len(result.merged_topics)}"
        )

        print(
            f"Tracing chunks: "
            f"{summary_row['tracing_chunks'] or 'None'}"
        )

        print(
            f"LLM fallback chunks: "
            f"{summary_row['llm_fallback_chunks'] or 'None'}"
        )

        print(
            "Primary topics: "
            f"{summary_row['primary_topics'] or 'None'}"
        )

        print(
            "Supporting topics: "
            f"{summary_row['supporting_topics'] or 'None'}"
        )

    summary_json = (
        args.output_dir
        / "module_3_batch_summary.json"
    )

    summary_csv = (
        args.output_dir
        / "module_3_batch_summary.csv"
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_json.write_text(
        json.dumps(
            summary_rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    save_summary_csv(
        rows=summary_rows,
        output_path=summary_csv,
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "MODULE 3 BATCH TEST COMPLETED"
    )

    print(
        "=" * 110
    )

    print(
        f"\nDetailed outputs:\n"
        f"{args.output_dir}"
    )

    print(
        f"\nSummary JSON:\n"
        f"{summary_json}"
    )

    print(
        f"\nSummary CSV:\n"
        f"{summary_csv}"
    )


if __name__ == "__main__":
    main()