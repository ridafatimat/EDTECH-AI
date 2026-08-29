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
    Save a complete human-readable Module 3 result for one transcript.

    The final lesson-level topics are grouped by role so batch output
    matches the single-transcript production pipeline.
    """

    primary_topics = [
        topic
        for topic in result.merged_topics
        if getattr(topic, "topic_role", None) == "primary"
    ]

    supporting_topics = [
        topic
        for topic in result.merged_topics
        if getattr(topic, "topic_role", None) == "supporting"
    ]

    unmapped_topics: list[Any] = []
    seen_unmapped: set[str] = set()

    for chunk_result in result.chunk_results:
        for signal in getattr(
            chunk_result,
            "unmapped_cs_signals",
            [],
        ):
            rough_topic = str(
                getattr(
                    signal,
                    "rough_topic",
                    getattr(signal, "domain", "Unmapped CS topic"),
                )
            ).strip()

            key = rough_topic.casefold()

            if not rough_topic or key in seen_unmapped:
                continue

            seen_unmapped.add(key)
            unmapped_topics.append(signal)

    classifications = Counter(
        getattr(
            chunk_result,
            "classification",
            "unknown",
        )
        for chunk_result in result.chunk_results
    )

    lines: list[str] = [
        "=" * 110,
        "AGENT 1 — MODULE 3 BATCH TOPIC EXTRACTION",
        "=" * 110,
        f"Source chunk file: {source_file}",
        f"Embedding model: {result.embedding_model}",
        (
            "Candidate keep threshold: "
            f"{result.candidate_keep_threshold}"
        ),
        f"Total chunks: {result.total_chunks}",
        (
            "CS-relevant chunks: "
            f"{result.cs_relevant_chunks}"
        ),
        (
            "Non-CS/no-new-topic chunks: "
            f"{getattr(result, 'non_cs_chunks', None)}"
        ),
        (
            "Official-topic chunks: "
            f"{classifications['official_aqa_topic']}"
        ),
        (
            "Mixed official + unmapped chunks: "
            f"{classifications['mixed_official_and_unmapped']}"
        ),
        (
            "Unmapped-CS chunks: "
            f"{classifications['cs_related_unmapped']}"
        ),
        (
            "Continuation/no-new-topic chunks: "
            f"{classifications['continuation_no_new_topic']}"
        ),
        (
            "No-topic chunks: "
            f"{classifications['no_topic']}"
        ),
        (
            "LLM fallback chunks: "
            f"{result.llm_fallback_chunk_ids}"
        ),
        "",
    ]

    for chunk_result in result.chunk_results:
        lines.extend(
            [
                "=" * 110,
                f"CHUNK {chunk_result.chunk_id}",
                "=" * 110,
                (
                    "Source words: "
                    f"{chunk_result.source_word_count}"
                ),
                (
                    "Classification: "
                    f"{chunk_result.classification}"
                ),
                (
                    "CS relevant: "
                    f"{chunk_result.is_cs_relevant}"
                ),
                (
                    "Creates new topic: "
                    f"{chunk_result.creates_new_topic}"
                ),
                (
                    "Chunk relevance score: "
                    f"{chunk_result.cs_relevance_score}"
                ),
                (
                    "Requires LLM fallback: "
                    f"{chunk_result.requires_llm_fallback}"
                ),
            ]
        )

        if chunk_result.notes:
            lines.append("Notes:")

            for note in chunk_result.notes:
                lines.append(
                    f"- {note}"
                )

        lines.extend(
            [
                "",
                "RETAINED TOPICS",
                "-" * 110,
            ]
        )

        if not chunk_result.topic_candidates:
            lines.append("None")

        for candidate in chunk_result.topic_candidates:
            lines.extend(
                [
                    f"- {candidate.topic}",
                    (
                        "  concept_id: "
                        f"{candidate.concept_id}"
                    ),
                    (
                        "  domain: "
                        f"{getattr(candidate, 'domain', None)}"
                    ),
                    (
                        "  official reference: "
                        f"{getattr(candidate, 'official_reference', None)}"
                    ),
                    (
                        "  confidence: "
                        f"{candidate.confidence}"
                    ),
                    (
                        "  salience score: "
                        f"{getattr(candidate, 'salience_score', None)}"
                    ),
                    (
                        "  keyword score: "
                        f"{candidate.keyword_score}"
                    ),
                    (
                        "  semantic score: "
                        f"{candidate.semantic_score}"
                    ),
                    (
                        "  method: "
                        f"{getattr(candidate, 'match_method', None)}"
                    ),
                    (
                        "  matched aliases: "
                        f"{candidate.matched_aliases}"
                    ),
                ]
            )

            if candidate.evidence:
                lines.append("  evidence:")

                for evidence in candidate.evidence:
                    lines.append(
                        f"    • {evidence}"
                    )

        chunk_unmapped = getattr(
            chunk_result,
            "unmapped_cs_signals",
            [],
        )

        if chunk_unmapped:
            lines.extend(
                [
                    "",
                    "UNMAPPED CS SIGNALS",
                    "-" * 110,
                ]
            )

            for signal in chunk_unmapped:
                lines.extend(
                    [
                        (
                            "- "
                            f"{getattr(signal, 'rough_topic', signal.domain)}"
                        ),
                        f"  domain: {signal.domain}",
                        f"  score: {signal.score}",
                        (
                            "  method: "
                            f"{getattr(signal, 'detection_method', None)}"
                        ),
                        (
                            "  matched aliases: "
                            f"{getattr(signal, 'matched_aliases', [])}"
                        ),
                        f"  evidence: {signal.evidence}",
                    ]
                )

        rejected = getattr(
            chunk_result,
            "rejected_candidates",
            [],
        )

        if rejected:
            lines.extend(
                [
                    "",
                    "REJECTED / LOW-CONFIDENCE CANDIDATES",
                    "-" * 110,
                ]
            )

            for candidate in rejected:
                lines.append(
                    f"- {candidate.topic}: "
                    f"{candidate.cs_relevance_score}"
                )

        lines.append("")

    lines.extend(
        [
            "=" * 110,
            "MERGED LESSON TOPICS",
            "=" * 110,
            "",
            "PRIMARY TOPICS",
            "-" * 110,
        ]
    )

    if not primary_topics:
        lines.append("None")
    else:
        for topic in primary_topics:
            lines.append(
                f"- {topic.topic}"
            )

    lines.extend(
        [
            "",
            "SUPPORTING TOPICS",
            "-" * 110,
        ]
    )

    if not supporting_topics:
        lines.append("None")
    else:
        for topic in supporting_topics:
            lines.append(
                f"- {topic.topic}"
            )

    lines.extend(
        [
            "",
            "UNMAPPED / EXTENDED TOPICS",
            "-" * 110,
        ]
    )

    if not unmapped_topics:
        lines.append("None")
    else:
        for signal in unmapped_topics:
            lines.append(
                "- "
                f"{getattr(signal, 'rough_topic', signal.domain)}"
            )

    lines.extend(
        [
            "",
            "DETAILED MERGED OFFICIAL TOPICS",
            "-" * 110,
        ]
    )

    if not result.merged_topics:
        lines.append(
            "No merged official AQA topics."
        )

    for topic in result.merged_topics:
        lines.extend(
            [
                f"- {topic.topic}",
                f"  concept_id: {topic.concept_id}",
                (
                    "  domain: "
                    f"{getattr(topic, 'domain', None)}"
                ),
                (
                    "  role: "
                    f"{getattr(topic, 'topic_role', None)}"
                ),
                f"  confidence: {topic.confidence}",
                (
                    "  ranking score: "
                    f"{getattr(topic, 'ranking_score', None)}"
                ),
                (
                    "  source chunks: "
                    f"{topic.source_chunk_ids}"
                ),
                (
                    "  support spans: "
                    f"{getattr(topic, 'support_span_count', None)}"
                ),
                (
                    "  mean semantic score: "
                    f"{getattr(topic, 'mean_semantic_score', None)}"
                ),
                (
                    "  mean salience score: "
                    f"{getattr(topic, 'mean_salience_score', None)}"
                ),
                (
                    "  coverage score: "
                    f"{getattr(topic, 'coverage_score', None)}"
                ),
                (
                    "  supporting candidates: "
                    f"{getattr(topic, 'supporting_candidate_count', None)}"
                ),
            ]
        )

        if topic.evidence:
            lines.append("  evidence:")

            for evidence in topic.evidence:
                lines.append(
                    f"    • {evidence}"
                )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines).strip() + "\n",
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