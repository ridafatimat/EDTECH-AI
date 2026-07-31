from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.services.topic_pipeline import (
    Module3TopicPipeline,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

TEST_DATA_DIR = (
    PROJECT_ROOT
    / "test_data"
)


def discover_default_input() -> Path:
    candidates = (
        TEST_DATA_DIR
        / "transcript_1_cleaned_chunks.json",
        TEST_DATA_DIR
        / "transcript_1_chunks.json",
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_chunks(
    input_path: Path,
) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Chunk JSON not found: {input_path}\n"
            "Run the Module 2 semantic chunking test first."
        )

    data = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

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
        chunks = None

    if not isinstance(
        chunks,
        list,
    ):
        raise ValueError(
            "Input JSON must be either a chunk list or "
            "an object containing a 'chunks' list."
        )

    return chunks


def model_to_dict(
    model,
) -> dict:
    if hasattr(
        model,
        "model_dump",
    ):
        return model.model_dump()

    return model.dict()


def save_readable_output(
    result,
    output_path: Path,
) -> None:
    lines: list[str] = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "AGENT 1 — MODULE 3 ROUGH TOPIC EXTRACTION"
    )

    lines.append(
        "=" * 100
    )

    lines.append(
        f"Embedding model: "
        f"{result.embedding_model}"
    )

    lines.append(
        f"Candidate keep threshold: "
        f"{result.candidate_keep_threshold}"
    )

    lines.append(
        f"Total chunks: "
        f"{result.total_chunks}"
    )

    lines.append(
        f"CS-relevant chunks: "
        f"{result.cs_relevant_chunks}"
    )

    lines.append(
        f"Non-CS/no-new-topic chunks: "
        f"{result.non_cs_chunks}"
    )

    lines.append(
        f"Official-topic chunks: {result.official_topic_chunks}"
    )
    lines.append(
        "Mixed official + unmapped chunks: "
        f"{result.mixed_official_unmapped_chunks}"
    )
    lines.append(
        f"Unmapped-CS chunks: {result.unmapped_cs_chunks}"
    )
    lines.append(
        f"Continuation/no-new-topic chunks: {result.continuation_chunks}"
    )
    lines.append(
        f"No-topic chunks: {result.no_topic_chunks}"
    )

    lines.append(
        f"LLM fallback chunks: "
        f"{result.llm_fallback_chunk_ids}"
    )

    lines.append("")

    for chunk_result in (
        result.chunk_results
    ):
        lines.append(
            "=" * 100
        )

        lines.append(
            f"CHUNK {chunk_result.chunk_id}"
        )

        lines.append(
            "=" * 100
        )

        lines.append(
            f"Source words: "
            f"{chunk_result.source_word_count}"
        )

        lines.append(
            f"Classification: {chunk_result.classification}"
        )

        lines.append(
            f"CS relevant: "
            f"{chunk_result.is_cs_relevant}"
        )

        lines.append(
            f"Creates new topic: {chunk_result.creates_new_topic}"
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

            for note in (
                chunk_result.notes
            ):
                lines.append(
                    f"- {note}"
                )

        lines.append(
            "\nRETAINED TOPICS"
        )

        lines.append(
            "-" * 100
        )

        if not (
            chunk_result
            .topic_candidates
        ):
            lines.append(
                "None"
            )

        for candidate in (
            chunk_result
            .topic_candidates
        ):
            lines.append(
                f"- {candidate.topic}"
            )

            lines.append(
                f"  concept_id: "
                f"{candidate.concept_id}"
            )

            lines.append(
                f"  domain: "
                f"{candidate.domain}"
            )

            lines.append(
                f"  official reference: {candidate.official_reference}"
            )

            lines.append(
                f"  confidence: "
                f"{candidate.confidence}"
            )

            lines.append(
                f"  salience score: {candidate.salience_score}"
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
                f"  method: "
                f"{candidate.extraction_method}"
            )

            lines.append(
                f"  matched aliases: "
                f"{candidate.matched_aliases}"
            )

            if candidate.evidence:
                lines.append(
                    "  evidence:"
                )

                for evidence in (
                    candidate.evidence
                ):
                    lines.append(
                        f"    • {evidence}"
                    )

        if chunk_result.has_unmapped_cs_content:
            lines.append("\nUNMAPPED CS SIGNALS")
            lines.append("-" * 100)
            for signal in chunk_result.unmapped_cs_signals:
                lines.append(
                    f"- {signal.rough_topic}"
                )
                lines.append(
                    f"  domain: {signal.domain}"
                )
                lines.append(
                    f"  score: {signal.score}"
                )
                lines.append(
                    f"  method: {signal.detection_method}"
                )
                lines.append(
                    f"  matched aliases: {signal.matched_aliases}"
                )
                lines.append(
                    f"  evidence: {signal.evidence}"
                )

        if (
            chunk_result
            .rejected_candidates
        ):
            lines.append(
                "\nREJECTED / LOW-CONFIDENCE CANDIDATES"
            )

            lines.append(
                "-" * 100
            )

            for candidate in (
                chunk_result
                .rejected_candidates
            ):
                lines.append(
                    f"- {candidate.topic}: "
                    f"{candidate.cs_relevance_score}"
                )

        lines.append("")

    lines.append(
        "=" * 100
    )

    lines.append(
        "MERGED LESSON TOPICS"
    )

    lines.append(
        "=" * 100
    )

    primary_topics = [
        topic
        for topic in result.merged_topics
        if topic.topic_role == "primary"
    ]

    supporting_topics = [
        topic
        for topic in result.merged_topics
        if topic.topic_role == "supporting"
    ]

    unmapped_topics: list[str] = []
    seen_unmapped_topics: set[str] = set()

    for chunk_result in result.chunk_results:
        for signal in chunk_result.unmapped_cs_signals:
            normalized_topic = " ".join(
                signal.rough_topic.lower().split()
            )

            if (
                not normalized_topic
                or normalized_topic in seen_unmapped_topics
            ):
                continue

            seen_unmapped_topics.add(
                normalized_topic
            )
            unmapped_topics.append(
                signal.rough_topic
            )

    lines.append("")
    lines.append("PRIMARY TOPICS")
    lines.append("-" * 100)

    if primary_topics:
        for topic in primary_topics:
            lines.append(
                f"- {topic.topic}"
            )
    else:
        lines.append("None")

    lines.append("")
    lines.append("SUPPORTING TOPICS")
    lines.append("-" * 100)

    if supporting_topics:
        for topic in supporting_topics:
            lines.append(
                f"- {topic.topic}"
            )
    else:
        lines.append("None")

    lines.append("")
    lines.append("UNMAPPED / EXTENDED TOPICS")
    lines.append("-" * 100)

    if unmapped_topics:
        for topic in unmapped_topics:
            lines.append(
                f"- {topic}"
            )
    else:
        lines.append("None")

    lines.append("")
    lines.append("DETAILED MERGED OFFICIAL TOPICS")
    lines.append("-" * 100)

    if not result.merged_topics:
        lines.append(
            "No retained official AQA topics."
        )

    for topic in result.merged_topics:
        lines.append(
            f"- {topic.topic}"
        )

        lines.append(
            f"  concept_id: "
            f"{topic.concept_id}"
        )

        lines.append(
            f"  domain: "
            f"{topic.domain}"
        )

        lines.append(
            f"  role: {topic.topic_role}"
        )

        lines.append(
            f"  confidence: "
            f"{topic.confidence}"
        )

        lines.append(
            f"  ranking score: {topic.ranking_score}"
        )

        lines.append(
            f"  source chunks: "
            f"{topic.source_chunk_ids}"
        )

        lines.append(
            f"  support spans: {topic.support_span_count}"
        )

        lines.append(
            f"  mean semantic score: {topic.mean_semantic_score}"
        )

        lines.append(
            f"  mean salience score: {topic.mean_salience_score}"
        )

        lines.append(
            f"  coverage score: {topic.coverage_score}"
        )

        lines.append(
            f"  supporting candidates: "
            f"{topic.supporting_candidate_count}"
        )

        if topic.evidence:
            lines.append(
                "  evidence:"
            )

            for evidence in (
                topic.evidence
            ):
                lines.append(
                    f"    • {evidence}"
                )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test Module 3 rough topic extraction "
            "on Module 2 chunk JSON."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=discover_default_input(),
        help=(
            "Path to Module 2 chunk JSON. "
            "Defaults to Transcript 1 chunks."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-text",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    chunks = load_chunks(
        args.input
    )

    output_json = (
        args.output_json
        or args.input.with_name(
            args.input.stem
            + "_topics.json"
        )
    )

    output_text = (
        args.output_text
        or args.input.with_name(
            args.input.stem
            + "_topics_readable.txt"
        )
    )

    print(
        "=" * 100
    )

    print(
        "AGENT 1 — MODULE 3 TOPIC EXTRACTION TEST"
    )

    print(
        "=" * 100
    )

    start = time.perf_counter()

    pipeline = Module3TopicPipeline()

    result = pipeline.process_chunks(
        chunks
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json.write_text(
        json.dumps(
            model_to_dict(
                result
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    save_readable_output(
        result=result,
        output_path=output_text,
    )

    print(
        f"\nProcessing time: "
        f"{elapsed:.3f} seconds"
    )

    print(
        f"Total chunks: "
        f"{result.total_chunks}"
    )

    print(
        f"CS-relevant chunks: "
        f"{result.cs_relevant_chunks}"
    )

    print(
        f"Non-CS chunks: "
        f"{result.non_cs_chunks}"
    )

    print(
        f"Merged topics: "
        f"{len(result.merged_topics)}"
    )

    print(
        f"LLM fallback chunks: "
        f"{result.llm_fallback_chunk_ids}"
    )

    primary_topics = [
        topic
        for topic in result.merged_topics
        if topic.topic_role == "primary"
    ]

    supporting_topics = [
        topic
        for topic in result.merged_topics
        if topic.topic_role == "supporting"
    ]

    unmapped_topics: list[str] = []
    seen_unmapped_topics: set[str] = set()

    for chunk_result in result.chunk_results:
        for signal in chunk_result.unmapped_cs_signals:
            normalized_topic = " ".join(
                signal.rough_topic.lower().split()
            )

            if (
                not normalized_topic
                or normalized_topic in seen_unmapped_topics
            ):
                continue

            seen_unmapped_topics.add(
                normalized_topic
            )
            unmapped_topics.append(
                signal.rough_topic
            )

    print(
        "\nPRIMARY TOPICS"
    )

    print(
        "-" * 100
    )

    if primary_topics:
        for topic in primary_topics:
            print(
                f"- {topic.topic}"
            )
    else:
        print("None")

    print(
        "\nSUPPORTING TOPICS"
    )

    print(
        "-" * 100
    )

    if supporting_topics:
        for topic in supporting_topics:
            print(
                f"- {topic.topic}"
            )
    else:
        print("None")

    print(
        "\nUNMAPPED / EXTENDED TOPICS"
    )

    print(
        "-" * 100
    )

    if unmapped_topics:
        for topic in unmapped_topics:
            print(
                f"- {topic}"
            )
    else:
        print("None")

    print(
        f"\nJSON saved to:\n"
        f"{output_json}"
    )

    print(
        f"\nReadable output saved to:\n"
        f"{output_text}"
    )

    print(
        "\nMODULE 3 TEST COMPLETED"
    )


if __name__ == "__main__":
    main()