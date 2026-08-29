from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from docx import Document
from dotenv import load_dotenv

from app.db.repositories.technical_correction_repository import (
    PostgreSQLTechnicalCorrectionRepository,
)
from app.db.session import session_scope
from app.services.semantic_chunker import (
    SemanticChunker,
    SemanticChunkingConfig,
)
from app.services.selective_technical_normalizer import (
    SelectiveTechnicalNormaliser,
)
from app.services.technical_correction_client import (
    GroqTechnicalCorrectionClient,
)
from app.services.topic_pipeline import (
    Module3TopicPipeline,
)
from app.services.transcript_preprocessor import (
    preprocess_transcript,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "test_outputs"
    / "pipeline_runs"
)


# ------------------------------------------------------------------
# Generic serialization helpers
# ------------------------------------------------------------------

def to_plain_data(value: Any) -> Any:
    """
    Convert Pydantic models, dataclasses and nested objects into values
    that json.dumps can safely serialize.
    """

    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "model_dump"):
        return to_plain_data(
            value.model_dump()
        )

    if hasattr(value, "dict"):
        try:
            return to_plain_data(
                value.dict()
            )
        except TypeError:
            pass

    if hasattr(value, "to_dict"):
        return to_plain_data(
            value.to_dict()
        )

    if is_dataclass(value):
        return to_plain_data(
            asdict(value)
        )

    if isinstance(value, dict):
        return {
            str(key): to_plain_data(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            to_plain_data(item)
            for item in value
        ]

    if isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "__dict__"):
        return {
            key: to_plain_data(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            to_plain_data(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text.strip() + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------------
# Path and input helpers
# ------------------------------------------------------------------

def safe_run_name(
    source_file: Path,
    requested_name: str | None,
) -> str:
    raw_name = (
        requested_name
        if requested_name
        else source_file.stem
    )

    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        raw_name.strip(),
    )

    cleaned = re.sub(
        r"_+",
        "_",
        cleaned,
    ).strip("_")

    return cleaned or "transcript_run"


def extract_docx_text(
    docx_path: Path,
) -> str:
    if not docx_path.exists():
        raise FileNotFoundError(
            f"Input DOCX not found: {docx_path}"
        )

    if docx_path.suffix.casefold() != ".docx":
        raise ValueError(
            "Agent 1 pipeline currently expects a .docx transcript."
        )

    document = Document(
        docx_path
    )

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    extracted = "\n".join(
        paragraphs
    ).strip()

    if not extracted:
        raise ValueError(
            "No transcript text was found inside the DOCX."
        )

    return extracted


def load_saved_chunks(
    chunks_path: Path,
) -> list[dict[str, Any]]:
    """
    Module 3 deliberately reloads Module 2's saved JSON instead of using
    the in-memory result. This verifies the real file-to-file pipeline.
    """

    data = json.loads(
        chunks_path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(data, dict):
        chunks = data.get(
            "chunks"
        )
    elif isinstance(data, list):
        chunks = data
    else:
        chunks = None

    if not isinstance(chunks, list):
        raise ValueError(
            "Module 2 output must contain a 'chunks' list."
        )

    return chunks


# ------------------------------------------------------------------
# Human-readable outputs
# ------------------------------------------------------------------

def save_chunks_readable(
    *,
    chunks_payload: dict[str, Any],
    output_path: Path,
) -> None:
    lines: list[str] = [
        "=" * 100,
        "AGENT 1 — MODULE 2 SEMANTIC CHUNKS",
        "=" * 100,
        "",
        (
            "Embedding model: "
            f"{chunks_payload.get('embedding_model')}"
        ),
        (
            "Semantic threshold: "
            f"{chunks_payload.get('semantic_threshold')}"
        ),
        (
            "Total sentences: "
            f"{chunks_payload.get('total_sentences')}"
        ),
        (
            "Total words: "
            f"{chunks_payload.get('total_words')}"
        ),
        (
            "Final chunks: "
            f"{len(chunks_payload.get('chunks', []))}"
        ),
        "",
    ]

    for chunk in chunks_payload.get(
        "chunks",
        [],
    ):
        lines.extend(
            [
                "-" * 100,
                (
                    "CHUNK "
                    f"{chunk.get('chunk_id')}"
                ),
                (
                    "Words: "
                    f"{chunk.get('word_count')}"
                ),
                (
                    "Sentences: "
                    f"{chunk.get('sentence_count')}"
                ),
                (
                    "Sentence range: "
                    f"{chunk.get('start_sentence')}"
                    " → "
                    f"{chunk.get('end_sentence')}"
                ),
                (
                    "Boundary reason: "
                    f"{chunk.get('boundary_reason')}"
                ),
                (
                    "Overlap words: "
                    f"{chunk.get('overlap_word_count', 0)}"
                ),
                "-" * 100,
                str(
                    chunk.get(
                        "text",
                        "",
                    )
                ),
                "",
            ]
        )

    write_text(
        output_path,
        "\n".join(lines),
    )


def save_topics_readable(
    *,
    topic_payload: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save the complete human-readable Module 3 report.

    The final lesson-level result is grouped into primary, supporting,
    and unmapped/extended topics. Chunk-level evidence is retained for
    debugging and evaluation.
    """

    chunk_results = topic_payload.get(
        "chunk_results",
        [],
    )

    merged_topics = topic_payload.get(
        "merged_topics",
        [],
    )

    primary_topics = [
        topic
        for topic in merged_topics
        if topic.get("topic_role") == "primary"
    ]

    supporting_topics = [
        topic
        for topic in merged_topics
        if topic.get("topic_role") == "supporting"
    ]

    # Preserve first-seen order while removing duplicate rough topics.
    unmapped_topics: list[dict[str, Any]] = []
    seen_unmapped: set[str] = set()

    for chunk_result in chunk_results:
        for signal in chunk_result.get(
            "unmapped_cs_signals",
            [],
        ):
            rough_topic = str(
                signal.get(
                    "rough_topic",
                    signal.get("domain", "Unmapped CS topic"),
                )
            ).strip()

            key = rough_topic.casefold()

            if not rough_topic or key in seen_unmapped:
                continue

            seen_unmapped.add(key)
            unmapped_topics.append(signal)

    classification_counts: dict[str, int] = {}

    for chunk_result in chunk_results:
        classification = str(
            chunk_result.get(
                "classification",
                "unknown",
            )
        )

        classification_counts[classification] = (
            classification_counts.get(classification, 0)
            + 1
        )

    lines: list[str] = [
        "=" * 100,
        "AGENT 1 — MODULE 3 ROUGH TOPIC EXTRACTION",
        "=" * 100,
        "",
        (
            "Embedding model: "
            f"{topic_payload.get('embedding_model')}"
        ),
        (
            "Candidate keep threshold: "
            f"{topic_payload.get('candidate_keep_threshold')}"
        ),
        (
            "Total chunks: "
            f"{topic_payload.get('total_chunks')}"
        ),
        (
            "CS-relevant chunks: "
            f"{topic_payload.get('cs_relevant_chunks')}"
        ),
        (
            "Non-CS/no-new-topic chunks: "
            f"{topic_payload.get('non_cs_chunks')}"
        ),
        (
            "Official-topic chunks: "
            f"{classification_counts.get('official_aqa_topic', 0)}"
        ),
        (
            "Mixed official + unmapped chunks: "
            f"{classification_counts.get('mixed_official_and_unmapped', 0)}"
        ),
        (
            "Unmapped-CS chunks: "
            f"{classification_counts.get('cs_related_unmapped', 0)}"
        ),
        (
            "Continuation/no-new-topic chunks: "
            f"{classification_counts.get('continuation_no_new_topic', 0)}"
        ),
        (
            "No-topic chunks: "
            f"{classification_counts.get('no_topic', 0)}"
        ),
        (
            "LLM fallback chunks: "
            f"{topic_payload.get('llm_fallback_chunk_ids', [])}"
        ),
        "",
    ]

    for chunk_result in chunk_results:
        lines.extend(
            [
                "=" * 100,
                (
                    "CHUNK "
                    f"{chunk_result.get('chunk_id')}"
                ),
                "=" * 100,
                (
                    "Source words: "
                    f"{chunk_result.get('source_word_count')}"
                ),
                (
                    "Classification: "
                    f"{chunk_result.get('classification')}"
                ),
                (
                    "CS relevant: "
                    f"{chunk_result.get('is_cs_relevant')}"
                ),
                (
                    "Creates new topic: "
                    f"{chunk_result.get('creates_new_topic')}"
                ),
                (
                    "Chunk relevance score: "
                    f"{chunk_result.get('cs_relevance_score')}"
                ),
                (
                    "Requires LLM fallback: "
                    f"{chunk_result.get('requires_llm_fallback')}"
                ),
            ]
        )

        notes = chunk_result.get(
            "notes",
            [],
        )

        if notes:
            lines.append("Notes:")

            for note in notes:
                lines.append(
                    f"- {note}"
                )

        lines.extend(
            [
                "",
                "RETAINED TOPICS",
                "-" * 100,
            ]
        )

        candidates = chunk_result.get(
            "topic_candidates",
            [],
        )

        if not candidates:
            lines.append("None")

        for candidate in candidates:
            lines.extend(
                [
                    (
                        "- "
                        f"{candidate.get('topic')}"
                    ),
                    (
                        "  concept_id: "
                        f"{candidate.get('concept_id')}"
                    ),
                    (
                        "  domain: "
                        f"{candidate.get('domain')}"
                    ),
                    (
                        "  official reference: "
                        f"{candidate.get('official_reference')}"
                    ),
                    (
                        "  confidence: "
                        f"{candidate.get('confidence')}"
                    ),
                    (
                        "  salience score: "
                        f"{candidate.get('salience_score')}"
                    ),
                    (
                        "  keyword score: "
                        f"{candidate.get('keyword_score')}"
                    ),
                    (
                        "  semantic score: "
                        f"{candidate.get('semantic_score')}"
                    ),
                    (
                        "  method: "
                        f"{candidate.get('match_method')}"
                    ),
                    (
                        "  matched aliases: "
                        f"{candidate.get('matched_aliases', [])}"
                    ),
                ]
            )

            evidence = candidate.get(
                "evidence",
                [],
            )

            if evidence:
                lines.append("  evidence:")

                for item in evidence:
                    lines.append(
                        f"    • {item}"
                    )

        chunk_unmapped = chunk_result.get(
            "unmapped_cs_signals",
            [],
        )

        if chunk_unmapped:
            lines.extend(
                [
                    "",
                    "UNMAPPED CS SIGNALS",
                    "-" * 100,
                ]
            )

            for signal in chunk_unmapped:
                lines.extend(
                    [
                        (
                            "- "
                            f"{signal.get('rough_topic', signal.get('domain'))}"
                        ),
                        (
                            "  domain: "
                            f"{signal.get('domain')}"
                        ),
                        (
                            "  score: "
                            f"{signal.get('score')}"
                        ),
                        (
                            "  method: "
                            f"{signal.get('detection_method')}"
                        ),
                        (
                            "  matched aliases: "
                            f"{signal.get('matched_aliases', [])}"
                        ),
                        (
                            "  evidence: "
                            f"{signal.get('evidence')}"
                        ),
                    ]
                )

        rejected = chunk_result.get(
            "rejected_candidates",
            [],
        )

        if rejected:
            lines.extend(
                [
                    "",
                    "REJECTED / LOW-CONFIDENCE CANDIDATES",
                    "-" * 100,
                ]
            )

            for candidate in rejected:
                lines.append(
                    "- "
                    f"{candidate.get('topic')}: "
                    f"{candidate.get('cs_relevance_score')}"
                )

        lines.append("")

    lines.extend(
        [
            "=" * 100,
            "MERGED LESSON TOPICS",
            "=" * 100,
            "",
            "PRIMARY TOPICS",
            "-" * 100,
        ]
    )

    if not primary_topics:
        lines.append("None")
    else:
        for topic in primary_topics:
            lines.append(
                f"- {topic.get('topic')}"
            )

    lines.extend(
        [
            "",
            "SUPPORTING TOPICS",
            "-" * 100,
        ]
    )

    if not supporting_topics:
        lines.append("None")
    else:
        for topic in supporting_topics:
            lines.append(
                f"- {topic.get('topic')}"
            )

    lines.extend(
        [
            "",
            "UNMAPPED / EXTENDED TOPICS",
            "-" * 100,
        ]
    )

    if not unmapped_topics:
        lines.append("None")
    else:
        for signal in unmapped_topics:
            lines.append(
                "- "
                f"{signal.get('rough_topic', signal.get('domain'))}"
            )

    lines.extend(
        [
            "",
            "DETAILED MERGED OFFICIAL TOPICS",
            "-" * 100,
        ]
    )

    if not merged_topics:
        lines.append(
            "No merged official AQA topics."
        )

    for topic in merged_topics:
        lines.extend(
            [
                (
                    "- "
                    f"{topic.get('topic')}"
                ),
                (
                    "  concept_id: "
                    f"{topic.get('concept_id')}"
                ),
                (
                    "  domain: "
                    f"{topic.get('domain')}"
                ),
                (
                    "  role: "
                    f"{topic.get('topic_role')}"
                ),
                (
                    "  confidence: "
                    f"{topic.get('confidence')}"
                ),
                (
                    "  ranking score: "
                    f"{topic.get('ranking_score')}"
                ),
                (
                    "  source chunks: "
                    f"{topic.get('source_chunk_ids')}"
                ),
                (
                    "  support spans: "
                    f"{topic.get('support_span_count')}"
                ),
                (
                    "  mean semantic score: "
                    f"{topic.get('mean_semantic_score')}"
                ),
                (
                    "  mean salience score: "
                    f"{topic.get('mean_salience_score')}"
                ),
                (
                    "  coverage score: "
                    f"{topic.get('coverage_score')}"
                ),
                (
                    "  supporting candidates: "
                    f"{topic.get('supporting_candidate_count')}"
                ),
            ]
        )

        evidence = topic.get(
            "evidence",
            [],
        )

        if evidence:
            lines.append("  evidence:")

            for item in evidence:
                lines.append(
                    f"    • {item}"
                )

    write_text(
        output_path,
        "\n".join(lines),
    )

# ------------------------------------------------------------------
# Manifest helpers
# ------------------------------------------------------------------

def new_manifest(
    *,
    run_name: str,
    source_file: Path,
    run_directory: Path,
) -> dict[str, Any]:
    now = datetime.now().astimezone().isoformat()

    return {
        "run_name": run_name,
        "source_file": str(
            source_file.resolve()
        ),
        "run_directory": str(
            run_directory.resolve()
        ),
        "created_at": now,
        "updated_at": now,
        "status": "running",
        "current_stage": "initialising",
        "stages": {
            "module_1_preprocessing": {
                "status": "pending",
                "outputs": {},
            },
            "module_1_technical_normalisation": {
                "status": "pending",
                "outputs": {},
            },
            "module_2_chunking": {
                "status": "pending",
                "outputs": {},
            },
            "module_3_topic_extraction": {
                "status": "pending",
                "outputs": {},
            },
        },
        "error": None,
    }


def save_manifest(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    manifest["updated_at"] = (
        datetime.now()
        .astimezone()
        .isoformat()
    )

    write_json(
        manifest_path,
        manifest,
    )


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def run_pipeline(
    *,
    source_file: Path,
    output_root: Path,
    run_name: str,
    no_llm: bool,
) -> Path:
    run_directory = (
        output_root
        / run_name
    )

    preprocessing_dir = (
        run_directory
        / "01_preprocessing"
    )

    chunking_dir = (
        run_directory
        / "02_chunking"
    )

    topics_dir = (
        run_directory
        / "03_topic_extraction"
    )

    manifest_path = (
        run_directory
        / "pipeline_manifest.json"
    )

    raw_output = (
        preprocessing_dir
        / "raw_extracted_transcript.txt"
    )

    deterministic_output = (
        preprocessing_dir
        / "deterministic_cleaned.txt"
    )

    preprocessing_audit = (
        preprocessing_dir
        / "preprocessing_audit.json"
    )

    final_cleaned_output = (
        preprocessing_dir
        / "cleaned_transcript.txt"
    )

    technical_audit = (
        preprocessing_dir
        / "technical_normalisation_audit.json"
    )

    chunks_output = (
        chunking_dir
        / "chunks.json"
    )

    chunks_readable_output = (
        chunking_dir
        / "chunks_readable.txt"
    )

    topics_output = (
        topics_dir
        / "topics.json"
    )

    topics_readable_output = (
        topics_dir
        / "topics_readable.txt"
    )

    manifest = new_manifest(
        run_name=run_name,
        source_file=source_file,
        run_directory=run_directory,
    )

    current_stage = "initialising"

    save_manifest(
        manifest_path=manifest_path,
        manifest=manifest,
    )

    try:
        # ----------------------------------------------------------
        # MODULE 1A — extraction + deterministic preprocessing
        # ----------------------------------------------------------
        current_stage = (
            "module_1_preprocessing"
        )

        manifest["current_stage"] = (
            current_stage
        )

        manifest["stages"][
            current_stage
        ]["status"] = "running"

        save_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
        )

        stage_start = time.perf_counter()

        raw_text = extract_docx_text(
            source_file
        )

        write_text(
            raw_output,
            raw_text,
        )

        preprocessing_result = (
            preprocess_transcript(
                raw_text=raw_text,
                source_name=source_file.name,
            )
        )

        deterministic_text = (
            preprocessing_result
            .cleaned_text
            .strip()
        )

        if not deterministic_text:
            raise ValueError(
                "Module 1 preprocessing produced empty text."
            )

        write_text(
            deterministic_output,
            deterministic_text,
        )

        write_json(
            preprocessing_audit,
            preprocessing_result,
        )

        manifest["stages"][
            current_stage
        ] = {
            "status": "completed",
            "duration_seconds": round(
                time.perf_counter()
                - stage_start,
                3,
            ),
            "outputs": {
                "raw_extracted_transcript": str(
                    raw_output.resolve()
                ),
                "deterministic_cleaned": str(
                    deterministic_output.resolve()
                ),
                "audit": str(
                    preprocessing_audit.resolve()
                ),
            },
        }

        save_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
        )

        # ----------------------------------------------------------
        # MODULE 1B — technical terminology normalisation
        # ----------------------------------------------------------
        current_stage = (
            "module_1_technical_normalisation"
        )

        manifest["current_stage"] = (
            current_stage
        )

        # Fix the initial manifest typo while keeping compatibility
        # with a partially written manifest from an interrupted run.
        manifest["stages"].pop(
            "module_1_technical_normalisation",
            None,
        )
        manifest["stages"].pop(
            "module_1_technical_normalisation",
            None,
        )
        manifest["stages"][
            current_stage
        ] = {
            "status": "running",
            "outputs": {},
        }

        save_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
        )

        stage_start = time.perf_counter()

        correction_client = (
            None
            if no_llm
            else GroqTechnicalCorrectionClient()
        )

        with session_scope() as session:
            repository = (
                PostgreSQLTechnicalCorrectionRepository(
                    session
                )
            )

            normaliser = (
                SelectiveTechnicalNormaliser(
                    repository=repository,
                    correction_client=(
                        correction_client
                    ),
                )
            )

            technical_result = (
                normaliser.normalise(
                    deterministic_text
                )
            )

        final_cleaned_text = (
            technical_result
            .normalised_text
            .strip()
        )

        if not final_cleaned_text:
            raise ValueError(
                "Technical normalisation produced empty text."
            )

        write_text(
            final_cleaned_output,
            final_cleaned_text,
        )

        write_json(
            technical_audit,
            technical_result,
        )

        manifest["stages"][
            current_stage
        ] = {
            "status": "completed",
            "duration_seconds": round(
                time.perf_counter()
                - stage_start,
                3,
            ),
            "outputs": {
                "cleaned_transcript": str(
                    final_cleaned_output.resolve()
                ),
                "audit": str(
                    technical_audit.resolve()
                ),
            },
        }

        save_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
        )

        # ----------------------------------------------------------
        # MODULE 2 — reads the SAVED cleaned transcript
        # ----------------------------------------------------------
        current_stage = (
            "module_2_chunking"
        )

        manifest["current_stage"] = (
            current_stage
        )

        manifest["stages"][
            current_stage
        ]["status"] = "running"

        save_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
        )

        stage_start = time.perf_counter()

        module_2_input_text = (
            final_cleaned_output
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        chunking_config = (
            SemanticChunkingConfig(
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
        )

        chunker = SemanticChunker(
            config=chunking_config
        )

        chunking_result = chunker.chunk(
            module_2_input_text
        )

        chunking_payload = to_plain_data(
            chunking_result
        )

        write_json(
            chunks_output,
            chunking_payload,
        )

        save_chunks_readable(
            chunks_payload=chunking_payload,
            output_path=(
                chunks_readable_output
            ),
        )

        manifest["stages"][
            current_stage
        ] = {
            "status": "completed",
            "duration_seconds": round(
                time.perf_counter()
                - stage_start,
                3,
            ),
            "input": str(
                final_cleaned_output.resolve()
            ),
            "outputs": {
                "chunks_json": str(
                    chunks_output.resolve()
                ),
                "chunks_readable": str(
                    chunks_readable_output.resolve()
                ),
            },
            "chunk_count": len(
                chunking_payload.get(
                    "chunks",
                    [],
                )
            ),
        }

        save_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
        )

        # ----------------------------------------------------------
        # MODULE 3 — reads the SAVED chunks.json
        # ----------------------------------------------------------
        current_stage = (
            "module_3_topic_extraction"
        )

        manifest["current_stage"] = (
            current_stage
        )

        manifest["stages"][
            current_stage
        ]["status"] = "running"

        save_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
        )

        stage_start = time.perf_counter()

        saved_chunks = load_saved_chunks(
            chunks_output
        )

        topic_pipeline = (
            Module3TopicPipeline()
        )

        topic_result = (
            topic_pipeline.process_chunks(
                saved_chunks
            )
        )

        topic_payload = to_plain_data(
            topic_result
        )

        write_json(
            topics_output,
            topic_payload,
        )

        save_topics_readable(
            topic_payload=topic_payload,
            output_path=(
                topics_readable_output
            ),
        )

        manifest["stages"][
            current_stage
        ] = {
            "status": "completed",
            "duration_seconds": round(
                time.perf_counter()
                - stage_start,
                3,
            ),
            "input": str(
                chunks_output.resolve()
            ),
            "outputs": {
                "topics_json": str(
                    topics_output.resolve()
                ),
                "topics_readable": str(
                    topics_readable_output.resolve()
                ),
            },
            "merged_topic_count": len(
                topic_payload.get(
                    "merged_topics",
                    [],
                )
            ),
        }

        manifest["status"] = "completed"
        manifest["current_stage"] = None
        manifest["error"] = None

        save_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
        )

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["current_stage"] = (
            current_stage
        )
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

        if current_stage in manifest["stages"]:
            manifest["stages"][
                current_stage
            ]["status"] = "failed"

        save_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
        )

        raise

    print()
    print("=" * 100)
    print("AGENT 1 PIPELINE COMPLETED")
    print("=" * 100)
    print(f"Run: {run_name}")
    print(f"Source: {source_file.resolve()}")
    print()
    print(
        "Module 1 cleaned transcript:"
    )
    print(
        final_cleaned_output.resolve()
    )
    print()
    print("Module 2 chunks:")
    print(
        chunks_output.resolve()
    )
    print()
    print("Module 3 topics:")
    print(
        topics_output.resolve()
    )
    print()
    print("Pipeline manifest:")
    print(
        manifest_path.resolve()
    )
    print("=" * 100)

    return run_directory


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Run Agent 1 as a saved file-to-file pipeline: "
            "DOCX → cleaned transcript → chunks → rough topics."
        )
    )

    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help=(
            "Path to the source transcript DOCX."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root folder for persistent pipeline outputs."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional folder name. Defaults to the source filename."
        ),
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Disable Groq for technical correction. "
            "Approved PostgreSQL memory and deterministic rules "
            "will still run."
        ),
    )

    args = parser.parse_args()

    source_file = args.file

    if not source_file.is_absolute():
        source_file = (
            PROJECT_ROOT
            / source_file
        )

    output_root = args.output_root

    if not output_root.is_absolute():
        output_root = (
            PROJECT_ROOT
            / output_root
        )

    run_name = safe_run_name(
        source_file=source_file,
        requested_name=args.run_name,
    )

    try:
        run_pipeline(
            source_file=source_file,
            output_root=output_root,
            run_name=run_name,
            no_llm=args.no_llm,
        )
    except Exception as exc:
        print()
        print("=" * 100)
        print("AGENT 1 PIPELINE FAILED")
        print("=" * 100)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print(
            "Check pipeline_manifest.json inside "
            "the run folder for the failed stage."
        )
        print("=" * 100)
        sys.exit(1)


if __name__ == "__main__":
    main()