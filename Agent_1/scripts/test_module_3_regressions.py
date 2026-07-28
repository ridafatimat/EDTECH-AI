from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from app.services.cs_relevance_filter import CSRelevanceFilter
from app.services.cs_unmapped_detector import CSUnmappedDetector
from app.services.topic_candidate_extractor import TopicCandidateExtractor
from app.services.topic_merger import TopicMerger
from app.schemas.topic import UnmappedCSSignal
from app.services.topic_pipeline import Module3TopicPipeline


def lexical_test_embeddings(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """
    Lightweight deterministic test embedding.

    Production still uses MiniLM. The regression suite avoids model loading
    so it remains fast and repeatable.
    """

    dimension = 512
    rows: list[np.ndarray] = []

    for text in texts:
        vector = np.zeros(dimension, dtype=np.float32)
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
        tokens = normalized.split()

        features = list(tokens)
        features.extend(
            " ".join(tokens[index:index + 2])
            for index in range(max(0, len(tokens) - 1))
        )
        features.extend(
            " ".join(tokens[index:index + 3])
            for index in range(max(0, len(tokens) - 2))
        )

        for feature in features:
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % dimension
            vector[index] += 1.0

        # Broad semantic anchors make the fake embedding useful for the
        # unmapped-CS detector while remaining deterministic and offline.
        semantic_groups = (
            {
                "programming", "code", "variable", "function", "method",
                "class", "object", "constructor", "attribute", "private",
                "public", "array", "loop",
            },
            {
                "algorithm", "search", "sorting", "sort", "trace",
                "efficiency", "decomposition", "abstraction",
            },
            {
                "binary", "hexadecimal", "bit", "byte", "encoding",
                "bitmap", "sound", "compression",
            },
            {
                "hardware", "software", "cpu", "memory", "storage",
                "operating", "system", "translator",
            },
            {
                "network", "protocol", "router", "switch", "security",
                "malware", "firewall", "encryption",
            },
            {
                "database", "sql", "table", "record", "field", "query",
            },
        )

        token_set = set(tokens)
        for group_index, group in enumerate(semantic_groups):
            overlap = len(token_set & group)
            if overlap:
                vector[group_index] += 4.0 * overlap

        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm

        rows.append(vector)

    return np.vstack(rows)


def build_test_pipeline() -> Module3TopicPipeline:
    extractor = TopicCandidateExtractor(
        embedding_function=lexical_test_embeddings
    )
    unmapped_detector = CSUnmappedDetector(
        embedding_function=lexical_test_embeddings
    )

    return Module3TopicPipeline(
        extractor=extractor,
        relevance_filter=CSRelevanceFilter(),
        unmapped_detector=unmapped_detector,
        merger=TopicMerger(),
    )


def test_pure_social_chunk_is_rejected() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 29,
                "overlap_word_count": 0,
                "text": (
                    "How was your holiday? The weather was cold and I "
                    "stayed in London. My family travelled but I stayed "
                    "at home."
                ),
            }
        ]
    )

    chunk = result.chunk_results[0]
    assert chunk.classification == "no_topic"
    assert not chunk.topic_candidates
    assert not chunk.has_unmapped_cs_content


def test_ambiguous_single_word_does_not_create_bits_topic() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 25,
                "overlap_word_count": 0,
                "text": (
                    "Wait a little bit and let me put this on the paper. "
                    "We will continue in a bit."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }
    assert "aqa_3_3_3_bits_bytes" not in ids


def test_incidental_integer_does_not_create_data_types_topic() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 35,
                "overlap_word_count": 0,
                "text": (
                    "The code declares integer k equals array zero, then "
                    "uses k while explaining how the values are swapped."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }
    assert "aqa_3_2_1_data_types" not in ids


def test_mixed_chunk_keeps_official_aqa_topics() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 70,
                "overlap_word_count": 0,
                "text": (
                    "We briefly talked about the holiday. Now look at the "
                    "array. The while loop uses K as the array index and "
                    "continues while K is less than array length. The array "
                    "is checked again in the next sentence."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }
    assert "aqa_3_2_2_iteration" in ids
    assert "aqa_3_2_6_arrays" in ids


def test_broad_sorting_topic_is_available_without_forcing_bubble_sort() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 55,
                "overlap_word_count": 0,
                "text": (
                    "We have sorting algorithms in the syllabus. This "
                    "example shows sorting by swapping values. We compare "
                    "and swap values so they are arranged in order."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }
    assert "aqa_3_1_4_sorting_algorithms" in ids


def test_same_evidence_does_not_create_duplicate_subroutine_labels() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 45,
                "overlap_word_count": 0,
                "text": (
                    "The function receives an array as input. This function "
                    "is called once and then returns the result."
                ),
            }
        ]
    )

    subroutine_ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
        if "subroutine" in candidate.topic.lower()
        or "function" in candidate.topic.lower()
    }
    assert len(subroutine_ids) <= 1


def test_overlap_continuation_does_not_create_new_topic() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 90,
                "overlap_word_count": 0,
                "text": (
                    "The array is checked using a while loop. The array "
                    "index increases while the condition remains true. "
                    "The array is then checked again."
                ),
            },
            {
                "chunk_id": 2,
                "word_count": 45,
                "overlap_word_count": 18,
                "text": (
                    "That result is not possible because the earlier "
                    "statement can only execute twice. The answer is two."
                ),
            },
        ]
    )

    second = result.chunk_results[1]
    assert second.classification == "continuation_no_new_topic"
    assert not second.creates_new_topic
    assert not second.is_cs_relevant


def test_unmapped_cs_content_is_flagged_without_fake_aqa_mapping() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 95,
                "overlap_word_count": 0,
                "text": (
                    "The constructor creates the object and initialises its "
                    "attributes. A second constructor accepts a size. The "
                    "private attributes cannot be accessed outside the "
                    "class, while public methods provide controlled access."
                ),
            }
        ]
    )

    chunk = result.chunk_results[0]
    assert chunk.classification in {
        "cs_related_unmapped",
        "mixed_official_and_unmapped",
    }
    assert chunk.has_unmapped_cs_content
    assert chunk.requires_llm_fallback


def test_adjacent_chunks_do_not_inflate_merged_confidence() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 60,
                "overlap_word_count": 0,
                "text": (
                    "The array index is less than the array length. The "
                    "array index is checked again."
                ),
            },
            {
                "chunk_id": 2,
                "word_count": 60,
                "overlap_word_count": 10,
                "text": (
                    "The array continues and the array index remains less "
                    "than the array length."
                ),
            },
            {
                "chunk_id": 3,
                "word_count": 60,
                "overlap_word_count": 10,
                "text": (
                    "The same array explanation continues with its array "
                    "index and array length."
                ),
            },
        ]
    )

    arrays = next(
        topic
        for topic in result.merged_topics
        if topic.concept_id == "aqa_3_2_6_arrays"
    )
    assert arrays.support_span_count == 1
    assert arrays.confidence <= 0.95


def test_catalogue_topics_keep_official_metadata() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 50,
                "overlap_word_count": 0,
                "text": (
                    "Binary search checks the middle item and discards half "
                    "of the sorted data."
                ),
            }
        ]
    )

    binary = next(
        candidate
        for candidate in result.chunk_results[0].topic_candidates
        if candidate.concept_id == "aqa_3_1_3_binary_search"
    )
    assert binary.official_reference == "3.1.3"
    assert binary.chapter_reference == "3.1"
    assert binary.official_title == "Searching algorithms"



def test_arraylist_compound_is_not_mapped_as_official_array() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 45,
                "overlap_word_count": 0,
                "text": (
                    "An ArrayList is a resizable list. The ArrayList can "
                    "grow while the program runs."
                ),
            }
        ]
    )

    chunk = result.chunk_results[0]
    ids = {
        candidate.concept_id
        for candidate in chunk.topic_candidates
    }

    assert "aqa_3_2_6_arrays" not in ids
    assert chunk.has_unmapped_cs_content
    assert any(
        signal.rough_topic == "Dynamic arrays and list collections"
        for signal in chunk.unmapped_cs_signals
    )


def test_real_arrays_and_arraylists_are_classified_separately() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 75,
                "overlap_word_count": 0,
                "text": (
                    "We finished one-dimensional arrays and practised array "
                    "index and array length questions. Next we introduce "
                    "ArrayLists, which are resizable list structures."
                ),
            }
        ]
    )

    chunk = result.chunk_results[0]
    ids = {
        candidate.concept_id
        for candidate in chunk.topic_candidates
    }

    assert "aqa_3_2_6_arrays" in ids
    assert chunk.classification == "mixed_official_and_unmapped"
    assert any(
        signal.rough_topic == "Dynamic arrays and list collections"
        for signal in chunk.unmapped_cs_signals
    )


def test_generic_code_tracing_language_maps_to_official_trace_topic() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 80,
                "overlap_word_count": 0,
                "text": (
                    "Follow the code step by step and track variable values. "
                    "Count statement executions and determine how many times "
                    "the loop runs before the condition becomes false."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }

    assert "aqa_3_1_1_algorithm_purpose_trace" in ids


def test_unmapped_signals_return_specific_rough_topics() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 95,
                "overlap_word_count": 0,
                "text": (
                    "The constructor initialises the object. Private "
                    "attributes are not accessible outside the class, and "
                    "public methods provide controlled access."
                ),
            }
        ]
    )

    rough_topics = {
        signal.rough_topic
        for signal in result.chunk_results[0].unmapped_cs_signals
    }

    assert "Object construction and initialisation" in rough_topics
    assert "Encapsulation and access modifiers" in rough_topics


def test_final_ranking_marks_keyword_dominated_topic_as_supporting() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 100,
                "overlap_word_count": 0,
                "text": (
                    "The array is traversed while the loop runs. Follow the "
                    "code step by step and track variable values. The index "
                    "is less than the array length."
                ),
            },
            {
                "chunk_id": 2,
                "word_count": 100,
                "overlap_word_count": 0,
                "text": (
                    "Continue the dry run and count statement executions. "
                    "Track variable values through the array while the loop "
                    "runs. One value is greater than another."
                ),
            },
            {
                "chunk_id": 3,
                "word_count": 80,
                "overlap_word_count": 0,
                "text": (
                    "The comparison uses less than and equal to, but the main "
                    "task is to trace the code and count how many times the "
                    "loop runs."
                ),
            },
        ]
    )

    tracing = next(
        topic
        for topic in result.merged_topics
        if topic.concept_id == "aqa_3_1_1_algorithm_purpose_trace"
    )
    relational = next(
        topic
        for topic in result.merged_topics
        if topic.concept_id == "aqa_3_2_4_relational_operations"
    )

    assert tracing.topic_role == "primary"
    assert relational.topic_role == "supporting"
    assert tracing.ranking_score > relational.ranking_score



def test_asr_style_execution_counts_map_to_tracing() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 95,
                "overlap_word_count": 0,
                "text": (
                    "The statement is executed four times and the loop ran "
                    "nine times. The index is updated after each pass. We "
                    "must decide whether that execution is possible."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }

    assert "aqa_3_1_1_algorithm_purpose_trace" in ids


def test_overlapped_same_tracing_topic_is_continuation() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 100,
                "overlap_word_count": 0,
                "text": (
                    "The statement is executed four times. Follow the code "
                    "and track how the index is updated after every pass."
                ),
            },
            {
                "chunk_id": 2,
                "word_count": 70,
                "overlap_word_count": 18,
                "text": (
                    "The loop ran twice, so the statement cannot execute "
                    "five times. That execution count is impossible."
                ),
            },
        ]
    )

    second = result.chunk_results[1]
    assert second.classification == "continuation_no_new_topic"
    assert not second.creates_new_topic
    assert not second.topic_candidates


def test_strong_lexical_unmapped_signal_skips_llm() -> None:
    pipeline = build_test_pipeline()

    signals = [
        UnmappedCSSignal(
            rough_topic="Object construction and initialisation",
            domain="Programming and software development",
            score=0.79,
            evidence="The constructor initialises the object.",
            matched_aliases=["constructor"],
            detection_method="lexical_semantic",
        )
    ]

    assert not pipeline._unmapped_requires_llm_fallback(signals)


def test_semantic_only_unmapped_signal_uses_llm() -> None:
    pipeline = build_test_pipeline()

    signals = [
        UnmappedCSSignal(
            rough_topic="Unmapped Computer Science content",
            domain="Programming and software development",
            score=0.74,
            evidence="This technical behaviour is discussed indirectly.",
            matched_aliases=[],
            detection_method="semantic",
        )
    ]

    assert pipeline._unmapped_requires_llm_fallback(signals)


def test_same_evidence_close_unmapped_topics_use_llm() -> None:
    pipeline = build_test_pipeline()

    evidence = "The object is created through a special class routine."
    signals = [
        UnmappedCSSignal(
            rough_topic="Object construction and initialisation",
            domain="Programming and software development",
            score=0.76,
            evidence=evidence,
            matched_aliases=["object creation"],
            detection_method="lexical_semantic",
        ),
        UnmappedCSSignal(
            rough_topic="Class lifecycle management",
            domain="Programming and software development",
            score=0.74,
            evidence=evidence,
            matched_aliases=["class routine"],
            detection_method="lexical_semantic",
        ),
    ]

    assert pipeline._unmapped_requires_llm_fallback(signals)

def main() -> None:
    tests = [
        test_pure_social_chunk_is_rejected,
        test_ambiguous_single_word_does_not_create_bits_topic,
        test_incidental_integer_does_not_create_data_types_topic,
        test_mixed_chunk_keeps_official_aqa_topics,
        test_broad_sorting_topic_is_available_without_forcing_bubble_sort,
        test_same_evidence_does_not_create_duplicate_subroutine_labels,
        test_overlap_continuation_does_not_create_new_topic,
        test_unmapped_cs_content_is_flagged_without_fake_aqa_mapping,
        test_adjacent_chunks_do_not_inflate_merged_confidence,
        test_catalogue_topics_keep_official_metadata,
        test_arraylist_compound_is_not_mapped_as_official_array,
        test_real_arrays_and_arraylists_are_classified_separately,
        test_generic_code_tracing_language_maps_to_official_trace_topic,
        test_asr_style_execution_counts_map_to_tracing,
        test_overlapped_same_tracing_topic_is_continuation,
        test_unmapped_signals_return_specific_rough_topics,
        test_strong_lexical_unmapped_signal_skips_llm,
        test_semantic_only_unmapped_signal_uses_llm,
        test_same_evidence_close_unmapped_topics_use_llm,
        test_final_ranking_marks_keyword_dominated_topic_as_supporting,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nALL MODULE 3 GENERIC-FIX REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()