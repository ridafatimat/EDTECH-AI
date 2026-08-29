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


def test_final_ranking_suppresses_keyword_dominated_supporting_topic() -> None:
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

    merged_ids = {
        topic.concept_id
        for topic in result.merged_topics
    }

    assert tracing.topic_role == "primary"
    assert "aqa_3_2_4_relational_operations" not in merged_ids



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
    assert any(
        candidate.concept_id == "aqa_3_1_1_algorithm_purpose_trace"
        for candidate in second.topic_candidates
    )


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


def test_binary_search_does_not_map_to_number_bases() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 70,
                "overlap_word_count": 0,
                "text": (
                    "Binary search checks the middle value of a sorted list. "
                    "If the target is smaller, it discards the upper half. "
                    "If the target is larger, it discards the lower half."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }

    assert "aqa_3_1_3_binary_search" in ids
    assert "aqa_3_3_1_number_bases" not in ids


def test_binary_number_context_does_not_map_to_binary_search() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 70,
                "overlap_word_count": 0,
                "text": (
                    "Binary is base two, decimal is base ten and "
                    "hexadecimal is base sixteen. Each number base uses "
                    "place values, and values can be converted between them."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }

    assert "aqa_3_3_1_number_bases" in ids
    assert "aqa_3_1_3_binary_search" not in ids


def test_mixed_binary_context_keeps_both_official_topics() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 115,
                "overlap_word_count": 0,
                "text": (
                    "First we revise number bases. Binary is base two, "
                    "decimal is base ten and hexadecimal is base sixteen. "
                    "We then move to searching algorithms. Binary search "
                    "checks the middle item of sorted data and discards "
                    "either the lower half or the upper half."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }

    assert "aqa_3_3_1_number_bases" in ids
    assert "aqa_3_1_3_binary_search" in ids


def test_incidental_sorting_mention_is_rejected() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 35,
                "overlap_word_count": 0,
                "text": (
                    "We also have sorting algorithms in our syllabus, by "
                    "the way. We will cover them in another lesson."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }

    assert "aqa_3_1_4_sorting_algorithms" not in ids


def test_explained_sorting_topic_is_retained() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 80,
                "overlap_word_count": 0,
                "text": (
                    "A sorting algorithm arranges values into order. For "
                    "example, we compare two adjacent values and swap them "
                    "when they are in the wrong order. Let's trace the list "
                    "step by step and explain why each swap happens."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }

    assert "aqa_3_1_4_sorting_algorithms" in ids


def test_relational_operator_phrases_need_operator_context() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 70,
                "overlap_word_count": 0,
                "text": (
                    "The while loop continues while k is less than array "
                    "length. We trace the loop and update the array index "
                    "after every pass."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }

    assert "aqa_3_2_4_relational_operations" not in ids


def test_relational_operators_survive_explicit_teaching_context() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 85,
                "overlap_word_count": 0,
                "text": (
                    "A relational operator compares two values. Less than, "
                    "greater than and equal to are comparison operators. For "
                    "example, x less than y creates a Boolean expression. "
                    "Let's decide whether each condition evaluates to true "
                    "or false."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }

    assert "aqa_3_2_4_relational_operations" in ids


def test_shared_evidence_prefers_stronger_main_topic() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 95,
                "overlap_word_count": 0,
                "text": (
                    "We trace this while loop step by step. The loop checks "
                    "whether k is less than the array length, then updates k. "
                    "How many times does the statement execute? Explain why "
                    "the loop stops and track every variable value."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }

    assert "aqa_3_1_1_algorithm_purpose_trace" in ids
    assert "aqa_3_2_4_relational_operations" not in ids


def test_high_coverage_topic_is_primary_and_single_chunk_topic_supporting() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 90,
                "overlap_word_count": 0,
                "text": (
                    "Linear search checks each item in order. For example, "
                    "we start at the first value and compare it with the "
                    "target."
                ),
            },
            {
                "chunk_id": 2,
                "word_count": 90,
                "overlap_word_count": 0,
                "text": (
                    "Continue the linear search by checking the next item. "
                    "Explain why the algorithm stops when the target is "
                    "found."
                ),
            },
            {
                "chunk_id": 3,
                "word_count": 90,
                "overlap_word_count": 0,
                "text": (
                    "Trace the linear search step by step through this list. "
                    "Count the comparisons made before finding the target."
                ),
            },
            {
                "chunk_id": 4,
                "word_count": 90,
                "overlap_word_count": 0,
                "text": (
                    "Now compare another linear search example and explain "
                    "how an unsuccessful search reaches the end of the list."
                ),
            },
            {
                "chunk_id": 5,
                "word_count": 75,
                "overlap_word_count": 0,
                "text": (
                    "A for loop repeats a fixed number of times. For example, "
                    "the loop prints each value once."
                ),
            },
        ]
    )

    linear = next(
        topic
        for topic in result.merged_topics
        if topic.concept_id == "aqa_3_1_3_linear_search"
    )
    iteration = next(
        topic
        for topic in result.merged_topics
        if topic.concept_id == "aqa_3_2_2_iteration"
    )

    assert linear.coverage_score > iteration.coverage_score
    assert linear.topic_role == "primary"
    assert iteration.topic_role == "supporting"


def test_generic_multi_topic_recap_does_not_create_primary_topics() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 45,
                "overlap_word_count": 0,
                "text": (
                    "Today we covered linear search, binary search, bubble "
                    "sort and arrays. Those were the topics in this lesson."
                ),
            }
        ]
    )

    assert not any(
        topic.topic_role == "primary"
        for topic in result.merged_topics
    )


def test_substantive_topic_survives_separate_generic_recap() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 90,
                "overlap_word_count": 0,
                "text": (
                    "Linear search checks each value in order until the target "
                    "is found. For example, we trace the list step by step and "
                    "count every comparison."
                ),
            },
            {
                "chunk_id": 2,
                "word_count": 45,
                "overlap_word_count": 0,
                "text": (
                    "To recap, we covered linear search, arrays, sorting and "
                    "iteration."
                ),
            },
        ]
    )

    linear = next(
        topic
        for topic in result.merged_topics
        if topic.concept_id == "aqa_3_1_3_linear_search"
    )

    assert linear.topic_role == "primary"

    other_ids = {
        topic.concept_id
        for topic in result.merged_topics
        if topic.concept_id != "aqa_3_1_3_linear_search"
    }

    assert "aqa_3_2_6_arrays" not in other_ids
    assert "aqa_3_2_2_iteration" not in other_ids


def test_brief_comparison_only_binary_search_is_rejected() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 115,
                "overlap_word_count": 0,
                "text": (
                    "Linear search starts at the beginning of the data set "
                    "and checks each item in turn until the target is found. "
                    "It does not require the data to be in order, unlike "
                    "another type of search, binary search. For example, "
                    "we trace a list step by step, compare each value with "
                    "the target, update the index and stop when found is "
                    "true or the end of the list is reached."
                ),
            }
        ]
    )

    retained_ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }
    rejected_ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].rejected_candidates
    }
    merged_ids = {
        topic.concept_id
        for topic in result.merged_topics
    }

    assert "aqa_3_1_3_linear_search" in retained_ids
    assert "aqa_3_1_3_binary_search" not in retained_ids
    assert "aqa_3_1_3_binary_search" in rejected_ids
    assert "aqa_3_1_3_binary_search" not in merged_ids


def test_detailed_merge_sort_comparison_is_retained() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 180,
                "overlap_word_count": 0,
                "text": (
                    "Bubble sort compares adjacent items and swaps them when "
                    "they are out of order. We trace several passes until no "
                    "more swaps are needed. Now compare merge sort and bubble "
                    "sort. Merge sort compares items from separate lists to "
                    "create new sorted lists. Merge sort is usually quicker "
                    "and is suitable for large data sets. Merge sort is more "
                    "difficult to program and its memory footprint can grow "
                    "while it executes. Bubble sort is slower but easier to "
                    "implement and has a known memory footprint."
                ),
            }
        ]
    )

    retained_ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }
    merged_ids = {
        topic.concept_id
        for topic in result.merged_topics
    }

    assert "aqa_3_1_4_bubble_sort" in retained_ids
    assert "aqa_3_1_4_merge_sort" in retained_ids
    assert "aqa_3_1_4_bubble_sort" in merged_ids
    assert "aqa_3_1_4_merge_sort" in merged_ids


def test_array_row_column_context_does_not_create_database_topic() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 120,
                "overlap_word_count": 0,
                "text": (
                    "A two dimensional array is visualised as a table. "
                    "One index selects the row and another index selects "
                    "the column. We access each array element using its "
                    "two indexes and trace several examples."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }
    assert "aqa_3_2_6_arrays" in ids
    assert "aqa_3_7_1_database_structure" not in ids


def test_database_row_column_context_keeps_database_topic() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 110,
                "overlap_word_count": 0,
                "text": (
                    "A relational database stores data in a database table. "
                    "Each row is a record and each column is a database "
                    "field. The primary key uniquely identifies each record."
                ),
            }
        ]
    )

    ids = {
        candidate.concept_id
        for candidate in result.chunk_results[0].topic_candidates
    }
    assert "aqa_3_7_1_database_structure" in ids


def test_continuation_preserves_existing_topic_evidence() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 120,
                "overlap_word_count": 0,
                "text": (
                    "Algorithm efficiency compares algorithms that solve "
                    "the same problem. One algorithm may be faster because "
                    "it executes fewer instructions."
                ),
            },
            {
                "chunk_id": 2,
                "word_count": 110,
                "overlap_word_count": 24,
                "text": (
                    "The same problem is solved again, but this version is "
                    "more efficient because the calculation runs once rather "
                    "than repeating inside a loop."
                ),
            },
        ]
    )

    second = result.chunk_results[1]
    assert second.classification == "continuation_no_new_topic"
    assert not second.creates_new_topic
    assert any(
        candidate.concept_id == "aqa_3_1_2_efficiency"
        for candidate in second.topic_candidates
    )
    efficiency = next(
        topic
        for topic in result.merged_topics
        if topic.concept_id == "aqa_3_1_2_efficiency"
    )
    assert 2 in efficiency.source_chunk_ids


def test_higher_dimensional_arrays_are_extended_not_fake_official_topic() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 130,
                "overlap_word_count": 0,
                "text": (
                    "GCSE covers one dimensional and two dimensional arrays. "
                    "Beyond the specification, a three dimensional array "
                    "uses three indexes and a four dimensional array uses "
                    "four indexes. These higher dimensional arrays continue "
                    "the same indexing idea."
                ),
            }
        ]
    )

    signals = result.chunk_results[0].unmapped_cs_signals
    assert any(
        signal.rough_topic == "Higher-dimensional arrays"
        for signal in signals
    )


def test_complexity_topics_are_preserved_as_extended_content() -> None:
    result = build_test_pipeline().process_chunks(
        [
            {
                "chunk_id": 1,
                "word_count": 170,
                "overlap_word_count": 0,
                "text": (
                    "Algorithm efficiency at GCSE asks which solution is "
                    "quicker. Beyond GCSE, time complexity describes how "
                    "running time grows, while space complexity describes "
                    "the amount of memory required. Big O notation classifies "
                    "constant time and linear time. Intractable problems do "
                    "not run in a practical amount of time as input grows."
                ),
            }
        ]
    )

    rough_topics = {
        signal.rough_topic
        for signal in result.chunk_results[0].unmapped_cs_signals
    }
    assert "Time complexity" in rough_topics
    assert "Space complexity" in rough_topics
    assert "Big O notation" in rough_topics
    assert "Tractable and intractable problems" in rough_topics


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
        test_binary_search_does_not_map_to_number_bases,
        test_binary_number_context_does_not_map_to_binary_search,
        test_incidental_sorting_mention_is_rejected,
        test_explained_sorting_topic_is_retained,
        test_relational_operator_phrases_need_operator_context,
        test_relational_operators_survive_explicit_teaching_context,
        test_shared_evidence_prefers_stronger_main_topic,
        test_mixed_binary_context_keeps_both_official_topics,
        test_arraylist_compound_is_not_mapped_as_official_array,
        test_real_arrays_and_arraylists_are_classified_separately,
        test_generic_code_tracing_language_maps_to_official_trace_topic,
        test_asr_style_execution_counts_map_to_tracing,
        test_overlapped_same_tracing_topic_is_continuation,
        test_unmapped_signals_return_specific_rough_topics,
        test_strong_lexical_unmapped_signal_skips_llm,
        test_semantic_only_unmapped_signal_uses_llm,
        test_same_evidence_close_unmapped_topics_use_llm,
        test_final_ranking_suppresses_keyword_dominated_supporting_topic,
        test_high_coverage_topic_is_primary_and_single_chunk_topic_supporting,
        test_generic_multi_topic_recap_does_not_create_primary_topics,
        test_substantive_topic_survives_separate_generic_recap,
        test_brief_comparison_only_binary_search_is_rejected,
        test_detailed_merge_sort_comparison_is_retained,
        test_array_row_column_context_does_not_create_database_topic,
        test_database_row_column_context_keeps_database_topic,
        test_continuation_preserves_existing_topic_evidence,
        test_higher_dimensional_arrays_are_extended_not_fake_official_topic,
        test_complexity_topics_are_preserved_as_extended_content,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("\nALL MODULE 3 GENERIC-FIX REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()