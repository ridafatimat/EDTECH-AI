from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from types import SimpleNamespace

from dotenv import load_dotenv

from app.services.detected_topic_edit_embedding_adapter import (
    Agent1EditMemoryEmbeddingAdapter,
)
from app.services.detected_topic_edit_memory_matcher import (
    DetectedTopicEditMemoryMatcher,
    EditMemoryMatchConfig,
)


SPEC_VERSION = "AQA-8525-v1.2-2022-11-29"


def normalize(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value).strip().casefold(),
    )


def evidence_hash(value: str) -> str:
    return hashlib.sha256(
        normalize(value).encode("utf-8")
    ).hexdigest()


def cosine_similarity(left, right) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match.")

    dot = sum(
        float(a) * float(b)
        for a, b in zip(left, right)
    )
    left_norm = math.sqrt(
        sum(float(value) ** 2 for value in left)
    )
    right_norm = math.sqrt(
        sum(float(value) ** 2 for value in right)
    )

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (left_norm * right_norm)


class FakeRepository:
    """
    Read-only in-memory records.

    Step 4 deliberately does not write to the real PostgreSQL table.
    """

    def __init__(self, records):
        self.records = list(records)

    def list_reusable(
        self,
        *,
        spec_version: str,
        source_concept_id: str | None = None,
        edit_actions=None,
        limit: int = 100,
    ):
        results = []

        for record in self.records:
            if record.spec_version != spec_version:
                continue
            if not record.reviewer_approved:
                continue
            if not record.is_active:
                continue
            if record.validation_status != "human_validated":
                continue
            if (
                source_concept_id is not None
                and record.source_concept_id != source_concept_id
            ):
                continue
            if (
                edit_actions is not None
                and record.edit_action not in set(edit_actions)
            ):
                continue

            results.append(record)

        return results[:limit]

    def list_reusable_additions(
        self,
        *,
        spec_version: str,
        limit: int = 100,
    ):
        return self.list_reusable(
            spec_version=spec_version,
            edit_actions=("add_topic",),
            limit=limit,
        )


def make_memory(
    *,
    memory_id: int,
    edit_action: str,
    evidence_text: str,
    source_concept_id=None,
    source_topic=None,
    source_role=None,
    target_concept_id=None,
    target_topic=None,
    target_role=None,
    reviewer_reason="Reviewer validated this edit.",
):
    return SimpleNamespace(
        id=memory_id,
        edit_action=edit_action,
        source_concept_id=source_concept_id,
        source_topic=source_topic,
        source_role=source_role,
        target_concept_id=target_concept_id,
        target_topic=target_topic,
        target_role=target_role,
        evidence_hash=evidence_hash(evidence_text),
        evidence_text=evidence_text,
        reviewer_reason=reviewer_reason,
        spec_version=SPEC_VERSION,
        reviewer_approved=True,
        is_active=True,
        validation_status="human_validated",
    )


@dataclass(frozen=True)
class PairDiagnostic:
    name: str
    stored_evidence: str
    new_evidence: str
    threshold: float
    expected: str  # "hit" or "miss"


def main() -> None:
    # Load Agent_1/.env when this script is run from the project root.
    load_dotenv()

    config = EditMemoryMatchConfig()

    adapter = Agent1EditMemoryEmbeddingAdapter()

    print("=" * 100)
    print("STEP 4 — REAL EMBEDDING DIAGNOSTICS")
    print("=" * 100)
    print()
    print("Embedding model:", adapter.model_name)
    print()
    print("Matcher thresholds (unchanged from Step 3):")
    print(
        "  remove / replace :",
        config.standard_similarity_threshold,
    )
    print(
        "  change role      :",
        config.role_change_similarity_threshold,
    )
    print(
        "  add topic        :",
        config.add_topic_similarity_threshold,
    )
    print(
        "  ambiguity margin :",
        config.ambiguity_margin,
    )
    print()

    sorting_incidental = (
        "Binary search requires ordered data. If the list is unsorted, "
        "some form of sorting would be needed before the search can begin."
    )
    sorting_paraphrase = (
        "A binary search only works on ordered items, so an unordered list "
        "would need to be sorted first before binary search is used."
    )
    bubble_sort_teaching = (
        "Bubble sort compares adjacent values and swaps them when they are "
        "in the wrong order. Repeated passes continue until no swaps remain."
    )

    arithmetic_incidental = (
        "The midpoint is calculated using left plus right and integer "
        "division by two, then the pointers are adjusted by one."
    )
    arithmetic_paraphrase = (
        "To perform binary search, add the left and right pointer positions, "
        "use integer division by two for the midpoint, and move a pointer."
    )
    arithmetic_teaching = (
        "This lesson teaches arithmetic operators including addition, "
        "subtraction, multiplication, division, integer division and modulo."
    )

    efficiency_focus = (
        "The lesson repeatedly compares two algorithms solving the same "
        "problem and explains why one needs fewer operations and is faster."
    )
    efficiency_paraphrase = (
        "Most of the lesson compares algorithms for the same task and "
        "explains their relative execution time and number of operations."
    )
    incidental_efficiency = (
        "The teacher briefly notes that binary search is faster than linear "
        "search before returning to the mechanics of the search algorithm."
    )

    boolean_teaching = (
        "The lesson explicitly teaches Boolean AND OR and NOT operators, "
        "including truth conditions and worked programming examples."
    )
    boolean_paraphrase = (
        "Students are taught how AND OR and NOT work with truth values and "
        "apply the Boolean operators in several code examples."
    )
    boolean_incidental = (
        "A found flag is set to false and checked in a while loop while the "
        "main lesson explains binary search pointer updates."
    )

    diagnostics = [
        PairDiagnostic(
            name="Sorting incidental paraphrase",
            stored_evidence=sorting_incidental,
            new_evidence=sorting_paraphrase,
            threshold=config.standard_similarity_threshold,
            expected="hit",
        ),
        PairDiagnostic(
            name="Sorting actual teaching negative",
            stored_evidence=sorting_incidental,
            new_evidence=bubble_sort_teaching,
            threshold=config.standard_similarity_threshold,
            expected="miss",
        ),
        PairDiagnostic(
            name="Arithmetic incidental paraphrase",
            stored_evidence=arithmetic_incidental,
            new_evidence=arithmetic_paraphrase,
            threshold=config.standard_similarity_threshold,
            expected="hit",
        ),
        PairDiagnostic(
            name="Arithmetic actual teaching negative",
            stored_evidence=arithmetic_incidental,
            new_evidence=arithmetic_teaching,
            threshold=config.standard_similarity_threshold,
            expected="miss",
        ),
        PairDiagnostic(
            name="Role-change similar lesson focus",
            stored_evidence=efficiency_focus,
            new_evidence=efficiency_paraphrase,
            threshold=config.role_change_similarity_threshold,
            expected="hit",
        ),
        PairDiagnostic(
            name="Role-change incidental mention negative",
            stored_evidence=efficiency_focus,
            new_evidence=incidental_efficiency,
            threshold=config.role_change_similarity_threshold,
            expected="miss",
        ),
        PairDiagnostic(
            name="Add-topic strong Boolean paraphrase",
            stored_evidence=boolean_teaching,
            new_evidence=boolean_paraphrase,
            threshold=config.add_topic_similarity_threshold,
            expected="hit",
        ),
        PairDiagnostic(
            name="Add-topic incidental Boolean negative",
            stored_evidence=boolean_teaching,
            new_evidence=boolean_incidental,
            threshold=config.add_topic_similarity_threshold,
            expected="miss",
        ),
    ]

    all_texts = []
    for diagnostic in diagnostics:
        all_texts.extend(
            [
                diagnostic.stored_evidence,
                diagnostic.new_evidence,
            ]
        )

    vectors = adapter.embed_texts(all_texts)

    print("-" * 100)
    print("RAW REAL-MODEL SIMILARITIES")
    print("-" * 100)

    pair_results = []
    vector_index = 0

    for diagnostic in diagnostics:
        stored_vector = vectors[vector_index]
        new_vector = vectors[vector_index + 1]
        vector_index += 2

        similarity = cosine_similarity(
            stored_vector,
            new_vector,
        )

        observed = (
            "hit"
            if similarity >= diagnostic.threshold
            else "miss"
        )
        passed = observed == diagnostic.expected

        pair_results.append(
            (
                diagnostic,
                similarity,
                observed,
                passed,
            )
        )

        print()
        print(diagnostic.name)
        print("  expected  :", diagnostic.expected)
        print("  threshold :", f"{diagnostic.threshold:.4f}")
        print("  similarity:", f"{similarity:.4f}")
        print("  observed  :", observed)
        print("  result    :", "PASS" if passed else "REVIEW")

    # ------------------------------------------------------------------
    # Real matcher smoke test using the same adapter.
    # ------------------------------------------------------------------

    sorting_memory = make_memory(
        memory_id=1,
        edit_action="remove_topic",
        source_concept_id="aqa_sorting",
        source_topic="Sorting algorithms",
        source_role="supporting",
        evidence_text=sorting_incidental,
        reviewer_reason=(
            "Sorting is only an incidental prerequisite and is not taught."
        ),
    )

    arithmetic_memory = make_memory(
        memory_id=2,
        edit_action="remove_topic",
        source_concept_id="aqa_arithmetic",
        source_topic="Arithmetic operations",
        source_role="supporting",
        evidence_text=arithmetic_incidental,
        reviewer_reason=(
            "Arithmetic is used inside binary search but is not taught."
        ),
    )

    role_memory = make_memory(
        memory_id=3,
        edit_action="change_role",
        source_concept_id="aqa_efficiency",
        source_topic="Time efficiency of algorithms",
        source_role="supporting",
        target_concept_id="aqa_efficiency",
        target_topic="Time efficiency of algorithms",
        target_role="primary",
        evidence_text=efficiency_focus,
        reviewer_reason=(
            "Time efficiency is the lesson's central teaching objective."
        ),
    )

    boolean_memory = make_memory(
        memory_id=4,
        edit_action="add_topic",
        target_concept_id="aqa_boolean",
        target_topic="Boolean operations",
        target_role="supporting",
        evidence_text=boolean_teaching,
        reviewer_reason=(
            "Boolean operations are explicitly taught but were missed."
        ),
    )

    matcher = DetectedTopicEditMemoryMatcher(
        repository=FakeRepository(
            [
                sorting_memory,
                arithmetic_memory,
                role_memory,
                boolean_memory,
            ]
        ),
        embedder=adapter,
        config=config,
    )

    print()
    print("-" * 100)
    print("REAL MATCHER SMOKE TEST")
    print("-" * 100)

    smoke_results = []

    cases = [
        (
            "sorting-positive",
            matcher.match_existing_topic_edit(
                spec_version=SPEC_VERSION,
                source_concept_id="aqa_sorting",
                current_evidence_text=sorting_paraphrase,
            ),
            "hit",
        ),
        (
            "sorting-negative",
            matcher.match_existing_topic_edit(
                spec_version=SPEC_VERSION,
                source_concept_id="aqa_sorting",
                current_evidence_text=bubble_sort_teaching,
            ),
            "miss",
        ),
        (
            "arithmetic-positive",
            matcher.match_existing_topic_edit(
                spec_version=SPEC_VERSION,
                source_concept_id="aqa_arithmetic",
                current_evidence_text=arithmetic_paraphrase,
            ),
            "hit",
        ),
        (
            "arithmetic-negative",
            matcher.match_existing_topic_edit(
                spec_version=SPEC_VERSION,
                source_concept_id="aqa_arithmetic",
                current_evidence_text=arithmetic_teaching,
            ),
            "miss",
        ),
        (
            "role-positive",
            matcher.match_existing_topic_edit(
                spec_version=SPEC_VERSION,
                source_concept_id="aqa_efficiency",
                current_evidence_text=efficiency_paraphrase,
            ),
            "hit",
        ),
        (
            "role-negative",
            matcher.match_existing_topic_edit(
                spec_version=SPEC_VERSION,
                source_concept_id="aqa_efficiency",
                current_evidence_text=incidental_efficiency,
            ),
            "miss",
        ),
    ]

    additions_positive = matcher.match_add_topic_memories(
        spec_version=SPEC_VERSION,
        current_chunk_evidence=[boolean_paraphrase],
        already_present_concept_ids=[],
    )
    additions_negative = matcher.match_add_topic_memories(
        spec_version=SPEC_VERSION,
        current_chunk_evidence=[boolean_incidental],
        already_present_concept_ids=[],
    )

    for name, result, expected in cases:
        observed = result.status
        passed = observed == expected
        smoke_results.append(passed)

        print()
        print(name)
        print("  expected:", expected)
        print("  observed:", observed)
        print(
            "  similarity:",
            (
                f"{result.best_similarity:.4f}"
                if result.best_similarity is not None
                else "None"
            ),
        )
        print("  result:", "PASS" if passed else "REVIEW")

    add_positive_pass = len(additions_positive) == 1
    add_negative_pass = len(additions_negative) == 0

    smoke_results.extend(
        [
            add_positive_pass,
            add_negative_pass,
        ]
    )

    print()
    print("add-topic-positive")
    print("  expected: one safe candidate")
    print("  observed:", len(additions_positive))
    print(
        "  result:",
        "PASS" if add_positive_pass else "REVIEW",
    )

    print()
    print("add-topic-negative")
    print("  expected: zero candidates")
    print("  observed:", len(additions_negative))
    print(
        "  result:",
        "PASS" if add_negative_pass else "REVIEW",
    )

    # ------------------------------------------------------------------
    # Final decision: diagnostics only. NO AUTOMATIC TUNING.
    # ------------------------------------------------------------------

    pair_pass = all(
        passed
        for _, _, _, passed in pair_results
    )
    smoke_pass = all(smoke_results)

    print()
    print("=" * 100)
    print("STEP 4 DIAGNOSTIC SUMMARY")
    print("=" * 100)
    print("Raw similarity expectations:", "PASS" if pair_pass else "REVIEW REQUIRED")
    print("Matcher smoke expectations :", "PASS" if smoke_pass else "REVIEW REQUIRED")
    print()
    print("No thresholds were changed by this script.")
    print("No database rows were written.")
    print("No Module 3 / Streamlit / Qdrant / Groq code was modified.")
    print()

    if pair_pass and smoke_pass:
        print(
            "READY_FOR_STEP5 = YES — current conservative thresholds "
            "separate these diagnostic positive/negative cases."
        )
    else:
        print(
            "READY_FOR_STEP5 = NO — inspect the real similarity scores "
            "before any wiring. Do NOT lower thresholds automatically."
        )


if __name__ == "__main__":
    main()
