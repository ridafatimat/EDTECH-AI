from __future__ import annotations

import math
import re
from types import SimpleNamespace

from app.services.detected_topic_edit_memory_matcher import (
    DetectedTopicEditMemoryMatcher,
    EditMemoryMatchConfig,
)


def normalize(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        value.strip().casefold(),
    )


class FakeRepository:
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


class MappingEmbedder:
    """
    Deterministic test embedder.

    The production matcher contains no topic-specific logic. These vectors
    only let the tests simulate clearly similar and clearly different
    semantic contexts without depending on an external model.
    """

    def __init__(self, mapping):
        self.mapping = {
            normalize(key): tuple(value)
            for key, value in mapping.items()
        }

    def embed_texts(self, texts):
        vectors = []

        for text in texts:
            key = normalize(text)

            if key not in self.mapping:
                raise KeyError(
                    "No deterministic test vector configured for: "
                    f"{text!r}"
                )

            vectors.append(self.mapping[key])

        return vectors


def unit_vector(angle_degrees: float):
    angle = math.radians(angle_degrees)
    return (
        math.cos(angle),
        math.sin(angle),
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
    spec_version="AQA-8525-v1.2-2022-11-29",
    reviewer_reason="Reviewer validated this contextual edit.",
    reviewer_approved=True,
    is_active=True,
    validation_status="human_validated",
):
    import hashlib

    evidence_hash = hashlib.sha256(
        normalize(evidence_text).encode("utf-8")
    ).hexdigest()

    return SimpleNamespace(
        id=memory_id,
        edit_action=edit_action,
        source_concept_id=source_concept_id,
        source_topic=source_topic,
        source_role=source_role,
        target_concept_id=target_concept_id,
        target_topic=target_topic,
        target_role=target_role,
        evidence_hash=evidence_hash,
        evidence_text=evidence_text,
        reviewer_reason=reviewer_reason,
        spec_version=spec_version,
        reviewer_approved=reviewer_approved,
        is_active=is_active,
        validation_status=validation_status,
    )


def main():
    spec = "AQA-8525-v1.2-2022-11-29"

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

    efficiency_role_memory = (
        "The lesson repeatedly compares two algorithms solving the same "
        "problem and explains why one needs fewer operations and is faster."
    )

    efficiency_role_paraphrase = (
        "Most of the lesson compares algorithms for the same task and "
        "explains their relative execution time and number of operations."
    )

    add_boolean_memory = (
        "The lesson explicitly teaches Boolean AND OR and NOT operators, "
        "including truth conditions and worked programming examples."
    )

    add_boolean_paraphrase = (
        "Students are taught how AND OR and NOT work with truth values and "
        "apply the Boolean operators in several code examples."
    )

    records = [
        make_memory(
            memory_id=1,
            edit_action="remove_topic",
            source_concept_id="aqa_sorting",
            source_topic="Sorting algorithms",
            source_role="supporting",
            evidence_text=sorting_incidental,
            reviewer_reason=(
                "Sorting is only an incidental prerequisite; no sorting "
                "algorithm is taught."
            ),
        ),
        make_memory(
            memory_id=2,
            edit_action="remove_topic",
            source_concept_id="aqa_arithmetic",
            source_topic="Arithmetic operations",
            source_role="supporting",
            evidence_text=arithmetic_incidental,
            reviewer_reason=(
                "Arithmetic is only used inside the search algorithm and "
                "is not independently taught."
            ),
        ),
        make_memory(
            memory_id=3,
            edit_action="change_role",
            source_concept_id="aqa_efficiency",
            source_topic="Time efficiency of algorithms",
            source_role="supporting",
            target_concept_id="aqa_efficiency",
            target_topic="Time efficiency of algorithms",
            target_role="primary",
            evidence_text=efficiency_role_memory,
            reviewer_reason=(
                "Time efficiency is the central teaching objective."
            ),
        ),
        make_memory(
            memory_id=4,
            edit_action="add_topic",
            target_concept_id="aqa_boolean",
            target_topic="Boolean operations",
            target_role="supporting",
            evidence_text=add_boolean_memory,
            reviewer_reason=(
                "Boolean operators are explicitly taught but were missed."
            ),
        ),
    ]

    vectors = {
        # Sorting incidental memory and paraphrase: very strong match.
        sorting_incidental: unit_vector(0.0),
        sorting_paraphrase: unit_vector(10.0),

        # Real Bubble Sort teaching is deliberately far from the incidental
        # binary-search prerequisite context.
        bubble_sort_teaching: unit_vector(70.0),

        # Arithmetic incidental memory and paraphrase: strong match.
        arithmetic_incidental: unit_vector(0.0),
        arithmetic_paraphrase: unit_vector(8.0),

        # Actual arithmetic lesson is different enough to miss.
        arithmetic_teaching: unit_vector(65.0),

        # Role correction context.
        efficiency_role_memory: unit_vector(0.0),
        efficiency_role_paraphrase: unit_vector(9.0),

        # Human-added Boolean topic context.
        add_boolean_memory: unit_vector(0.0),
        add_boolean_paraphrase: unit_vector(8.0),
    }

    matcher = DetectedTopicEditMemoryMatcher(
        repository=FakeRepository(records),
        embedder=MappingEmbedder(vectors),
    )

    # 1. Similar incidental sorting context may safely reuse removal.
    result = matcher.match_existing_topic_edit(
        spec_version=spec,
        source_concept_id="aqa_sorting",
        current_evidence_text=sorting_paraphrase,
    )
    assert result.status == "hit"
    assert result.match is not None
    assert result.match.memory_id == 1
    assert result.match.edit_action == "remove_topic"

    # 2. Actual Bubble Sort teaching must NOT reuse the removal.
    result = matcher.match_existing_topic_edit(
        spec_version=spec,
        source_concept_id="aqa_sorting",
        current_evidence_text=bubble_sort_teaching,
    )
    assert result.status == "miss"
    assert result.match is None

    # 3. Incidental midpoint arithmetic may reuse the prior removal.
    result = matcher.match_existing_topic_edit(
        spec_version=spec,
        source_concept_id="aqa_arithmetic",
        current_evidence_text=arithmetic_paraphrase,
    )
    assert result.status == "hit"
    assert result.match is not None
    assert result.match.memory_id == 2

    # 4. A genuine arithmetic lesson must NOT reuse incidental removal.
    result = matcher.match_existing_topic_edit(
        spec_version=spec,
        source_concept_id="aqa_arithmetic",
        current_evidence_text=arithmetic_teaching,
    )
    assert result.status == "miss"

    # 5. Similar emphasis can reuse a human role correction.
    result = matcher.match_existing_topic_edit(
        spec_version=spec,
        source_concept_id="aqa_efficiency",
        current_evidence_text=efficiency_role_paraphrase,
    )
    assert result.status == "hit"
    assert result.match is not None
    assert result.match.edit_action == "change_role"
    assert result.match.target_role == "primary"

    # 6. Different specification version must never reuse.
    result = matcher.match_existing_topic_edit(
        spec_version="FUTURE-SPEC",
        source_concept_id="aqa_sorting",
        current_evidence_text=sorting_paraphrase,
    )
    assert result.status == "miss"

    # 7. Human-added topics can be recovered only on a very strong chunk match.
    additions = matcher.match_add_topic_memories(
        spec_version=spec,
        current_chunk_evidence=[add_boolean_paraphrase],
        already_present_concept_ids=[],
    )
    assert len(additions) == 1
    assert additions[0].target_concept_id == "aqa_boolean"

    # Already present concepts must never be added twice.
    additions = matcher.match_add_topic_memories(
        spec_version=spec,
        current_chunk_evidence=[add_boolean_paraphrase],
        already_present_concept_ids=["aqa_boolean"],
    )
    assert additions == []

    # 8. Exact evidence is always recognized without semantic weakening.
    result = matcher.match_existing_topic_edit(
        spec_version=spec,
        source_concept_id="aqa_sorting",
        current_evidence_text=sorting_incidental,
    )
    assert result.status == "hit"
    assert result.match is not None
    assert result.match.match_type == "exact_evidence"

    # 9. Too-short evidence must abstain.
    result = matcher.match_existing_topic_edit(
        spec_version=spec,
        source_concept_id="aqa_sorting",
        current_evidence_text="sorting mentioned",
    )
    assert result.status == "miss"

    # 10. Conflicting strong memories must abstain rather than choose.
    conflicting_record = make_memory(
        memory_id=5,
        edit_action="replace_topic",
        source_concept_id="aqa_sorting",
        source_topic="Sorting algorithms",
        source_role="supporting",
        target_concept_id="aqa_search",
        target_topic="Searching algorithms",
        target_role="supporting",
        evidence_text=(
            "Binary search needs ordered data and a sorting method might "
            "be required first when the input list is unordered."
        ),
        reviewer_reason="Reviewer chose a different correction.",
    )

    conflicting_text = conflicting_record.evidence_text

    conflict_vectors = dict(vectors)
    conflict_vectors[conflicting_text] = unit_vector(11.0)

    conflict_matcher = DetectedTopicEditMemoryMatcher(
        repository=FakeRepository(
            [records[0], conflicting_record]
        ),
        embedder=MappingEmbedder(conflict_vectors),
        config=EditMemoryMatchConfig(
            standard_similarity_threshold=0.90,
            role_change_similarity_threshold=0.92,
            add_topic_similarity_threshold=0.94,
            ambiguity_margin=0.03,
        ),
    )

    result = conflict_matcher.match_existing_topic_edit(
        spec_version=spec,
        source_concept_id="aqa_sorting",
        current_evidence_text=sorting_paraphrase,
    )
    assert result.status == "ambiguous"
    assert result.match is None

    print("Detected-topic contextual edit-memory matcher tests passed")


if __name__ == "__main__":
    main()
