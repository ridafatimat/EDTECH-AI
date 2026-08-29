from __future__ import annotations

from copy import deepcopy

from app.services.detected_topic_edit_overlay import (
    DetectedTopicEditOverlay,
    EditMemoryCandidate,
    OverlayTopic,
    ReasonValidation,
)


SPEC = "AQA-8525-v1.2-2022-11-29"


class FakeCandidateProvider:
    def __init__(
        self,
        *,
        existing=None,
        additions=None,
    ):
        self.existing = existing or {}
        self.additions = list(additions or [])

    def candidates_for_existing_topic(
        self,
        *,
        spec_version,
        topic,
        current_evidence,
    ):
        return list(
            self.existing.get(
                topic.concept_id,
                [],
            )
        )

    def candidates_for_additions(
        self,
        *,
        spec_version,
        current_chunk_evidence,
        already_present_concept_ids,
    ):
        return list(self.additions)


class FakeReasonValidator:
    def __init__(self, decisions):
        self.decisions = dict(decisions)

    def validate(
        self,
        *,
        candidate,
        current_evidence,
    ):
        return self.decisions.get(
            candidate.memory_id,
            ReasonValidation(
                decision="uncertain",
                confidence=0.50,
                safe_for_automatic_reuse=False,
                explanation="No deterministic test decision configured.",
            ),
        )


def safe_result(
    explanation="Human rationale still applies.",
):
    return ReasonValidation(
        decision="compatible",
        confidence=0.97,
        safe_for_automatic_reuse=True,
        explanation=explanation,
    )


def unsafe_result(
    decision="incompatible",
):
    return ReasonValidation(
        decision=decision,
        confidence=0.95,
        safe_for_automatic_reuse=False,
        explanation="Human rationale does not safely apply.",
    )


def base_topics():
    return [
        OverlayTopic(
            concept_id="aqa_binary_search",
            topic="Binary search",
            role="primary",
            official_reference="3.1.3",
            confidence=0.73,
            ranking_score=0.75,
            source_chunk_ids=(1, 2, 3, 4),
        ),
        OverlayTopic(
            concept_id="aqa_sorting",
            topic="Sorting algorithms",
            role="supporting",
            official_reference="3.1.4",
            confidence=0.61,
            ranking_score=0.40,
            source_chunk_ids=(3,),
        ),
        OverlayTopic(
            concept_id="aqa_efficiency",
            topic="Time efficiency of algorithms",
            role="supporting",
            official_reference="3.1.2",
            confidence=0.60,
            ranking_score=0.37,
            source_chunk_ids=(3,),
        ),
        OverlayTopic(
            concept_id="aqa_subroutine_statement",
            topic="Subroutine statements",
            role="supporting",
            official_reference="3.2.2",
            confidence=0.66,
            ranking_score=0.46,
            source_chunk_ids=(4,),
        ),
    ]


def evidence_map():
    return {
        "aqa_binary_search": (
            "Binary search is taught through midpoint calculations and "
            "pointer updates."
        ),
        "aqa_sorting": (
            "Sorting is only mentioned because binary search requires "
            "ordered data."
        ),
        "aqa_efficiency": (
            "The lesson repeatedly compares execution efficiency and "
            "operation counts."
        ),
        "aqa_subroutine_statement": (
            "The lesson explains functions, procedures, parameters and "
            "return values."
        ),
    }


def main():
    # ------------------------------------------------------------------
    # TEST 1: All four actions apply safely to a COPY.
    # ------------------------------------------------------------------
    original = base_topics()
    original_snapshot = deepcopy(original)

    remove_sorting = EditMemoryCandidate(
        memory_id=1,
        spec_version=SPEC,
        edit_action="remove_topic",
        source_concept_id="aqa_sorting",
        source_topic="Sorting algorithms",
        source_role="supporting",
        reviewer_reason="Sorting is only incidental.",
        stored_evidence="Old incidental sorting context.",
    )

    change_efficiency_role = EditMemoryCandidate(
        memory_id=2,
        spec_version=SPEC,
        edit_action="change_role",
        source_concept_id="aqa_efficiency",
        source_topic="Time efficiency of algorithms",
        source_role="supporting",
        target_concept_id="aqa_efficiency",
        target_topic="Time efficiency of algorithms",
        target_role="primary",
        reviewer_reason="Efficiency is the central lesson focus.",
        stored_evidence="Old efficiency-focused context.",
    )

    replace_subroutine = EditMemoryCandidate(
        memory_id=3,
        spec_version=SPEC,
        edit_action="replace_topic",
        source_concept_id="aqa_subroutine_statement",
        source_topic="Subroutine statements",
        source_role="supporting",
        target_concept_id="aqa_subroutines",
        target_topic="Subroutines, procedures and functions",
        target_role="supporting",
        target_official_reference="3.2.10",
        reviewer_reason="The generic mapping is too broad.",
        stored_evidence="Old functions/procedures teaching context.",
    )

    add_boolean = EditMemoryCandidate(
        memory_id=4,
        spec_version=SPEC,
        edit_action="add_topic",
        target_concept_id="aqa_boolean",
        target_topic="Boolean operations",
        target_role="supporting",
        target_official_reference="3.2.5",
        reviewer_reason="Boolean operators were explicitly taught but missed.",
        stored_evidence="Old Boolean teaching context.",
        current_source_chunk_ids=(5,),
    )

    provider = FakeCandidateProvider(
        existing={
            "aqa_sorting": [remove_sorting],
            "aqa_efficiency": [change_efficiency_role],
            "aqa_subroutine_statement": [replace_subroutine],
        },
        additions=[add_boolean],
    )

    validator = FakeReasonValidator(
        {
            1: safe_result(),
            2: safe_result(),
            3: safe_result(),
            4: safe_result(),
        }
    )

    overlay = DetectedTopicEditOverlay(
        candidate_provider=provider,
        reason_validator=validator,
    )

    result = overlay.apply(
        topics=original,
        spec_version=SPEC,
        evidence_by_concept_id=evidence_map(),
        current_chunk_evidence=[
            "AND OR and NOT are explicitly taught using truth values."
        ],
    )

    # Original list and objects must be unchanged.
    assert original == original_snapshot

    result_by_id = {
        topic.concept_id: topic
        for topic in result.topics
    }

    assert "aqa_sorting" not in result_by_id

    assert result_by_id["aqa_efficiency"].role == "primary"
    assert result_by_id["aqa_efficiency"].memory_applied is True
    assert result_by_id["aqa_efficiency"].memory_id == 2

    assert "aqa_subroutine_statement" not in result_by_id
    assert "aqa_subroutines" in result_by_id
    assert (
        result_by_id["aqa_subroutines"].official_reference
        == "3.2.10"
    )

    assert "aqa_boolean" in result_by_id
    assert result_by_id["aqa_boolean"].memory_action == "add_topic"
    assert result_by_id["aqa_boolean"].source_chunk_ids == (5,)

    assert len(result.applied) == 4

    # ------------------------------------------------------------------
    # TEST 2: Incompatible / uncertain -> original topic untouched.
    # ------------------------------------------------------------------
    original = base_topics()
    original_snapshot = deepcopy(original)

    provider = FakeCandidateProvider(
        existing={
            "aqa_sorting": [remove_sorting],
            "aqa_efficiency": [change_efficiency_role],
        }
    )

    validator = FakeReasonValidator(
        {
            1: unsafe_result("incompatible"),
            2: unsafe_result("uncertain"),
        }
    )

    result = DetectedTopicEditOverlay(
        candidate_provider=provider,
        reason_validator=validator,
    ).apply(
        topics=original,
        spec_version=SPEC,
        evidence_by_concept_id=evidence_map(),
    )

    assert original == original_snapshot
    assert list(result.topics) == original_snapshot
    assert len(result.applied) == 0

    # ------------------------------------------------------------------
    # TEST 3: Wrong spec -> no edit, even if validator would allow it.
    # ------------------------------------------------------------------
    wrong_spec_candidate = EditMemoryCandidate(
        memory_id=10,
        spec_version="OLD-SPEC",
        edit_action="remove_topic",
        source_concept_id="aqa_sorting",
        source_topic="Sorting algorithms",
        source_role="supporting",
        reviewer_reason="Old-spec reason.",
        stored_evidence="Old evidence.",
    )

    original = base_topics()
    original_snapshot = deepcopy(original)

    result = DetectedTopicEditOverlay(
        candidate_provider=FakeCandidateProvider(
            existing={
                "aqa_sorting": [wrong_spec_candidate],
            }
        ),
        reason_validator=FakeReasonValidator(
            {
                10: safe_result(),
            }
        ),
    ).apply(
        topics=original,
        spec_version=SPEC,
        evidence_by_concept_id=evidence_map(),
    )

    assert original == original_snapshot
    assert list(result.topics) == original_snapshot
    assert len(result.applied) == 0

    # ------------------------------------------------------------------
    # TEST 4: Multiple candidates -> conflict -> no edit.
    # ------------------------------------------------------------------
    conflicting_replace = EditMemoryCandidate(
        memory_id=11,
        spec_version=SPEC,
        edit_action="replace_topic",
        source_concept_id="aqa_sorting",
        source_topic="Sorting algorithms",
        source_role="supporting",
        target_concept_id="aqa_searching",
        target_topic="Searching algorithms",
        target_role="supporting",
        reviewer_reason="Different human correction.",
        stored_evidence="Different reviewed evidence.",
    )

    original = base_topics()
    original_snapshot = deepcopy(original)

    result = DetectedTopicEditOverlay(
        candidate_provider=FakeCandidateProvider(
            existing={
                "aqa_sorting": [
                    remove_sorting,
                    conflicting_replace,
                ],
            }
        ),
        reason_validator=FakeReasonValidator(
            {
                1: safe_result(),
                11: safe_result(),
            }
        ),
    ).apply(
        topics=original,
        spec_version=SPEC,
        evidence_by_concept_id=evidence_map(),
    )

    assert original == original_snapshot
    assert list(result.topics) == original_snapshot
    assert len(result.applied) == 0

    # ------------------------------------------------------------------
    # TEST 5: Duplicate add target -> fail closed.
    # ------------------------------------------------------------------
    duplicate_add = EditMemoryCandidate(
        memory_id=12,
        spec_version=SPEC,
        edit_action="add_topic",
        target_concept_id="aqa_boolean",
        target_topic="Boolean operations",
        target_role="primary",
        target_official_reference="3.2.5",
        reviewer_reason="Conflicting add memory.",
        stored_evidence="Different Boolean context.",
        current_source_chunk_ids=(6,),
    )

    original = base_topics()
    original_snapshot = deepcopy(original)

    result = DetectedTopicEditOverlay(
        candidate_provider=FakeCandidateProvider(
            additions=[
                add_boolean,
                duplicate_add,
            ]
        ),
        reason_validator=FakeReasonValidator(
            {
                4: safe_result(),
                12: safe_result(),
            }
        ),
    ).apply(
        topics=original,
        spec_version=SPEC,
        evidence_by_concept_id=evidence_map(),
        current_chunk_evidence=[
            "Boolean operators are taught here."
        ],
    )

    assert original == original_snapshot
    assert "aqa_boolean" not in {
        topic.concept_id
        for topic in result.topics
    }
    assert len(result.applied) == 0

    # ------------------------------------------------------------------
    # TEST 6: Existing topic cannot be added again.
    # ------------------------------------------------------------------
    existing_boolean = OverlayTopic(
        concept_id="aqa_boolean",
        topic="Boolean operations",
        role="supporting",
        official_reference="3.2.5",
        confidence=0.70,
        ranking_score=0.50,
        source_chunk_ids=(5,),
    )

    original = [
        *base_topics(),
        existing_boolean,
    ]
    original_snapshot = deepcopy(original)

    result = DetectedTopicEditOverlay(
        candidate_provider=FakeCandidateProvider(
            additions=[add_boolean]
        ),
        reason_validator=FakeReasonValidator(
            {
                4: safe_result(),
            }
        ),
    ).apply(
        topics=original,
        spec_version=SPEC,
        evidence_by_concept_id={
            **evidence_map(),
            "aqa_boolean": "Boolean operators are already detected.",
        },
        current_chunk_evidence=[
            "Boolean operators are explicitly taught."
        ],
    )

    assert original == original_snapshot
    boolean_count = sum(
        topic.concept_id == "aqa_boolean"
        for topic in result.topics
    )
    assert boolean_count == 1

    # ------------------------------------------------------------------
    # TEST 7: Missing evidence -> abstain.
    # ------------------------------------------------------------------
    original = base_topics()
    original_snapshot = deepcopy(original)

    result = DetectedTopicEditOverlay(
        candidate_provider=FakeCandidateProvider(
            existing={
                "aqa_sorting": [remove_sorting],
            }
        ),
        reason_validator=FakeReasonValidator(
            {
                1: safe_result(),
            }
        ),
    ).apply(
        topics=original,
        spec_version=SPEC,
        evidence_by_concept_id={},
    )

    assert original == original_snapshot
    assert list(result.topics) == original_snapshot
    assert len(result.applied) == 0

    print("Controlled in-memory detected-topic edit overlay tests passed")


if __name__ == "__main__":
    main()
