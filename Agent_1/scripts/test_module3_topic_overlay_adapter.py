from __future__ import annotations

from copy import deepcopy

from app.schemas.topic import (
    MergedTopic,
    Module3Result,
)
from app.services.detected_topic_edit_overlay import (
    DetectedTopicEditOverlay,
    EditMemoryCandidate,
    ReasonValidation,
)
from app.services.module3_topic_overlay_adapter import (
    ActualModule3TopicOverlayAdapter,
    AddedTopicMaterialization,
    OfficialConceptMetadata,
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
        self.additions = list(
            additions or []
        )

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
    def __init__(self, safe_memory_ids):
        self.safe_memory_ids = set(
            safe_memory_ids
        )

    def validate(
        self,
        *,
        candidate,
        current_evidence,
    ):
        safe = (
            candidate.memory_id
            in self.safe_memory_ids
        )

        return ReasonValidation(
            decision=(
                "compatible"
                if safe
                else "incompatible"
            ),
            confidence=0.97,
            safe_for_automatic_reuse=safe,
            explanation=(
                "Deterministic Step 4.9 regression decision."
            ),
        )


def make_topic(
    *,
    concept_id: str,
    topic: str,
    official_reference: str,
    topic_role: str,
    ranking_score: float,
    source_chunk_ids,
    evidence,
):
    chapter_reference = ".".join(
        official_reference.split(".")[:2]
    )

    return MergedTopic(
        concept_id=concept_id,
        topic=topic,
        domain="Programming and algorithms",
        official_reference=official_reference,
        chapter_reference=chapter_reference,
        official_title=topic,
        paper="Paper 2",
        source_pages=[10, 11],
        confidence=0.72,
        ranking_score=ranking_score,
        topic_role=topic_role,
        source_chunk_ids=list(
            source_chunk_ids
        ),
        support_span_count=1,
        mean_semantic_score=0.71,
        mean_keyword_score=0.48,
        mean_salience_score=0.64,
        coverage_score=0.50,
        evidence=list(evidence),
        supporting_candidate_count=max(
            1,
            len(source_chunk_ids),
        ),
    )


def original_topics():
    return [
        make_topic(
            concept_id="aqa_binary_search",
            topic="Binary search",
            official_reference="3.1.3",
            topic_role="primary",
            ranking_score=0.7553,
            source_chunk_ids=(1, 2, 3, 4),
            evidence=(
                "Binary search is explained through midpoint updates.",
            ),
        ),
        make_topic(
            concept_id="aqa_sorting",
            topic="Sorting algorithms",
            official_reference="3.1.4",
            topic_role="supporting",
            ranking_score=0.4019,
            source_chunk_ids=(3,),
            evidence=(
                "Sorting is mentioned before binary search.",
            ),
        ),
        make_topic(
            concept_id="aqa_efficiency",
            topic="Time efficiency of algorithms",
            official_reference="3.1.2",
            topic_role="supporting",
            ranking_score=0.3669,
            source_chunk_ids=(3,),
            evidence=(
                "Algorithm efficiency is compared repeatedly.",
            ),
        ),
        make_topic(
            concept_id="aqa_subroutine_statement",
            topic="Subroutine statements",
            official_reference="3.2.2",
            topic_role="supporting",
            ranking_score=0.4647,
            source_chunk_ids=(4,),
            evidence=(
                "Functions and procedures are taught.",
            ),
        ),
    ]


def make_module3_result():
    # Step 4.9 tests the REAL Module3Result shape. Chunk results can remain
    # empty because the adapter must preserve them exactly rather than change
    # chunk-level processing.
    return Module3Result(
        chunk_results=[],
        merged_topics=original_topics(),
        total_chunks=8,
        cs_relevant_chunks=8,
        non_cs_chunks=0,
        official_topic_chunks=8,
        mixed_official_unmapped_chunks=0,
        unmapped_cs_chunks=0,
        continuation_chunks=0,
        no_topic_chunks=0,
        llm_fallback_chunk_ids=[],
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        candidate_keep_threshold=0.50,
    )


def evidence_map():
    return {
        "aqa_binary_search": (
            "Binary search is taught through midpoint calculations."
        ),
        "aqa_sorting": (
            "Sorting is only mentioned because binary search needs "
            "ordered data."
        ),
        "aqa_efficiency": (
            "Efficiency is repeatedly compared as the lesson focus."
        ),
        "aqa_subroutine_statement": (
            "Functions, procedures, parameters and return values are taught."
        ),
    }


def assert_same_except(
    before: MergedTopic,
    after: MergedTopic,
    *,
    allowed_changes: set[str],
):
    before_dump = before.model_dump(
        mode="python"
    )
    after_dump = after.model_dump(
        mode="python"
    )

    for field_name, before_value in before_dump.items():
        if field_name in allowed_changes:
            continue

        assert after_dump[field_name] == before_value, (
            f"Unexpected field change: {field_name}"
        )


def main():
    # ------------------------------------------------------------------
    # 1. No-memory round trip through the REAL schema.
    # ------------------------------------------------------------------
    original_result = make_module3_result()
    original_snapshot = deepcopy(
        original_result.model_dump(
            mode="python"
        )
    )

    overlay = DetectedTopicEditOverlay(
        candidate_provider=FakeCandidateProvider(),
        reason_validator=FakeReasonValidator(
            safe_memory_ids=set()
        ),
    )

    overlay_result = overlay.apply(
        topics=(
            ActualModule3TopicOverlayAdapter
            .to_overlay_topics(
                original_result.merged_topics
            )
        ),
        spec_version=SPEC,
        evidence_by_concept_id=evidence_map(),
    )

    roundtrip = (
        ActualModule3TopicOverlayAdapter
        .materialize_module3_result(
            original_result=original_result,
            overlay_result=overlay_result,
            sort_result=False,
        )
    )

    assert (
        roundtrip.model_dump(mode="python")
        == original_snapshot
    )

    assert (
        original_result.model_dump(mode="python")
        == original_snapshot
    )

    # ------------------------------------------------------------------
    # 2. Controlled all-four-action regression.
    # ------------------------------------------------------------------
    original_result = make_module3_result()
    original_snapshot = deepcopy(
        original_result.model_dump(
            mode="python"
        )
    )

    remove_sorting = EditMemoryCandidate(
        memory_id=1,
        spec_version=SPEC,
        edit_action="remove_topic",
        source_concept_id="aqa_sorting",
        source_topic="Sorting algorithms",
        source_role="supporting",
        reviewer_reason="Sorting is incidental.",
        stored_evidence="Stored incidental sorting evidence.",
    )

    change_efficiency = EditMemoryCandidate(
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
        stored_evidence="Stored efficiency-focused evidence.",
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
        reviewer_reason="The generic subroutine mapping is too broad.",
        stored_evidence="Stored function/procedure evidence.",
    )

    add_boolean = EditMemoryCandidate(
        memory_id=4,
        spec_version=SPEC,
        edit_action="add_topic",
        target_concept_id="aqa_boolean",
        target_topic="Boolean operations",
        target_role="supporting",
        target_official_reference="3.2.5",
        reviewer_reason="Boolean operations were explicitly taught but missed.",
        stored_evidence="Stored Boolean teaching evidence.",
        current_source_chunk_ids=(5,),
    )

    overlay = DetectedTopicEditOverlay(
        candidate_provider=FakeCandidateProvider(
            existing={
                "aqa_sorting": [remove_sorting],
                "aqa_efficiency": [change_efficiency],
                "aqa_subroutine_statement": [replace_subroutine],
            },
            additions=[add_boolean],
        ),
        reason_validator=FakeReasonValidator(
            safe_memory_ids={
                1,
                2,
                3,
                4,
            }
        ),
    )

    overlay_result = overlay.apply(
        topics=(
            ActualModule3TopicOverlayAdapter
            .to_overlay_topics(
                original_result.merged_topics
            )
        ),
        spec_version=SPEC,
        evidence_by_concept_id=evidence_map(),
        current_chunk_evidence=[
            "AND, OR and NOT are explicitly taught with truth values."
        ],
    )

    official_metadata = {
        "aqa_subroutines": OfficialConceptMetadata(
            concept_id="aqa_subroutines",
            topic="Subroutines, procedures and functions",
            domain="Programming and algorithms",
            official_reference="3.2.10",
            chapter_reference="3.2",
            official_title="Subroutines, procedures and functions",
            paper="Paper 2",
            source_pages=(20, 21),
        ),
    }

    added_topics = {
        4: AddedTopicMaterialization(
            memory_id=4,
            metadata=OfficialConceptMetadata(
                concept_id="aqa_boolean",
                topic="Boolean operations",
                domain="Programming and algorithms",
                official_reference="3.2.5",
                chapter_reference="3.2",
                official_title="Boolean operations",
                paper="Paper 2",
                source_pages=(15,),
            ),
            confidence=0.78,
            ranking_score=0.44,
            source_chunk_ids=(5,),
            support_span_count=1,
            mean_semantic_score=0.80,
            mean_keyword_score=0.70,
            mean_salience_score=0.76,
            coverage_score=0.25,
            evidence=(
                "AND, OR and NOT are explicitly taught with truth values.",
            ),
            supporting_candidate_count=1,
        )
    }

    updated = (
        ActualModule3TopicOverlayAdapter
        .materialize_module3_result(
            original_result=original_result,
            overlay_result=overlay_result,
            official_metadata=official_metadata,
            added_topics=added_topics,
            sort_result=True,
        )
    )

    # Original full result is unchanged.
    assert (
        original_result.model_dump(mode="python")
        == original_snapshot
    )

    # Every top-level Module3Result field except merged_topics is preserved.
    updated_dump = updated.model_dump(
        mode="python"
    )

    for field_name, before_value in original_snapshot.items():
        if field_name == "merged_topics":
            continue
        assert updated_dump[field_name] == before_value

    original_by_id = {
        topic.concept_id: topic
        for topic in original_result.merged_topics
    }

    updated_by_id = {
        topic.concept_id: topic
        for topic in updated.merged_topics
    }

    # remove_topic
    assert "aqa_sorting" not in updated_by_id

    # untouched topic exactly preserved
    assert (
        updated_by_id["aqa_binary_search"].model_dump(mode="python")
        == original_by_id["aqa_binary_search"].model_dump(mode="python")
    )

    # change_role: ONLY topic_role changes
    assert (
        updated_by_id["aqa_efficiency"].topic_role
        == "primary"
    )
    assert_same_except(
        original_by_id["aqa_efficiency"],
        updated_by_id["aqa_efficiency"],
        allowed_changes={"topic_role"},
    )

    # replace_topic: official identity changes, evidence/scores stay exact.
    assert "aqa_subroutine_statement" not in updated_by_id
    assert "aqa_subroutines" in updated_by_id

    replaced = updated_by_id["aqa_subroutines"]
    source = original_by_id[
        "aqa_subroutine_statement"
    ]

    assert replaced.topic == (
        "Subroutines, procedures and functions"
    )
    assert replaced.official_reference == "3.2.10"
    assert replaced.source_chunk_ids == source.source_chunk_ids
    assert replaced.evidence == source.evidence
    assert replaced.confidence == source.confidence
    assert replaced.ranking_score == source.ranking_score
    assert replaced.coverage_score == source.coverage_score
    assert replaced.mean_semantic_score == source.mean_semantic_score
    assert replaced.mean_keyword_score == source.mean_keyword_score
    assert replaced.mean_salience_score == source.mean_salience_score

    # add_topic exists only because all required real-schema metrics were
    # supplied explicitly. The adapter did not invent defaults.
    added = updated_by_id["aqa_boolean"]

    assert added.topic == "Boolean operations"
    assert added.official_reference == "3.2.5"
    assert added.topic_role == "supporting"
    assert added.source_chunk_ids == [5]
    assert added.confidence == 0.78
    assert added.ranking_score == 0.44
    assert added.evidence == [
        "AND, OR and NOT are explicitly taught with truth values."
    ]

    # Role-first ordering should now put both primary topics before supporting.
    roles = [
        topic.topic_role
        for topic in updated.merged_topics
    ]

    first_supporting = (
        roles.index("supporting")
        if "supporting" in roles
        else len(roles)
    )

    assert all(
        role == "primary"
        for role in roles[:first_supporting]
    )

    # ------------------------------------------------------------------
    # 3. add_topic MUST fail if real-schema metrics are not supplied.
    #    This prevents the adapter from inventing scores/evidence.
    # ------------------------------------------------------------------
    add_only_overlay = DetectedTopicEditOverlay(
        candidate_provider=FakeCandidateProvider(
            additions=[add_boolean]
        ),
        reason_validator=FakeReasonValidator(
            safe_memory_ids={4}
        ),
    )

    original_result = make_module3_result()

    add_only_result = add_only_overlay.apply(
        topics=(
            ActualModule3TopicOverlayAdapter
            .to_overlay_topics(
                original_result.merged_topics
            )
        ),
        spec_version=SPEC,
        evidence_by_concept_id=evidence_map(),
        current_chunk_evidence=[
            "AND, OR and NOT are explicitly taught."
        ],
    )

    try:
        (
            ActualModule3TopicOverlayAdapter
            .materialize_module3_result(
                original_result=original_result,
                overlay_result=add_only_result,
                added_topics={},
            )
        )
    except ValueError as exc:
        assert "explicit validated MergedTopic metrics" in str(exc)
    else:
        raise AssertionError(
            "Expected add_topic materialization to fail without explicit "
            "real-schema metrics."
        )

    print(
        "Actual Module 3 schema overlay adapter regression tests passed"
    )


if __name__ == "__main__":
    main()
