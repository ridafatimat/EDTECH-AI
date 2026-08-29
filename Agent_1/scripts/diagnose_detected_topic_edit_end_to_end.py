from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from sqlalchemy import func, select

from app.db.models.detected_topic_edit_memory import (
    DetectedTopicEditMemory,
)
from app.db.repositories.detected_topic_edit_memory_repository import (
    DetectedTopicEditMemoryRepository,
)
from app.db.session import (
    get_engine,
    get_session_factory,
)
from app.schemas.topic import (
    ChunkTopicResult,
    MergedTopic,
    Module3Result,
    TopicCandidate,
)
from app.services.cs_concept_catalog import (
    get_concept,
)
from app.services.detected_topic_edit_end_to_end import (
    DetectedTopicEditEndToEndService,
)
from app.services.detected_topic_edit_memory_service import (
    DetectedTopicEdit,
    DetectedTopicEditMemoryService,
)


BASE_SPEC = "AQA-8525-v1.2-2022-11-29"


def concept_topic(
    *,
    concept_id: str,
    role: str,
    evidence: str,
    ranking_score: float,
    source_chunk_ids,
) -> MergedTopic:
    concept = get_concept(
        concept_id
    )

    return MergedTopic(
        concept_id=concept.concept_id,
        topic=concept.label,
        domain=concept.domain,
        official_reference=(
            concept.official_reference
        ),
        chapter_reference=(
            concept.chapter_reference
        ),
        official_title=(
            concept.official_title
        ),
        paper=concept.paper,
        source_pages=list(
            concept.source_pages
        ),
        confidence=0.78,
        ranking_score=ranking_score,
        topic_role=role,
        source_chunk_ids=list(
            source_chunk_ids
        ),
        support_span_count=1,
        mean_semantic_score=0.76,
        mean_keyword_score=0.60,
        mean_salience_score=0.72,
        coverage_score=0.50,
        evidence=[evidence],
        supporting_candidate_count=1,
    )


def candidate(
    *,
    concept_id: str,
    evidence: str,
    cs_relevance_score: float,
    semantic_score: float,
    keyword_score: float,
    salience_score: float,
    cs_relevant: bool,
) -> TopicCandidate:
    concept = get_concept(
        concept_id
    )

    return TopicCandidate(
        concept_id=concept.concept_id,
        topic=concept.label,
        domain=concept.domain,
        official_reference=(
            concept.official_reference
        ),
        chapter_reference=(
            concept.chapter_reference
        ),
        official_title=(
            concept.official_title
        ),
        paper=concept.paper,
        source_pages=list(
            concept.source_pages
        ),
        confidence=0.76,
        keyword_score=keyword_score,
        semantic_score=semantic_score,
        salience_score=salience_score,
        extraction_method="keyword_embedding",
        evidence=[evidence],
        cs_relevance_score=cs_relevance_score,
        cs_relevant=cs_relevant,
    )


SORTING_ID = "aqa_3_1_4_sorting_algorithms"
BINARY_ID = "aqa_3_1_3_binary_search"
EFFICIENCY_ID = "aqa_3_1_2_efficiency"
SUBROUTINE_GENERIC_ID = "aqa_3_2_2_subroutine_statement"
SUBROUTINE_SPECIFIC_ID = "aqa_3_2_10_subroutines"
BOOLEAN_ID = "aqa_3_2_5_boolean_operations"


SORTING_INCIDENTAL = (
    "Binary search requires ordered data. If the list is unsorted, "
    "sorting would be needed before the search can begin, but no sorting "
    "algorithm is taught."
)

SORTING_REAL_TEACHING = (
    "Bubble sort compares adjacent values and swaps them when they are in "
    "the wrong order. Students trace repeated passes until no swaps remain."
)

EFFICIENCY_FOCUS = (
    "The lesson repeatedly compares two algorithms solving the same problem "
    "and explains why one needs fewer operations and is faster."
)

SUBROUTINE_TEACHING = (
    "The lesson explains functions and procedures, their parameters, local "
    "variables and returned values."
)

BOOLEAN_TEACHING = (
    "The lesson explicitly teaches Boolean AND, OR and NOT operators with "
    "truth values and worked programming examples."
)


def positive_module3_result() -> Module3Result:
    boolean_rejected = candidate(
        concept_id=BOOLEAN_ID,
        evidence=BOOLEAN_TEACHING,
        cs_relevance_score=0.77,
        semantic_score=0.81,
        keyword_score=0.73,
        salience_score=0.78,
        cs_relevant=False,
    )

    chunk_results = [
        ChunkTopicResult(
            chunk_id=1,
            source_word_count=80,
            classification="official_aqa_topic",
            is_cs_relevant=True,
            creates_new_topic=True,
            cs_relevance_score=0.84,
            topic_candidates=[
                candidate(
                    concept_id=BINARY_ID,
                    evidence=(
                        "Binary search is taught using midpoint calculations "
                        "and repeated halving."
                    ),
                    cs_relevance_score=0.84,
                    semantic_score=0.82,
                    keyword_score=0.80,
                    salience_score=0.82,
                    cs_relevant=True,
                )
            ],
        ),
        ChunkTopicResult(
            chunk_id=2,
            source_word_count=70,
            classification="official_aqa_topic",
            is_cs_relevant=True,
            creates_new_topic=True,
            cs_relevance_score=0.76,
            topic_candidates=[
                candidate(
                    concept_id=EFFICIENCY_ID,
                    evidence=EFFICIENCY_FOCUS,
                    cs_relevance_score=0.76,
                    semantic_score=0.79,
                    keyword_score=0.67,
                    salience_score=0.81,
                    cs_relevant=True,
                )
            ],
        ),
        ChunkTopicResult(
            chunk_id=3,
            source_word_count=75,
            classification="official_aqa_topic",
            is_cs_relevant=True,
            creates_new_topic=True,
            cs_relevance_score=0.74,
            topic_candidates=[
                candidate(
                    concept_id=SORTING_ID,
                    evidence=SORTING_INCIDENTAL,
                    cs_relevance_score=0.74,
                    semantic_score=0.64,
                    keyword_score=0.59,
                    salience_score=0.48,
                    cs_relevant=True,
                )
            ],
        ),
        ChunkTopicResult(
            chunk_id=4,
            source_word_count=85,
            classification="official_aqa_topic",
            is_cs_relevant=True,
            creates_new_topic=True,
            cs_relevance_score=0.79,
            topic_candidates=[
                candidate(
                    concept_id=SUBROUTINE_GENERIC_ID,
                    evidence=SUBROUTINE_TEACHING,
                    cs_relevance_score=0.79,
                    semantic_score=0.77,
                    keyword_score=0.71,
                    salience_score=0.75,
                    cs_relevant=True,
                )
            ],
        ),
        ChunkTopicResult(
            chunk_id=5,
            source_word_count=65,
            classification="mixed_official_and_unmapped",
            is_cs_relevant=True,
            creates_new_topic=False,
            cs_relevance_score=0.0,
            topic_candidates=[],
            rejected_candidates=[
                boolean_rejected,
            ],
            has_unmapped_cs_content=False,
            requires_llm_fallback=False,
        ),
    ]

    return Module3Result(
        chunk_results=chunk_results,
        merged_topics=[
            concept_topic(
                concept_id=BINARY_ID,
                role="primary",
                evidence=(
                    "Binary search is taught using midpoint calculations "
                    "and repeated halving."
                ),
                ranking_score=0.78,
                source_chunk_ids=(1,),
            ),
            concept_topic(
                concept_id=SORTING_ID,
                role="supporting",
                evidence=SORTING_INCIDENTAL,
                ranking_score=0.40,
                source_chunk_ids=(3,),
            ),
            concept_topic(
                concept_id=EFFICIENCY_ID,
                role="supporting",
                evidence=EFFICIENCY_FOCUS,
                ranking_score=0.55,
                source_chunk_ids=(2,),
            ),
            concept_topic(
                concept_id=SUBROUTINE_GENERIC_ID,
                role="supporting",
                evidence=SUBROUTINE_TEACHING,
                ranking_score=0.47,
                source_chunk_ids=(4,),
            ),
        ],
        total_chunks=5,
        cs_relevant_chunks=5,
        non_cs_chunks=0,
        official_topic_chunks=4,
        mixed_official_unmapped_chunks=1,
        unmapped_cs_chunks=0,
        continuation_chunks=0,
        no_topic_chunks=0,
        llm_fallback_chunk_ids=[],
        embedding_model=(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        candidate_keep_threshold=0.50,
    )


def negative_sorting_result() -> Module3Result:
    result = positive_module3_result()

    # Keep only Binary Search + Sorting so this is a focused negative test.
    retained = [
        topic.model_copy(
            deep=True
        )
        for topic in result.merged_topics
        if topic.concept_id in {
            BINARY_ID,
            SORTING_ID,
        }
    ]

    retained = [
        (
            topic.model_copy(
                deep=True,
                update={
                    "evidence": [
                        SORTING_REAL_TEACHING
                    ]
                },
            )
            if topic.concept_id == SORTING_ID
            else topic
        )
        for topic in retained
    ]

    focused_chunks = [
        result.chunk_results[0].model_copy(
            deep=True
        ),
        ChunkTopicResult(
            chunk_id=2,
            source_word_count=90,
            classification="official_aqa_topic",
            is_cs_relevant=True,
            creates_new_topic=True,
            cs_relevance_score=0.88,
            topic_candidates=[
                candidate(
                    concept_id=SORTING_ID,
                    evidence=SORTING_REAL_TEACHING,
                    cs_relevance_score=0.88,
                    semantic_score=0.87,
                    keyword_score=0.86,
                    salience_score=0.90,
                    cs_relevant=True,
                )
            ],
        ),
    ]

    return Module3Result(
        chunk_results=focused_chunks,
        merged_topics=retained,
        total_chunks=2,
        cs_relevant_chunks=2,
        non_cs_chunks=0,
        official_topic_chunks=2,
        mixed_official_unmapped_chunks=0,
        unmapped_cs_chunks=0,
        continuation_chunks=0,
        no_topic_chunks=0,
        llm_fallback_chunk_ids=[],
        embedding_model=result.embedding_model,
        candidate_keep_threshold=(
            result.candidate_keep_threshold
        ),
    )


def seed_test_memories(
    *,
    memory_service: DetectedTopicEditMemoryService,
    spec_version: str,
) -> list[int]:
    edits = [
        DetectedTopicEdit(
            edit_action="remove_topic",
            source_concept_id=SORTING_ID,
            source_topic=get_concept(
                SORTING_ID
            ).label,
            source_role="supporting",
            evidence_text=SORTING_INCIDENTAL,
            source_chunk_ids=(3,),
            reviewer_reason=(
                "Sorting is only mentioned as a prerequisite for binary "
                "search; no sorting algorithm is independently taught."
            ),
            source_transcript="__STEP5_ROLLBACK_TEST__",
            spec_version=spec_version,
            reviewed_by="step5-controlled-test",
        ),
        DetectedTopicEdit(
            edit_action="change_role",
            source_concept_id=EFFICIENCY_ID,
            source_topic=get_concept(
                EFFICIENCY_ID
            ).label,
            source_role="supporting",
            target_concept_id=EFFICIENCY_ID,
            target_topic=get_concept(
                EFFICIENCY_ID
            ).label,
            target_role="primary",
            evidence_text=EFFICIENCY_FOCUS,
            source_chunk_ids=(2,),
            reviewer_reason=(
                "Time efficiency is the central teaching objective rather "
                "than merely a supporting comparison."
            ),
            source_transcript="__STEP5_ROLLBACK_TEST__",
            spec_version=spec_version,
            reviewed_by="step5-controlled-test",
        ),
        DetectedTopicEdit(
            edit_action="replace_topic",
            source_concept_id=(
                SUBROUTINE_GENERIC_ID
            ),
            source_topic=get_concept(
                SUBROUTINE_GENERIC_ID
            ).label,
            source_role="supporting",
            target_concept_id=(
                SUBROUTINE_SPECIFIC_ID
            ),
            target_topic=get_concept(
                SUBROUTINE_SPECIFIC_ID
            ).label,
            target_role="supporting",
            evidence_text=SUBROUTINE_TEACHING,
            source_chunk_ids=(4,),
            reviewer_reason=(
                "The lesson teaches functions and procedures, parameters, "
                "local variables and returned values; the generic "
                "subroutine-statement mapping is too broad."
            ),
            source_transcript="__STEP5_ROLLBACK_TEST__",
            spec_version=spec_version,
            reviewed_by="step5-controlled-test",
        ),
        DetectedTopicEdit(
            edit_action="add_topic",
            target_concept_id=BOOLEAN_ID,
            target_topic=get_concept(
                BOOLEAN_ID
            ).label,
            target_role="supporting",
            evidence_text=BOOLEAN_TEACHING,
            source_chunk_ids=(5,),
            reviewer_reason=(
                "Boolean operators are explicitly taught but were missed "
                "from the detected final topic list."
            ),
            source_transcript="__STEP5_ROLLBACK_TEST__",
            spec_version=spec_version,
            reviewed_by="step5-controlled-test",
        ),
    ]

    ids = []

    for edit in edits:
        record = memory_service.remember(
            edit
        )
        ids.append(
            int(record.id)
        )

    return ids


def count_spec_rows(
    session,
    spec_version: str,
) -> int:
    return int(
        session.execute(
            select(
                func.count(
                    DetectedTopicEditMemory.id
                )
            ).where(
                DetectedTopicEditMemory.spec_version
                == spec_version
            )
        ).scalar_one()
    )


def main() -> None:
    print("=" * 100)
    print("STEP 5 — CONTROLLED END-TO-END EDIT-MEMORY CHAIN")
    print("=" * 100)
    print()
    print(
        "This diagnostic uses REAL PostgreSQL, REAL MiniLM and REAL Groq."
    )
    print(
        "Temporary PostgreSQL test memories are inserted inside ONE "
        "transaction and ROLLED BACK at the end."
    )
    print(
        "No hit_count / last_used_at update is performed."
    )
    print(
        "No Streamlit / Module 3 notebook / Agent 2 code is modified."
    )
    print()

    # Confirm the Step 1 table already exists. We do not create schema here.
    engine = get_engine()

    if not engine.dialect.has_table(
        engine.connect(),
        DetectedTopicEditMemory.__tablename__,
    ):
        raise RuntimeError(
            "detected_topic_edit_memory table does not exist. "
            "Complete Step 1 database setup first."
        )

    # Unique test spec means real user memories are excluded by hard filter.
    test_spec = (
        BASE_SPEC
        + "-STEP5-"
        + uuid4().hex[:10]
    )

    session_factory = get_session_factory()
    session = session_factory()

    rollback_verified = False

    try:
        repo = DetectedTopicEditMemoryRepository(
            session
        )
        memory_service = (
            DetectedTopicEditMemoryService(
                repo
            )
        )

        before_count = count_spec_rows(
            session,
            test_spec,
        )

        assert before_count == 0

        test_memory_ids = seed_test_memories(
            memory_service=memory_service,
            spec_version=test_spec,
        )

        inserted_count = count_spec_rows(
            session,
            test_spec,
        )

        assert inserted_count == 4

        print(
            "Temporary test memories inserted:",
            inserted_count,
        )
        print(
            "Temporary memory IDs:",
            test_memory_ids,
        )

        # --------------------------------------------------------------
        # POSITIVE: all four edits should flow through the complete chain.
        # --------------------------------------------------------------
        print()
        print("-" * 100)
        print("POSITIVE CONTROLLED CHAIN")
        print("-" * 100)

        original = positive_module3_result()
        original_snapshot = deepcopy(
            original.model_dump(
                mode="python"
            )
        )

        service = DetectedTopicEditEndToEndService(
            repository=repo
        )

        positive = service.apply(
            module3_result=original,
            spec_version=test_spec,
        )

        # Original real Module3Result remains untouched.
        assert (
            original.model_dump(
                mode="python"
            )
            == original_snapshot
        )

        updated_by_id = {
            topic.concept_id: topic
            for topic in positive.module3_result.merged_topics
        }

        # remove_topic
        assert SORTING_ID not in updated_by_id

        # change_role
        assert (
            updated_by_id[
                EFFICIENCY_ID
            ].topic_role
            == "primary"
        )

        # replace_topic
        assert (
            SUBROUTINE_GENERIC_ID
            not in updated_by_id
        )
        assert (
            SUBROUTINE_SPECIFIC_ID
            in updated_by_id
        )

        # add_topic
        assert BOOLEAN_ID in updated_by_id
        assert (
            updated_by_id[
                BOOLEAN_ID
            ].source_chunk_ids
            == [5]
        )

        # Existing Module 3 candidate evidence produced the add metrics.
        assert (
            updated_by_id[
                BOOLEAN_ID
            ].confidence
            > 0.0
        )
        assert (
            updated_by_id[
                BOOLEAN_ID
            ].evidence
            == [BOOLEAN_TEACHING]
        )

        actions = {
            edit.action
            for edit in (
                positive.overlay_result.applied
            )
        }

        assert actions == {
            "remove_topic",
            "change_role",
            "replace_topic",
            "add_topic",
        }

        print(
            "Applied actions:",
            sorted(actions),
        )

        print(
            "Final topic IDs:",
            [
                topic.concept_id
                for topic in (
                    positive.module3_result
                    .merged_topics
                )
            ],
        )

        print()
        print("Retrieval diagnostics:")

        for line in (
            positive.retrieval_diagnostics
        ):
            print(" -", line)

        print()
        print(
            "POSITIVE_CHAIN = PASS"
        )

        # --------------------------------------------------------------
        # NEGATIVE: real sorting lesson must survive the old removal memory.
        # --------------------------------------------------------------
        print()
        print("-" * 100)
        print("NEGATIVE SAFETY CHAIN — ACTUAL SORTING TEACHING")
        print("-" * 100)

        negative_input = (
            negative_sorting_result()
        )
        negative_snapshot = deepcopy(
            negative_input.model_dump(
                mode="python"
            )
        )

        negative_service = (
            DetectedTopicEditEndToEndService(
                repository=repo
            )
        )

        negative = negative_service.apply(
            module3_result=negative_input,
            spec_version=test_spec,
        )

        assert (
            negative_input.model_dump(
                mode="python"
            )
            == negative_snapshot
        )

        negative_ids = {
            topic.concept_id
            for topic in (
                negative.module3_result
                .merged_topics
            )
        }

        # This is the key false-positive guard.
        assert SORTING_ID in negative_ids

        unsafe_sorting_removals = [
            edit
            for edit in negative.overlay_result.applied
            if (
                edit.action == "remove_topic"
                and edit.source_concept_id
                == SORTING_ID
            )
        ]

        assert unsafe_sorting_removals == []

        print(
            "Sorting topic retained:",
            SORTING_ID in negative_ids,
        )

        print()
        print("Retrieval diagnostics:")

        for line in (
            negative.retrieval_diagnostics
        ):
            print(" -", line)

        print()
        print(
            "NEGATIVE_SAFETY_CHAIN = PASS"
        )

        # Verify Step 5 never updates usage counters.
        records = [
            repo.get(memory_id)
            for memory_id in test_memory_ids
        ]

        assert all(
            record is not None
            and int(record.hit_count or 0) == 0
            and record.last_used_at is None
            for record in records
        )

        print()
        print(
            "Usage counters unchanged: PASS"
        )

    finally:
        # CRITICAL: all temporary memory rows disappear here.
        session.rollback()
        session.close()

    verification_session = (
        session_factory()
    )

    try:
        remaining = count_spec_rows(
            verification_session,
            test_spec,
        )

        rollback_verified = (
            remaining == 0
        )

    finally:
        verification_session.close()

    print()
    print("=" * 100)
    print("STEP 5 SUMMARY")
    print("=" * 100)
    print(
        "Temporary PostgreSQL rows remaining:",
        0 if rollback_verified else "UNEXPECTED ROWS",
    )
    print(
        "Transaction rollback verified:",
        rollback_verified,
    )
    print(
        "Original Module3Result mutation:",
        "NONE",
    )
    print(
        "Persistent hit-count changes:",
        "NONE",
    )
    print()

    if rollback_verified:
        print(
            "READY_FOR_STREAMLIT_WIRING = YES"
        )
    else:
        print(
            "READY_FOR_STREAMLIT_WIRING = NO"
        )


if __name__ == "__main__":
    main()
