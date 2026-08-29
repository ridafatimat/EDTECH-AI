from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from app.services.detected_topic_edit_memory_service import (
    DetectedTopicEdit,
    DetectedTopicEditMemoryService,
)


class FakeRepository:
    def __init__(self) -> None:
        self.records = []
        self.next_id = 1

    def upsert(self, values: dict):
        for record in self.records:
            if record.cache_key == values["cache_key"]:
                for key, value in values.items():
                    setattr(record, key, value)
                return record

        record = SimpleNamespace(
            id=self.next_id,
            hit_count=0,
            last_used_at=None,
            **values,
        )
        self.next_id += 1
        self.records.append(record)
        return record

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

    def mark_used(self, memory_id: int):
        for record in self.records:
            if record.id == memory_id:
                record.hit_count += 1
                return record
        raise LookupError(memory_id)


def expect_value_error(callback) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def main() -> None:
    repository = FakeRepository()
    service = DetectedTopicEditMemoryService(repository)

    sorting_evidence = (
        "If the items are not in order, you first have to perform some "
        "form of sorting algorithm before starting a binary search."
    )

    sorting_edit = DetectedTopicEdit(
        edit_action="remove_topic",
        source_concept_id="aqa_3_1_4_sorting_algorithms",
        source_topic="Sorting algorithms",
        source_role="supporting",
        evidence_text=sorting_evidence,
        source_chunk_ids=(3,),
        reviewer_reason=(
            "Sorting is only mentioned as a prerequisite for binary search; "
            "no sorting algorithm is actually taught."
        ),
        source_transcript="0_00.docx",
        spec_version="AQA-8525-v1.2-2022-11-29",
        reviewed_by="streamlit",
    )

    first = service.remember(sorting_edit)
    assert first.id == 1
    assert first.edit_action == "remove_topic"
    assert first.reviewer_approved is True
    assert first.is_active is True

    # Repeating the same exact reviewed edit must not create duplicate memory.
    second = service.remember(sorting_edit)
    assert second.id == first.id
    assert len(repository.records) == 1

    exact_match = service.find_exact_evidence_match(
        spec_version="AQA-8525-v1.2-2022-11-29",
        evidence_text=sorting_evidence,
        source_concept_id="aqa_3_1_4_sorting_algorithms",
        edit_actions=("remove_topic",),
    )
    assert exact_match is not None
    assert exact_match.memory_id == first.id

    # Different evidence must NOT be reused in Step 2.
    unrelated_sorting_lesson = (
        "Bubble sort repeatedly compares adjacent items and swaps them "
        "until the list is sorted."
    )
    assert service.find_exact_evidence_match(
        spec_version="AQA-8525-v1.2-2022-11-29",
        evidence_text=unrelated_sorting_lesson,
        source_concept_id="aqa_3_1_4_sorting_algorithms",
        edit_actions=("remove_topic",),
    ) is None

    # Different specification version must never reuse the old edit.
    assert service.find_exact_evidence_match(
        spec_version="SOME-FUTURE-SPEC",
        evidence_text=sorting_evidence,
        source_concept_id="aqa_3_1_4_sorting_algorithms",
        edit_actions=("remove_topic",),
    ) is None

    role_edit = DetectedTopicEdit(
        edit_action="change_role",
        source_concept_id="aqa_3_1_2_efficiency",
        source_topic="Time efficiency of algorithms",
        source_role="supporting",
        target_role="primary",
        evidence_text=(
            "The lesson repeatedly compares execution speed and explains "
            "why one algorithm is more time-efficient than another."
        ),
        source_chunk_ids=(1, 2, 3),
        reviewer_reason=(
            "Time efficiency is the central teaching objective in this lesson."
        ),
        source_transcript="efficiency_lesson.docx",
        spec_version="AQA-8525-v1.2-2022-11-29",
        reviewed_by="streamlit",
    )

    role_record = service.remember(role_edit)
    assert role_record.source_concept_id == "aqa_3_1_2_efficiency"
    assert role_record.target_concept_id == "aqa_3_1_2_efficiency"
    assert role_record.source_role == "supporting"
    assert role_record.target_role == "primary"

    expect_value_error(
        lambda: service.remember(
            DetectedTopicEdit(
                edit_action="remove_topic",
                source_concept_id="aqa_x",
                source_topic="X",
                source_role="supporting",
                evidence_text="Some evidence.",
                reviewer_reason="",
                spec_version="AQA-8525-v1.2-2022-11-29",
            )
        )
    )

    expect_value_error(
        lambda: service.remember(
            DetectedTopicEdit(
                edit_action="change_role",
                source_concept_id="aqa_x",
                source_topic="X",
                source_role="supporting",
                target_role="supporting",
                evidence_text="Some evidence.",
                reviewer_reason="Role should change.",
                spec_version="AQA-8525-v1.2-2022-11-29",
            )
        )
    )

    print("Detected-topic edit memory isolated tests passed")


if __name__ == "__main__":
    main()
