from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence

from app.schemas.topic import MergedTopic, Module3Result
from app.services.detected_topic_edit_overlay import (
    AppliedOverlayEdit,
    OverlayResult,
    OverlayTopic,
)


@dataclass(frozen=True, slots=True)
class OfficialConceptMetadata:
    """
    Official AQA metadata required when a human edit changes concept identity.

    Existing-topic role changes do not need this.
    Replacements and human-added topics do.

    No metadata is invented by the adapter.
    """

    concept_id: str
    topic: str
    domain: str

    official_reference: str
    chapter_reference: str
    official_title: str
    paper: str

    source_pages: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class AddedTopicMaterialization:
    """
    Explicit validated values required to construct a NEW real MergedTopic.

    The current Module 3 schema requires ranking/confidence/evidence metrics.
    Step 4.9 therefore refuses to invent them.

    A later production integration must decide where these values come from
    (for example, reviewed source evidence / an existing official candidate).
    Until then, add_topic materialization is allowed only when the caller
    supplies every required value explicitly.
    """

    memory_id: int
    metadata: OfficialConceptMetadata

    confidence: float
    ranking_score: float

    source_chunk_ids: tuple[int, ...]
    support_span_count: int

    mean_semantic_score: float
    mean_keyword_score: float
    mean_salience_score: float
    coverage_score: float

    evidence: tuple[str, ...]
    supporting_candidate_count: int


class ActualModule3TopicOverlayAdapter:
    """
    Adapter between the real Agent 1 `MergedTopic` Pydantic schema and the
    neutral Step 4.8 `OverlayTopic`.

    Safety goals:
    - real Module 3 objects are copied, never mutated;
    - untouched topics are byte-for-field equivalent after round-trip;
    - role changes alter only `topic_role`;
    - replacements preserve source evidence/scores/chunks while replacing only
      official identity fields and requested role;
    - add_topic requires explicit materialization values;
    - Module3Result is copied with only `merged_topics` changed;
    - audit metadata stays in OverlayResult, not inside the existing schema.
    """

    @staticmethod
    def _topic_dump(topic: MergedTopic) -> dict:
        return topic.model_dump(mode="python")

    @staticmethod
    def _copy_topic(topic: MergedTopic) -> MergedTopic:
        return topic.model_copy(deep=True)

    @classmethod
    def to_overlay_topic(
        cls,
        topic: MergedTopic,
    ) -> OverlayTopic:
        return OverlayTopic(
            concept_id=topic.concept_id,
            topic=topic.topic,
            role=topic.topic_role,
            official_reference=topic.official_reference,
            confidence=topic.confidence,
            ranking_score=topic.ranking_score,
            source_chunk_ids=tuple(topic.source_chunk_ids),
        )

    @classmethod
    def to_overlay_topics(
        cls,
        topics: Sequence[MergedTopic],
    ) -> tuple[OverlayTopic, ...]:
        return tuple(
            cls.to_overlay_topic(topic)
            for topic in topics
        )

    @staticmethod
    def _applied_by_memory(
        overlay_result: OverlayResult,
    ) -> dict[int, AppliedOverlayEdit]:
        mapping: dict[int, AppliedOverlayEdit] = {}

        for edit in overlay_result.applied:
            if edit.memory_id in mapping:
                raise ValueError(
                    "OverlayResult contains duplicate applied memory IDs."
                )
            mapping[int(edit.memory_id)] = edit

        return mapping

    @staticmethod
    def _validate_metadata(
        metadata: OfficialConceptMetadata,
        *,
        expected_concept_id: str,
    ) -> None:
        if metadata.concept_id != expected_concept_id:
            raise ValueError(
                "Official metadata concept_id does not match overlay target."
            )

        required_text = {
            "topic": metadata.topic,
            "domain": metadata.domain,
            "official_reference": metadata.official_reference,
            "chapter_reference": metadata.chapter_reference,
            "official_title": metadata.official_title,
            "paper": metadata.paper,
        }

        missing = [
            key
            for key, value in required_text.items()
            if not str(value).strip()
        ]

        if missing:
            raise ValueError(
                "Official concept metadata is incomplete: "
                + ", ".join(missing)
            )

    @classmethod
    def _materialize_unchanged(
        cls,
        overlay_topic: OverlayTopic,
        originals_by_concept: Mapping[str, MergedTopic],
    ) -> MergedTopic:
        original = originals_by_concept.get(
            overlay_topic.concept_id
        )

        if original is None:
            raise ValueError(
                "Unedited overlay topic has no matching original MergedTopic."
            )

        # A supposedly untouched overlay topic must still represent the same
        # identity and role. If not, fail rather than silently accepting drift.
        expected = cls.to_overlay_topic(original)

        comparable_expected = (
            expected.concept_id,
            expected.topic,
            expected.role,
            expected.official_reference,
            expected.confidence,
            expected.ranking_score,
            expected.source_chunk_ids,
        )
        comparable_actual = (
            overlay_topic.concept_id,
            overlay_topic.topic,
            overlay_topic.role,
            overlay_topic.official_reference,
            overlay_topic.confidence,
            overlay_topic.ranking_score,
            overlay_topic.source_chunk_ids,
        )

        if comparable_expected != comparable_actual:
            raise ValueError(
                "An overlay topic marked as unedited differs from its "
                "original Module 3 topic."
            )

        return cls._copy_topic(original)

    @classmethod
    def _materialize_role_change(
        cls,
        *,
        overlay_topic: OverlayTopic,
        applied_edit: AppliedOverlayEdit,
        originals_by_concept: Mapping[str, MergedTopic],
    ) -> MergedTopic:
        source_id = applied_edit.source_concept_id

        if not source_id:
            raise ValueError(
                "change_role audit entry has no source_concept_id."
            )

        original = originals_by_concept.get(source_id)

        if original is None:
            raise ValueError(
                "change_role source topic is missing from original topics."
            )

        if overlay_topic.concept_id != original.concept_id:
            raise ValueError(
                "change_role unexpectedly changed concept identity."
            )

        if overlay_topic.role not in {"primary", "supporting"}:
            raise ValueError(
                "change_role produced an invalid topic role."
            )

        # Pydantic's model_copy preserves every other Module 3 field.
        changed = original.model_copy(
            deep=True,
            update={
                "topic_role": overlay_topic.role,
            },
        )

        original_dump = cls._topic_dump(original)
        changed_dump = cls._topic_dump(changed)

        for field_name, original_value in original_dump.items():
            if field_name == "topic_role":
                continue
            if changed_dump[field_name] != original_value:
                raise AssertionError(
                    "change_role changed an unrelated Module 3 field: "
                    f"{field_name}"
                )

        return changed

    @classmethod
    def _materialize_replacement(
        cls,
        *,
        overlay_topic: OverlayTopic,
        applied_edit: AppliedOverlayEdit,
        originals_by_concept: Mapping[str, MergedTopic],
        official_metadata: Mapping[str, OfficialConceptMetadata],
    ) -> MergedTopic:
        source_id = applied_edit.source_concept_id

        if not source_id:
            raise ValueError(
                "replace_topic audit entry has no source_concept_id."
            )

        original = originals_by_concept.get(source_id)

        if original is None:
            raise ValueError(
                "replace_topic source topic is missing from originals."
            )

        metadata = official_metadata.get(
            overlay_topic.concept_id
        )

        if metadata is None:
            raise ValueError(
                "replace_topic requires official target metadata."
            )

        cls._validate_metadata(
            metadata,
            expected_concept_id=overlay_topic.concept_id,
        )

        if overlay_topic.role not in {"primary", "supporting"}:
            raise ValueError(
                "replace_topic produced an invalid role."
            )

        # Preserve extraction/ranking/evidence values because the human edit
        # corrects official identity; it does not rerun Module 3 scoring.
        changed = original.model_copy(
            deep=True,
            update={
                "concept_id": metadata.concept_id,
                "topic": metadata.topic,
                "domain": metadata.domain,
                "official_reference": metadata.official_reference,
                "chapter_reference": metadata.chapter_reference,
                "official_title": metadata.official_title,
                "paper": metadata.paper,
                "source_pages": list(metadata.source_pages),
                "topic_role": overlay_topic.role,
            },
        )

        # Force Pydantic validation of the complete resulting object.
        validated = MergedTopic.model_validate(
            changed.model_dump(mode="python")
        )

        # Critical preservation checks.
        preserved_fields = (
            "confidence",
            "ranking_score",
            "source_chunk_ids",
            "support_span_count",
            "mean_semantic_score",
            "mean_keyword_score",
            "mean_salience_score",
            "coverage_score",
            "evidence",
            "supporting_candidate_count",
        )

        for field_name in preserved_fields:
            if (
                getattr(validated, field_name)
                != getattr(original, field_name)
            ):
                raise AssertionError(
                    "replace_topic changed an extraction/evidence field: "
                    f"{field_name}"
                )

        return validated

    @classmethod
    def _materialize_addition(
        cls,
        *,
        overlay_topic: OverlayTopic,
        materialization: AddedTopicMaterialization,
    ) -> MergedTopic:
        if materialization.memory_id != overlay_topic.memory_id:
            raise ValueError(
                "Added-topic materialization memory_id does not match overlay."
            )

        cls._validate_metadata(
            materialization.metadata,
            expected_concept_id=overlay_topic.concept_id,
        )

        if overlay_topic.role not in {"primary", "supporting"}:
            raise ValueError(
                "add_topic produced an invalid role."
            )

        if overlay_topic.topic != materialization.metadata.topic:
            raise ValueError(
                "Overlay add-topic label differs from official metadata."
            )

        # Cross-transcript reuse note:
        # overlay_topic.source_chunk_ids come from the OLD reviewed transcript.
        # materialization.source_chunk_ids come from CURRENT Module 3 candidate
        # evidence and are therefore authoritative for the new topic.
        topic = MergedTopic(
            concept_id=materialization.metadata.concept_id,
            topic=materialization.metadata.topic,
            domain=materialization.metadata.domain,
            official_reference=(
                materialization.metadata.official_reference
            ),
            chapter_reference=(
                materialization.metadata.chapter_reference
            ),
            official_title=materialization.metadata.official_title,
            paper=materialization.metadata.paper,
            source_pages=list(materialization.metadata.source_pages),
            confidence=materialization.confidence,
            ranking_score=materialization.ranking_score,
            topic_role=overlay_topic.role,
            source_chunk_ids=list(materialization.source_chunk_ids),
            support_span_count=materialization.support_span_count,
            mean_semantic_score=materialization.mean_semantic_score,
            mean_keyword_score=materialization.mean_keyword_score,
            mean_salience_score=materialization.mean_salience_score,
            coverage_score=materialization.coverage_score,
            evidence=list(materialization.evidence),
            supporting_candidate_count=(
                materialization.supporting_candidate_count
            ),
        )

        return topic

    @staticmethod
    def _sort_final_topics(
        topics: Sequence[MergedTopic],
    ) -> list[MergedTopic]:
        """
        Preserve the Module 3 presentation contract:
        primary first, then lesson-relative ranking/coverage/semantic strength.

        Scores themselves are never recalculated here.
        """

        role_priority = {
            "primary": 1,
            "supporting": 0,
        }

        return sorted(
            list(topics),
            key=lambda topic: (
                role_priority[topic.topic_role],
                topic.ranking_score,
                topic.coverage_score,
                topic.mean_semantic_score,
                topic.confidence,
            ),
            reverse=True,
        )

    @classmethod
    def materialize_topics(
        cls,
        *,
        original_topics: Sequence[MergedTopic],
        overlay_result: OverlayResult,
        official_metadata: Mapping[
            str,
            OfficialConceptMetadata,
        ] | None = None,
        added_topics: Mapping[
            int,
            AddedTopicMaterialization,
        ] | None = None,
        sort_result: bool = True,
    ) -> list[MergedTopic]:
        """
        Convert the neutral OverlayResult back into REAL Module 3 MergedTopic
        objects without modifying the original objects.
        """

        official_metadata = official_metadata or {}
        added_topics = added_topics or {}

        original_copies = [
            topic.model_copy(deep=True)
            for topic in original_topics
        ]

        originals_by_concept = {
            topic.concept_id: topic
            for topic in original_copies
        }

        if len(originals_by_concept) != len(original_copies):
            raise ValueError(
                "Original Module 3 topics contain duplicate concept IDs."
            )

        applied_by_memory = cls._applied_by_memory(
            overlay_result
        )

        final_topics: list[MergedTopic] = []

        for overlay_topic in overlay_result.topics:
            if not overlay_topic.memory_applied:
                materialized = cls._materialize_unchanged(
                    overlay_topic,
                    originals_by_concept,
                )
                final_topics.append(materialized)
                continue

            if overlay_topic.memory_id is None:
                raise ValueError(
                    "Edited overlay topic is missing memory_id."
                )

            applied_edit = applied_by_memory.get(
                int(overlay_topic.memory_id)
            )

            if applied_edit is None:
                raise ValueError(
                    "Edited overlay topic has no matching audit entry."
                )

            action = overlay_topic.memory_action

            if action == "change_role":
                materialized = cls._materialize_role_change(
                    overlay_topic=overlay_topic,
                    applied_edit=applied_edit,
                    originals_by_concept=originals_by_concept,
                )

            elif action == "replace_topic":
                materialized = cls._materialize_replacement(
                    overlay_topic=overlay_topic,
                    applied_edit=applied_edit,
                    originals_by_concept=originals_by_concept,
                    official_metadata=official_metadata,
                )

            elif action == "add_topic":
                addition = added_topics.get(
                    int(overlay_topic.memory_id)
                )

                if addition is None:
                    raise ValueError(
                        "add_topic cannot be materialized without explicit "
                        "validated MergedTopic metrics."
                    )

                materialized = cls._materialize_addition(
                    overlay_topic=overlay_topic,
                    materialization=addition,
                )

            else:
                raise ValueError(
                    "Unexpected memory action on a returned overlay topic: "
                    f"{action!r}"
                )

            final_topics.append(materialized)

        concept_ids = [
            topic.concept_id
            for topic in final_topics
        ]

        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError(
                "Materialized Module 3 topics contain duplicate concept IDs."
            )

        if sort_result:
            return cls._sort_final_topics(
                final_topics
            )

        return final_topics

    @classmethod
    def materialize_module3_result(
        cls,
        *,
        original_result: Module3Result,
        overlay_result: OverlayResult,
        official_metadata: Mapping[
            str,
            OfficialConceptMetadata,
        ] | None = None,
        added_topics: Mapping[
            int,
            AddedTopicMaterialization,
        ] | None = None,
        sort_result: bool = True,
    ) -> Module3Result:
        """
        Return a NEW real Module3Result whose only intended change is the
        `merged_topics` list.

        Chunk results, counts, fallback IDs, embedding model and candidate
        threshold remain exactly as produced by the working Module 3 pipeline.
        """

        original_snapshot = original_result.model_dump(
            mode="python"
        )

        final_topics = cls.materialize_topics(
            original_topics=original_result.merged_topics,
            overlay_result=overlay_result,
            official_metadata=official_metadata,
            added_topics=added_topics,
            sort_result=sort_result,
        )

        updated = original_result.model_copy(
            deep=True,
            update={
                "merged_topics": final_topics,
            },
        )

        # Validate full current schema.
        updated = Module3Result.model_validate(
            updated.model_dump(mode="python")
        )

        # Prove every Module3Result field except merged_topics is unchanged.
        updated_dump = updated.model_dump(
            mode="python"
        )

        for field_name, original_value in original_snapshot.items():
            if field_name == "merged_topics":
                continue

            if updated_dump[field_name] != original_value:
                raise AssertionError(
                    "Overlay adapter changed an unrelated Module3Result field: "
                    f"{field_name}"
                )

        return updated
