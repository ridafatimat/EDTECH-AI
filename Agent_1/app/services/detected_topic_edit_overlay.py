from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence


VALID_ROLES = frozenset({"primary", "supporting"})
VALID_ACTIONS = frozenset({
    "remove_topic",
    "add_topic",
    "replace_topic",
    "change_role",
})


@dataclass(frozen=True, slots=True)
class OverlayTopic:
    """
    Neutral in-memory representation used ONLY for the controlled overlay test.

    It deliberately does not import or modify Module 3's real topic schema.
    A later adapter can translate between the real Module 3 topic object and
    this neutral representation once the overlay behaviour is proven safe.
    """

    concept_id: str
    topic: str
    role: str

    official_reference: str | None = None
    confidence: float | None = None
    ranking_score: float | None = None
    source_chunk_ids: tuple[int, ...] = ()

    memory_applied: bool = False
    memory_id: int | None = None
    memory_action: str | None = None


@dataclass(frozen=True, slots=True)
class EditMemoryCandidate:
    """
    One plausible reviewer-approved memory candidate supplied to the overlay.

    Candidate retrieval/matching is a separate concern. Step 4.8 assumes a
    candidate was already retrieved and then applies hard fail-closed checks.
    """

    memory_id: int
    spec_version: str
    edit_action: str

    source_concept_id: str | None = None
    source_topic: str | None = None
    source_role: str | None = None

    target_concept_id: str | None = None
    target_topic: str | None = None
    target_role: str | None = None
    target_official_reference: str | None = None

    reviewer_reason: str = ""
    stored_evidence: str = ""

    # For add_topic, these are the CURRENT transcript chunks that matched.
    current_source_chunk_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ReasonValidation:
    """
    Minimal result required from the already-tested Step 4.6/4.7 reason gate.
    """

    decision: str
    confidence: float
    safe_for_automatic_reuse: bool
    explanation: str


class CandidateProviderProtocol(Protocol):
    """
    Controlled candidate provider boundary.

    Step 4.8 tests orchestration only. The production Step 3 matcher and
    PostgreSQL repository are NOT wired here yet.
    """

    def candidates_for_existing_topic(
        self,
        *,
        spec_version: str,
        topic: OverlayTopic,
        current_evidence: str,
    ) -> Sequence[EditMemoryCandidate]:
        ...

    def candidates_for_additions(
        self,
        *,
        spec_version: str,
        current_chunk_evidence: Sequence[str],
        already_present_concept_ids: Sequence[str],
    ) -> Sequence[EditMemoryCandidate]:
        ...


class ReasonValidatorProtocol(Protocol):
    def validate(
        self,
        *,
        candidate: EditMemoryCandidate,
        current_evidence: str,
    ) -> ReasonValidation:
        ...


@dataclass(frozen=True, slots=True)
class AppliedOverlayEdit:
    memory_id: int
    action: str
    source_concept_id: str | None
    target_concept_id: str | None
    explanation: str


@dataclass(frozen=True, slots=True)
class SkippedOverlayEdit:
    memory_id: int | None
    action: str | None
    source_concept_id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """
    Output of a controlled overlay run.

    `topics` is a NEW tuple. The original input objects/list are never mutated.
    """

    topics: tuple[OverlayTopic, ...]
    applied: tuple[AppliedOverlayEdit, ...]
    skipped: tuple[SkippedOverlayEdit, ...]


class DetectedTopicEditOverlay:
    """
    Fail-closed, in-memory final-topic edit overlay.

    Safety guarantees:
    - operates on a deep-copied topic list;
    - does not mutate original Module 3-like inputs;
    - does not write PostgreSQL;
    - does not mark memories as used;
    - does not call Qdrant/Groq itself;
    - does not modify mapping memory;
    - wrong spec -> skip;
    - 0 candidates -> keep original;
    - >1 candidates -> conflict -> keep original;
    - reason validator unsafe/uncertain/incompatible -> keep original;
    - malformed edit -> keep original;
    - duplicate add -> skip;
    - unknown action -> skip.
    """

    def __init__(
        self,
        *,
        candidate_provider: CandidateProviderProtocol,
        reason_validator: ReasonValidatorProtocol,
    ) -> None:
        self.candidate_provider = candidate_provider
        self.reason_validator = reason_validator

    @staticmethod
    def _validate_topic(topic: OverlayTopic) -> None:
        if not str(topic.concept_id).strip():
            raise ValueError("Topic concept_id must not be empty.")

        if not str(topic.topic).strip():
            raise ValueError("Topic label must not be empty.")

        if topic.role not in VALID_ROLES:
            raise ValueError(
                "Topic role must be 'primary' or 'supporting'."
            )

    @staticmethod
    def _candidate_spec_matches(
        candidate: EditMemoryCandidate,
        *,
        spec_version: str,
    ) -> bool:
        return (
            str(candidate.spec_version).strip()
            == str(spec_version).strip()
        )

    @staticmethod
    def _candidate_action_valid(
        candidate: EditMemoryCandidate,
    ) -> bool:
        return candidate.edit_action in VALID_ACTIONS

    @staticmethod
    def _mark_memory_applied(
        topic: OverlayTopic,
        candidate: EditMemoryCandidate,
    ) -> OverlayTopic:
        return replace(
            topic,
            memory_applied=True,
            memory_id=int(candidate.memory_id),
            memory_action=candidate.edit_action,
        )

    def _apply_existing_topic_candidate(
        self,
        *,
        topic: OverlayTopic,
        candidate: EditMemoryCandidate,
        validation: ReasonValidation,
    ) -> OverlayTopic | None:
        """
        Returns:
        - OverlayTopic for keep/replace/change_role
        - None for remove_topic

        Raises ValueError for malformed edit instructions.
        """

        action = candidate.edit_action

        if candidate.source_concept_id != topic.concept_id:
            raise ValueError(
                "Candidate source concept does not match current topic."
            )

        if action == "remove_topic":
            return None

        if action == "change_role":
            if candidate.target_role not in VALID_ROLES:
                raise ValueError(
                    "change_role requires a valid target role."
                )

            if (
                candidate.target_concept_id is not None
                and candidate.target_concept_id != topic.concept_id
            ):
                raise ValueError(
                    "change_role cannot change the official concept."
                )

            changed = replace(
                topic,
                role=candidate.target_role,
            )
            return self._mark_memory_applied(
                changed,
                candidate,
            )

        if action == "replace_topic":
            if not candidate.target_concept_id:
                raise ValueError(
                    "replace_topic requires target_concept_id."
                )

            if not candidate.target_topic:
                raise ValueError(
                    "replace_topic requires target_topic."
                )

            target_role = (
                candidate.target_role
                if candidate.target_role in VALID_ROLES
                else topic.role
            )

            changed = replace(
                topic,
                concept_id=candidate.target_concept_id,
                topic=candidate.target_topic,
                role=target_role,
                official_reference=(
                    candidate.target_official_reference
                    if candidate.target_official_reference is not None
                    else topic.official_reference
                ),
            )

            return self._mark_memory_applied(
                changed,
                candidate,
            )

        # add_topic is not valid on an already-present source topic.
        raise ValueError(
            f"Action {action!r} is not valid for an existing topic."
        )

    def apply(
        self,
        *,
        topics: Sequence[OverlayTopic],
        spec_version: str,
        evidence_by_concept_id: dict[str, str],
        current_chunk_evidence: Sequence[str] = (),
    ) -> OverlayResult:
        spec_version = str(spec_version).strip()

        if not spec_version:
            raise ValueError("spec_version is required.")

        # Critical safety rule: operate only on copies.
        working = deepcopy(list(topics))

        for topic in working:
            self._validate_topic(topic)

        applied: list[AppliedOverlayEdit] = []
        skipped: list[SkippedOverlayEdit] = []

        # --------------------------------------------------------------
        # A. Existing-topic edits
        # --------------------------------------------------------------
        output_topics: list[OverlayTopic] = []

        for topic in working:
            current_evidence = str(
                evidence_by_concept_id.get(
                    topic.concept_id,
                    "",
                )
            ).strip()

            if not current_evidence:
                output_topics.append(topic)
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=None,
                        action=None,
                        source_concept_id=topic.concept_id,
                        reason=(
                            "No current evidence supplied; overlay abstained."
                        ),
                    )
                )
                continue

            candidates = list(
                self.candidate_provider.candidates_for_existing_topic(
                    spec_version=spec_version,
                    topic=topic,
                    current_evidence=current_evidence,
                )
            )

            # 0 candidates = normal safe miss.
            if not candidates:
                output_topics.append(topic)
                continue

            # Overlay itself refuses conflicts even if an upstream matcher
            # accidentally provides multiple candidates.
            if len(candidates) != 1:
                output_topics.append(topic)
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=None,
                        action=None,
                        source_concept_id=topic.concept_id,
                        reason=(
                            "Multiple edit-memory candidates were supplied; "
                            "automatic overlay abstained."
                        ),
                    )
                )
                continue

            candidate = candidates[0]

            if not self._candidate_spec_matches(
                candidate,
                spec_version=spec_version,
            ):
                output_topics.append(topic)
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=topic.concept_id,
                        reason="Candidate spec_version does not match.",
                    )
                )
                continue

            if not self._candidate_action_valid(candidate):
                output_topics.append(topic)
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=topic.concept_id,
                        reason="Unknown edit action; overlay abstained.",
                    )
                )
                continue

            # Narrow no-op guard: if a historical change_role memory already
            # targets the exact role produced by fresh Module 3, there is no
            # state disagreement to resolve. Keep the fresh topic unchanged,
            # record the memory as already satisfied, and do NOT invoke the
            # contextual validator or ask a human to make a meaningless choice.
            # All retrieval, thresholds, validation and edit behaviour remain
            # unchanged for every memory that would actually alter the result.
            if (
                candidate.edit_action == "change_role"
                and candidate.target_role in VALID_ROLES
                and candidate.target_role == topic.role
                and (
                    candidate.target_concept_id is None
                    or candidate.target_concept_id == topic.concept_id
                )
            ):
                output_topics.append(topic)
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=topic.concept_id,
                        reason=(
                            "Historical change_role outcome already matches the "
                            "fresh Module 3 role; no edit or human review required."
                        ),
                    )
                )
                continue

            validation = self.reason_validator.validate(
                candidate=candidate,
                current_evidence=current_evidence,
            )

            if not validation.safe_for_automatic_reuse:
                output_topics.append(topic)
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=topic.concept_id,
                        reason=(
                            "Reason validator did not authorize automatic "
                            f"reuse: {validation.decision}."
                        ),
                    )
                )
                continue

            try:
                changed = self._apply_existing_topic_candidate(
                    topic=topic,
                    candidate=candidate,
                    validation=validation,
                )
            except ValueError as exc:
                output_topics.append(topic)
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=topic.concept_id,
                        reason=f"Malformed edit memory: {exc}",
                    )
                )
                continue

            if changed is not None:
                output_topics.append(changed)

            applied.append(
                AppliedOverlayEdit(
                    memory_id=candidate.memory_id,
                    action=candidate.edit_action,
                    source_concept_id=topic.concept_id,
                    target_concept_id=(
                        changed.concept_id
                        if changed is not None
                        else None
                    ),
                    explanation=validation.explanation,
                )
            )

        # --------------------------------------------------------------
        # B. Add-topic edits
        # --------------------------------------------------------------
        present_concepts = {
            topic.concept_id
            for topic in output_topics
        }

        addition_candidates = list(
            self.candidate_provider.candidates_for_additions(
                spec_version=spec_version,
                current_chunk_evidence=current_chunk_evidence,
                already_present_concept_ids=tuple(
                    sorted(present_concepts)
                ),
            )
        )

        # Fail closed on duplicate candidate targets.
        target_counts: dict[str, int] = {}
        for candidate in addition_candidates:
            if candidate.target_concept_id:
                target_counts[candidate.target_concept_id] = (
                    target_counts.get(
                        candidate.target_concept_id,
                        0,
                    )
                    + 1
                )

        for candidate in addition_candidates:
            if not self._candidate_spec_matches(
                candidate,
                spec_version=spec_version,
            ):
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=None,
                        reason="Add candidate spec_version does not match.",
                    )
                )
                continue

            if candidate.edit_action != "add_topic":
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=None,
                        reason=(
                            "Non-add action supplied by addition provider."
                        ),
                    )
                )
                continue

            if not candidate.target_concept_id:
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=None,
                        reason="add_topic has no target_concept_id.",
                    )
                )
                continue

            if target_counts.get(
                candidate.target_concept_id,
                0,
            ) > 1:
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=None,
                        reason=(
                            "Multiple add memories target the same concept; "
                            "automatic overlay abstained."
                        ),
                    )
                )
                continue

            if candidate.target_concept_id in present_concepts:
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=None,
                        reason=(
                            "Target topic is already present; duplicate add "
                            "was skipped."
                        ),
                    )
                )
                continue

            if not candidate.target_topic:
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=None,
                        reason="add_topic has no target_topic.",
                    )
                )
                continue

            if candidate.target_role not in VALID_ROLES:
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=None,
                        reason="add_topic has an invalid target role.",
                    )
                )
                continue

            current_evidence = "\n".join(
                str(text).strip()
                for text in current_chunk_evidence
                if str(text).strip()
            )

            if not current_evidence:
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=None,
                        reason=(
                            "No current chunk evidence was supplied for "
                            "add-topic validation."
                        ),
                    )
                )
                continue

            validation = self.reason_validator.validate(
                candidate=candidate,
                current_evidence=current_evidence,
            )

            if not validation.safe_for_automatic_reuse:
                skipped.append(
                    SkippedOverlayEdit(
                        memory_id=candidate.memory_id,
                        action=candidate.edit_action,
                        source_concept_id=None,
                        reason=(
                            "Reason validator did not authorize add_topic: "
                            f"{validation.decision}."
                        ),
                    )
                )
                continue

            new_topic = OverlayTopic(
                concept_id=candidate.target_concept_id,
                topic=candidate.target_topic,
                role=candidate.target_role,
                official_reference=(
                    candidate.target_official_reference
                ),
                confidence=None,
                ranking_score=None,
                source_chunk_ids=tuple(
                    candidate.current_source_chunk_ids
                ),
                memory_applied=True,
                memory_id=candidate.memory_id,
                memory_action="add_topic",
            )

            output_topics.append(new_topic)
            present_concepts.add(
                new_topic.concept_id
            )

            applied.append(
                AppliedOverlayEdit(
                    memory_id=candidate.memory_id,
                    action="add_topic",
                    source_concept_id=None,
                    target_concept_id=new_topic.concept_id,
                    explanation=validation.explanation,
                )
            )

        return OverlayResult(
            topics=tuple(output_topics),
            applied=tuple(applied),
            skipped=tuple(skipped),
        )