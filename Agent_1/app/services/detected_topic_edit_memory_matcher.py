from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.db.models.detected_topic_edit_memory import DetectedTopicEditMemory


class EditMemoryRepositoryProtocol(Protocol):
    def list_reusable(
        self,
        *,
        spec_version: str,
        source_concept_id: str | None = None,
        edit_actions: Sequence[str] | None = None,
        limit: int = 100,
    ) -> list[DetectedTopicEditMemory]:
        ...

    def list_reusable_additions(
        self,
        *,
        spec_version: str,
        limit: int = 100,
    ) -> list[DetectedTopicEditMemory]:
        ...


class TextEmbedderProtocol(Protocol):
    """
    Minimal adapter interface.

    Step 3 deliberately does not import or modify the project's existing
    embedding service. Later wiring can adapt the existing embedder to this
    interface without changing the matcher.
    """

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        ...


@dataclass(frozen=True, slots=True)
class EditMemoryMatchConfig:
    """
    Conservative defaults for final-topic edit reuse.

    These thresholds are separate from all existing Agent 1 thresholds.

    Precision is intentionally preferred over recall:
    a false automatic edit is more harmful than a safe memory miss.
    """

    # remove_topic / replace_topic
    standard_similarity_threshold: float = 0.90

    # Role changes are more contextual, so require a stronger match.
    role_change_similarity_threshold: float = 0.92

    # Added topics have no source concept anchor, so they are strictest.
    add_topic_similarity_threshold: float = 0.94

    # If two different outcomes are both strong and too close, abstain.
    ambiguity_margin: float = 0.03

    # Do not semantically reuse tiny evidence fragments.
    minimum_evidence_characters: int = 40

    max_candidates: int = 100


@dataclass(frozen=True, slots=True)
class ContextualEditMemoryMatch:
    memory_id: int
    edit_action: str

    source_concept_id: str | None
    source_topic: str | None
    source_role: str | None

    target_concept_id: str | None
    target_topic: str | None
    target_role: str | None

    reviewer_reason: str
    evidence_similarity: float

    match_type: str
    # "exact_evidence" or "semantic_context"


@dataclass(frozen=True, slots=True)
class ContextualEditMatchResult:
    status: str
    # hit | miss | ambiguous

    match: ContextualEditMemoryMatch | None

    candidates_evaluated: int
    strong_candidates: int

    best_similarity: float | None
    reason: str


class DetectedTopicEditMemoryMatcher:
    """
    Conservative contextual matcher for reviewer-approved final-topic edits.

    IMPORTANT:
    - It never modifies Module 3 output.
    - It never writes to PostgreSQL.
    - It never marks a memory as used.
    - It never changes existing Agent 1 thresholds.
    - It never calls Qdrant or Groq.
    - It contains no topic-specific rules.

    Step 3 only answers:
        "Is there one unambiguous, strongly compatible edit memory?"

    Applying the returned edit is deliberately postponed to the wiring step.
    """

    def __init__(
        self,
        *,
        repository: EditMemoryRepositoryProtocol,
        embedder: TextEmbedderProtocol,
        config: EditMemoryMatchConfig | None = None,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.config = config or EditMemoryMatchConfig()

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        if value is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            str(value).strip().casefold(),
        )

    @classmethod
    def _evidence_hash(cls, value: str) -> str:
        normalized = cls._normalize_text(value)
        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _cosine_similarity(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        if len(left) != len(right):
            raise ValueError(
                "Embedding vectors must have the same dimension."
            )

        if not left:
            return 0.0

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

        similarity = dot / (left_norm * right_norm)

        # Numerical safety only.
        return max(-1.0, min(1.0, float(similarity)))

    @staticmethod
    def _outcome_key(
        memory: DetectedTopicEditMemory,
    ) -> tuple[
        str,
        str | None,
        str | None,
    ]:
        """
        Two memories are considered the same outcome only when they would
        perform the same action and produce the same target concept/role.
        """

        return (
            str(memory.edit_action),
            memory.target_concept_id,
            memory.target_role,
        )

    def _threshold_for_action(
        self,
        action: str,
    ) -> float:
        if action == "add_topic":
            return self.config.add_topic_similarity_threshold

        if action == "change_role":
            return self.config.role_change_similarity_threshold

        return self.config.standard_similarity_threshold

    def _to_match(
        self,
        memory: DetectedTopicEditMemory,
        *,
        similarity: float,
        match_type: str,
    ) -> ContextualEditMemoryMatch:
        return ContextualEditMemoryMatch(
            memory_id=int(memory.id),
            edit_action=memory.edit_action,
            source_concept_id=memory.source_concept_id,
            source_topic=memory.source_topic,
            source_role=memory.source_role,
            target_concept_id=memory.target_concept_id,
            target_topic=memory.target_topic,
            target_role=memory.target_role,
            reviewer_reason=memory.reviewer_reason,
            evidence_similarity=float(similarity),
            match_type=match_type,
        )

    def _evaluate_candidates(
        self,
        *,
        new_evidence_text: str,
        candidates: Sequence[DetectedTopicEditMemory],
    ) -> ContextualEditMatchResult:
        normalized_new = self._normalize_text(new_evidence_text)

        if (
            len(normalized_new)
            < self.config.minimum_evidence_characters
        ):
            return ContextualEditMatchResult(
                status="miss",
                match=None,
                candidates_evaluated=0,
                strong_candidates=0,
                best_similarity=None,
                reason=(
                    "Current evidence is too short for safe contextual "
                    "edit-memory reuse."
                ),
            )

        if not candidates:
            return ContextualEditMatchResult(
                status="miss",
                match=None,
                candidates_evaluated=0,
                strong_candidates=0,
                best_similarity=None,
                reason="No compatible reusable edit memories were found.",
            )

        new_hash = self._evidence_hash(normalized_new)

        exact = [
            memory
            for memory in candidates
            if memory.evidence_hash == new_hash
        ]

        if exact:
            exact_outcomes = {
                self._outcome_key(memory)
                for memory in exact
            }

            if len(exact_outcomes) != 1:
                return ContextualEditMatchResult(
                    status="ambiguous",
                    match=None,
                    candidates_evaluated=len(candidates),
                    strong_candidates=len(exact),
                    best_similarity=1.0,
                    reason=(
                        "Conflicting reviewer-approved edit memories exist "
                        "for the same exact evidence; automatic reuse abstained."
                    ),
                )

            # Multiple exact memories with the same outcome are harmless.
            # Prefer the most recently returned candidate; repository order
            # is expected to be newest first.
            memory = exact[0]

            return ContextualEditMatchResult(
                status="hit",
                match=self._to_match(
                    memory,
                    similarity=1.0,
                    match_type="exact_evidence",
                ),
                candidates_evaluated=len(candidates),
                strong_candidates=len(exact),
                best_similarity=1.0,
                reason=(
                    "Reviewer-approved edit memory matched the exact "
                    "normalized evidence."
                ),
            )

        texts = [
            normalized_new,
            *[
                self._normalize_text(memory.evidence_text)
                for memory in candidates
            ],
        ]

        vectors = list(self.embedder.embed_texts(texts))

        if len(vectors) != len(texts):
            raise ValueError(
                "Embedder returned an unexpected number of vectors."
            )

        query_vector = vectors[0]

        scored: list[
            tuple[
                DetectedTopicEditMemory,
                float,
            ]
        ] = []

        for memory, vector in zip(candidates, vectors[1:]):
            similarity = self._cosine_similarity(
                query_vector,
                vector,
            )
            scored.append((memory, similarity))

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        strong = [
            (memory, similarity)
            for memory, similarity in scored
            if similarity
            >= self._threshold_for_action(memory.edit_action)
        ]

        best_similarity = (
            scored[0][1]
            if scored
            else None
        )

        if not strong:
            return ContextualEditMatchResult(
                status="miss",
                match=None,
                candidates_evaluated=len(candidates),
                strong_candidates=0,
                best_similarity=best_similarity,
                reason=(
                    "No reviewer-approved edit memory passed the "
                    "conservative contextual similarity threshold."
                ),
            )

        best_memory, best_score = strong[0]
        best_outcome = self._outcome_key(best_memory)

        competing = [
            (memory, score)
            for memory, score in strong[1:]
            if (
                self._outcome_key(memory) != best_outcome
                and (best_score - score)
                <= self.config.ambiguity_margin
            )
        ]

        if competing:
            return ContextualEditMatchResult(
                status="ambiguous",
                match=None,
                candidates_evaluated=len(candidates),
                strong_candidates=len(strong),
                best_similarity=best_score,
                reason=(
                    "Multiple strong reviewer-approved edit memories "
                    "support different outcomes within the ambiguity margin; "
                    "automatic reuse abstained."
                ),
            )

        return ContextualEditMatchResult(
            status="hit",
            match=self._to_match(
                best_memory,
                similarity=best_score,
                match_type="semantic_context",
            ),
            candidates_evaluated=len(candidates),
            strong_candidates=len(strong),
            best_similarity=best_score,
            reason=(
                "One unambiguous reviewer-approved edit memory passed "
                "the conservative contextual similarity gate."
            ),
        )

    def match_existing_topic_edit(
        self,
        *,
        spec_version: str,
        source_concept_id: str,
        current_evidence_text: str,
        allowed_actions: Sequence[str] = (
            "remove_topic",
            "replace_topic",
            "change_role",
        ),
    ) -> ContextualEditMatchResult:
        """
        Match an edit for a topic Module 3 already detected.

        Hard guards happen before semantic matching:
        - same specification version
        - same official source concept
        - only allowed edit actions
        - repository already restricts to approved + active + human-validated
        """

        source_concept_id = str(source_concept_id).strip()
        spec_version = str(spec_version).strip()

        if not source_concept_id:
            raise ValueError(
                "source_concept_id is required for an existing-topic edit."
            )

        if not spec_version:
            raise ValueError(
                "spec_version is required for edit-memory matching."
            )

        candidates = self.repository.list_reusable(
            spec_version=spec_version,
            source_concept_id=source_concept_id,
            edit_actions=allowed_actions,
            limit=self.config.max_candidates,
        )

        return self._evaluate_candidates(
            new_evidence_text=current_evidence_text,
            candidates=candidates,
        )

    def match_add_topic_memories(
        self,
        *,
        spec_version: str,
        current_chunk_evidence: Sequence[str],
        already_present_concept_ids: Sequence[str] = (),
    ) -> list[ContextualEditMemoryMatch]:
        """
        Find safe human-added topics for the current transcript.

        Addition memory is intentionally stricter because there is no source
        concept anchor. Every returned addition must:
        - be same spec
        - be approved + active + human-validated
        - have a target concept not already present
        - strongly match at least one current chunk
        - be unambiguous for that target concept

        The matcher only RETURNS candidate additions; it does not apply them.
        """

        spec_version = str(spec_version).strip()

        if not spec_version:
            raise ValueError(
                "spec_version is required for add-topic memory matching."
            )

        chunk_texts = [
            str(text).strip()
            for text in current_chunk_evidence
            if len(self._normalize_text(text))
            >= self.config.minimum_evidence_characters
        ]

        if not chunk_texts:
            return []

        present = {
            str(concept_id).strip()
            for concept_id in already_present_concept_ids
            if str(concept_id).strip()
        }

        candidates = self.repository.list_reusable_additions(
            spec_version=spec_version,
            limit=self.config.max_candidates,
        )

        candidates = [
            memory
            for memory in candidates
            if (
                memory.target_concept_id
                and memory.target_concept_id not in present
            )
        ]

        if not candidates:
            return []

        # Group by target concept so one concept cannot be added twice.
        by_target: dict[
            str,
            list[DetectedTopicEditMemory],
        ] = {}

        for memory in candidates:
            by_target.setdefault(
                str(memory.target_concept_id),
                [],
            ).append(memory)

        matches: list[ContextualEditMemoryMatch] = []

        for target_concept_id, memories in by_target.items():
            # Evaluate each chunk separately and keep the strongest safe hit.
            best: ContextualEditMemoryMatch | None = None

            for chunk_text in chunk_texts:
                result = self._evaluate_candidates(
                    new_evidence_text=chunk_text,
                    candidates=memories,
                )

                if result.status != "hit" or result.match is None:
                    continue

                if (
                    best is None
                    or result.match.evidence_similarity
                    > best.evidence_similarity
                ):
                    best = result.match

            if best is not None:
                matches.append(best)

        matches.sort(
            key=lambda match: match.evidence_similarity,
            reverse=True,
        )

        return matches
