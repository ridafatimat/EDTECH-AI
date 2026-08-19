from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence

from app.db.models.detected_topic_edit_memory import (
    DetectedTopicEditMemory,
)
from app.db.repositories.detected_topic_edit_memory_repository import (
    DetectedTopicEditMemoryRepository,
)


VALID_EDIT_ACTIONS = frozenset(
    {
        "remove_topic",
        "add_topic",
        "replace_topic",
        "change_role",
    }
)

VALID_TOPIC_ROLES = frozenset(
    {
        "primary",
        "supporting",
    }
)


@dataclass(frozen=True, slots=True)
class DetectedTopicEdit:
    """
    One reviewer-approved final-topic edit.

    This structure is independent of Streamlit and Module 3 so the memory
    layer can be tested without changing the working pipeline.
    """

    edit_action: str

    source_concept_id: str | None = None
    source_topic: str | None = None
    source_role: str | None = None

    target_concept_id: str | None = None
    target_topic: str | None = None
    target_role: str | None = None

    evidence_text: str = ""
    source_chunk_ids: tuple[int, ...] = ()

    reviewer_reason: str = ""

    source_transcript: str | None = None
    spec_version: str = ""

    reviewed_by: str | None = None


@dataclass(frozen=True, slots=True)
class ExactEditMemoryMatch:
    """
    Conservative exact-evidence match.

    This is deliberately NOT the future semantic/paraphrase matcher.
    It is useful for isolated Step 2 validation and is safe because both
    concept/spec filters and the exact evidence hash must agree.
    """

    memory_id: int
    edit_action: str
    source_concept_id: str | None
    target_concept_id: str | None
    source_role: str | None
    target_role: str | None
    reviewer_reason: str


class EditMemoryRepositoryProtocol(Protocol):
    def upsert(self, values: dict) -> DetectedTopicEditMemory:
        ...

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

    def mark_used(
        self,
        memory_id: int,
    ) -> DetectedTopicEditMemory:
        ...


class DetectedTopicEditMemoryService:
    """
    Validation and memory-record preparation for final-topic human edits.

    IMPORTANT:
    - This Step 2 service does not alter Module 3 output.
    - It does not change existing mapping-memory thresholds.
    - It does not call Qdrant or Groq.
    - It does not perform broad semantic reuse yet.

    The future contextual matching policy will be added and tested separately.
    """

    def __init__(
        self,
        repository: EditMemoryRepositoryProtocol,
    ) -> None:
        self.repository = repository

    @classmethod
    def normalize_text(
        cls,
        value: str | None,
    ) -> str:
        if value is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            str(value).strip().casefold(),
        )

    @classmethod
    def normalize_role(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = cls.normalize_text(value)

        if normalized not in VALID_TOPIC_ROLES:
            raise ValueError(
                "Topic role must be 'primary' or 'supporting'."
            )

        return normalized

    @classmethod
    def normalize_action(
        cls,
        value: str,
    ) -> str:
        normalized = cls.normalize_text(value)

        if normalized not in VALID_EDIT_ACTIONS:
            raise ValueError(
                "Unsupported detected-topic edit action: "
                f"{value!r}."
            )

        return normalized

    @classmethod
    def normalized_evidence(
        cls,
        evidence_text: str,
    ) -> str:
        return cls.normalize_text(evidence_text)

    @classmethod
    def evidence_hash(
        cls,
        evidence_text: str,
    ) -> str:
        normalized = cls.normalized_evidence(evidence_text)

        if not normalized:
            raise ValueError(
                "Detected-topic edit memory requires transcript evidence."
            )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def normalized_chunk_ids(
        values: Sequence[int],
    ) -> list[int]:
        return sorted(
            {
                int(value)
                for value in values
                if int(value) >= 0
            }
        )

    @classmethod
    def validate_edit(
        cls,
        edit: DetectedTopicEdit,
    ) -> DetectedTopicEdit:
        action = cls.normalize_action(edit.edit_action)

        source_concept_id = (
            str(edit.source_concept_id).strip()
            if edit.source_concept_id is not None
            else None
        )
        source_topic = (
            str(edit.source_topic).strip()
            if edit.source_topic is not None
            else None
        )
        target_concept_id = (
            str(edit.target_concept_id).strip()
            if edit.target_concept_id is not None
            else None
        )
        target_topic = (
            str(edit.target_topic).strip()
            if edit.target_topic is not None
            else None
        )

        source_role = cls.normalize_role(edit.source_role)
        target_role = cls.normalize_role(edit.target_role)

        evidence_text = str(edit.evidence_text or "").strip()
        reviewer_reason = str(edit.reviewer_reason or "").strip()
        spec_version = str(edit.spec_version or "").strip()

        if not evidence_text:
            raise ValueError(
                "A final-topic edit requires supporting transcript evidence."
            )

        if not reviewer_reason:
            raise ValueError(
                "A final-topic edit requires a reviewer reason."
            )

        if not spec_version:
            raise ValueError(
                "A final-topic edit requires the current spec_version."
            )

        if action == "remove_topic":
            if not source_concept_id or not source_topic:
                raise ValueError(
                    "remove_topic requires the original concept and topic."
                )

        elif action == "add_topic":
            if not target_concept_id or not target_topic:
                raise ValueError(
                    "add_topic requires the added official concept and topic."
                )
            if target_role is None:
                raise ValueError(
                    "add_topic requires the added topic role."
                )

        elif action == "replace_topic":
            if not source_concept_id or not source_topic:
                raise ValueError(
                    "replace_topic requires the original concept and topic."
                )
            if not target_concept_id or not target_topic:
                raise ValueError(
                    "replace_topic requires the replacement concept and topic."
                )
            if source_concept_id == target_concept_id:
                raise ValueError(
                    "replace_topic must change the official concept."
                )

        elif action == "change_role":
            if not source_concept_id or not source_topic:
                raise ValueError(
                    "change_role requires the original concept and topic."
                )
            if source_role is None or target_role is None:
                raise ValueError(
                    "change_role requires both old and new roles."
                )
            if source_role == target_role:
                raise ValueError(
                    "change_role requires a different target role."
                )

            # Role changes do not change the official concept.
            if target_concept_id is None:
                target_concept_id = source_concept_id
            if target_topic is None:
                target_topic = source_topic

            if target_concept_id != source_concept_id:
                raise ValueError(
                    "change_role cannot change the official concept."
                )

        return DetectedTopicEdit(
            edit_action=action,
            source_concept_id=source_concept_id,
            source_topic=source_topic,
            source_role=source_role,
            target_concept_id=target_concept_id,
            target_topic=target_topic,
            target_role=target_role,
            evidence_text=evidence_text,
            source_chunk_ids=tuple(
                cls.normalized_chunk_ids(edit.source_chunk_ids)
            ),
            reviewer_reason=reviewer_reason,
            source_transcript=(
                str(edit.source_transcript).strip()
                if edit.source_transcript is not None
                else None
            ),
            spec_version=spec_version,
            reviewed_by=(
                str(edit.reviewed_by).strip()
                if edit.reviewed_by is not None
                else None
            ),
        )

    @classmethod
    def build_cache_key(
        cls,
        edit: DetectedTopicEdit,
    ) -> str:
        validated = cls.validate_edit(edit)

        identity = {
            "edit_action": validated.edit_action,
            "source_concept_id": validated.source_concept_id,
            "source_role": validated.source_role,
            "target_concept_id": validated.target_concept_id,
            "target_role": validated.target_role,
            "evidence_hash": cls.evidence_hash(
                validated.evidence_text
            ),
            "spec_version": validated.spec_version,
        }

        serialized = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def remember(
        self,
        edit: DetectedTopicEdit,
    ) -> DetectedTopicEditMemory:
        """
        Persist a reviewer-approved edit.

        This method only stores validated human feedback. It does not apply the
        feedback to a current or future Module 3 result.
        """

        validated = self.validate_edit(edit)
        now = datetime.now(timezone.utc)

        values = {
            "cache_key": self.build_cache_key(validated),
            "edit_action": validated.edit_action,
            "source_concept_id": validated.source_concept_id,
            "source_topic": validated.source_topic,
            "source_role": validated.source_role,
            "target_concept_id": validated.target_concept_id,
            "target_topic": validated.target_topic,
            "target_role": validated.target_role,
            "evidence_hash": self.evidence_hash(
                validated.evidence_text
            ),
            "evidence_text": validated.evidence_text,
            "source_chunk_ids": list(validated.source_chunk_ids),
            "reviewer_reason": validated.reviewer_reason,
            "source_transcript": validated.source_transcript,
            "spec_version": validated.spec_version,
            "reviewed_by": validated.reviewed_by,
            "reviewed_at": now,
            "validation_status": "human_validated",
            "reviewer_approved": True,
            "is_active": True,
        }

        return self.repository.upsert(values)

    def find_exact_evidence_match(
        self,
        *,
        spec_version: str,
        evidence_text: str,
        source_concept_id: str | None,
        edit_actions: Sequence[str],
    ) -> ExactEditMemoryMatch | None:
        """
        Step 2's intentionally conservative lookup.

        An exact evidence hash is required. This proves the isolated storage
        layer works without introducing a broad semantic-reuse rule yet.
        """

        expected_hash = self.evidence_hash(evidence_text)

        candidates = self.repository.list_reusable(
            spec_version=str(spec_version).strip(),
            source_concept_id=source_concept_id,
            edit_actions=edit_actions,
            limit=100,
        )

        exact = [
            candidate
            for candidate in candidates
            if candidate.evidence_hash == expected_hash
        ]

        if len(exact) != 1:
            # Zero = no hit.
            # More than one = deliberately abstain rather than guess.
            return None

        candidate = exact[0]

        return ExactEditMemoryMatch(
            memory_id=int(candidate.id),
            edit_action=candidate.edit_action,
            source_concept_id=candidate.source_concept_id,
            target_concept_id=candidate.target_concept_id,
            source_role=candidate.source_role,
            target_role=candidate.target_role,
            reviewer_reason=candidate.reviewer_reason,
        )
