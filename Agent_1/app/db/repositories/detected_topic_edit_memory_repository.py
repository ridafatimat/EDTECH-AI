from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.detected_topic_edit_memory import (
    DetectedTopicEditMemory,
)


class DetectedTopicEditMemoryRepository:
    """
    Persistence-only repository for final-topic edit memory.

    This class deliberately does not decide whether a memory is semantically
    compatible with a new transcript. Compatibility belongs to the service
    layer so database access and matching policy remain separate.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, memory_id: int) -> DetectedTopicEditMemory | None:
        return self.session.get(
            DetectedTopicEditMemory,
            int(memory_id),
        )

    def get_by_cache_key(
        self,
        cache_key: str,
    ) -> DetectedTopicEditMemory | None:
        statement = (
            select(DetectedTopicEditMemory)
            .where(
                DetectedTopicEditMemory.cache_key == str(cache_key)
            )
            .limit(1)
        )
        return self.session.execute(statement).scalar_one_or_none()

    def upsert(
        self,
        values: Mapping[str, Any],
    ) -> DetectedTopicEditMemory:
        """
        Insert a new memory or refresh the same exact reviewed edit.

        ``cache_key`` defines identity. Repeated review of the same edit/evidence
        does not create uncontrolled duplicate reusable memories.
        """

        cache_key = str(values["cache_key"])
        record = self.get_by_cache_key(cache_key)

        if record is None:
            record = DetectedTopicEditMemory(**dict(values))
            self.session.add(record)
        else:
            protected_fields = {
                "id",
                "created_at",
                "hit_count",
                "last_used_at",
            }

            for field_name, value in values.items():
                if field_name in protected_fields:
                    continue
                setattr(record, field_name, value)

            record.updated_at = datetime.now(timezone.utc)

        self.session.flush()
        return record

    def list_reusable(
        self,
        *,
        spec_version: str,
        source_concept_id: str | None = None,
        edit_actions: Iterable[str] | None = None,
        limit: int = 100,
    ) -> list[DetectedTopicEditMemory]:
        """
        Return reviewer-approved, active memories for compatibility evaluation.

        This method only performs hard safety filtering:
        - same spec version
        - reviewer approved
        - active
        - human validated

        It does NOT declare a semantic memory hit.
        """

        statement = select(DetectedTopicEditMemory).where(
            DetectedTopicEditMemory.spec_version == str(spec_version),
            DetectedTopicEditMemory.reviewer_approved.is_(True),
            DetectedTopicEditMemory.is_active.is_(True),
            DetectedTopicEditMemory.validation_status == "human_validated",
        )

        if source_concept_id is not None:
            statement = statement.where(
                DetectedTopicEditMemory.source_concept_id
                == str(source_concept_id)
            )

        if edit_actions is not None:
            actions = tuple(
                str(action).strip()
                for action in edit_actions
                if str(action).strip()
            )
            if not actions:
                return []
            statement = statement.where(
                DetectedTopicEditMemory.edit_action.in_(actions)
            )

        statement = (
            statement
            .order_by(
                DetectedTopicEditMemory.updated_at.desc(),
                DetectedTopicEditMemory.id.desc(),
            )
            .limit(max(1, int(limit)))
        )

        return list(self.session.execute(statement).scalars().all())

    def list_reusable_additions(
        self,
        *,
        spec_version: str,
        limit: int = 100,
    ) -> list[DetectedTopicEditMemory]:
        """
        Addition memories cannot be filtered by a source concept because the
        source topic was missing. They are still restricted to same spec,
        approved, active, human-validated records.
        """

        return self.list_reusable(
            spec_version=spec_version,
            edit_actions=("add_topic",),
            limit=limit,
        )

    def mark_used(
        self,
        memory_id: int,
    ) -> DetectedTopicEditMemory:
        record = self.get(memory_id)
        if record is None:
            raise LookupError(
                f"Detected-topic edit memory {memory_id} was not found."
            )

        record.hit_count = int(record.hit_count or 0) + 1
        record.last_used_at = datetime.now(timezone.utc)
        record.updated_at = datetime.now(timezone.utc)

        self.session.flush()
        return record

    def set_active(
        self,
        memory_id: int,
        *,
        is_active: bool,
    ) -> DetectedTopicEditMemory:
        record = self.get(memory_id)
        if record is None:
            raise LookupError(
                f"Detected-topic edit memory {memory_id} was not found."
            )

        record.is_active = bool(is_active)
        record.updated_at = datetime.now(timezone.utc)

        self.session.flush()
        return record
