from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, text

from app.db.models.detected_topic_edit_memory import DetectedTopicEditMemory
from app.db.session import session_scope


VALID_REUSE_DECISIONS = frozenset({"approve_reuse", "reject_reuse"})


@dataclass(frozen=True, slots=True)
class ReuseFeedbackDecision:
    memory_id: int
    decision: str
    reviewer_reason: str
    current_evidence_hash: str
    pipeline_run_id: str | None
    source_transcript: str | None
    source_concept_id: str | None
    spec_version: str
    reviewed_by: str | None


class DetectedTopicEditReuseFeedbackStore:
    """
    Explicit human decisions for ambiguous/conflicting final-topic edit memory.

    This store does NOT replace detected_topic_edit_memory. It records whether a
    previously reviewer-approved memory should or should not be reused for the
    *current evidence context*.

    Safety policy:
    - approve_reuse applies only to the exact normalized current evidence hash;
    - reject_reuse suppresses that historical memory only for the exact current
      evidence hash;
    - no broad semantic generalisation is introduced here.
    """

    def __init__(self) -> None:
        self._schema_ready = False

    @staticmethod
    def normalize_evidence(value: str | None) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().casefold())

    @classmethod
    def evidence_hash(cls, value: str | None) -> str:
        normalized = cls.normalize_evidence(value)
        if not normalized:
            raise ValueError("Reuse feedback requires current transcript evidence.")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return

        ddl = text(
            """
            CREATE TABLE IF NOT EXISTS detected_topic_edit_reuse_feedback (
                id BIGSERIAL PRIMARY KEY,
                memory_id BIGINT NOT NULL,
                pipeline_run_id TEXT,
                source_transcript TEXT,
                source_concept_id TEXT,
                current_evidence_hash TEXT NOT NULL,
                current_evidence_text TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewer_reason TEXT NOT NULL,
                spec_version TEXT NOT NULL,
                reviewed_by TEXT,
                reviewed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT detected_topic_edit_reuse_feedback_decision_ck
                    CHECK (decision IN ('approve_reuse', 'reject_reuse')),
                CONSTRAINT detected_topic_edit_reuse_feedback_context_uq
                    UNIQUE (memory_id, current_evidence_hash, spec_version)
            )
            """
        )

        index_sql = text(
            """
            CREATE INDEX IF NOT EXISTS
                ix_detected_topic_edit_reuse_feedback_run
            ON detected_topic_edit_reuse_feedback (pipeline_run_id, reviewed_at DESC)
            """
        )

        with session_scope() as session:
            session.execute(ddl)
            session.execute(index_sql)

        self._schema_ready = True

    def record(
        self,
        *,
        memory_id: int,
        current_evidence: str,
        decision: str,
        reviewer_reason: str,
        spec_version: str,
        pipeline_run_id: str | None = None,
        source_transcript: str | None = None,
        source_concept_id: str | None = None,
        reviewed_by: str | None = "streamlit",
    ) -> ReuseFeedbackDecision:
        normalized_decision = str(decision or "").strip().casefold()
        if normalized_decision not in VALID_REUSE_DECISIONS:
            raise ValueError(
                "Reuse feedback decision must be 'approve_reuse' or 'reject_reuse'."
            )

        reason = str(reviewer_reason or "").strip()
        if not reason:
            raise ValueError("A reviewer reason is required for reuse feedback.")

        spec = str(spec_version or "").strip()
        if not spec:
            raise ValueError("spec_version is required for reuse feedback.")

        evidence_text = str(current_evidence or "").strip()
        evidence_hash = self.evidence_hash(evidence_text)

        self._ensure_schema()

        statement = text(
            """
            INSERT INTO detected_topic_edit_reuse_feedback (
                memory_id,
                pipeline_run_id,
                source_transcript,
                source_concept_id,
                current_evidence_hash,
                current_evidence_text,
                decision,
                reviewer_reason,
                spec_version,
                reviewed_by,
                reviewed_at,
                updated_at
            ) VALUES (
                :memory_id,
                :pipeline_run_id,
                :source_transcript,
                :source_concept_id,
                :current_evidence_hash,
                :current_evidence_text,
                :decision,
                :reviewer_reason,
                :spec_version,
                :reviewed_by,
                NOW(),
                NOW()
            )
            ON CONFLICT (memory_id, current_evidence_hash, spec_version)
            DO UPDATE SET
                pipeline_run_id = EXCLUDED.pipeline_run_id,
                source_transcript = EXCLUDED.source_transcript,
                source_concept_id = EXCLUDED.source_concept_id,
                current_evidence_text = EXCLUDED.current_evidence_text,
                decision = EXCLUDED.decision,
                reviewer_reason = EXCLUDED.reviewer_reason,
                reviewed_by = EXCLUDED.reviewed_by,
                reviewed_at = NOW(),
                updated_at = NOW()
            """
        )

        with session_scope() as session:
            session.execute(
                statement,
                {
                    "memory_id": int(memory_id),
                    "pipeline_run_id": str(pipeline_run_id or "").strip() or None,
                    "source_transcript": str(source_transcript or "").strip() or None,
                    "source_concept_id": str(source_concept_id or "").strip() or None,
                    "current_evidence_hash": evidence_hash,
                    "current_evidence_text": evidence_text,
                    "decision": normalized_decision,
                    "reviewer_reason": reason,
                    "spec_version": spec,
                    "reviewed_by": str(reviewed_by or "").strip() or None,
                },
            )

        return ReuseFeedbackDecision(
            memory_id=int(memory_id),
            decision=normalized_decision,
            reviewer_reason=reason,
            current_evidence_hash=evidence_hash,
            pipeline_run_id=str(pipeline_run_id or "").strip() or None,
            source_transcript=str(source_transcript or "").strip() or None,
            source_concept_id=str(source_concept_id or "").strip() or None,
            spec_version=spec,
            reviewed_by=str(reviewed_by or "").strip() or None,
        )

    def get_decision(
        self,
        *,
        memory_id: int,
        current_evidence: str,
        spec_version: str,
    ) -> ReuseFeedbackDecision | None:
        evidence_hash = self.evidence_hash(current_evidence)
        spec = str(spec_version or "").strip()
        self._ensure_schema()

        statement = text(
            """
            SELECT
                memory_id,
                decision,
                reviewer_reason,
                current_evidence_hash,
                pipeline_run_id,
                source_transcript,
                source_concept_id,
                spec_version,
                reviewed_by
            FROM detected_topic_edit_reuse_feedback
            WHERE memory_id = :memory_id
              AND current_evidence_hash = :current_evidence_hash
              AND spec_version = :spec_version
            LIMIT 1
            """
        )

        with session_scope() as session:
            row = session.execute(
                statement,
                {
                    "memory_id": int(memory_id),
                    "current_evidence_hash": evidence_hash,
                    "spec_version": spec,
                },
            ).mappings().first()

        if row is None:
            return None

        return ReuseFeedbackDecision(
            memory_id=int(row["memory_id"]),
            decision=str(row["decision"]),
            reviewer_reason=str(row["reviewer_reason"]),
            current_evidence_hash=str(row["current_evidence_hash"]),
            pipeline_run_id=row.get("pipeline_run_id"),
            source_transcript=row.get("source_transcript"),
            source_concept_id=row.get("source_concept_id"),
            spec_version=str(row["spec_version"]),
            reviewed_by=row.get("reviewed_by"),
        )

    @staticmethod
    def _memory_to_dict(record: DetectedTopicEditMemory) -> dict[str, Any]:
        return {
            "memory_id": int(record.id),
            "edit_action": record.edit_action,
            "source_concept_id": record.source_concept_id,
            "source_topic": record.source_topic,
            "source_role": record.source_role,
            "target_concept_id": record.target_concept_id,
            "target_topic": record.target_topic,
            "target_role": record.target_role,
            "reviewer_reason": record.reviewer_reason,
            "stored_evidence": record.evidence_text,
            "source_chunk_ids": list(record.source_chunk_ids or []),
            "spec_version": record.spec_version,
            "reviewed_by": record.reviewed_by,
            "reviewed_at": (
                record.reviewed_at.isoformat()
                if getattr(record, "reviewed_at", None) is not None
                else None
            ),
        }

    def memory_snapshot(self, memory_id: int) -> dict[str, Any] | None:
        with session_scope() as session:
            record = session.get(DetectedTopicEditMemory, int(memory_id))
            if record is None:
                return None
            return self._memory_to_dict(record)

    def reusable_memories_for_source(
        self,
        *,
        source_concept_id: str,
        spec_version: str,
    ) -> list[dict[str, Any]]:
        with session_scope() as session:
            statement = (
                select(DetectedTopicEditMemory)
                .where(
                    DetectedTopicEditMemory.source_concept_id
                    == str(source_concept_id),
                    DetectedTopicEditMemory.spec_version == str(spec_version),
                    DetectedTopicEditMemory.reviewer_approved.is_(True),
                    DetectedTopicEditMemory.is_active.is_(True),
                    DetectedTopicEditMemory.validation_status == "human_validated",
                )
                .order_by(DetectedTopicEditMemory.id.desc())
            )
            records = session.execute(statement).scalars().all()
            return [self._memory_to_dict(record) for record in records]


    def feedback_for_evidence(
        self,
        *,
        current_evidence: str,
        spec_version: str,
    ) -> list[dict[str, Any]]:
        """Return latest explicit decision per memory for an exact context.

        This is used by the Streamlit layer to bridge a stable lesson-context
        fingerprint onto the backend's current evidence representation.  It
        does not broaden reuse: the same exact normalized context hash and
        spec_version are still required.
        """
        evidence_hash = self.evidence_hash(current_evidence)
        spec = str(spec_version or "").strip()
        self._ensure_schema()
        statement = text(
            """
            SELECT
                id,
                memory_id,
                pipeline_run_id,
                source_transcript,
                source_concept_id,
                current_evidence_hash,
                decision,
                reviewer_reason,
                spec_version,
                reviewed_by,
                reviewed_at
            FROM detected_topic_edit_reuse_feedback
            WHERE current_evidence_hash = :current_evidence_hash
              AND spec_version = :spec_version
            ORDER BY reviewed_at DESC, id DESC
            """
        )
        with session_scope() as session:
            rows = session.execute(
                statement,
                {
                    "current_evidence_hash": evidence_hash,
                    "spec_version": spec,
                },
            ).mappings().all()

        output: list[dict[str, Any]] = []
        seen: set[int] = set()
        for row in rows:
            try:
                memory_id = int(row["memory_id"])
            except (TypeError, ValueError, KeyError):
                continue
            if memory_id in seen:
                continue
            seen.add(memory_id)
            output.append(dict(row))
        return output

    def reusable_add_memories(
        self,
        *,
        spec_version: str,
    ) -> list[dict[str, Any]]:
        """Return active reviewer-approved human add-topic memories."""
        with session_scope() as session:
            statement = (
                select(DetectedTopicEditMemory)
                .where(
                    DetectedTopicEditMemory.edit_action == "add_topic",
                    DetectedTopicEditMemory.spec_version == str(spec_version),
                    DetectedTopicEditMemory.reviewer_approved.is_(True),
                    DetectedTopicEditMemory.is_active.is_(True),
                    DetectedTopicEditMemory.validation_status == "human_validated",
                )
                .order_by(DetectedTopicEditMemory.id.desc())
            )
            records = session.execute(statement).scalars().all()
            return [self._memory_to_dict(record) for record in records]

    def feedback_for_run(self, pipeline_run_id: str) -> list[dict[str, Any]]:
        """Return the latest explicit decision per memory for one pipeline run.

        A previous UI version may have written the same human decision against
        more than one evidence hash while evidence canonicalization was being
        fixed. Current-run authority is per memory, so downstream UI/runtime
        helpers should see only the newest explicit decision for that memory.
        Historical rows remain in PostgreSQL for audit.
        """
        self._ensure_schema()
        statement = text(
            """
            SELECT
                id,
                memory_id,
                source_concept_id,
                decision,
                reviewer_reason,
                current_evidence_hash,
                spec_version,
                reviewed_by,
                reviewed_at
            FROM detected_topic_edit_reuse_feedback
            WHERE pipeline_run_id = :pipeline_run_id
            ORDER BY reviewed_at DESC, id DESC
            """
        )
        with session_scope() as session:
            rows = session.execute(
                statement,
                {"pipeline_run_id": str(pipeline_run_id)},
            ).mappings().all()

        output: list[dict[str, Any]] = []
        seen_memory_ids: set[int] = set()
        for row in rows:
            try:
                memory_id = int(row["memory_id"])
            except (TypeError, ValueError, KeyError):
                continue
            if memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(memory_id)
            output.append(dict(row))

        return output