from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from dotenv import load_dotenv

from orchestration.state_resolver import resolve_agent1_state
from mcp_server.schemas.agent1 import (
    Agent2TopicApprovalRequest,
    DetectedTopicEditAction,
    DetectedTopicEditRequest,
    TopicReviewRequest,
)


_PENDING_STATUSES = {"candidate", "pending", "awaiting_review", "needs_review"}


class Agent1HitlAdapter:
    """Thin bridge to the existing Agent 1 HITL/self-improving implementation.

    The adapter deliberately does not reimplement topic mapping, memory matching,
    Qdrant retrieval, or detected-topic edit reuse. It reads the artifacts that
    Agent 1 already writes and calls the existing PostgreSQL/runtime paths.
    """

    def __init__(self, frontend_project_root: Path):
        self.frontend_project_root = Path(frontend_project_root).resolve()
        self.runs_root = self.frontend_project_root / "runs"
        self._edit_runtime_module: ModuleType | None = None
        self._edit_capture_module: ModuleType | None = None

    # ------------------------------------------------------------------
    # Path / JSON helpers
    # ------------------------------------------------------------------
    def _run_dir(self, run_id: str) -> Path:
        run_dir = self.runs_root / str(run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Agent 1 run directory was not found: {run_dir}")
        return run_dir

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read JSON artifact {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected a JSON object in {path}.")
        return value

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _artifact_paths(self, run_id: str) -> tuple[Path, str, Path, Path, Path]:
        run_dir = self._run_dir(run_id)
        snapshot = resolve_agent1_state(run_dir)
        transcript_name = str(snapshot.transcript_name or "").strip()
        if not transcript_name:
            raise RuntimeError("Could not resolve transcript name for this Agent 1 run.")

        output_dir = run_dir / "output" / transcript_name
        module2_path = output_dir / "02_chunking.json"
        module3_path = output_dir / "03_topic_mapping.json"
        if not module3_path.is_file():
            raise FileNotFoundError(f"Module 3 output was not found: {module3_path}")
        return run_dir, transcript_name, output_dir, module2_path, module3_path

    # ------------------------------------------------------------------
    # Existing runtime / backend loading
    # ------------------------------------------------------------------
    def _agent1_code_root(self) -> Path:
        root = self.frontend_project_root
        for candidate in (root, root.parent, root.parent.parent):
            if (candidate / "app" / "services").is_dir() and (candidate / "app" / "db").is_dir():
                return candidate
        raise RuntimeError(
            "Could not locate Agent_1/app from the Streamlit frontend project root."
        )

    def _prepare_agent1_imports(self) -> Path:
        code_root = self._agent1_code_root()
        code_root_text = str(code_root)
        sys.path[:] = [entry for entry in sys.path if str(entry) != code_root_text]
        sys.path.insert(0, code_root_text)
        load_dotenv(code_root / ".env", override=False)
        load_dotenv(self.frontend_project_root / ".env", override=False)
        return code_root

    def _load_edit_runtime(self) -> ModuleType:
        if self._edit_runtime_module is not None:
            return self._edit_runtime_module

        path = self.frontend_project_root / "frontend" / "detected_topic_edit_runtime.py"
        if not path.is_file():
            raise FileNotFoundError(
                f"Existing detected-topic edit runtime was not found: {path}"
            )
        spec = importlib.util.spec_from_file_location("agent1_existing_edit_runtime", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load existing detected-topic edit runtime: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._edit_runtime_module = module
        return module

    def _load_edit_capture(self) -> ModuleType:
        """Load the existing non-UI detected-topic edit persistence helper.

        We intentionally do NOT import frontend/app.py because importing the
        Streamlit app executes UI code. This helper module is already the
        backend-safe persistence entry point used by the current frontend.
        """

        if self._edit_capture_module is not None:
            return self._edit_capture_module

        path = self.frontend_project_root / "frontend" / "detected_topic_edit_capture.py"
        if not path.is_file():
            raise FileNotFoundError(
                f"Existing detected-topic edit capture helper was not found: {path}"
            )
        spec = importlib.util.spec_from_file_location("agent1_existing_edit_capture", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load existing detected-topic edit capture helper: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._edit_capture_module = module
        return module

    # ------------------------------------------------------------------
    # PostgreSQL review-state reconciliation
    # ------------------------------------------------------------------
    def get_review_status_reconciliation(self, run_id: str) -> dict[str, Any]:
        """Reconcile Module 3 review ids against live PostgreSQL.

        PostgreSQL is authoritative when it is reachable. If a historical JSON
        artifact references a review id that no longer exists in PostgreSQL,
        that item is marked as an integrity issue rather than being treated as
        a writable pending review. This prevents an old run from being mutated
        against a missing/recreated database row.
        """

        _, _, _, _, module3_path = self._artifact_paths(run_id)
        payload = self._read_json(module3_path)
        items = payload.get("topic_review_items") or []
        review_ids: list[int] = []
        invalid_item_count = 0
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    review_ids.append(int(item.get("id")))
                except (TypeError, ValueError):
                    invalid_item_count += 1

        if not review_ids:
            return {
                "statuses": {},
                "missing_ids": [],
                "invalid_item_count": invalid_item_count,
                "db_available": True,
            }

        self._prepare_agent1_imports()
        from app.db.repositories.topic_human_review_repository import (
            TopicHumanReviewRepository,
        )
        from app.db.session import session_scope

        statuses: dict[int, str] = {}
        missing_ids: list[int] = []
        with session_scope() as session:
            repo = TopicHumanReviewRepository(session)
            for review_id in review_ids:
                row = repo.get_by_id(review_id)
                if row is None:
                    missing_ids.append(int(review_id))
                else:
                    statuses[int(review_id)] = str(row.status).strip().casefold()

        return {
            "statuses": statuses,
            "missing_ids": sorted(set(missing_ids)),
            "invalid_item_count": invalid_item_count,
            "db_available": True,
        }

    def get_review_status_overrides(self, run_id: str) -> dict[int, str]:
        """Return DB-authoritative status overrides for controller state.

        Missing live rows are explicitly mapped to ``missing_db_row`` so the
        deterministic state resolver enters a fail-closed integrity state.
        """

        reconciliation = self.get_review_status_reconciliation(run_id)
        overrides = {
            int(review_id): str(status).strip().casefold()
            for review_id, status in (reconciliation.get("statuses") or {}).items()
        }
        for review_id in reconciliation.get("missing_ids") or []:
            overrides[int(review_id)] = "missing_db_row"
        return overrides

    # ------------------------------------------------------------------
    # Read-side tools
    # ------------------------------------------------------------------
    def get_effective_topics(self, run_id: str) -> dict[str, Any]:
        """Return the same effective topic list the current Streamlit UI uses.

        This includes the existing read-only detected-topic edit-memory overlay
        and saved human ADD operations, while leaving the raw Module 3 JSON
        untouched.
        """

        run_dir, transcript_name, _, module2_path, module3_path = self._artifact_paths(run_id)
        module2_json = self._read_json(module2_path) if module2_path.is_file() else {}
        module3_json = self._read_json(module3_path)
        raw_module3_result = module3_json.get("module3_result") or {}
        if not isinstance(raw_module3_result, dict):
            raw_module3_result = {}

        runtime = self._load_edit_runtime()
        runtime_result = runtime.apply_detected_topic_edit_runtime(
            module3_result_payload=raw_module3_result,
            module3_json=module3_json,
            run_dir=run_dir,
            transcript_name=transcript_name,
            frontend_project_root=self.frontend_project_root,
        )

        effective_module3 = runtime_result.get("module3_result") or raw_module3_result
        if not isinstance(effective_module3, dict):
            effective_module3 = dict(raw_module3_result)

        merged_topics = effective_module3.get("merged_topics") or []
        if not isinstance(merged_topics, list):
            merged_topics = []
        merged_topics = [dict(item) for item in merged_topics if isinstance(item, dict)]

        additions = module3_json.get("topic_output_additions") or []
        if isinstance(additions, list):
            present = {
                str(item.get("concept_id") or "").strip()
                for item in merged_topics
                if isinstance(item, dict)
            }
            for addition in additions:
                if not isinstance(addition, dict):
                    continue
                concept_id = str(addition.get("concept_id") or "").strip()
                if concept_id and concept_id not in present:
                    merged_topics.append(dict(addition))
                    present.add(concept_id)

        chunks = module2_json.get("chunks") or []
        if not isinstance(chunks, list):
            chunks = []

        return {
            "transcript_name": transcript_name,
            "topics": merged_topics,
            "topic_count": len(merged_topics),
            "chunks": [dict(item) for item in chunks if isinstance(item, dict)],
            "runtime": {
                key: value
                for key, value in runtime_result.items()
                if key != "module3_result"
            },
            "spec_version": (
                module3_json.get("spec_version")
                or module3_json.get("specification_version")
                or runtime_result.get("spec_version")
            ),
        }

    def get_pending_topic_reviews(self, run_id: str) -> dict[str, Any]:
        _, transcript_name, _, _, module3_path = self._artifact_paths(run_id)
        payload = self._read_json(module3_path)
        items = payload.get("topic_review_items") or []
        if not isinstance(items, list):
            items = []

        reconciliation_error: str | None = None
        db_available = False
        db_statuses: dict[int, str] = {}
        missing_ids: set[int] = set()
        try:
            reconciliation = self.get_review_status_reconciliation(run_id)
            db_statuses = {
                int(key): str(value).strip().casefold()
                for key, value in (reconciliation.get("statuses") or {}).items()
            }
            missing_ids = {int(value) for value in (reconciliation.get("missing_ids") or [])}
            db_available = bool(reconciliation.get("db_available", True))
        except Exception as exc:
            # DB unreachable: fail closed to the artifact. The human gate stays
            # active, but write methods still rely on a real DB row/UPDATE.
            reconciliation_error = f"{type(exc).__name__}: {exc}"

        pending: list[dict[str, Any]] = []
        resolved: list[dict[str, Any]] = []
        orphaned: list[dict[str, Any]] = []
        mismatches: list[dict[str, Any]] = []

        for raw in items:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            artifact_status = str(
                item.get("status") or item.get("review_status") or "pending"
            ).strip().casefold()
            try:
                review_id = int(item.get("id"))
            except (TypeError, ValueError):
                review_id = None

            if db_available and review_id is None:
                effective_status = "invalid_review_id"
                db_status = None
                status_source = "artifact_integrity_error"
                target = orphaned
            elif db_available and review_id in missing_ids:
                effective_status = "missing_db_row"
                db_status = None
                status_source = "postgresql_missing"
                target = orphaned
            else:
                db_status = db_statuses.get(review_id) if review_id is not None else None
                effective_status = str(db_status or artifact_status).strip().casefold()
                status_source = "postgresql" if db_status is not None else "artifact_fail_closed"
                target = pending if effective_status in _PENDING_STATUSES else resolved

            item["artifact_status"] = artifact_status
            item["db_status"] = db_status
            item["status"] = effective_status
            item["status_source"] = status_source

            if db_status is not None and db_status != artifact_status:
                mismatches.append(
                    {
                        "review_id": review_id,
                        "artifact_status": artifact_status,
                        "db_status": db_status,
                    }
                )
            elif db_available and target is orphaned:
                mismatches.append(
                    {
                        "review_id": review_id,
                        "artifact_status": artifact_status,
                        "db_status": None,
                        "integrity_status": effective_status,
                    }
                )

            target.append(item)

        return {
            "transcript_name": transcript_name,
            "pending": pending,
            "pending_count": len(pending),
            "resolved": resolved,
            "resolved_count": len(resolved),
            "orphaned": orphaned,
            "orphaned_count": len(orphaned),
            "total_count": len(pending) + len(resolved) + len(orphaned),
            "status_mismatches": mismatches,
            "status_mismatch_count": len(mismatches),
            "status_authority": (
                "postgresql" if db_available else "artifact_fail_closed"
            ),
            "db_available": db_available,
            "reconciliation_error": reconciliation_error,
            "safe_real_write_available": db_available and len(pending) > 0,
        }

    def get_approved_topics(self, run_id: str) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        path = run_dir / "output" / "integration" / "approved_topics.json"
        if not path.is_file():
            return {"path": str(path), "topics": [], "topic_count": 0, "payload": {}}
        payload = self._read_json(path)
        topics = payload.get("topics") or []
        if not isinstance(topics, list):
            topics = []
        topics = [dict(item) for item in topics if isinstance(item, dict)]
        return {
            "path": str(path),
            "topics": topics,
            "topic_count": len(topics),
            "payload": payload,
        }

    # ------------------------------------------------------------------
    # Mapping-review write path. PostgreSQL trigger remains authoritative.
    # ------------------------------------------------------------------
    @staticmethod
    def _matching_record_id(value: Any, record_id: int) -> bool:
        try:
            return int(value) == int(record_id)
        except (TypeError, ValueError):
            return False

    def _persist_topic_review_status_in_json(
        self,
        *,
        run_dir: Path,
        record_id: int,
        status: str,
        corrected_decision: str | None = None,
        corrected_mapped_concept_id: str | None = None,
        correction_reason: str | None = None,
    ) -> int:
        """Mirror the existing Streamlit helper so state artifacts stay aligned."""

        changed_records = 0
        output_root = Path(run_dir) / "output"

        for json_path in output_root.glob("*/03_topic_mapping.json"):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue

            changed = 0
            review_items = payload.get("topic_review_items", [])
            if isinstance(review_items, list):
                for item in review_items:
                    if isinstance(item, dict) and self._matching_record_id(item.get("id"), record_id):
                        item["status"] = status
                        if status == "corrected":
                            item["corrected_decision"] = corrected_decision
                            item["corrected_mapped_concept_id"] = corrected_mapped_concept_id
                            item["correction_reason"] = correction_reason
                        changed += 1

            llm_results = payload.get("llm_results", [])
            if isinstance(llm_results, list):
                for item in llm_results:
                    if isinstance(item, dict) and self._matching_record_id(item.get("review_id"), record_id):
                        item["review_status"] = status
                        if status == "corrected":
                            item["decision"] = corrected_decision
                            item["mapped_concept_id"] = corrected_mapped_concept_id
                            item["reason"] = correction_reason
                        changed += 1

            if changed:
                self._write_json(json_path, payload)
                changed_records += changed

        return changed_records

    def submit_topic_review(self, request: TopicReviewRequest) -> list[dict[str, Any]]:
        """Persist human Module 3 review decisions through the existing DB contract.

        The same topic_human_review table is updated. Existing PostgreSQL
        trigger v2 remains responsible for decision logging and promotion of
        approved/corrected mappings into reusable memory.
        """

        run_dir = self._run_dir(request.run_id)

        # Safety boundary: a controller/human may only update review records
        # that are actually pending in THIS run's Module 3 artifact. This
        # prevents an unrelated PostgreSQL review_id from being mutated if a
        # caller supplies the wrong id.
        pending_payload = self.get_pending_topic_reviews(request.run_id)
        pending_ids: set[int] = set()
        for item in pending_payload.get("pending", []):
            if not isinstance(item, dict):
                continue
            try:
                pending_ids.add(int(item.get("id")))
            except (TypeError, ValueError):
                continue

        requested_ids = {int(item.review_id) for item in request.decisions}
        invalid_ids = sorted(requested_ids - pending_ids)
        if invalid_ids:
            raise ValueError(
                "Review id(s) are not pending for this Agent 1 run: "
                + ", ".join(str(value) for value in invalid_ids)
            )

        self._prepare_agent1_imports()

        try:
            from app.db.session import session_scope
            from sqlalchemy import text
        except ImportError as exc:
            raise RuntimeError(
                "Could not import the existing Agent 1 database session."
            ) from exc

        statement = text(
            """
            UPDATE topic_human_review
            SET
                status = :status,
                corrected_decision = :corrected_decision,
                corrected_mapped_concept_id = :corrected_mapped_concept_id,
                correction_reason = :correction_reason,
                review_notes = :review_notes,
                reviewed_by = :reviewed_by,
                reviewed_at = NOW(),
                updated_at = NOW()
            WHERE id = :record_id
              AND LOWER(status) IN ('candidate', 'pending', 'awaiting_review', 'needs_review')
            RETURNING
                id,
                cache_key,
                original_topic,
                proposed_decision,
                proposed_mapped_concept_id,
                corrected_decision,
                corrected_mapped_concept_id,
                correction_reason,
                confidence,
                status,
                spec_version,
                reviewed_at
            """
        )

        rows: list[dict[str, Any]] = []
        with session_scope() as session:
            for decision in request.decisions:
                if decision.action.value == "approve":
                    status = "approved"
                    corrected_decision = None
                    corrected_concept_id = None
                    correction_reason = None
                elif decision.action.value == "correct":
                    status = "corrected"
                    corrected_decision = decision.corrected_decision
                    corrected_concept_id = decision.corrected_mapped_concept_id
                    correction_reason = str(decision.reason or "").strip()
                else:
                    status = "rejected"
                    corrected_decision = None
                    corrected_concept_id = None
                    correction_reason = None

                review_notes = str(decision.review_notes or "").strip() or None
                # Preserve the existing UI behavior: Reject does not require a
                # reason, but if the reviewer supplies one we retain it as notes.
                if status == "rejected" and not review_notes:
                    review_notes = str(decision.reason or "").strip() or None

                row = session.execute(
                    statement,
                    {
                        "record_id": int(decision.review_id),
                        "status": status,
                        "corrected_decision": corrected_decision,
                        "corrected_mapped_concept_id": corrected_concept_id,
                        "correction_reason": correction_reason,
                        "review_notes": review_notes,
                        "reviewed_by": request.reviewed_by,
                    },
                ).mappings().one_or_none()
                if row is None:
                    raise KeyError(
                        f"Topic review record {decision.review_id} was not found in a pending "
                        "database state; no review was overwritten."
                    )
                rows.append(dict(row))

        # Keep current-run artifacts aligned with the committed DB state.
        for decision, row in zip(request.decisions, rows, strict=True):
            self._persist_topic_review_status_in_json(
                run_dir=run_dir,
                record_id=int(decision.review_id),
                status=str(row.get("status") or ""),
                corrected_decision=row.get("corrected_decision"),
                corrected_mapped_concept_id=row.get("corrected_mapped_concept_id"),
                correction_reason=row.get("correction_reason"),
            )

        return rows

    # ------------------------------------------------------------------
    # Phase 5B: final detected-topic edit memory
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_chunk_ids(values: Any) -> list[int]:
        output: list[int] = []
        for value in values or []:
            try:
                chunk_id = int(value)
            except (TypeError, ValueError):
                continue
            if chunk_id not in output:
                output.append(chunk_id)
        return output

    @staticmethod
    def _topic_role(topic: dict[str, Any]) -> str:
        role = str(topic.get("topic_role") or topic.get("role") or "supporting").strip().casefold()
        return role if role in {"primary", "supporting"} else "supporting"

    def _catalogue_by_id(self) -> dict[str, dict[str, Any]]:
        self._prepare_agent1_imports()
        from app.services.syllabus_store import get_syllabus_store

        return {
            str(concept.concept_id): {
                "concept_id": concept.concept_id,
                "topic": concept.label,
                "domain": concept.domain,
                "official_reference": concept.official_reference,
                "chapter_reference": concept.chapter_reference,
                "official_title": concept.official_title,
                "paper": concept.paper,
                "source_pages": list(concept.source_pages),
            }
            for concept in get_syllabus_store().get_all_concepts()
        }

    def _evidence_from_chunks(
        self,
        *,
        chunks: list[dict[str, Any]],
        source_chunk_ids: list[int],
    ) -> str:
        chunk_text_by_id: dict[int, str] = {}
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            try:
                chunk_id = int(chunk.get("chunk_id"))
            except (TypeError, ValueError):
                continue
            text_value = str(chunk.get("text") or "").strip()
            if text_value:
                chunk_text_by_id[chunk_id] = text_value

        missing = [chunk_id for chunk_id in source_chunk_ids if not chunk_text_by_id.get(chunk_id)]
        if missing:
            raise ValueError(
                "Current transcript evidence is missing for chunk id(s): "
                + ", ".join(str(value) for value in missing)
            )
        evidence = "\n\n".join(chunk_text_by_id[chunk_id] for chunk_id in source_chunk_ids).strip()
        if not evidence:
            raise ValueError(
                "Could not recover current transcript evidence for this human correction."
            )
        return evidence

    def _append_detected_topic_decision_log(
        self,
        *,
        run_id: str,
        transcript_name: str,
        normalized_topic: str,
        source_chunk_ids: list[int],
        action: str,
        decision: str,
        mapped_concept_id: str | None,
        reason: str,
        reviewed_by: str,
        details: dict[str, Any],
        spec_version: str,
    ) -> None:
        """Mirror the existing frontend audit insert without importing app.py."""

        self._prepare_agent1_imports()
        from app.db.session import session_scope
        from sqlalchemy import text

        statement = text(
            """
            INSERT INTO topic_mapping_decision_log (
                pipeline_run_id,
                normalized_topic,
                source_transcript,
                source_chunk_ids,
                decision_stage,
                actor_type,
                action,
                decision,
                mapped_concept_id,
                reason,
                decided_by,
                details,
                spec_version
            )
            VALUES (
                :pipeline_run_id,
                :normalized_topic,
                :source_transcript,
                CAST(:source_chunk_ids AS jsonb),
                'detected_topic_review',
                'human',
                :action,
                :decision,
                :mapped_concept_id,
                :reason,
                :decided_by,
                CAST(:details AS jsonb),
                :spec_version
            )
            """
        )
        with session_scope() as session:
            session.execute(
                statement,
                {
                    "pipeline_run_id": run_id,
                    "normalized_topic": str(normalized_topic).strip().casefold(),
                    "source_transcript": transcript_name,
                    "source_chunk_ids": json.dumps(source_chunk_ids),
                    "action": action,
                    "decision": decision,
                    "mapped_concept_id": mapped_concept_id,
                    "reason": reason,
                    "decided_by": reviewed_by,
                    "details": json.dumps(details, ensure_ascii=False),
                    "spec_version": spec_version,
                },
            )

    def _promote_replacement_to_mapping_memory(
        self,
        *,
        run_id: str,
        transcript_name: str,
        original_topic: dict[str, Any],
        replacement_concept_id: str,
        reason: str,
        source_chunk_ids: list[int],
        evidence_text: str,
        reviewed_by: str,
        spec_version: str,
        catalogue: dict[str, dict[str, Any]],
    ) -> int:
        """Preserve the current Streamlit replacement -> mapping-memory path.

        The existing PostgreSQL trigger remains authoritative for the decision
        log and topic_mapping_memory promotion. This code only creates the same
        pending human-review row and immediately resolves it as a human
        correction, matching the current frontend behavior.
        """

        self._prepare_agent1_imports()
        from app.db.session import session_scope
        from sqlalchemy import text

        original_label = str(original_topic.get("topic") or "").strip()
        original_concept_id = str(original_topic.get("concept_id") or "").strip() or None
        normalized_topic = original_label.casefold()
        if not normalized_topic:
            raise ValueError("The detected topic has no reusable topic label.")

        replacement = catalogue.get(str(replacement_concept_id))
        if replacement is None:
            raise ValueError("Select a valid official AQA replacement topic.")

        evidence_hash = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
        cache_key = hashlib.sha256(
            (
                "detected-topic-correction-v1|"
                + spec_version
                + "|"
                + transcript_name
                + "|"
                + normalized_topic
                + "|"
                + evidence_hash
            ).encode("utf-8")
        ).hexdigest()

        candidate_ids: list[str] = []
        for concept_id in (original_concept_id, str(replacement_concept_id)):
            if concept_id and concept_id not in candidate_ids:
                candidate_ids.append(concept_id)

        qdrant_candidates = [
            {
                "concept_id": concept_id,
                "label": catalogue[concept_id].get("topic"),
                "official_reference": catalogue[concept_id].get("official_reference"),
                "source": "detected_topic_editor",
            }
            for concept_id in candidate_ids
            if concept_id in catalogue
        ]
        try:
            confidence_value = float(original_topic.get("confidence"))
        except (TypeError, ValueError):
            confidence_value = 0.0

        insert_statement = text(
            """
            INSERT INTO topic_human_review (
                cache_key, normalized_topic, original_topic, evidence_hash,
                evidence_text, source_transcript, source_chunk_ids,
                memory_lookup_result, candidate_concept_ids, qdrant_candidates,
                proposed_decision, proposed_mapped_concept_id, confidence,
                confidence_band, reason, model_name, prompt_version, status,
                corrected_decision, corrected_mapped_concept_id,
                correction_reason, review_notes, reviewed_by, reviewed_at,
                spec_version, created_at, updated_at
            ) VALUES (
                :cache_key, :normalized_topic, :original_topic, :evidence_hash,
                :evidence_text, :source_transcript, CAST(:source_chunk_ids AS jsonb),
                'manual_edit', CAST(:candidate_concept_ids AS jsonb),
                CAST(:qdrant_candidates AS jsonb), 'mapped',
                :proposed_mapped_concept_id, :confidence, 'human_review',
                'Reviewer identified an incorrect retained official AQA mapping.',
                'detected_topic_editor', 'detected-topic-correction-v1', 'pending',
                NULL, NULL, NULL, NULL, NULL, NULL, :spec_version, NOW(), NOW()
            )
            ON CONFLICT (
                source_transcript, normalized_topic, spec_version
            ) WHERE source_transcript IS NOT NULL
            DO UPDATE SET
                cache_key = EXCLUDED.cache_key,
                original_topic = EXCLUDED.original_topic,
                evidence_hash = EXCLUDED.evidence_hash,
                evidence_text = EXCLUDED.evidence_text,
                source_chunk_ids = EXCLUDED.source_chunk_ids,
                memory_lookup_result = EXCLUDED.memory_lookup_result,
                candidate_concept_ids = EXCLUDED.candidate_concept_ids,
                qdrant_candidates = EXCLUDED.qdrant_candidates,
                proposed_decision = EXCLUDED.proposed_decision,
                proposed_mapped_concept_id = EXCLUDED.proposed_mapped_concept_id,
                confidence = EXCLUDED.confidence,
                confidence_band = EXCLUDED.confidence_band,
                reason = EXCLUDED.reason,
                model_name = EXCLUDED.model_name,
                prompt_version = EXCLUDED.prompt_version,
                status = 'pending',
                corrected_decision = NULL,
                corrected_mapped_concept_id = NULL,
                correction_reason = NULL,
                review_notes = NULL,
                reviewed_by = NULL,
                reviewed_at = NULL,
                updated_at = NOW()
            RETURNING id
            """
        )
        update_statement = text(
            """
            UPDATE topic_human_review
            SET
                status = 'corrected',
                corrected_decision = 'mapped',
                corrected_mapped_concept_id = :replacement_concept_id,
                correction_reason = :reason,
                review_notes = 'Correction originated from the final detected-topic editor.',
                reviewed_by = :reviewed_by,
                reviewed_at = NOW(),
                updated_at = NOW()
            WHERE id = :record_id
              AND LOWER(status) IN ('candidate', 'pending', 'awaiting_review', 'needs_review')
            RETURNING id
            """
        )

        with session_scope() as session:
            review_id = session.execute(
                insert_statement,
                {
                    "cache_key": cache_key,
                    "normalized_topic": normalized_topic,
                    "original_topic": original_label,
                    "evidence_hash": evidence_hash,
                    "evidence_text": evidence_text,
                    "source_transcript": transcript_name,
                    "source_chunk_ids": json.dumps(source_chunk_ids),
                    "candidate_concept_ids": json.dumps(candidate_ids),
                    "qdrant_candidates": json.dumps(qdrant_candidates),
                    "proposed_mapped_concept_id": original_concept_id,
                    "confidence": confidence_value,
                    "spec_version": spec_version,
                },
            ).scalar_one()
            updated_id = session.execute(
                update_statement,
                {
                    "record_id": int(review_id),
                    "replacement_concept_id": str(replacement_concept_id),
                    "reason": reason,
                    "reviewed_by": reviewed_by,
                },
            ).scalar_one_or_none()
            if updated_id is None:
                raise RuntimeError(
                    "Replacement correction review could not be resolved; no mapping memory was promoted."
                )
        return int(review_id)

    def submit_detected_topic_edit(self, request: DetectedTopicEditRequest) -> dict[str, Any]:
        """Apply one explicit final-topic human edit through existing memory services.

        Semantic detection/ranking is untouched. The adapter only performs the
        same deterministic current-run curation around Agent 1's existing
        ``persist_detected_topic_edit_memory`` service and, for replacements,
        preserves the existing human-review -> mapping-memory promotion path.
        """

        run_dir, transcript_name, _, module2_path, module3_path = self._artifact_paths(request.run_id)
        module2_json = self._read_json(module2_path)
        module3_json = self._read_json(module3_path)
        chunks = module2_json.get("chunks") or []
        if not isinstance(chunks, list):
            chunks = []
        chunks = [dict(item) for item in chunks if isinstance(item, dict)]
        module3_result = module3_json.get("module3_result") or {}
        if not isinstance(module3_result, dict):
            raise RuntimeError("Module 3 result is missing from the current run JSON.")
        merged_topics = module3_result.get("merged_topics") or []
        if not isinstance(merged_topics, list):
            raise RuntimeError("Merged topics are missing from the current run JSON.")

        effective = self.get_effective_topics(request.run_id)
        effective_topics = effective.get("topics") or []
        if not isinstance(effective_topics, list):
            effective_topics = []
        spec_version = str(
            module3_json.get("spec_version")
            or module3_json.get("specification_version")
            or effective.get("spec_version")
            or os.getenv("AQA_SPEC_VERSION")
            or "AQA-8525-v1.2-2022-11-29"
        ).strip()
        reason = str(request.reason).strip()
        reviewed_by = str(request.reviewed_by).strip()
        catalogue = self._catalogue_by_id()
        capture = self._load_edit_capture()

        action = request.action
        now = datetime.now(timezone.utc).isoformat()

        if action is DetectedTopicEditAction.ADD_TOPIC:
            target_id = str(request.target_concept_id or "").strip()
            target = catalogue.get(target_id)
            if target is None:
                raise ValueError("Select a valid official AQA topic.")
            present_ids = {
                str(item.get("concept_id") or "").strip()
                for item in effective_topics
                if isinstance(item, dict) and str(item.get("concept_id") or "").strip()
            }
            if target_id in present_ids:
                raise ValueError("That official AQA topic is already in the effective topic list.")
            source_chunk_ids = self._clean_chunk_ids(request.source_chunk_ids)
            evidence_text = self._evidence_from_chunks(
                chunks=chunks,
                source_chunk_ids=source_chunk_ids,
            )
            memory_record = capture.persist_detected_topic_edit_memory(
                edit_action="add_topic",
                source_concept_id=None,
                source_topic=None,
                source_role=None,
                target_concept_id=target_id,
                target_topic=str(target["topic"]),
                target_role=str(request.target_role),
                evidence_text=evidence_text,
                source_chunk_ids=source_chunk_ids,
                reviewer_reason=reason,
                source_transcript=transcript_name,
                spec_version=spec_version,
                reviewed_by=reviewed_by,
            )
            additions = module3_json.setdefault("topic_output_additions", [])
            if not isinstance(additions, list):
                raise RuntimeError("Current-run human topic additions are malformed.")
            if any(
                isinstance(item, dict)
                and str(item.get("concept_id") or "").strip() == target_id
                for item in additions
            ):
                raise ValueError("That topic has already been manually added to this run.")
            added_topic = {
                "concept_id": target_id,
                "topic": target["topic"],
                "domain": target["domain"],
                "official_reference": target["official_reference"],
                "chapter_reference": target["chapter_reference"],
                "official_title": target["official_title"],
                "paper": target["paper"],
                "source_pages": target["source_pages"],
                "confidence": None,
                "ranking_score": None,
                "topic_role": str(request.target_role),
                "source_chunk_ids": source_chunk_ids,
                "evidence": [evidence_text],
                "human_edited": True,
                "human_added_topic": True,
                "human_edit_action": "add_topic",
                "human_edit_reason": reason,
                "detected_topic_edit_memory_id": memory_record["memory_id"],
            }
            additions.append(added_topic)
            edit_record = {
                "timestamp": now,
                "action": "add_topic",
                "replacement_topic": target["topic"],
                "replacement_concept_id": target_id,
                "new_role": str(request.target_role),
                "source_chunk_ids": source_chunk_ids,
                "reason": reason,
                "reviewed_by": reviewed_by,
                "detected_topic_edit_memory_id": memory_record["memory_id"],
                "stored_as_contextual_edit_memory": True,
            }
            module3_json.setdefault("topic_output_edits", []).append(edit_record)
            self._write_json(module3_path, module3_json)
            self._append_detected_topic_decision_log(
                run_id=request.run_id,
                transcript_name=transcript_name,
                normalized_topic=str(target["topic"]),
                source_chunk_ids=source_chunk_ids,
                action="add_topic",
                decision="added",
                mapped_concept_id=target_id,
                reason=reason,
                reviewed_by=reviewed_by,
                details=edit_record,
                spec_version=spec_version,
            )
            return {
                "action": "add_topic",
                "edit_record": edit_record,
                "memory": memory_record,
                "effective_topic_count_before": len(effective_topics),
                "source_chunk_ids": source_chunk_ids,
                "artifact_path": str(module3_path),
            }

        if request.topic_index is None or request.topic_index >= len(effective_topics):
            raise ValueError("topic_index is not present in the current effective topic list.")
        selected = effective_topics[int(request.topic_index)]
        if not isinstance(selected, dict):
            raise ValueError("The selected effective topic is malformed.")
        if bool(selected.get("human_added_topic")):
            raise ValueError(
                "Current-run manually added topics are not editable through the existing detected-topic selector."
            )
        selected_concept_id = str(selected.get("concept_id") or "").strip()
        if request.source_concept_id and str(request.source_concept_id).strip() != selected_concept_id:
            raise ValueError(
                "source_concept_id does not match the current topic at topic_index; refresh before editing."
            )
        if not selected_concept_id:
            raise ValueError("The selected topic has no stable concept_id.")

        raw_index: int | None = None
        for index, candidate in enumerate(merged_topics):
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("concept_id") or "").strip() == selected_concept_id:
                raw_index = index
                break
        if raw_index is None:
            raise ValueError(
                "This effective topic could not be matched safely to the fresh Module 3 output. "
                "No edit was written; refresh the run and try again."
            )

        if "merged_topics_original" not in module3_result:
            module3_result["merged_topics_original"] = json.loads(
                json.dumps(merged_topics, ensure_ascii=False)
            )
        original = dict(merged_topics[raw_index])
        source_chunk_ids = self._clean_chunk_ids(original.get("source_chunk_ids") or [])
        evidence_text = self._evidence_from_chunks(
            chunks=chunks,
            source_chunk_ids=source_chunk_ids,
        )
        source_topic = str(original.get("topic") or "").strip() or None
        source_role = self._topic_role(original)
        target_concept_id: str | None = None
        target_topic: str | None = None
        target_role: str | None = None
        decision: str
        mapped_concept_id: str | None

        edit_record: dict[str, Any] = {
            "timestamp": now,
            "action": action.value,
            "topic_index_before_edit": raw_index,
            "original_topic": original.get("topic"),
            "original_concept_id": original.get("concept_id"),
            "original_role": original.get("topic_role"),
            "source_chunk_ids": source_chunk_ids,
            "reason": reason,
            "reviewed_by": reviewed_by,
        }

        if action is DetectedTopicEditAction.CHANGE_ROLE:
            target_role = str(request.target_role)
            if target_role == source_role:
                raise ValueError("Choose a different role before saving the correction.")
            updated = dict(original)
            updated["topic_role"] = target_role
            updated["human_edited"] = True
            updated["human_edit_action"] = "change_role"
            updated["human_edit_reason"] = reason
            merged_topics[raw_index] = updated
            edit_record["new_role"] = target_role
            decision = "mapped"
            mapped_concept_id = selected_concept_id
            target_concept_id = selected_concept_id
            target_topic = source_topic

        elif action is DetectedTopicEditAction.REPLACE_TOPIC:
            target_concept_id = str(request.target_concept_id or "").strip()
            replacement = catalogue.get(target_concept_id)
            if replacement is None:
                raise ValueError("Select a valid official AQA topic.")
            if target_concept_id == selected_concept_id:
                raise ValueError("Choose a different official AQA topic.")
            for other_index, other in enumerate(merged_topics):
                if other_index == raw_index or not isinstance(other, dict):
                    continue
                if str(other.get("concept_id") or "").strip() == target_concept_id:
                    raise ValueError("That official AQA topic is already in the final topic list.")
            target_role = str(request.target_role or source_role)
            target_topic = str(replacement["topic"])
            updated = dict(original)
            updated.update(
                {
                    "concept_id": target_concept_id,
                    "topic": target_topic,
                    "domain": replacement["domain"],
                    "official_reference": replacement["official_reference"],
                    "chapter_reference": replacement["chapter_reference"],
                    "official_title": replacement["official_title"],
                    "paper": replacement["paper"],
                    "source_pages": replacement["source_pages"],
                    "topic_role": target_role,
                    "human_edited": True,
                    "human_edit_action": "replace_topic",
                    "human_edit_reason": reason,
                    "human_original_topic": original.get("topic"),
                    "human_original_concept_id": original.get("concept_id"),
                }
            )
            merged_topics[raw_index] = updated
            edit_record.update(
                {
                    "replacement_topic": target_topic,
                    "replacement_concept_id": target_concept_id,
                    "new_role": target_role,
                }
            )
            decision = "mapped"
            mapped_concept_id = target_concept_id

        elif action is DetectedTopicEditAction.REMOVE_TOPIC:
            merged_topics.pop(raw_index)
            decision = "removed"
            mapped_concept_id = None

        else:
            raise ValueError("Unsupported detected-topic edit action.")

        memory_record = capture.persist_detected_topic_edit_memory(
            edit_action=action.value,
            source_concept_id=selected_concept_id,
            source_topic=source_topic,
            source_role=source_role,
            target_concept_id=target_concept_id,
            target_topic=target_topic,
            target_role=target_role,
            evidence_text=evidence_text,
            source_chunk_ids=source_chunk_ids,
            reviewer_reason=reason,
            source_transcript=transcript_name,
            spec_version=spec_version,
            reviewed_by=reviewed_by,
        )
        edit_record["detected_topic_edit_memory_id"] = memory_record["memory_id"]
        edit_record["stored_as_contextual_edit_memory"] = True
        module3_json.setdefault("topic_output_edits", []).append(edit_record)
        self._write_json(module3_path, module3_json)

        self._append_detected_topic_decision_log(
            run_id=request.run_id,
            transcript_name=transcript_name,
            normalized_topic=str(source_topic or "detected topic"),
            source_chunk_ids=source_chunk_ids,
            action=action.value,
            decision=decision,
            mapped_concept_id=mapped_concept_id,
            reason=reason,
            reviewed_by=reviewed_by,
            details=edit_record,
            spec_version=spec_version,
        )

        mapping_review_id: int | None = None
        if action is DetectedTopicEditAction.REPLACE_TOPIC:
            mapping_review_id = self._promote_replacement_to_mapping_memory(
                run_id=request.run_id,
                transcript_name=transcript_name,
                original_topic=original,
                replacement_concept_id=str(target_concept_id),
                reason=reason,
                source_chunk_ids=source_chunk_ids,
                evidence_text=evidence_text,
                reviewed_by=reviewed_by,
                spec_version=spec_version,
                catalogue=catalogue,
            )
            edit_record["topic_human_review_id"] = mapping_review_id
            edit_record["promoted_as_human_correction"] = True
            # Mirror the extra fields in the saved artifact just as the current
            # Streamlit helper does after promotion.
            module3_json = self._read_json(module3_path)
            edits = module3_json.get("topic_output_edits") or []
            if isinstance(edits, list) and edits:
                last = edits[-1]
                if isinstance(last, dict) and last.get("detected_topic_edit_memory_id") == memory_record["memory_id"]:
                    last["topic_human_review_id"] = mapping_review_id
                    last["promoted_as_human_correction"] = True
                    self._write_json(module3_path, module3_json)

        return {
            "action": action.value,
            "edit_record": edit_record,
            "memory": memory_record,
            "mapping_review_id": mapping_review_id,
            "source_chunk_ids": source_chunk_ids,
            "artifact_path": str(module3_path),
        }

    # ------------------------------------------------------------------
    # Agent 1 -> Agent 2 handoff approval
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_chunk_id(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _build_agent2_topic_handoff(
        self,
        *,
        merged_topics: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        chunk_text_by_id: dict[int, str] = {}
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_id = self._normalise_chunk_id(chunk.get("chunk_id"))
            if chunk_id is None:
                continue
            chunk_text_by_id[chunk_id] = str(chunk.get("text") or "").strip()

        payload: list[dict[str, Any]] = []
        for topic_index, topic in enumerate(merged_topics, start=1):
            if not isinstance(topic, dict):
                continue

            raw_chunk_ids = topic.get("source_chunk_ids") or topic.get("source_chunks") or []
            source_chunks: list[int] = []
            for raw_chunk_id in raw_chunk_ids:
                chunk_id = self._normalise_chunk_id(raw_chunk_id)
                if chunk_id is not None and chunk_id not in source_chunks:
                    source_chunks.append(chunk_id)

            source_chunk_texts = [
                chunk_text_by_id[chunk_id]
                for chunk_id in source_chunks
                if chunk_text_by_id.get(chunk_id)
            ]
            topic_name = str(topic.get("topic") or topic.get("detected_topic") or "").strip()
            role = str(topic.get("topic_role") or topic.get("role") or "supporting").strip().casefold()
            if role not in {"primary", "supporting"}:
                role = "supporting"

            payload.append(
                {
                    "topic_index": topic_index,
                    "concept_id": topic.get("concept_id"),
                    "topic": topic_name,
                    "detected_topic": topic_name,
                    "role": role,
                    "topic_role": role,
                    "official_reference": str(topic.get("official_reference") or "").strip(),
                    "official_title": topic.get("official_title"),
                    "chapter_reference": topic.get("chapter_reference"),
                    "domain": topic.get("domain"),
                    "paper": topic.get("paper"),
                    "confidence": topic.get("confidence"),
                    "ranking_score": topic.get("ranking_score"),
                    "source_chunks": source_chunks,
                    "source_chunk_texts": source_chunk_texts,
                    "source_chunk_text_count": len(source_chunk_texts),
                    "source_chunk_count": len(source_chunks),
                    "missing_source_chunk_ids": [
                        chunk_id for chunk_id in source_chunks if not chunk_text_by_id.get(chunk_id)
                    ],
                    "evidence": topic.get("evidence") or [],
                }
            )
        return payload

    def _write_handoff(
        self,
        *,
        run_dir: Path,
        transcript_name: str,
        topics: list[dict[str, Any]],
        approved_only: bool,
        reviewed_by: str | None = None,
    ) -> Path:
        integration_dir = run_dir / "output" / "integration"
        integration_dir.mkdir(parents=True, exist_ok=True)
        output_path = integration_dir / (
            "approved_topics.json" if approved_only else "agent1_topics_with_evidence.json"
        )
        payload = {
            "schema_version": "agent1-agent2-topic-handoff-v1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": run_dir.name,
            "transcript": transcript_name,
            "source": {
                "module2_output": "02_chunking.json",
                "module3_output": "03_topic_mapping.json",
                "notebook_logic_changed": False,
                "handoff_built_by": "mcp_hitl_adapter",
            },
            "approved_only": bool(approved_only),
            "topic_count": len(topics),
            "actual_chunk_evidence_available": bool(
                topics and all(topic.get("source_chunk_texts") for topic in topics)
            ),
            "topics": topics,
        }
        if reviewed_by:
            payload["reviewed_by"] = reviewed_by
        self._write_json(output_path, payload)
        return output_path

    def save_agent2_topic_approval(self, request: Agent2TopicApprovalRequest) -> dict[str, Any]:
        effective = self.get_effective_topics(request.run_id)
        run_dir = self._run_dir(request.run_id)
        transcript_name = str(effective["transcript_name"])
        all_topics = self._build_agent2_topic_handoff(
            merged_topics=effective["topics"],
            chunks=effective["chunks"],
        )

        self._write_handoff(
            run_dir=run_dir,
            transcript_name=transcript_name,
            topics=all_topics,
            approved_only=False,
        )

        topic_by_index = {int(item["topic_index"]): item for item in all_topics}
        approved: list[dict[str, Any]] = []
        for selection in request.selections:
            if not selection.approved:
                continue
            if selection.topic_index not in topic_by_index:
                raise ValueError(
                    f"Topic index {selection.topic_index} is not present in the current effective topic list."
                )
            topic = dict(topic_by_index[selection.topic_index])
            if selection.topic is not None:
                topic_name = selection.topic.strip()
                if not topic_name:
                    raise ValueError("Approved topic label cannot be blank.")
                topic["topic"] = topic_name
                topic["detected_topic"] = topic_name
            if selection.role is not None:
                topic["role"] = selection.role
                topic["topic_role"] = selection.role
            if selection.official_reference is not None:
                reference = selection.official_reference.strip()
                topic["official_reference"] = reference
            topic["approved_for_agent2"] = True
            approved.append(topic)

        if not approved:
            raise ValueError("Approve at least one topic for Agent 2.")
        if not any(str(topic.get("role") or "").casefold() == "primary" for topic in approved):
            raise ValueError("At least one approved topic must have the primary role.")

        invalid_references = [
            str(topic.get("official_reference") or "").strip() or "<blank>"
            for topic in approved
            if not re.fullmatch(r"\d+(?:\.\d+){1,3}", str(topic.get("official_reference") or "").strip())
        ]
        if invalid_references:
            raise ValueError(
                "Invalid official reference(s): " + ", ".join(invalid_references)
            )

        output_path = self._write_handoff(
            run_dir=run_dir,
            transcript_name=transcript_name,
            topics=approved,
            approved_only=True,
            reviewed_by=request.reviewed_by,
        )
        return {
            "path": str(output_path),
            "topics": approved,
            "topic_count": len(approved),
            "primary_count": sum(
                str(topic.get("role") or "").casefold() == "primary"
                for topic in approved
            ),
        }
