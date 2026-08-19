from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_SPEC_VERSION = "AQA-8525-v1.2-2022-11-29"


def _agent1_code_root(frontend_project_root: Path) -> Path:
    root = Path(frontend_project_root).resolve()
    for candidate in (root, root.parent, root.parent.parent):
        if (candidate / "app" / "services").is_dir() and (candidate / "app" / "db").is_dir():
            return candidate
    raise RuntimeError("Could not locate Agent_1/app from the Streamlit frontend project root.")


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Expected Pydantic model/dict, received {type(value).__name__}.")


def _serialise_audit_item(item: Any) -> dict[str, Any]:
    fields = (
        "memory_id",
        "action",
        "source_concept_id",
        "target_concept_id",
        "explanation",
        "reason",
    )
    row: dict[str, Any] = {}
    for field_name in fields:
        if hasattr(item, field_name):
            value = getattr(item, field_name)
            if value is not None:
                row[field_name] = value
        elif isinstance(item, dict) and field_name in item:
            row[field_name] = item[field_name]
    if not row:
        row["value"] = str(item)
    return row


def apply_detected_topic_edit_runtime(
    *,
    module3_result_payload: dict[str, Any],
    module3_json: dict[str, Any] | None,
    run_dir: Path,
    transcript_name: str,
    frontend_project_root: Path,
) -> dict[str, Any]:
    original_payload = json.loads(
        json.dumps(module3_result_payload, ensure_ascii=False)
    )

    module3_json = module3_json or {}
    spec_version = str(
        module3_json.get("spec_version")
        or module3_json.get("specification_version")
        or os.getenv("AQA_SPEC_VERSION")
        or DEFAULT_SPEC_VERSION
    ).strip()

    session = None

    try:
        code_root = _agent1_code_root(Path(frontend_project_root))

        # Streamlit puts frontend/ at the front of sys.path. That folder
        # contains frontend/app.py, while the real backend package is
        # Agent_1/app/. If Agent_1 is already present later in sys.path,
        # the previous 'if not in sys.path' guard does not move it forward.
        # Then 'import app.db...' can resolve frontend/app.py and recursively
        # execute the Streamlit UI, causing duplicate widget IDs.
        #
        # Always move Agent_1 root to sys.path[0].
        code_root_text = str(code_root)
        sys.path[:] = [
            entry
            for entry in sys.path
            if str(entry) != code_root_text
        ]
        sys.path.insert(0, code_root_text)

        load_dotenv(code_root / ".env", override=False)
        load_dotenv(Path(frontend_project_root) / ".env", override=False)

        # These are the exact classes/functions used by the Step 5 diagnostic
        # that already passed in the user's environment.
        from app.db.session import get_session_factory
        from app.db.repositories.detected_topic_edit_memory_repository import (
            DetectedTopicEditMemoryRepository,
        )
        from app.schemas.topic import Module3Result
        from app.services.detected_topic_edit_end_to_end import (
            DetectedTopicEditEndToEndService,
        )

        fresh_result = Module3Result.model_validate(original_payload)

        session_factory = get_session_factory()
        session = session_factory()

        repository = DetectedTopicEditMemoryRepository(session)

        service = DetectedTopicEditEndToEndService(
            repository=repository
        )

        service_result = service.apply(
            module3_result=fresh_result,
            spec_version=spec_version,
        )

        final_result = Module3Result.model_validate(
            _model_dump(service_result.module3_result)
        )
        final_payload = _model_dump(final_result)

        applied = [
            _serialise_audit_item(item)
            for item in service_result.overlay_result.applied
        ]
        skipped = [
            _serialise_audit_item(item)
            for item in service_result.overlay_result.skipped
        ]
        diagnostics = list(service_result.retrieval_diagnostics)

        # Read-only runtime: ensure the DB session cannot persist anything.
        session.rollback()
        session.close()
        session = None

        return {
            "status": "applied" if applied else "no_change",
            "module3_result": final_payload,
            "applied": applied,
            "skipped": skipped,
            "retrieval_diagnostics": diagnostics,
            "applied_count": len(applied),
            "skipped_count": len(skipped),
            "error": None,
            "spec_version": spec_version,
            "persistent_usage_counter_updates": False,
            "saved_module3_json_mutated": False,
        }

    except Exception as exc:
        if session is not None:
            try:
                session.rollback()
            finally:
                session.close()

        return {
            "status": "fallback",
            "module3_result": original_payload,
            "applied": [],
            "skipped": [],
            "retrieval_diagnostics": [],
            "applied_count": 0,
            "skipped_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "spec_version": spec_version,
            "persistent_usage_counter_updates": False,
            "saved_module3_json_mutated": False,
        }
