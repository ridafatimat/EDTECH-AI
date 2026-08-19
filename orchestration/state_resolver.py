from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .guardrails import allowed_tools_for_state
from .workflow_state import HumanGate, WorkflowState


_PENDING_REVIEW_STATUSES = {
    "candidate",
    "pending",
    "awaiting_review",
    "needs_review",
}
_RESOLVED_REVIEW_STATUSES = {"approved", "corrected", "rejected"}
_INTEGRITY_REVIEW_STATUSES = {"missing_db_row", "orphaned_missing_db", "invalid_review_id"}


@dataclass(frozen=True)
class WorkflowSnapshot:
    run_id: str | None
    transcript_name: str | None
    state: WorkflowState
    human_gate: HumanGate
    human_action_required: bool
    manifest_status: str | None
    module1_complete: bool
    module2_complete: bool
    module3_complete: bool
    topic_review_count: int
    pending_topic_review_count: int
    approved_topic_count: int
    allowed_tools: tuple[str, ...]
    reason: str
    topic_review_integrity_issue_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["human_gate"] = self.human_gate.value
        payload["allowed_tools"] = list(self.allowed_tools)
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _resolve_transcript_name(run_dir: Path, manifest: dict[str, Any]) -> str | None:
    value = str(manifest.get("transcript_name") or "").strip()
    if value:
        return value

    output_root = run_dir / "output"
    if not output_root.is_dir():
        return None

    ignored = {"integration", "agent2"}
    candidates = [
        p.name
        for p in output_root.iterdir()
        if p.is_dir() and p.name not in ignored
    ]
    if len(candidates) == 1:
        return candidates[0]

    # create_pipeline_run() creates the input file before run_pipeline() writes
    # pipeline_manifest.json. For MCP stage-by-stage execution, infer the same
    # transcript stem from the single input file so RAW_TRANSCRIPT_READY is
    # resolvable before Module 1 runs.
    input_dir = run_dir / "input"
    input_files = [p for p in input_dir.glob("*") if p.is_file()] if input_dir.is_dir() else []
    if len(input_files) == 1:
        return input_files[0].stem.strip() or None
    return None


def _review_status(
    item: dict[str, Any],
    status_overrides: Mapping[int, str] | None = None,
) -> str:
    """Return the effective review status.

    ``03_topic_mapping.json`` is a run snapshot, while PostgreSQL is the
    authoritative store for human review decisions.  Runtime callers can pass
    DB-derived status overrides so an old artifact cannot keep a review gate
    open after the review has already been resolved in PostgreSQL.
    """

    if status_overrides:
        try:
            review_id = int(item.get("id"))
        except (TypeError, ValueError):
            review_id = None
        if review_id is not None and review_id in status_overrides:
            return str(status_overrides[review_id]).strip().casefold()

    # Current Agent 1 writes `status` in topic_review_items and also updates
    # review_status in related payloads. Support both without changing Agent 1.
    return str(item.get("status") or item.get("review_status") or "pending").strip().casefold()


def _snapshot(
    *,
    run_id: str | None,
    transcript_name: str | None,
    state: WorkflowState,
    human_gate: HumanGate,
    manifest_status: str | None,
    module1_complete: bool,
    module2_complete: bool,
    module3_complete: bool,
    topic_review_count: int,
    pending_topic_review_count: int,
    approved_topic_count: int,
    reason: str,
    topic_review_integrity_issue_count: int = 0,
) -> WorkflowSnapshot:
    return WorkflowSnapshot(
        run_id=run_id,
        transcript_name=transcript_name,
        state=state,
        human_gate=human_gate,
        human_action_required=human_gate is not HumanGate.NONE,
        manifest_status=manifest_status,
        module1_complete=module1_complete,
        module2_complete=module2_complete,
        module3_complete=module3_complete,
        topic_review_count=topic_review_count,
        pending_topic_review_count=pending_topic_review_count,
        approved_topic_count=approved_topic_count,
        allowed_tools=allowed_tools_for_state(state),
        reason=reason,
        topic_review_integrity_issue_count=topic_review_integrity_issue_count,
    )


def resolve_agent1_state(
    run_dir: Path,
    *,
    transcript_name: str | None = None,
    topic_review_status_overrides: Mapping[int, str] | None = None,
) -> WorkflowSnapshot:
    """Resolve state from deterministic Agent 1 run artifacts.

    No LLM is used here and no database writes occur. The resolver reads the
    artifacts your existing pipeline already creates. Topic review updates are
    persisted back into 03_topic_mapping.json by the current Streamlit review
    path, so the mandatory review gate can be detected without reimplementing
    the HITL/self-improvement service.
    """

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return _snapshot(
            run_id=None,
            transcript_name=transcript_name,
            state=WorkflowState.NO_RUN,
            human_gate=HumanGate.NONE,
            manifest_status=None,
            module1_complete=False,
            module2_complete=False,
            module3_complete=False,
            topic_review_count=0,
            pending_topic_review_count=0,
            approved_topic_count=0,
            reason="Run directory does not exist.",
        )

    manifest = _read_json(run_dir / "pipeline_manifest.json")
    run_id = str(manifest.get("job_id") or run_dir.name).strip() or run_dir.name
    manifest_status = str(manifest.get("status") or "").strip().casefold() or None
    resolved_transcript = transcript_name or _resolve_transcript_name(run_dir, manifest)

    input_files = [p for p in (run_dir / "input").glob("*") if p.is_file()] if (run_dir / "input").is_dir() else []

    if manifest_status == "failed":
        # Compatibility recovery for the old Streamlit quiz integration:
        # a downstream Agent 2 / quiz failure could incorrectly change the
        # already-completed Agent 1 pipeline manifest to "failed".
        recovery_output = (
            run_dir
            / "output"
            / str(resolved_transcript or "")
        )

        agent1_outputs_complete = bool(
            resolved_transcript
            and _nonempty(
                recovery_output
                / "01_preprocessing.json"
            )
            and _nonempty(
                recovery_output
                / "01_cleaned_transcript.txt"
            )
            and _nonempty(
                recovery_output
                / "02_chunking.json"
            )
            and _nonempty(
                recovery_output
                / "03_topic_mapping.json"
            )
        )

        if not agent1_outputs_complete:
            return _snapshot(
                run_id=run_id,
                transcript_name=resolved_transcript,
                state=WorkflowState.TOOL_FAILED,
                human_gate=HumanGate.NONE,
                manifest_status=manifest_status,
                module1_complete=False,
                module2_complete=False,
                module3_complete=False,
                topic_review_count=0,
                pending_topic_review_count=0,
                approved_topic_count=0,
                reason="Existing Agent 1 pipeline manifest reports failure.",
            )

        # Treat the completed Agent 1 artifacts as authoritative for state
        # resolution. The next LangGraph sync will write "completed" back.
        manifest_status = "completed"

    if not input_files:
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.NO_RUN,
            human_gate=HumanGate.NONE,
            manifest_status=manifest_status,
            module1_complete=False,
            module2_complete=False,
            module3_complete=False,
            topic_review_count=0,
            pending_topic_review_count=0,
            approved_topic_count=0,
            reason="Run exists but no transcript input file is present.",
        )

    if not resolved_transcript:
        return _snapshot(
            run_id=run_id,
            transcript_name=None,
            state=WorkflowState.RAW_TRANSCRIPT_READY,
            human_gate=HumanGate.NONE,
            manifest_status=manifest_status,
            module1_complete=False,
            module2_complete=False,
            module3_complete=False,
            topic_review_count=0,
            pending_topic_review_count=0,
            approved_topic_count=0,
            reason="Transcript input exists; output transcript name has not been resolved yet.",
        )

    transcript_output = run_dir / "output" / resolved_transcript
    module1_complete = _nonempty(transcript_output / "01_preprocessing.json") and _nonempty(
        transcript_output / "01_cleaned_transcript.txt"
    )
    module2_complete = _nonempty(transcript_output / "02_chunking.json")
    module3_complete = _nonempty(transcript_output / "03_topic_mapping.json")

    if not module1_complete:
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.RAW_TRANSCRIPT_READY,
            human_gate=HumanGate.NONE,
            manifest_status=manifest_status,
            module1_complete=False,
            module2_complete=False,
            module3_complete=False,
            topic_review_count=0,
            pending_topic_review_count=0,
            approved_topic_count=0,
            reason="Raw transcript is available and Module 1 outputs are absent.",
        )

    if not module2_complete:
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.PREPROCESSING_COMPLETE,
            human_gate=HumanGate.NONE,
            manifest_status=manifest_status,
            module1_complete=True,
            module2_complete=False,
            module3_complete=False,
            topic_review_count=0,
            pending_topic_review_count=0,
            approved_topic_count=0,
            reason="Module 1 outputs exist; semantic chunks are not available yet.",
        )

    if not module3_complete:
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.CHUNKS_READY,
            human_gate=HumanGate.NONE,
            manifest_status=manifest_status,
            module1_complete=True,
            module2_complete=True,
            module3_complete=False,
            topic_review_count=0,
            pending_topic_review_count=0,
            approved_topic_count=0,
            reason="Module 2 chunks exist; Module 3 topic mapping has not completed.",
        )

    module3_payload = _read_json(transcript_output / "03_topic_mapping.json")
    review_items = module3_payload.get("topic_review_items") or []
    if not isinstance(review_items, list):
        review_items = []
    review_items = [item for item in review_items if isinstance(item, dict)]
    statuses = [
        _review_status(item, topic_review_status_overrides)
        for item in review_items
    ]
    pending_count = sum(status in _PENDING_REVIEW_STATUSES for status in statuses)
    integrity_issue_count = sum(status in _INTEGRITY_REVIEW_STATUSES for status in statuses)

    if integrity_issue_count > 0:
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.REVIEW_STATE_INCONSISTENT,
            human_gate=HumanGate.TOPIC_REVIEW_INTEGRITY,
            manifest_status=manifest_status,
            module1_complete=True,
            module2_complete=True,
            module3_complete=True,
            topic_review_count=len(review_items),
            pending_topic_review_count=pending_count,
            approved_topic_count=0,
            reason=(
                f"Module 3 contains {integrity_issue_count} review item(s) whose ids "
                "are not present in the live PostgreSQL review table. The controller "
                "is fail-closed: review writes and Agent 2 are blocked."
            ),
            topic_review_integrity_issue_count=integrity_issue_count,
        )

    if pending_count > 0:
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.AWAITING_TOPIC_MAPPING_REVIEW,
            human_gate=HumanGate.TOPIC_MAPPING_REVIEW,
            manifest_status=manifest_status,
            module1_complete=True,
            module2_complete=True,
            module3_complete=True,
            topic_review_count=len(review_items),
            pending_topic_review_count=pending_count,
            approved_topic_count=0,
            reason=(
                f"Module 3 completed but {pending_count} topic mapping review item(s) "
                "still require a human decision. Agent 2 is blocked."
            ),
        )

    # Count currently retained official topics from the current run artifact.
    # Human additions are stored separately in topic_output_additions by the
    # existing Agent 1 final-topic editor, so include them as current-run topics.
    module3_result = module3_payload.get("module3_result") or {}
    if not isinstance(module3_result, dict):
        module3_result = {}
    merged_topics = module3_result.get("merged_topics") or []
    if not isinstance(merged_topics, list):
        merged_topics = []
    retained_concept_ids = {
        str(item.get("concept_id") or "").strip()
        for item in merged_topics
        if isinstance(item, dict) and str(item.get("concept_id") or "").strip()
    }
    retained_topic_count = sum(1 for item in merged_topics if isinstance(item, dict))

    additions = module3_payload.get("topic_output_additions") or []
    if isinstance(additions, list):
        for item in additions:
            if not isinstance(item, dict):
                continue
            concept_id = str(item.get("concept_id") or "").strip()
            if concept_id and concept_id not in retained_concept_ids:
                retained_topic_count += 1
                retained_concept_ids.add(concept_id)

    approved_path = run_dir / "output" / "integration" / "approved_topics.json"
    approved_payload = _read_json(approved_path)
    approved_topics = approved_payload.get("topics") or []
    if not isinstance(approved_topics, list):
        approved_topics = []
    approved_topics = [item for item in approved_topics if isinstance(item, dict)]

    # An Agent 2 handoff is valid only if it was created after the current
    # Module 3 artifact. Human mapping decisions update 03_topic_mapping.json,
    # so an older approval file must not silently unlock Agent 2.
    try:
        approval_is_stale = bool(
            approved_topics
            and approved_path.is_file()
            and approved_path.stat().st_mtime < (transcript_output / "03_topic_mapping.json").stat().st_mtime
        )
    except OSError:
        approval_is_stale = bool(approved_topics)

    if approval_is_stale:
        approved_topics = []

    if retained_topic_count == 0 and not approved_topics:
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.NO_RETAINED_TOPICS,
            human_gate=HumanGate.NONE,
            manifest_status=manifest_status,
            module1_complete=True,
            module2_complete=True,
            module3_complete=True,
            topic_review_count=len(review_items),
            pending_topic_review_count=0,
            approved_topic_count=0,
            reason=(
                "Module 3 mapping reviews are resolved, but no official AQA topics "
                "are retained. Agent 2 is blocked because there is nothing to approve "
                "or retrieve against. A human may add a missed official topic if needed."
            ),
        )

    # If all mapping reviews are resolved (or none were required), current
    # Agent 1 still has a distinct human handoff approval before Agent 2.
    if not approved_topics:
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.AWAITING_AGENT2_TOPIC_APPROVAL,
            human_gate=HumanGate.AGENT2_TOPIC_APPROVAL,
            manifest_status=manifest_status,
            module1_complete=True,
            module2_complete=True,
            module3_complete=True,
            topic_review_count=len(review_items),
            pending_topic_review_count=0,
            approved_topic_count=0,
            reason=(
                "Module 3 is complete and mapping reviews are resolved, but no "
                "human-approved Agent 1 -> Agent 2 topic handoff exists yet."
            ),
        )

    # ------------------------------------------------------------------
    # Phase 8: resolve current Agent 2 execution artifacts without changing
    # Agent 2 logic.  Freshness is anchored to the current human-approved topic
    # handoff and current assessment_request.json so stale previous assessments
    # can never unlock a later run/request.
    # ------------------------------------------------------------------
    integration_dir = run_dir / "output" / "integration"
    assessment_request_path = integration_dir / "assessment_request.json"
    agent2_output_dir = run_dir / "output" / "agent2"
    attempt_path = agent2_output_dir / "agent2_frontend_last_attempt.json"
    agent2_manifest_path = agent2_output_dir / "agent2_execution_manifest.json"
    current_run_path = agent2_output_dir / "agent2_current_run.json"

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime if path.is_file() else 0.0
        except OSError:
            return 0.0

    approval_mtime = _mtime(approved_path)
    request_mtime = _mtime(assessment_request_path)
    request_is_current = bool(
        request_mtime > 0
        and (approval_mtime <= 0 or request_mtime >= approval_mtime)
    )

    attempt = _read_json(attempt_path)
    attempt_status = str(attempt.get("status") or "").strip().casefold()
    attempt_is_current = bool(
        _mtime(attempt_path) > 0
        and (not request_is_current or _mtime(attempt_path) >= request_mtime)
    )

    # A genuine current execution failure remains a TOOL_FAILED state.  Expected
    # no-result outcomes are represented by Agent 2 packages and handled below.
    if attempt_is_current and attempt_status == "failed":
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.TOOL_FAILED,
            human_gate=HumanGate.NONE,
            manifest_status=manifest_status,
            module1_complete=True,
            module2_complete=True,
            module3_complete=True,
            topic_review_count=len(review_items),
            pending_topic_review_count=0,
            approved_topic_count=len(approved_topics),
            reason="The current Agent 2 execution attempt failed.",
        )

    manifest2 = _read_json(agent2_manifest_path)
    current_run = _read_json(current_run_path)

    def _candidate_package_path() -> Path | None:
        candidates: list[Path] = []
        for raw in [
            manifest2.get("package_path"),
            current_run.get("assessment_package_json"),
        ]:
            text_value = str(raw or "").strip()
            if not text_value:
                continue
            path = Path(text_value)
            if not path.is_absolute():
                path = agent2_output_dir / path
            candidates.append(path)
        if agent2_output_dir.is_dir():
            try:
                candidates.extend(
                    sorted(
                        agent2_output_dir.glob("agent2_assessment_package_*.json"),
                        key=lambda path: path.stat().st_mtime,
                        reverse=True,
                    )
                )
            except OSError:
                pass
        for path in candidates:
            if not path.is_file():
                continue
            # If a current request exists, reject any package older than it.
            if request_is_current and _mtime(path) < request_mtime:
                continue
            # Without a request artifact, still reject packages older than the
            # current human-approved topic handoff.
            if not request_is_current and approval_mtime and _mtime(path) < approval_mtime:
                continue
            return path
        return None

    package_path = _candidate_package_path()
    if package_path is not None:
        package = _read_json(package_path)
        questions = package.get("questions") or []
        if not isinstance(questions, list):
            questions = []
        assessment_generated_raw = package.get("assessment_generated")
        assessment_generated = (
            bool(questions)
            if assessment_generated_raw is None
            else bool(assessment_generated_raw)
        )

        if not assessment_generated or not questions:
            return _snapshot(
                run_id=run_id,
                transcript_name=resolved_transcript,
                state=WorkflowState.NO_SAFE_ASSESSMENT,
                human_gate=HumanGate.NONE,
                manifest_status=manifest_status,
                module1_complete=True,
                module2_complete=True,
                module3_complete=True,
                topic_review_count=len(review_items),
                pending_topic_review_count=0,
                approved_topic_count=len(approved_topics),
                reason=(
                    "Agent 2 completed the current request safely but returned no "
                    "compatible assessment. No weak or wrong-paper substitute was used."
                ),
            )

        # Agent 2 currently has no formal HITL/self-improving release gate.
        # A current non-empty package is therefore an ordinary ready state.
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.ASSESSMENT_READY,
            human_gate=HumanGate.NONE,
            manifest_status=manifest_status,
            module1_complete=True,
            module2_complete=True,
            module3_complete=True,
            topic_review_count=len(review_items),
            pending_topic_review_count=0,
            approved_topic_count=len(approved_topics),
            reason=(
                f"Agent 2 retrieved {len(questions)} current question(s); "
                "the assessment package is ready for Streamlit to display."
            ),
        )

    if request_is_current:
        return _snapshot(
            run_id=run_id,
            transcript_name=resolved_transcript,
            state=WorkflowState.ASSESSMENT_REQUEST_READY,
            human_gate=HumanGate.NONE,
            manifest_status=manifest_status,
            module1_complete=True,
            module2_complete=True,
            module3_complete=True,
            topic_review_count=len(review_items),
            pending_topic_review_count=0,
            approved_topic_count=len(approved_topics),
            reason=(
                "A current Agent 2 assessment request exists and retrieval may run."
            ),
        )

    return _snapshot(
        run_id=run_id,
        transcript_name=resolved_transcript,
        state=WorkflowState.TOPICS_APPROVED,
        human_gate=HumanGate.NONE,
        manifest_status=manifest_status,
        module1_complete=True,
        module2_complete=True,
        module3_complete=True,
        topic_review_count=len(review_items),
        pending_topic_review_count=0,
        approved_topic_count=len(approved_topics),
        reason=(
            f"{len(approved_topics)} topic(s) are approved for Agent 2; retrieval is allowed."
        ),
    )
