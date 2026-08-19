from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Coroutine, TypeVar


_T = TypeVar("_T")

_HUMAN_UI_TOOLS = {
    "submit_topic_review",
    "submit_detected_topic_edit",
    "save_agent2_topic_approval",
    "submit_agent2_quiz_review",
}

_AGENT1_MODULES = (
    (
        "Module 1 — Preprocessing",
        ("01_cleaned_transcript.txt", "01_preprocessing.json"),
    ),
    ("Module 2 — Semantic Chunking", ("02_chunking.json",)),
    ("Module 3 — Topic Mapping", ("03_topic_mapping.json",)),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _edtech_root(frontend_root: Path | str) -> Path:
    """Resolve the shared EDTECH root without assuming a drive/user name."""

    root = Path(frontend_root).resolve()
    candidates = [root, *root.parents]
    for candidate in candidates:
        if (candidate / "mcp_server").is_dir() and (candidate / "orchestration").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the EDTECH root containing mcp_server/ and orchestration/ "
        f"from frontend root {root}."
    )


def _prepare_imports(frontend_root: Path | str) -> Path:
    root = _edtech_root(frontend_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _load_existing_pipeline_runner(frontend_root: Path | str) -> ModuleType:
    frontend_root = Path(frontend_root).resolve()
    path = frontend_root / "frontend" / "pipeline_runner.py"
    if not path.is_file():
        raise FileNotFoundError(f"Existing Agent 1 pipeline runner not found: {path}")

    module_name = "edtech_phase10_existing_pipeline_runner"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load existing pipeline runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_coro_sync(factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """Run an async MCP call safely from a synchronous Streamlit script thread."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    result: list[_T] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(asyncio.run(factory()))
        except BaseException as exc:  # pragma: no cover - defensive thread bridge
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    if not result:
        raise RuntimeError("Async controller bridge finished without a result.")
    return result[0]


@contextmanager
def _temporary_agent2_environment(
    *,
    agent2_project_root: str | None = None,
    agent2_notebook_path: str | None = None,
):
    updates = {
        "EDTECH_AGENT2_PROJECT_ROOT": str(agent2_project_root or "").strip(),
        "EDTECH_AGENT2_NOTEBOOK05": str(agent2_notebook_path or "").strip(),
    }
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def create_controller_run(
    *,
    frontend_root: Path | str,
    filename: str,
    content: bytes,
) -> dict[str, Any]:
    """Create the same run folder/input as the existing frontend runner.

    No preprocessing/chunking/mapping operation is executed here.  The controller
    will decide which MCP stage to call from the resulting RAW_TRANSCRIPT_READY
    state.
    """

    frontend_root = Path(frontend_root).resolve()
    runner = _load_existing_pipeline_runner(frontend_root)
    run = runner.create_pipeline_run(frontend_root, filename, content)

    manifest = {
        "job_id": run.job_id,
        "transcript_name": run.transcript_name,
        "input_file": str(run.input_file),
        "output_root": str(run.output_root),
        "status": "controller_running",
        "orchestration": "mcp_controller_phase10",
        "modules": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run.run_dir / "pipeline_manifest.json", manifest)
    return {
        "run_id": run.job_id,
        "run_dir": str(run.run_dir),
        "transcript_name": run.transcript_name,
        "input_file": str(run.input_file),
    }


def _sync_pipeline_manifest(
    *,
    frontend_root: Path | str,
    run_id: str,
    controller_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep the legacy Streamlit results renderer compatible with MCP stage runs.

    The manifest is orchestration metadata only. Stage completion is derived from
    the exact output files the existing notebooks create; no algorithm result is
    fabricated or modified.
    """

    frontend_root = Path(frontend_root).resolve()
    run_dir = frontend_root / "runs" / str(run_id)
    manifest_path = run_dir / "pipeline_manifest.json"
    manifest = _read_json(manifest_path)

    transcript_name = str(manifest.get("transcript_name") or "").strip()
    if not transcript_name:
        input_files = list((run_dir / "input").glob("*")) if (run_dir / "input").is_dir() else []
        if len(input_files) == 1:
            transcript_name = input_files[0].stem
    output_dir = run_dir / "output" / transcript_name

    module_rows: list[dict[str, Any]] = []
    completed_count = 0
    for label, expected in _AGENT1_MODULES:
        complete = bool(expected) and all(_nonempty(output_dir / name) for name in expected)
        if complete:
            completed_count += 1
        module_rows.append(
            {
                "module": label,
                "status": "completed" if complete else "pending",
                "seconds": None,
                "error": None,
                "orchestrated_by": "mcp_controller_phase10",
            }
        )

    failed = False
    error_message = None
    if controller_result:
        for step in controller_result.get("steps") or []:
            if isinstance(step, dict) and step.get("tool_success") is False:
                failed = True
                error_message = str(step.get("stop_reason") or "MCP tool returned failure.")
                break

    if completed_count == len(_AGENT1_MODULES):
        # pipeline_manifest.json describes the Agent 1 transcript pipeline.
        # A later Agent 2 / quiz failure must not erase completed Agent 1 state.
        status = "completed"
    elif failed:
        status = "failed"
    else:
        status = "controller_running"

    manifest.update(
        {
            "job_id": str(manifest.get("job_id") or run_id),
            "transcript_name": transcript_name,
            "status": status,
            "modules": module_rows,
            "orchestration": "mcp_controller_phase10",
            "controller_last_sync_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    if error_message:
        manifest["controller_error"] = error_message
    else:
        manifest.pop("controller_error", None)

    if status == "completed":
        manifest["outputs"] = [
            "01_cleaned_transcript.txt",
            "01_preprocessing.json",
            "02_chunking.json",
            "03_topic_mapping.json",
        ]

    _write_json(manifest_path, manifest)
    return manifest


async def _run_controller_async(
    *,
    frontend_root: Path,
    run_id: str,
    user_request: str,
    max_steps: int,
) -> dict[str, Any]:
    from mcp import Client
    from mcp_server.server import create_mcp_server
    from orchestration.controller import EDTechController
    from orchestration.controller_state import Agent1ControllerStateProvider

    provider = Agent1ControllerStateProvider(frontend_root)
    controller = EDTechController(state_provider=provider, max_steps=max_steps)
    server = create_mcp_server(frontend_root=frontend_root)

    async with Client(server, raise_exceptions=True) as client:
        result = await controller.run_until_pause(
            client=client,
            run_id=run_id,
            user_request=user_request,
            max_steps=max_steps,
        )
    return result.model_dump(mode="json")


def run_controller_request(
    *,
    frontend_root: Path | str,
    run_id: str,
    user_request: str,
    max_steps: int = 8,
    agent2_project_root: str | None = None,
    agent2_notebook_path: str | None = None,
) -> dict[str, Any]:
    """Execute one user goal through the real controller + in-memory MCP server."""

    frontend_root = Path(frontend_root).resolve()
    _prepare_imports(frontend_root)

    with _temporary_agent2_environment(
        agent2_project_root=agent2_project_root,
        agent2_notebook_path=agent2_notebook_path,
    ):
        result = _run_coro_sync(
            lambda: _run_controller_async(
                frontend_root=frontend_root,
                run_id=str(run_id),
                user_request=str(user_request),
                max_steps=int(max_steps),
            )
        )

    _sync_pipeline_manifest(
        frontend_root=frontend_root,
        run_id=str(run_id),
        controller_result=result,
    )
    return result


def controller_snapshot(*, frontend_root: Path | str, run_id: str) -> dict[str, Any]:
    frontend_root = Path(frontend_root).resolve()
    _prepare_imports(frontend_root)
    from orchestration.controller_state import Agent1ControllerStateProvider

    return Agent1ControllerStateProvider(frontend_root).get_snapshot(run_id).to_dict()


async def _call_human_tool_async(
    *,
    frontend_root: Path,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if tool_name not in _HUMAN_UI_TOOLS:
        raise ValueError(
            f"{tool_name!r} is not an approved human-UI MCP write tool."
        )

    from mcp import Client
    from mcp_server.server import create_mcp_server
    from orchestration.controller import extract_mcp_structured_result

    server = create_mcp_server(frontend_root=frontend_root)
    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool(tool_name, arguments)
    payload = extract_mcp_structured_result(result)
    if getattr(result, "is_error", False) or not bool(payload.get("success")):
        raise RuntimeError(
            str(payload.get("error") or payload.get("message") or f"{tool_name} failed")
        )
    return payload


def _call_human_tool(
    *,
    frontend_root: Path | str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    frontend_root = Path(frontend_root).resolve()
    _prepare_imports(frontend_root)
    return _run_coro_sync(
        lambda: _call_human_tool_async(
            frontend_root=frontend_root,
            tool_name=tool_name,
            arguments=arguments,
        )
    )


def submit_human_topic_review(
    *,
    frontend_root: Path | str,
    run_id: str,
    review_id: int,
    status: str,
    reviewed_by: str = "streamlit",
    corrected_decision: str | None = None,
    corrected_mapped_concept_id: str | None = None,
    correction_reason: str | None = None,
    review_notes: str | None = None,
) -> dict[str, Any]:
    status_value = str(status).strip().casefold()
    action_map = {
        "approved": "approve",
        "corrected": "correct",
        "rejected": "reject",
    }
    if status_value not in action_map:
        raise ValueError("Status must be approved, corrected, or rejected.")

    decision: dict[str, Any] = {
        "review_id": int(review_id),
        "action": action_map[status_value],
    }
    if status_value == "corrected":
        decision.update(
            {
                "corrected_decision": corrected_decision,
                "corrected_mapped_concept_id": corrected_mapped_concept_id,
                "reason": correction_reason,
                "review_notes": review_notes,
            }
        )

    payload = _call_human_tool(
        frontend_root=frontend_root,
        tool_name="submit_topic_review",
        arguments={
            "run_id": str(run_id),
            "decisions": [decision],
            "reviewed_by": str(reviewed_by),
        },
    )
    updated = payload.get("data", {}).get("updated_reviews") or []
    return dict(updated[0]) if updated and isinstance(updated[0], dict) else payload


def submit_human_detected_topic_edit(
    *,
    frontend_root: Path | str,
    run_id: str,
    action: str,
    reason: str,
    topic_index: int | None = None,
    source_concept_id: str | None = None,
    target_concept_id: str | None = None,
    target_role: str | None = None,
    source_chunk_ids: list[int] | None = None,
    reviewed_by: str = "streamlit",
) -> dict[str, Any]:
    payload = _call_human_tool(
        frontend_root=frontend_root,
        tool_name="submit_detected_topic_edit",
        arguments={
            "run_id": str(run_id),
            "action": str(action),
            "reason": str(reason),
            "topic_index": topic_index,
            "source_concept_id": source_concept_id,
            "target_concept_id": target_concept_id,
            "target_role": target_role,
            "source_chunk_ids": list(source_chunk_ids or []),
            "reviewed_by": str(reviewed_by),
        },
    )
    data = dict(payload.get("data") or {})
    edit_record = dict(data.get("edit_record") or {})
    memory = data.get("memory") or {}
    if isinstance(memory, dict):
        memory_id = memory.get("memory_id") or memory.get("id")
        if memory_id is not None:
            edit_record.setdefault("detected_topic_edit_memory_id", memory_id)
    if not edit_record:
        edit_record = data
    return edit_record


def submit_human_agent2_topic_approval(
    *,
    frontend_root: Path | str,
    run_id: str,
    selections: list[dict[str, Any]],
    reviewed_by: str = "streamlit",
) -> dict[str, Any]:
    payload = _call_human_tool(
        frontend_root=frontend_root,
        tool_name="save_agent2_topic_approval",
        arguments={
            "run_id": str(run_id),
            "selections": selections,
            "reviewed_by": str(reviewed_by),
        },
    )
    return dict(payload.get("data") or {})


def build_assessment_request_text(
    *,
    paper: str,
    number_of_questions: int,
    target_total_marks: int,
    minimum_question_marks: int,
    maximum_question_marks: int,
    minimum_primary_questions: int,
    minimum_supporting_questions: int,
    cover_all_approved_topics: bool,
    include_code_questions: bool,
    include_visual_questions: bool,
    programming_language: str,
) -> str:
    """Serialize UI controls into an explicit, auditable controller request."""

    paper_value = str(paper or "Any").strip()
    if paper_value not in {"Paper 1", "Paper 2", "Any"}:
        paper_value = "Any"
    paper_phrase = "" if paper_value == "Any" else f" {paper_value}"

    parts = [
        f"Generate {int(number_of_questions)}{paper_phrase} questions for {int(target_total_marks)} marks.",
        f"Minimum question marks {int(minimum_question_marks)}.",
        f"Maximum question marks {int(maximum_question_marks)}.",
        f"Minimum primary questions {int(minimum_primary_questions)}.",
        f"Minimum supporting questions {int(minimum_supporting_questions)}.",
        (
            "Cover all approved topics."
            if cover_all_approved_topics
            else "Do not cover all approved topics."
        ),
        (
            "Include code questions."
            if include_code_questions
            else "Without code questions."
        ),
        (
            "Include visual questions."
            if include_visual_questions
            else "Without visual questions."
        ),
    ]
    if str(programming_language).strip().casefold() == "python":
        parts.append("Programming language Python.")
    return " ".join(parts)


def _build_agent2_filter_request_text(
    *,
    action_phrase: str,
    paper: str,
    number_of_questions: int,
    target_total_marks: int,
    minimum_question_marks: int,
    maximum_question_marks: int,
    minimum_primary_questions: int,
    minimum_supporting_questions: int,
    cover_all_approved_topics: bool,
    include_code_questions: bool,
    include_visual_questions: bool,
    programming_language: str,
) -> str:
    paper_value = str(paper or "Any").strip()
    if paper_value not in {"Paper 1", "Paper 2", "Any"}:
        paper_value = "Any"
    paper_phrase = "" if paper_value == "Any" else f" {paper_value}"
    parts = [
        f"{action_phrase}: {int(number_of_questions)}{paper_phrase} questions for {int(target_total_marks)} marks.",
        f"Minimum question marks {int(minimum_question_marks)}.",
        f"Maximum question marks {int(maximum_question_marks)}.",
        f"Minimum primary questions {int(minimum_primary_questions)}.",
        f"Minimum supporting questions {int(minimum_supporting_questions)}.",
        (
            "Cover all approved topics."
            if cover_all_approved_topics
            else "Do not cover all approved topics."
        ),
        (
            "Include code questions."
            if include_code_questions
            else "Without code questions."
        ),
        (
            "Include visual questions."
            if include_visual_questions
            else "Without visual questions."
        ),
    ]
    if str(programming_language).strip().casefold() == "python":
        parts.append("Programming language Python.")
    return " ".join(parts)


def build_complete_quiz_request_text(**kwargs: Any) -> str:
    return _build_agent2_filter_request_text(
        action_phrase="Generate complete quiz",
        **kwargs,
    )


def build_missing_quiz_request_text() -> str:
    return (
        "Generate missing quiz coverage. "
        "Reuse the exact current Notebook 05 assessment request and only fill the shortfall."
    )


def submit_human_agent2_quiz_review(
    *,
    frontend_root: Path | str,
    run_id: str,
    quiz_mode: str,
    decision: str,
    reason: str,
    reviewed_by: str = "streamlit",
) -> dict[str, Any]:
    payload = _call_human_tool(
        frontend_root=frontend_root,
        tool_name="submit_agent2_quiz_review",
        arguments={
            "run_id": str(run_id),
            "quiz_mode": str(quiz_mode),
            "decision": str(decision),
            "reason": str(reason),
            "reviewed_by": str(reviewed_by),
        },
    )
    return dict(payload.get("data") or {})
