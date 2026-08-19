from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, TypeVar

from .controller_bridge import (
    build_assessment_request_text,
    build_complete_quiz_request_text,
    build_missing_quiz_request_text,
    create_controller_run,
    submit_human_agent2_quiz_review,
    submit_human_agent2_topic_approval,
    submit_human_detected_topic_edit,
    submit_human_topic_review,
)

_T = TypeVar("_T")

_AGENT1_MODULES = (
    ("Module 1 — Preprocessing", ("01_cleaned_transcript.txt", "01_preprocessing.json")),
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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _edtech_root(frontend_root: Path | str) -> Path:
    root = Path(frontend_root).resolve()
    for candidate in [root, *root.parents]:
        if (candidate / "mcp_server").is_dir() and (candidate / "langgraph_orchestration").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate EDTECH root containing mcp_server/ and langgraph_orchestration/."
    )


def _prepare_imports(frontend_root: Path | str) -> Path:
    root = _edtech_root(frontend_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _run_coro_sync(factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """Run async LangGraph/Postgres safely from synchronous Streamlit.

    Psycopg async connections require SelectorEventLoop on Windows. Streamlit
    normally has no running asyncio loop in its script thread, so the coroutine
    runs in that same thread and UI callbacks can update placeholders live.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if sys.platform == "win32":
            return asyncio.run(factory(), loop_factory=asyncio.SelectorEventLoop)
        return asyncio.run(factory())

    # Defensive fallback for unusual hosts that already own the current loop.
    # Live UI callbacks are not executed from this worker thread; collected
    # updates are still returned and rendered on the next Streamlit pass.
    result: list[_T] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            if sys.platform == "win32":
                result.append(asyncio.run(factory(), loop_factory=asyncio.SelectorEventLoop))
            else:
                result.append(asyncio.run(factory()))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    if not result:
        raise RuntimeError("LangGraph Streamlit bridge finished without a result.")
    return result[0]


@contextmanager
def _temporary_agent2_environment(
    *,
    agent2_project_root: str | None = None,
    agent2_notebook_path: str | None = None,
    agent2_notebook08_path: str | None = None,
):
    updates = {
        "EDTECH_AGENT2_PROJECT_ROOT": str(agent2_project_root or "").strip(),
        "EDTECH_AGENT2_NOTEBOOK05": str(agent2_notebook_path or "").strip(),
        "EDTECH_AGENT2_NOTEBOOK08": str(agent2_notebook08_path or "").strip(),
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


def _sync_langgraph_manifest(
    *,
    frontend_root: Path | str,
    run_id: str,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frontend_root = Path(frontend_root).resolve()
    run_dir = frontend_root / "runs" / str(run_id)
    manifest_path = run_dir / "pipeline_manifest.json"
    manifest = _read_json(manifest_path)
    transcript_name = str(manifest.get("transcript_name") or "").strip()
    if not transcript_name:
        inputs = list((run_dir / "input").glob("*")) if (run_dir / "input").is_dir() else []
        if len(inputs) == 1:
            transcript_name = inputs[0].stem
    output_dir = run_dir / "output" / transcript_name

    rows: list[dict[str, Any]] = []
    completed = 0
    for label, expected in _AGENT1_MODULES:
        ok = all(_nonempty(output_dir / name) for name in expected)
        completed += int(ok)
        rows.append({
            "module": label,
            "status": "completed" if ok else "pending",
            "seconds": None,
            "error": None,
            "orchestrated_by": "langgraph_mcp_step4",
        })

    failed = bool(
        result
        and result.get("node_status") in {"failed", "blocked"}
        and result.get("tool_success") is False
    )

    # pipeline_manifest.json describes the Agent 1 transcript pipeline.
    # Once all three Agent 1 modules exist, a later Agent 2 retrieval/quiz
    # failure must never downgrade that completed pipeline to "failed".
    if completed == len(_AGENT1_MODULES):
        status = "completed"
    elif failed:
        status = "failed"
    else:
        status = "langgraph_running"

    manifest.update({
        "job_id": str(manifest.get("job_id") or run_id),
        "transcript_name": transcript_name,
        "status": status,
        "modules": rows,
        "orchestration": "langgraph_mcp_step4",
        "langgraph_last_sync_utc": datetime.now(timezone.utc).isoformat(),
    })
    if status == "completed":
        manifest["outputs"] = [
            "01_cleaned_transcript.txt", "01_preprocessing.json",
            "02_chunking.json", "03_topic_mapping.json",
        ]

    if failed and completed == len(_AGENT1_MODULES):
        manifest["downstream_last_error"] = str(
            (result or {}).get("tool_error")
            or (result or {}).get("stop_reason")
            or "A downstream Agent 2 / quiz action failed."
        )
        manifest["downstream_last_error_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
    elif not failed:
        manifest.pop("downstream_last_error", None)
        manifest.pop("downstream_last_error_utc", None)

    _write_json(manifest_path, manifest)
    return manifest


def create_langgraph_run(
    *, frontend_root: Path | str, filename: str, content: bytes
) -> dict[str, Any]:
    run = create_controller_run(frontend_root=frontend_root, filename=filename, content=content)
    manifest_path = Path(run["run_dir"]) / "pipeline_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["orchestration"] = "langgraph_mcp_step4"
    manifest["status"] = "langgraph_running"
    _write_json(manifest_path, manifest)
    return run


def langgraph_snapshot(*, frontend_root: Path | str, run_id: str) -> dict[str, Any]:
    frontend_root = Path(frontend_root).resolve()
    _prepare_imports(frontend_root)
    from orchestration.controller_state import Agent1ControllerStateProvider

    return Agent1ControllerStateProvider(frontend_root).get_snapshot(str(run_id)).to_dict()


def _interrupt_payloads(state_snapshot: Any) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for task in tuple(getattr(state_snapshot, "tasks", ()) or ()):
        for item in tuple(getattr(task, "interrupts", ()) or ()):
            raw = getattr(item, "value", item)
            if isinstance(raw, dict):
                values.append(dict(raw))
    return values


async def _run_langgraph_async(
    *,
    frontend_root: Path,
    run_id: str,
    user_request: str,
    max_steps: int,
    mode: str,
    agent2_action: str | None,
    on_update: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    from langgraph.types import Command
    from mcp import Client
    from mcp_server.server import create_mcp_server
    from orchestration.controller_state import Agent1ControllerStateProvider
    from langgraph_orchestration.checkpointing import async_postgres_checkpointer, thread_config
    from langgraph_orchestration.graph import build_hitl_graph

    provider = Agent1ControllerStateProvider(frontend_root)
    server = create_mcp_server(frontend_root=frontend_root)
    config = thread_config(run_id)
    updates: list[dict[str, Any]] = []

    async with async_postgres_checkpointer(setup=False) as checkpointer:
        async with Client(server, raise_exceptions=True) as client:
            graph = build_hitl_graph(
                provider,
                client,
                checkpointer=checkpointer,
                default_max_steps=int(max_steps),
            )
            if mode == "resume":
                graph_input: Any = Command(resume={
                    "human_action_applied": True,
                    "source": "streamlit_human_ui",
                })
            else:
                graph_input = {
                    "run_id": str(run_id),
                    "user_request": str(user_request),
                    "agent2_action": str(agent2_action or ""),
                    "events": [],
                    "completed_nodes": [],
                    "called_tools": [],
                    "execution_steps": 0,
                    "max_execution_steps": int(max_steps),
                }

            async for update in graph.astream(
                graph_input,
                config=config,
                stream_mode="updates",
            ):
                if not isinstance(update, dict):
                    continue
                update_dict = dict(update)
                updates.append(update_dict)
                if on_update is not None:
                    on_update(update_dict)

            persisted = await graph.aget_state(config)
            values = dict(getattr(persisted, "values", {}) or {})
            interrupts = _interrupt_payloads(persisted)

    if interrupts and on_update is not None:
        on_update({"__interrupt__": interrupts})

    final = {
        **values,
        "run_id": str(run_id),
        "updates": updates,
        "interrupts": interrupts,
        "interrupt_count": len(interrupts),
        "human_action_required": bool(interrupts) or bool(values.get("human_action_required")),
        "final_state": str(values.get("workflow_state") or ""),
        "orchestration": "langgraph_mcp_step4",
        "checkpoint_backend": "postgresql",
    }
    return final


def run_langgraph_request(
    *,
    frontend_root: Path | str,
    run_id: str,
    user_request: str,
    max_steps: int = 8,
    mode: str = "start",
    agent2_action: str | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    agent2_project_root: str | None = None,
    agent2_notebook_path: str | None = None,
    agent2_notebook08_path: str | None = None,
) -> dict[str, Any]:
    """Execute/resume the real LangGraph -> MCP workflow with Postgres checkpoints."""
    if mode not in {"start", "resume"}:
        raise ValueError("mode must be 'start' or 'resume'")
    frontend_root = Path(frontend_root).resolve()
    _prepare_imports(frontend_root)

    with _temporary_agent2_environment(
        agent2_project_root=agent2_project_root,
        agent2_notebook_path=agent2_notebook_path,
        agent2_notebook08_path=agent2_notebook08_path,
    ):
        result = _run_coro_sync(lambda: _run_langgraph_async(
            frontend_root=frontend_root,
            run_id=str(run_id),
            user_request=str(user_request),
            max_steps=int(max_steps),
            mode=mode,
            agent2_action=agent2_action,
            on_update=on_update,
        ))

    _sync_langgraph_manifest(frontend_root=frontend_root, run_id=str(run_id), result=result)
    return result


__all__ = [
    "build_assessment_request_text",
    "build_complete_quiz_request_text",
    "build_missing_quiz_request_text",
    "create_langgraph_run",
    "langgraph_snapshot",
    "run_langgraph_request",
    "submit_human_agent2_quiz_review",
    "submit_human_agent2_topic_approval",
    "submit_human_detected_topic_edit",
    "submit_human_topic_review",
]
