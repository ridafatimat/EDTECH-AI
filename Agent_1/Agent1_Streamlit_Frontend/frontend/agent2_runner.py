from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import nbformat
from dotenv import dotenv_values


AGENT2_NOTEBOOK_FILENAMES = (
    # Current user-selected final filename
    "05_agent1_topics_to_ranked_assessment_retrieval.ipynb",

    # Previous documented filename retained for compatibility
    (
        "05_agent1_topics_to_ranked_assessment_retrieval_"
        "PHASE3_BLOCK_AWARE_FINAL.ipynb"
    ),
)


@dataclass(frozen=True)
class Agent2ExecutionResult:
    agent2_project_root: Path
    source_notebook: Path
    prepared_notebook: Path
    executed_notebook: Path
    output_dir: Path
    package_path: Path
    release_readiness_path: Path | None
    log_path: Path
    manifest_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_paths(values: list[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[str] = set()
    for value in values:
        try:
            resolved = value.expanduser().resolve()
        except OSError:
            resolved = value.expanduser()
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(resolved)
    return output


def agent2_project_candidates(frontend_project_root: Path) -> list[Path]:
    root = Path(frontend_project_root).resolve()
    configured = str(os.getenv("AGENT2_PROJECT_ROOT", "")).strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            root.parent / "Agent2",
            root.parent / "Agent_2",
            root.parent.parent / "Agent2",
            root.parent.parent / "Agent_2",
            root.parent.parent / "Agent 2",
            Path.cwd() / "Agent2",
            Path.cwd() / "Agent_2",
        ]
    )
    return _unique_paths(candidates)


def resolve_agent2_project_root(
    frontend_project_root: Path,
    explicit_path: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(agent2_project_candidates(frontend_project_root))
    checked: list[str] = []
    for candidate in _unique_paths(candidates):
        checked.append(str(candidate))
        if not candidate.is_dir():
            continue
        has_environment = candidate.joinpath(".env").is_file() or bool(
            os.getenv("AGENT2_DATABASE_URL", "").strip()
        )
        has_expected_structure = any(
            path.is_dir()
            for path in [
                candidate / "cache",
                candidate / "notebooks",
                candidate / "Notebooks",
                candidate / "OUTPUT",
            ]
        )
        if has_environment and has_expected_structure:
            return candidate
    raise FileNotFoundError(
        "Could not locate the Agent 2 project folder. Set AGENT2_PROJECT_ROOT "
        "or enter the folder in Streamlit.\n\nChecked:\n- "
        + "\n- ".join(checked)
    )


def _notebook_candidates(
    *,
    agent2_project_root: Path,
    frontend_project_root: Path,
) -> list[Path]:
    root = Path(agent2_project_root)
    frontend_root = Path(frontend_project_root)
    direct: list[Path] = []

    for filename in AGENT2_NOTEBOOK_FILENAMES:
        direct.extend(
            [
                root / "notebooks" / filename,
                root / "Notebooks" / filename,
                root / filename,
                frontend_root / "Agent2_Notebooks" / filename,
            ]
        )

    discovered: list[Path] = []

    for parent in [
        root / "notebooks",
        root / "Notebooks",
        root,
        frontend_root / "Agent2_Notebooks",
    ]:
        if not parent.is_dir():
            continue

        # Prefer the exact current filename, but also support
        # any Notebook 05 retrieval filename for compatibility.
        discovered.extend(
            sorted(
                parent.glob(
                    "05_agent1_topics_to_ranked_"
                    "assessment_retrieval*.ipynb"
                )
            )
        )

    return _unique_paths(
        direct + discovered
    )


def resolve_agent2_notebook(
    *,
    agent2_project_root: Path,
    frontend_project_root: Path,
    explicit_path: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        _notebook_candidates(
            agent2_project_root=agent2_project_root,
            frontend_project_root=frontend_project_root,
        )
    )
    checked: list[str] = []
    for candidate in _unique_paths(candidates):
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the existing Agent 2 Notebook 05. Expected "
        "'05_agent1_topics_to_ranked_assessment_retrieval.ipynb', or enter "
        "its full path in Streamlit.\n\nChecked:\n- "
        + "\n- ".join(checked)
    )


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}.")
    return value


def _integration_input_source() -> str:
    return '''
# ============================================================
# STREAMLIT INTEGRATION INPUTS
#
# This code exists only in the temporary executed copy created
# by the frontend. The original Notebook 05 is not modified.
# All retrieval, adaptive threshold, visual rendering and
# Phase 3 logic below remains unchanged.
# ============================================================

approved_topics_path = Path(
    os.environ["AGENT2_APPROVED_TOPICS_PATH"]
).resolve()
assessment_request_path = Path(
    os.environ["AGENT2_ASSESSMENT_REQUEST_PATH"]
).resolve()
integration_output_dir = Path(
    os.environ["AGENT2_RUN_OUTPUT_DIR"]
).resolve()

approved_topics_payload = json.loads(
    approved_topics_path.read_text(encoding="utf-8")
)
assessment_request_payload = json.loads(
    assessment_request_path.read_text(encoding="utf-8")
)

AGENT1_TOPIC_OUTPUT = approved_topics_payload.get("topics", [])
ASSESSMENT_REQUEST = assessment_request_payload.get(
    "assessment_request",
    assessment_request_payload,
)
LESSON_SUMMARY = str(
    assessment_request_payload.get("lesson_summary", "")
).strip()

if not LESSON_SUMMARY:
    approved_topic_names = [
        str(
            item.get("topic")
            or item.get("detected_topic")
            or ""
        ).strip()
        for item in AGENT1_TOPIC_OUTPUT
    ]
    approved_topic_names = [value for value in approved_topic_names if value]
    LESSON_SUMMARY = (
        "The approved Agent 1 lesson topics are: "
        + ", ".join(approved_topic_names)
        + "."
    )

AGENT1_SOURCE_CHUNK_TEXTS = {}
for topic_item in AGENT1_TOPIC_OUTPUT:
    chunk_ids = topic_item.get("source_chunks", []) or []
    chunk_texts = topic_item.get("source_chunk_texts", []) or []
    for chunk_id, chunk_text in zip(chunk_ids, chunk_texts):
        try:
            normalized_chunk_id = int(chunk_id)
        except (TypeError, ValueError):
            continue
        cleaned_chunk_text = str(chunk_text or "").strip()
        if cleaned_chunk_text:
            AGENT1_SOURCE_CHUNK_TEXTS[normalized_chunk_id] = cleaned_chunk_text

OUTPUT_DIR = integration_output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Images are still rendered and saved. Disabling inline previews
# prevents the executed notebook from becoming very large.
DISPLAY_RENDERED_IMAGES_IN_NOTEBOOK = False

print("Streamlit integration inputs loaded.")
print(f"Approved topics: {len(AGENT1_TOPIC_OUTPUT)}")
print(f"Agent 2 output directory: {OUTPUT_DIR}")
print(
    "Topics containing actual chunk text: "
    f"{sum(bool(item.get('source_chunk_texts')) for item in AGENT1_TOPIC_OUTPUT)}"
)
display(pd.DataFrame(AGENT1_TOPIC_OUTPUT))
display(pd.DataFrame([ASSESSMENT_REQUEST]))
'''.strip()


def prepare_integration_notebook(
    *,
    source_notebook: Path,
    prepared_notebook: Path,
) -> Path:
    source_notebook = Path(source_notebook).resolve()
    notebook = nbformat.read(source_notebook, as_version=4)
    target_cell_index = None
    for index, cell in enumerate(notebook.cells):
        if (
            cell.cell_type == "code"
            and "AGENT1_TOPIC_OUTPUT = [" in cell.source
            and "ASSESSMENT_REQUEST = {" in cell.source
            and 'print(f"Project root:' in cell.source
        ):
            target_cell_index = index
            break
    if target_cell_index is None:
        raise RuntimeError(
            "Notebook 05 sample input cell could not be located. "
            "No original notebook file was changed."
        )
    source = notebook.cells[target_cell_index].source
    start = source.index("AGENT1_TOPIC_OUTPUT = [")
    end = source.index('print(f"Project root:')
    notebook.cells[target_cell_index].source = (
        source[:start] + _integration_input_source() + "\n\n\n" + source[end:]
    )
    notebook.metadata["agent2_streamlit_integration"] = {
        "source_notebook": str(source_notebook),
        "source_sha256": _sha256(source_notebook),
        "original_notebook_modified": False,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    prepared_notebook.parent.mkdir(parents=True, exist_ok=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, prepared_notebook)
    return prepared_notebook


def _execution_environment(
    *,
    frontend_project_root: Path,
    agent2_project_root: Path,
    approved_topics_path: Path,
    assessment_request_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    for env_path in [
        Path(agent2_project_root) / ".env",
        Path(frontend_project_root) / ".env",
    ]:
        if not env_path.is_file():
            continue
        for key, value in dotenv_values(env_path).items():
            if key and value is not None and not str(env.get(key, "")).strip():
                env[key] = str(value)
    if not str(env.get("AGENT2_DATABASE_URL", "")).strip():
        fallback_database_url = str(env.get("DATABASE_URL", "")).strip()
        if fallback_database_url:
            env["AGENT2_DATABASE_URL"] = fallback_database_url
    env.update(
        {
            "AGENT2_FRONTEND_MODE": "1",
            "AGENT2_PROJECT_ROOT": str(Path(agent2_project_root).resolve()),
            "AGENT2_APPROVED_TOPICS_PATH": str(Path(approved_topics_path).resolve()),
            "AGENT2_ASSESSMENT_REQUEST_PATH": str(
                Path(assessment_request_path).resolve()
            ),
            "AGENT2_RUN_OUTPUT_DIR": str(Path(output_dir).resolve()),
            "PYTHONUTF8": "1",
        }
    )
    return env


def _execution_helper_source() -> str:
    """Return the isolated notebook execution helper."""
    return 'from __future__ import annotations\n\nimport sys\nimport traceback\nfrom pathlib import Path\n\nimport nbformat\nfrom jupyter_client.kernelspec import KernelSpecManager\nfrom nbconvert.preprocessors import ExecutePreprocessor\n\n\ndef resolve_kernel_name(notebook) -> str:\n    requested = str(\n        notebook.metadata.get("kernelspec", {}).get("name", "")\n    ).strip()\n\n    available = KernelSpecManager().find_kernel_specs()\n\n    if requested and requested in available:\n        return requested\n\n    if "python3" in available:\n        return "python3"\n\n    if available:\n        return sorted(available)[0]\n\n    raise RuntimeError(\n        "No Jupyter kernels are installed. Run: python -m ipykernel install --user --name python3"\n    )\n\n\ndef main() -> int:\n    if len(sys.argv) != 4:\n        raise RuntimeError(\n            "Expected: prepared_notebook executed_notebook project_root"\n        )\n\n    prepared_path = Path(sys.argv[1]).resolve()\n    executed_path = Path(sys.argv[2]).resolve()\n    project_root = Path(sys.argv[3]).resolve()\n\n    notebook = nbformat.read(prepared_path, as_version=4)\n    kernel_name = resolve_kernel_name(notebook)\n\n    executor = ExecutePreprocessor(\n        timeout=-1,\n        kernel_name=kernel_name,\n        allow_errors=False,\n    )\n\n    resources = {\n        "metadata": {\n            "path": str(project_root),\n        }\n    }\n\n    print(f"Using Jupyter kernel: {kernel_name}")\n    print(f"Available kernels: {sorted(KernelSpecManager().find_kernel_specs())}")\n\n    try:\n        executor.preprocess(\n            notebook,\n            resources=resources,\n        )\n    except Exception:\n        executed_path.parent.mkdir(parents=True, exist_ok=True)\n        nbformat.write(notebook, executed_path)\n        traceback.print_exc()\n        return 1\n\n    executed_path.parent.mkdir(parents=True, exist_ok=True)\n    nbformat.write(notebook, executed_path)\n\n    print(f"Notebook executed from project root: {project_root}")\n    print(f"Executed notebook saved to: {executed_path}")\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


def run_agent2_notebook(
    *,
    frontend_project_root: Path,
    run_dir: Path,
    approved_topics_path: Path,
    assessment_request_path: Path,
    agent2_project_root: Path,
    source_notebook: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> Agent2ExecutionResult:
    frontend_project_root = Path(frontend_project_root).resolve()
    run_dir = Path(run_dir).resolve()
    agent2_project_root = Path(agent2_project_root).resolve()
    source_notebook = Path(source_notebook).resolve()

    approved_payload = _load_json(approved_topics_path)
    approved_topics = approved_payload.get("topics", [])
    if not isinstance(approved_topics, list) or not approved_topics:
        raise RuntimeError("The approved Agent 1 handoff contains no topics.")

    request_payload = _load_json(assessment_request_path)
    if not isinstance(
        request_payload.get("assessment_request", request_payload), dict
    ):
        raise RuntimeError("The Agent 2 assessment request is invalid.")

    execution_dir = run_dir / "executed_notebooks" / "agent2"
    output_dir = run_dir / "output" / "agent2"
    logs_dir = run_dir / "logs" / "agent2"
    for directory in [execution_dir, output_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    prepared_notebook = execution_dir / "prepared_agent2_notebook_05.ipynb"
    executed_notebook = execution_dir / "executed_agent2_notebook_05.ipynb"
    log_path = logs_dir / "agent2_notebook_05.log"
    manifest_path = output_dir / "agent2_execution_manifest.json"
    source_hash_before = _sha256(source_notebook)

    if progress_callback:
        progress_callback("Preparing a temporary Notebook 05 execution copy")
    prepare_integration_notebook(
        source_notebook=source_notebook,
        prepared_notebook=prepared_notebook,
    )

    env = _execution_environment(
        frontend_project_root=frontend_project_root,
        agent2_project_root=agent2_project_root,
        approved_topics_path=approved_topics_path,
        assessment_request_path=assessment_request_path,
        output_dir=output_dir,
    )
    # Execute through a temporary helper so the Jupyter kernel
    # receives Agent2 as its true working directory. Setting only
    # subprocess cwd is not enough because nbconvert otherwise
    # starts the kernel beside the temporary notebook copy.
    execution_helper = (
        execution_dir
        / "execute_agent2_notebook_from_project_root.py"
    )
    execution_helper.write_text(
        _execution_helper_source(),
        encoding="utf-8",
    )

    model_cache_dir = (
        agent2_project_root
        / "cache"
        / "models"
    )
    model_cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep all model caches under the real Agent2 project.
    env["SENTENCE_TRANSFORMERS_HOME"] = str(
        model_cache_dir
    )
    env["HF_HOME"] = str(
        agent2_project_root
        / "cache"
        / "huggingface"
    )

    command = [
        sys.executable,
        str(execution_helper),
        str(prepared_notebook),
        str(executed_notebook),
        str(agent2_project_root),
    ]

    if progress_callback:
        progress_callback(
            "Running Notebook 05 from the Agent 2 project root"
        )

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=agent2_project_root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(
        "Agent 2 Notebook 05 Streamlit execution\n"
        + "=" * 100
        + "\n"
        + f"Source notebook: {source_notebook}\n"
        + f"Source SHA-256 before: {source_hash_before}\n"
        + f"Prepared copy: {prepared_notebook}\n"
        + f"Executed copy: {executed_notebook}\n"
        + f"Agent 2 project root: {agent2_project_root}\n"
        + f"Kernel working directory: {agent2_project_root}\n"
        + f"Model cache directory: {model_cache_dir}\n"
        + f"Execution helper: {execution_helper}\n"
        + "Kernel selection: notebook metadata -> python3 -> first installed kernel\n"
        + f"Output directory: {output_dir}\n"
        + f"Elapsed seconds: {elapsed:.3f}\n"
        + f"Return code: {completed.returncode}\n\n"
        + "STDOUT\n"
        + "=" * 100
        + "\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + "=" * 100
        + "\n"
        + completed.stderr,
        encoding="utf-8",
    )

    source_hash_after = _sha256(source_notebook)
    if source_hash_before != source_hash_after:
        raise RuntimeError(
            "The original Agent 2 notebook hash changed unexpectedly."
        )
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-35:])
        raise RuntimeError(
            "Agent 2 Notebook 05 failed. The original notebook was not "
            "modified. Open the Agent 2 log for details.\n\n" + tail
        )

    package_files = sorted(
        output_dir.glob("agent2_assessment_package_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not package_files:
        raise RuntimeError(
            "Notebook 05 completed but no assessment package JSON was created."
        )
    package_path = package_files[0]
    release_files = sorted(
        output_dir.glob("agent2_assessment_release_readiness_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    release_path = release_files[0] if release_files else None
    output_files = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    manifest = {
        "schema_version": "agent1-agent2-notebook-execution-v1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "agent2_project_root": str(agent2_project_root),
        "source_notebook": str(source_notebook),
        "source_notebook_sha256_before": source_hash_before,
        "source_notebook_sha256_after": source_hash_after,
        "original_notebook_modified": False,
        "prepared_notebook": str(prepared_notebook),
        "executed_notebook": str(executed_notebook),
        "approved_topics_path": str(Path(approved_topics_path).resolve()),
        "assessment_request_path": str(Path(assessment_request_path).resolve()),
        "output_dir": str(output_dir),
        "package_path": str(package_path),
        "release_readiness_path": str(release_path) if release_path else None,
        "log_path": str(log_path),
        "elapsed_seconds": round(elapsed, 3),
        "output_files": output_files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if progress_callback:
        progress_callback("Agent 2 assessment package generated")
    return Agent2ExecutionResult(
        agent2_project_root=agent2_project_root,
        source_notebook=source_notebook,
        prepared_notebook=prepared_notebook,
        executed_notebook=executed_notebook,
        output_dir=output_dir,
        package_path=package_path,
        release_readiness_path=release_path,
        log_path=log_path,
        manifest_path=manifest_path,
    )