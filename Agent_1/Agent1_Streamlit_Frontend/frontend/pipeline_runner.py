from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


NOTEBOOKS = (
    ("Module 1 — Preprocessing", "Module1 Preprocessing.ipynb"),
    ("Module 2 — Semantic Chunking", "Module2 Chunking.ipynb"),
    ("Module 3 — Topic Mapping", "Module3 Topic Mapping.ipynb"),
)


@dataclass(frozen=True)
class PipelineRun:
    job_id: str
    transcript_name: str
    run_dir: Path
    input_file: Path
    output_root: Path
    transcript_output: Path
    executed_notebooks_dir: Path
    logs_dir: Path


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" ._")
    return stem or "transcript"


def create_pipeline_run(project_root: Path, filename: str, content: bytes) -> PipelineRun:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"job_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir = project_root / "runs" / job_id
    input_dir = run_dir / "input"
    output_root = run_dir / "output"
    executed_notebooks_dir = run_dir / "executed_notebooks"
    logs_dir = run_dir / "logs"

    for directory in (input_dir, output_root, executed_notebooks_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    transcript_name = safe_stem(filename)
    suffix = Path(filename).suffix.casefold() or ".pdf"
    input_file = input_dir / f"{transcript_name}{suffix}"
    input_file.write_bytes(content)

    return PipelineRun(
        job_id=job_id,
        transcript_name=transcript_name,
        run_dir=run_dir,
        input_file=input_file,
        output_root=output_root,
        transcript_output=output_root / transcript_name,
        executed_notebooks_dir=executed_notebooks_dir,
        logs_dir=logs_dir,
    )


def _run_notebook(
    *,
    project_root: Path,
    notebook_path: Path,
    run: PipelineRun,
    label: str,
    env: dict[str, str],
) -> None:
    executed_name = f"executed_{notebook_path.name}"
    log_path = run.logs_dir / f"{notebook_path.stem}.log"

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output",
        executed_name,
        "--output-dir",
        str(run.executed_notebooks_dir),
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.allow_errors=False",
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started

    log_text = (
        f"Label: {label}\n"
        f"Notebook: {notebook_path}\n"
        f"Command: {' '.join(command)}\n"
        f"Elapsed seconds: {elapsed:.3f}\n"
        f"Return code: {completed.returncode}\n\n"
        "STDOUT\n"
        "=" * 100
        + "\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + "=" * 100
        + "\n"
        + completed.stderr
    )
    log_path.write_text(log_text, encoding="utf-8")

    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-30:])
        raise RuntimeError(
            f"{label} failed. Open {log_path.name} for the complete log.\n\n{tail}"
        )


def run_pipeline(
    *,
    project_root: Path,
    run: PipelineRun,
    progress_callback: Callable[[int, str], None] | None = None,
) -> PipelineRun:
    notebooks_dir = project_root / "Notebooks"

    missing = [
        notebooks_dir / notebook_name
        for _, notebook_name in NOTEBOOKS
        if not (notebooks_dir / notebook_name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing frontend-ready notebook(s):\n"
            + "\n".join(f"- {path}" for path in missing)
        )

    env = os.environ.copy()
    env.update(
        {
            "AGENT1_FRONTEND_MODE": "1",
            "AGENT1_INPUT_FILE": str(run.input_file.resolve()),
            "AGENT1_TRANSCRIPT_NAME": run.transcript_name,
            "AGENT1_OUTPUT_ROOT": str(run.output_root.resolve()),
            "PYTHONUTF8": "1",
        }
    )

    manifest = {
        "job_id": run.job_id,
        "transcript_name": run.transcript_name,
        "input_file": str(run.input_file),
        "output_root": str(run.output_root),
        "status": "running",
        "modules": [],
    }
    manifest_path = run.run_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for index, (label, notebook_name) in enumerate(NOTEBOOKS, start=1):
        if progress_callback:
            progress_callback(index - 1, f"Running {label}")
        started = time.perf_counter()
        try:
            _run_notebook(
                project_root=project_root,
                notebook_path=notebooks_dir / notebook_name,
                run=run,
                label=label,
                env=env,
            )
            status = "completed"
            error = None
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            manifest["modules"].append(
                {
                    "module": label,
                    "status": status,
                    "seconds": round(time.perf_counter() - started, 3),
                    "error": error,
                }
            )
            manifest["status"] = "failed"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            raise

        manifest["modules"].append(
            {
                "module": label,
                "status": status,
                "seconds": round(time.perf_counter() - started, 3),
                "error": error,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    expected = [
        "01_cleaned_transcript.txt",
        "01_preprocessing.json",
        "01_preprocessing.pdf",
        "02_chunking.json",
        "02_chunking.pdf",
        "03_topics_readable.pdf",
        "04_llm_mapping.pdf",
        "05_final_topic_summary.pdf",
        "03_topic_mapping.json",
    ]
    missing_outputs = [
        name
        for name in expected
        if not (run.transcript_output / name).is_file()
        or (run.transcript_output / name).stat().st_size == 0
    ]
    if missing_outputs:
        manifest["status"] = "failed"
        manifest["missing_outputs"] = missing_outputs
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise RuntimeError(
            "Pipeline finished but output files are missing or empty: "
            + ", ".join(missing_outputs)
        )

    manifest["status"] = "completed"
    manifest["outputs"] = expected
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if progress_callback:
        progress_callback(len(NOTEBOOKS), "Pipeline completed")

    return run


def rerun_module3(
    *,
    project_root: Path,
    run_dir: Path,
    transcript_name: str,
    progress_callback: Callable[[str], None] | None = None,
) -> PipelineRun:
    """
    Re-execute only Module 3 for an existing completed frontend run.

    This is used after a reviewer supplies a missing rough-topic label.
    Module 1 and Module 2 outputs are reused exactly as they are; they are
    neither re-executed nor modified.
    """

    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "pipeline_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Pipeline manifest was not found: {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    input_file_text = str(manifest.get("input_file") or "").strip()
    input_file = Path(input_file_text) if input_file_text else None
    if input_file is None or not input_file.is_file():
        candidates = [path for path in (run_dir / "input").glob("*") if path.is_file()]
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Could not resolve the original transcript input for this run."
            )
        input_file = candidates[0]

    output_root = run_dir / "output"
    transcript_output = output_root / transcript_name
    module2_json = transcript_output / "02_chunking.json"
    if not module2_json.is_file():
        raise FileNotFoundError(
            "Module 2 output is missing, so Module 3 cannot be rerun safely."
        )

    notebooks_dir = project_root / "Notebooks"
    notebook_path = notebooks_dir / "Module3 Topic Mapping.ipynb"
    if not notebook_path.is_file():
        raise FileNotFoundError(
            f"Module 3 notebook was not found: {notebook_path}"
        )

    run = PipelineRun(
        job_id=str(manifest.get("job_id") or run_dir.name),
        transcript_name=transcript_name,
        run_dir=run_dir,
        input_file=input_file,
        output_root=output_root,
        transcript_output=transcript_output,
        executed_notebooks_dir=run_dir / "executed_notebooks",
        logs_dir=run_dir / "logs",
    )

    run.executed_notebooks_dir.mkdir(parents=True, exist_ok=True)
    run.logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "AGENT1_FRONTEND_MODE": "1",
            "AGENT1_INPUT_FILE": str(run.input_file.resolve()),
            "AGENT1_TRANSCRIPT_NAME": run.transcript_name,
            "AGENT1_OUTPUT_ROOT": str(run.output_root.resolve()),
            "PYTHONUTF8": "1",
        }
    )

    if progress_callback:
        progress_callback("Re-running Module 3 with reviewer topic label")

    started = time.perf_counter()
    try:
        _run_notebook(
            project_root=project_root,
            notebook_path=notebook_path,
            run=run,
            label="Module 3 — Topic Mapping (review rerun)",
            env=env,
        )
    except Exception as exc:
        manifest.setdefault("module3_reruns", []).append(
            {
                "status": "failed",
                "seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise

    expected = [
        "03_topics_readable.pdf",
        "04_llm_mapping.pdf",
        "05_final_topic_summary.pdf",
        "03_topic_mapping.json",
    ]
    missing_outputs = [
        name
        for name in expected
        if not (transcript_output / name).is_file()
        or (transcript_output / name).stat().st_size == 0
    ]
    if missing_outputs:
        raise RuntimeError(
            "Module 3 rerun completed but output files are missing or empty: "
            + ", ".join(missing_outputs)
        )

    manifest.setdefault("module3_reruns", []).append(
        {
            "status": "completed",
            "seconds": round(time.perf_counter() - started, 3),
            "reason": "human_topic_label_override",
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if progress_callback:
        progress_callback("Module 3 rerun completed")

    return run
