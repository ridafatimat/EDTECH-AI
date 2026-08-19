from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class Agent1NotebookAdapter:
    """Thin adapter over the existing Agent 1 frontend runner.

    It deliberately reuses PipelineRun and `_run_notebook` from the existing
    `frontend/pipeline_runner.py`. No notebook cell or Agent 1 algorithm is
    copied into MCP.
    """

    MODULES: dict[str, tuple[str, str, tuple[str, ...]]] = {
        "preprocessing": (
            "Module 1 — Preprocessing",
            "Module1 Preprocessing.ipynb",
            ("01_cleaned_transcript.txt", "01_preprocessing.json"),
        ),
        "chunking": (
            "Module 2 — Semantic Chunking",
            "Module2 Chunking.ipynb",
            ("02_chunking.json",),
        ),
        "topic_mapping": (
            "Module 3 — Topic Mapping",
            "Module3 Topic Mapping.ipynb",
            ("03_topic_mapping.json",),
        ),
    }

    def __init__(self, frontend_project_root: Path):
        self.frontend_project_root = Path(frontend_project_root).resolve()
        self._runner = self._load_pipeline_runner()

    def _load_pipeline_runner(self) -> ModuleType:
        path = self.frontend_project_root / "frontend" / "pipeline_runner.py"
        if not path.is_file():
            raise FileNotFoundError(f"Agent 1 pipeline runner was not found: {path}")
        spec = importlib.util.spec_from_file_location("agent1_existing_pipeline_runner", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load Agent 1 pipeline runner: {path}")
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolves annotation/module metadata through sys.modules
        # while the module is executing. Register this temporary adapter name
        # before exec_module, then reuse the loaded existing runner.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def _resolve_run(self, run_id: str) -> Any:
        run_dir = self.frontend_project_root / "runs" / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Agent 1 run directory was not found: {run_dir}")

        manifest_path = run_dir / "pipeline_manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    manifest = value
            except (OSError, json.JSONDecodeError):
                pass

        input_dir = run_dir / "input"
        inputs = [p for p in input_dir.glob("*") if p.is_file()] if input_dir.is_dir() else []
        if len(inputs) != 1:
            raise RuntimeError(
                f"Expected exactly one transcript input in {input_dir}, found {len(inputs)}."
            )
        input_file = inputs[0]

        transcript_name = str(manifest.get("transcript_name") or "").strip()
        if not transcript_name:
            transcript_name = self._runner.safe_stem(input_file.name)

        output_root = run_dir / "output"
        executed = run_dir / "executed_notebooks"
        logs = run_dir / "logs"
        for directory in (output_root, executed, logs):
            directory.mkdir(parents=True, exist_ok=True)

        return self._runner.PipelineRun(
            job_id=str(manifest.get("job_id") or run_id),
            transcript_name=transcript_name,
            run_dir=run_dir,
            input_file=input_file,
            output_root=output_root,
            transcript_output=output_root / transcript_name,
            executed_notebooks_dir=executed,
            logs_dir=logs,
        )

    def execute(self, *, run_id: str, module: str) -> tuple[Path, str]:
        if module not in self.MODULES:
            raise ValueError(f"Unknown Agent 1 module: {module}")

        run = self._resolve_run(run_id)
        label, notebook_name, expected_outputs = self.MODULES[module]
        notebook_path = self.frontend_project_root / "Notebooks" / notebook_name
        if not notebook_path.is_file():
            raise FileNotFoundError(f"Agent 1 notebook was not found: {notebook_path}")

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

        self._runner._run_notebook(
            project_root=self.frontend_project_root,
            notebook_path=notebook_path,
            run=run,
            label=label,
            env=env,
        )

        missing = [
            name
            for name in expected_outputs
            if not (run.transcript_output / name).is_file()
            or (run.transcript_output / name).stat().st_size == 0
        ]
        if missing:
            raise RuntimeError(
                f"{label} completed but expected output(s) are missing or empty: "
                + ", ".join(missing)
            )

        return run.run_dir, run.transcript_name
