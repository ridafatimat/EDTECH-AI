from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mcp_server.schemas.visuals import RenderVisualsRequest
from mcp_server.adapters.mcp_final_pdf_adapter import finalize_mcp_quiz_pdf


VisualFamily = Literal["logic", "technical", "structured"]


VISUAL_FAMILY_TYPES: dict[str, frozenset[str]] = {
    "logic": frozenset({"logic_gate_diagram"}),
    "technical": frozenset(
        {
            "network_diagram",
            "simple_flowchart",
            "cpu_block_diagram",
        }
    ),
    "structured": frozenset(
        {
            "truth_table",
            "code_block",
            "trace_table",
            "array_grid",
            "database_table",
            "memory_grid",
            "binary_register",
        }
    ),
}

VISUAL_FAMILY_TOOL_NAMES = {
    "logic": "render_logic_visual",
    "technical": "render_technical_visual",
    "structured": "render_structured_visual",
}


class VisualRendererAdapter:
    """MCP adapter over the existing Notebook 08 visual tool layer.

    The renderer implementation stays in Notebook 08. This adapter only reads
    Notebook 06's current handoff, filters it to one semantic renderer family,
    executes Notebook 08, then merges the family result into an MCP result
    manifest. No syllabus/topic routing lives here.
    """

    def __init__(self, frontend_project_root: Path):
        self.frontend_project_root = Path(frontend_project_root).resolve()

    def _run_dir(self, run_id: str) -> Path:
        run_dir = self.frontend_project_root / "runs" / str(run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Agent 1 run not found: {run_dir}")
        return run_dir

    def _quiz_output_dir(self, run_id: str, quiz_mode: str) -> Path:
        path = self._run_dir(run_id) / "output" / "agent2_quiz" / str(quiz_mode)
        if not path.is_dir():
            raise FileNotFoundError(
                "Notebook 06 quiz output directory does not exist yet: "
                f"{path}"
            )
        return path

    def _resolve_agent2_project_root(self) -> Path:
        explicit = str(os.getenv("EDTECH_AGENT2_PROJECT_ROOT", "") or "").strip()
        if explicit:
            root = Path(explicit).expanduser().resolve()
        else:
            root = (self.frontend_project_root.parents[1] / "Agent2").resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                "Agent 2 project root not found: "
                f"{root}. Set EDTECH_AGENT2_PROJECT_ROOT if needed."
            )
        return root

    def _resolve_visual_notebook(self, agent2_root: Path) -> Path:
        explicit = str(os.getenv("EDTECH_AGENT2_NOTEBOOK08", "") or "").strip()
        candidates: list[Path] = []
        if explicit:
            candidates.append(Path(explicit))

        filenames = [
            "08_visual_generation_tool_layer.ipynb",
            "08_visual_generation_tool_layer_v2_2_docs_sync.ipynb",
            "08_visual_generation_tool_layer_v2_1_fail_closed.ipynb",
            "08_visual_generation_tool_layer_v2_kroki_schemdraw_local.ipynb",
        ]
        for folder in [
            agent2_root / "Notebooks",
            agent2_root / "notebooks",
            agent2_root,
        ]:
            for filename in filenames:
                candidates.append(folder / filename)

        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                resolved = candidate.expanduser()
            if resolved.is_file():
                return resolved

        raise FileNotFoundError(
            "Could not locate Agent 2 Notebook 08. Set EDTECH_AGENT2_NOTEBOOK08 "
            "or place 08_visual_generation_tool_layer.ipynb under Agent2/Notebooks."
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def _handoff_path(self, run_id: str, quiz_mode: str) -> Path:
        return self._quiz_output_dir(run_id, quiz_mode) / "visual_tool_handoff.json"

    def _family_handoff(
        self,
        *,
        handoff: dict[str, Any],
        family: VisualFamily,
    ) -> dict[str, Any]:
        questions = handoff.get("questions", [])
        if not isinstance(questions, list):
            raise ValueError("visual_tool_handoff.json field 'questions' must be a list.")

        allowed_types = VISUAL_FAMILY_TYPES[family]
        selected = [
            deepcopy(question)
            for question in questions
            if isinstance(question, dict)
            and str(question.get("visual_requirement", "none") or "none").strip()
            in allowed_types
        ]

        result = deepcopy(handoff)
        result["questions"] = selected
        result["mcp_visual_family"] = family
        result["mcp_visual_tool"] = VISUAL_FAMILY_TOOL_NAMES[family]
        return result

    def _execute_notebook08_family(
        self,
        *,
        request: RenderVisualsRequest,
        family: VisualFamily,
        filtered_handoff_path: Path,
    ) -> dict[str, Any]:
        agent2_root = self._resolve_agent2_project_root()
        source_notebook = self._resolve_visual_notebook(agent2_root)
        quiz_output_dir = self._quiz_output_dir(request.run_id, request.quiz_mode)
        mcp_root = quiz_output_dir / "mcp_visuals"
        family_output_dir = mcp_root / family
        execution_dir = (
            self._run_dir(request.run_id)
            / "executed_notebooks"
            / "agent2_visuals"
            / request.quiz_mode
            / family
        )
        logs_dir = (
            self._run_dir(request.run_id)
            / "logs"
            / "agent2_visuals"
            / request.quiz_mode
        )

        family_output_dir.mkdir(parents=True, exist_ok=True)
        execution_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        executed_notebook = execution_dir / f"executed_08_visual_{family}.ipynb"
        log_path = logs_dir / f"08_visual_{family}.log"

        env = os.environ.copy()
        env.update(
            {
                "AGENT2_PROJECT_ROOT": str(agent2_root),
                "AGENT2_QUIZ_OUTPUT_DIR": str(quiz_output_dir),
                "AGENT2_VISUAL_HANDOFF_PATH": str(filtered_handoff_path),
                "AGENT2_VISUAL_OUTPUT_DIR": str(family_output_dir),
                "AGENT2_VISUAL_PATCH_NOTEBOOK06_MANIFEST": "0",
                "AGENT2_VISUAL_RUN_SMOKE_TESTS": "0",
                "PYTHONUTF8": "1",
            }
        )

        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(source_notebook),
            "--output",
            executed_notebook.name,
            "--output-dir",
            str(execution_dir),
            "--ExecutePreprocessor.timeout=-1",
            "--ExecutePreprocessor.allow_errors=False",
        ]

        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=agent2_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - started

        log_path.write_text(
            "Agent 2 Notebook 08 MCP visual execution\n"
            + "=" * 100
            + "\n"
            + f"Family: {family}\n"
            + f"Tool: {VISUAL_FAMILY_TOOL_NAMES[family]}\n"
            + f"Source notebook: {source_notebook}\n"
            + f"Handoff: {filtered_handoff_path}\n"
            + f"Output directory: {family_output_dir}\n"
            + f"Elapsed seconds: {elapsed:.3f}\n"
            + f"Return code: {completed.returncode}\n\nSTDOUT\n"
            + completed.stdout
            + "\n\nSTDERR\n"
            + completed.stderr,
            encoding="utf-8",
        )

        if completed.returncode != 0:
            tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-50:])
            raise RuntimeError(
                "Agent 2 Notebook 08 visual rendering failed. Open the visual log "
                "for details.\n\n"
                + tail
            )

        results_path = family_output_dir / "notebook08_visual_results.json"
        if not results_path.is_file():
            raise RuntimeError(
                "Notebook 08 completed but notebook08_visual_results.json was not created."
            )

        family_results = self._read_json(results_path)
        family_results["mcp_status"] = "MCP_WIRED"
        family_results["mcp_tool_name"] = VISUAL_FAMILY_TOOL_NAMES[family]
        family_results["mcp_visual_family"] = family
        family_results["mcp_executed_at_utc"] = datetime.now(timezone.utc).isoformat()
        family_results["mcp_elapsed_seconds"] = round(elapsed, 3)
        family_results["mcp_log_path"] = str(log_path)
        family_results["mcp_executed_notebook"] = str(executed_notebook)
        self._write_json(results_path, family_results)
        return family_results

    @staticmethod
    def _question_key(
        question: dict[str, Any],
        fallback_index: int = 0,
    ) -> tuple[str, int]:
        return (
            str(question.get("generated_question_id", "") or ""),
            int(question.get("question_index", fallback_index) or fallback_index),
        )

    def _aggregate_results(
        self,
        *,
        request: RenderVisualsRequest,
        handoff: dict[str, Any],
        family_results: dict[str, Any],
    ) -> dict[str, Any]:
        quiz_output_dir = self._quiz_output_dir(request.run_id, request.quiz_mode)
        mcp_root = quiz_output_dir / "mcp_visuals"
        aggregate_path = mcp_root / "notebook08_visual_results.json"
        existing = self._read_json(aggregate_path)

        result_lookup: dict[tuple[str, int], dict[str, Any]] = {}
        question_lookup: dict[tuple[str, int], dict[str, Any]] = {}

        for source in [existing, family_results]:
            for index, result in enumerate(source.get("render_results", []), start=1):
                if not isinstance(result, dict):
                    continue
                key = (
                    str(result.get("generated_question_id", "") or ""),
                    int(result.get("question_index", index) or index),
                )
                result_lookup[key] = deepcopy(result)

            for index, question in enumerate(source.get("questions", []), start=1):
                if not isinstance(question, dict):
                    continue
                question_lookup[self._question_key(question, index)] = deepcopy(question)

        full_questions = handoff.get("questions", [])
        if not isinstance(full_questions, list):
            full_questions = []

        required_keys: list[tuple[str, int]] = []
        for index, question in enumerate(full_questions, start=1):
            if not isinstance(question, dict):
                continue
            visual_type = str(
                question.get("visual_requirement", "none") or "none"
            ).strip()
            if visual_type != "none":
                required_keys.append(self._question_key(question, index))

        required_key_set = set(required_keys)
        missing_required = [
            {
                "generated_question_id": key[0],
                "question_index": key[1],
                "status": "pending_mcp_tool",
            }
            for key in required_keys
            if key not in result_lookup
        ]
        required_failures = [
            result
            for key, result in result_lookup.items()
            if key in required_key_set
            and str(result.get("status", "")) != "rendered"
        ]

        if missing_required:
            integrity_status = "PENDING"
            release_eligible = False
        elif required_failures:
            integrity_status = "BLOCKED"
            release_eligible = False
        else:
            integrity_status = "PASS"
            release_eligible = True

        tools_called = sorted(
            {
                str(result.get("tool_name", "") or "")
                for result in result_lookup.values()
                if str(result.get("tool_name", "") or "").strip()
            }
        )

        aggregate = {
            "schema_version": "agent2-notebook08-mcp-results-v1.0.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_handoff_path": str(
                self._handoff_path(request.run_id, request.quiz_mode)
            ),
            "mcp_status": "MCP_WIRED",
            "mcp_tools_called": tools_called,
            "visual_integrity_gate": {
                "status": integrity_status,
                "release_eligible": release_eligible,
                "pending_required_visuals": missing_required,
                "required_visual_failures": required_failures,
            },
            "summary": {
                "question_count": len(full_questions),
                "visual_required_count": len(required_keys),
                "rendered_count": sum(
                    1
                    for key, result in result_lookup.items()
                    if key in required_key_set
                    and str(result.get("status", "")) == "rendered"
                ),
                "pending_count": len(missing_required),
                "failed_count": len(required_failures),
            },
            "render_results": list(result_lookup.values()),
            "questions": list(question_lookup.values()),
        }

        self._write_json(aggregate_path, aggregate)
        self._write_patched_manifest(request=request, aggregate=aggregate)
        return aggregate

    def _write_patched_manifest(
        self,
        *,
        request: RenderVisualsRequest,
        aggregate: dict[str, Any],
    ) -> Path:
        quiz_output_dir = self._quiz_output_dir(request.run_id, request.quiz_mode)
        source_manifest_path = quiz_output_dir / "final_quiz_manifest.json"
        if not source_manifest_path.is_file():
            raise FileNotFoundError(
                "Notebook 06 final_quiz_manifest.json was not found: "
                f"{source_manifest_path}"
            )

        manifest = deepcopy(self._read_json(source_manifest_path))
        rendered_lookup = {
            self._question_key(question, index): question
            for index, question in enumerate(aggregate.get("questions", []), start=1)
            if isinstance(question, dict)
        }

        def patch_questions(rows: Any) -> Any:
            if not isinstance(rows, list):
                return rows
            patched = []
            for index, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    patched.append(row)
                    continue
                candidate = deepcopy(row)
                key = self._question_key(candidate, index)
                rendered = rendered_lookup.get(key)
                if rendered is None and key[0]:
                    for lookup_key, lookup_value in rendered_lookup.items():
                        if lookup_key[0] == key[0]:
                            rendered = lookup_value
                            break
                if (
                    isinstance(rendered, dict)
                    and rendered.get("notebook08_visual_status") == "rendered"
                ):
                    candidate["visual_path"] = rendered.get("notebook08_visual_path")
                    candidate["visual_renderer"] = rendered.get(
                        "notebook08_visual_renderer"
                    )
                    candidate["visual_asset_sha256"] = rendered.get(
                        "notebook08_visual_asset_sha256"
                    )
                    candidate["visual_tool_name"] = rendered.get(
                        "notebook08_visual_tool_name"
                    )
                    candidate["visual_backend"] = rendered.get(
                        "notebook08_visual_backend"
                    )
                    candidate["visual_engine"] = rendered.get(
                        "notebook08_visual_engine"
                    )
                    candidate["visual_architecture_phase"] = (
                        "mcp_notebook08_tool_layer"
                    )
                patched.append(candidate)
            return patched

        for field in ["candidate_questions", "questions"]:
            manifest[field] = patch_questions(manifest.get(field, []))

        aggregate_path = quiz_output_dir / "mcp_visuals" / "notebook08_visual_results.json"
        manifest.setdefault("output_files", {})[
            "mcp_notebook08_visual_results"
        ] = str(aggregate_path)
        manifest.setdefault("source_artifacts", {})[
            "mcp_notebook08_visual_results"
        ] = str(aggregate_path)
        manifest["visual_integrity_gate"] = aggregate.get(
            "visual_integrity_gate", {}
        )
        manifest["visual_tool_architecture"] = {
            "mcp_status": "MCP_WIRED",
            "routing_owner": "LangGraph + MCP",
            "renderer_notebook": "Notebook 08",
            "semantic_tools": [
                "render_logic_visual",
                "render_technical_visual",
                "render_structured_visual",
            ],
            "tools_called": aggregate.get("mcp_tools_called", []),
            "renderer_policy": {
                "logic": "schemdraw",
                "technical": "kroki",
                "structured": "local_structured",
            },
            "separate_image_generation_api_used": False,
            "paid_ai_image_generation_used": False,
        }

        integrity = aggregate.get("visual_integrity_gate", {}) or {}
        if integrity.get("status") == "BLOCKED":
            manifest["release_ready"] = False
            manifest["visual_release_block_reason"] = (
                "One or more required MCP-routed Notebook 08 visuals failed "
                "validation or rendering."
            )

        patched_path = (
            quiz_output_dir
            / "mcp_visuals"
            / "final_quiz_manifest_with_mcp_visuals.json"
        )
        self._write_json(patched_path, manifest)
        return patched_path

    def render_family(
        self,
        request: RenderVisualsRequest,
        *,
        family: VisualFamily,
    ) -> dict[str, Any]:
        if family not in VISUAL_FAMILY_TYPES:
            raise ValueError(f"Unsupported visual family: {family}")

        handoff_path = self._handoff_path(request.run_id, request.quiz_mode)
        if not handoff_path.is_file():
            raise FileNotFoundError(
                "Notebook 06 visual_tool_handoff.json was not found. Run Notebook 06 "
                "successfully before MCP visual rendering. "
                f"Expected: {handoff_path}"
            )

        handoff = self._read_json(handoff_path)
        filtered = self._family_handoff(handoff=handoff, family=family)
        selected_questions = filtered.get("questions", [])

        quiz_output_dir = self._quiz_output_dir(request.run_id, request.quiz_mode)
        handoff_dir = quiz_output_dir / "mcp_visuals" / "handoffs"
        handoff_dir.mkdir(parents=True, exist_ok=True)
        filtered_handoff_path = handoff_dir / f"{family}_visual_handoff.json"
        self._write_json(filtered_handoff_path, filtered)

        if selected_questions:
            family_results = self._execute_notebook08_family(
                request=request,
                family=family,
                filtered_handoff_path=filtered_handoff_path,
            )
        else:
            family_results = {
                "schema_version": "agent2-notebook08-mcp-family-noop-v1.0.0",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "mcp_status": "MCP_WIRED",
                "mcp_tool_name": VISUAL_FAMILY_TOOL_NAMES[family],
                "mcp_visual_family": family,
                "render_results": [],
                "questions": [],
                "summary": {
                    "question_count": 0,
                    "visual_required_count": 0,
                    "rendered_count": 0,
                    "failed_count": 0,
                },
            }

        aggregate = self._aggregate_results(
            request=request,
            handoff=handoff,
            family_results=family_results,
        )
        patched_manifest_path = (
            quiz_output_dir
            / "mcp_visuals"
            / "final_quiz_manifest_with_mcp_visuals.json"
        )

        final_pdf_result: dict[str, Any] = {}
        aggregate_gate = (
            aggregate.get("visual_integrity_gate", {}) or {}
        )

        # Final PDF is assembled only after every required MCP visual family
        # has completed and the aggregate visual gate reaches PASS.
        if str(
            aggregate_gate.get("status", "") or ""
        ).strip().upper() == "PASS":
            final_pdf_result = finalize_mcp_quiz_pdf(
                quiz_output_dir=quiz_output_dir,
                patched_manifest_path=patched_manifest_path,
                visual_results_path=(
                    quiz_output_dir
                    / "mcp_visuals"
                    / "notebook08_visual_results.json"
                ),
            )

        family_summary = family_results.get("summary", {}) or {}
        return {
            "run_id": request.run_id,
            "quiz_mode": request.quiz_mode,
            "visual_family": family,
            "tool_name": VISUAL_FAMILY_TOOL_NAMES[family],
            "selected_question_count": len(selected_questions),
            "rendered_count": int(family_summary.get("rendered_count", 0) or 0),
            "failed_count": int(family_summary.get("failed_count", 0) or 0),
            "aggregate_status": (
                aggregate.get("visual_integrity_gate", {}) or {}
            ).get("status", ""),
            "aggregate_results_path": str(
                quiz_output_dir / "mcp_visuals" / "notebook08_visual_results.json"
            ),
            "patched_manifest_path": str(patched_manifest_path),
            "family_handoff_path": str(filtered_handoff_path),
            "final_pdf_status": str(
                final_pdf_result.get("status", "") or ""
            ),
            "final_pdf_path": str(
                final_pdf_result.get("final_pdf_path", "") or ""
            ),
            "final_pdf_uses_notebook08_assets": bool(
                final_pdf_result.get("uses_notebook08_assets", False)
            ),
        }

    def render_logic_visual(self, request: RenderVisualsRequest) -> dict[str, Any]:
        return self.render_family(request, family="logic")

    def render_technical_visual(
        self, request: RenderVisualsRequest
    ) -> dict[str, Any]:
        return self.render_family(request, family="technical")

    def render_structured_visual(
        self, request: RenderVisualsRequest
    ) -> dict[str, Any]:
        return self.render_family(request, family="structured")
