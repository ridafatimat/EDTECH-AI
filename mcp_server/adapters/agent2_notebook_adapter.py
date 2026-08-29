from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from mcp_server.schemas.agent2 import (
    GenerateAgent2CompleteQuizRequest,
    GenerateAgent2MissingQuizRequest,
    RunAgent2RetrievalRequest,
    SubmitAgent2QuizReviewRequest,
)


class Agent2NotebookAdapter:
    """Thin adapter over existing Agent 2 notebooks.

    Notebook 05 remains the official retrieval/ranking path. Notebook 06 provides
    complete_quiz and fill_shortfall modes. MCP only orchestrates those existing
    notebook capabilities; PostgreSQL/Qdrant retrieval logic is not reimplemented.
    """

    def __init__(self, frontend_project_root: Path):
        self.frontend_project_root = Path(frontend_project_root).resolve()
        self.runner = self._load_runner()

    # ------------------------------------------------------------------
    # Existing-code loading / paths
    # ------------------------------------------------------------------
    def _load_runner(self) -> ModuleType:
        runner_path = self.frontend_project_root / "frontend" / "agent2_runner.py"
        if not runner_path.is_file():
            raise FileNotFoundError(
                "Existing Agent 2 runner not found at " f"{runner_path}."
            )

        module_name = "edtech_existing_agent2_runner_for_mcp"
        existing = sys.modules.get(module_name)
        if existing is not None:
            return existing

        spec = importlib.util.spec_from_file_location(module_name, runner_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load Agent 2 runner: {runner_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _run_dir(self, run_id: str) -> Path:
        run_dir = self.frontend_project_root / "runs" / str(run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Agent 1 run not found: {run_dir}")
        return run_dir

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
    def _resolve_file(value: Any, *, output_dir: Path | None = None) -> Path | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if path.is_file():
            return path.resolve()
        if output_dir is not None and not path.is_absolute():
            candidate = output_dir / path
            if candidate.is_file():
                return candidate.resolve()
        return None

    def approved_topics_path(self, run_id: str) -> Path:
        return (
            self._run_dir(run_id)
            / "output"
            / "integration"
            / "approved_topics.json"
        )

    def assessment_request_path(self, run_id: str) -> Path:
        path = (
            self._run_dir(run_id)
            / "output"
            / "integration"
            / "assessment_request.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _transcript_name(self, run_dir: Path) -> str:
        manifest = self._read_json(run_dir / "pipeline_manifest.json")
        value = str(manifest.get("transcript_name") or "").strip()
        if value:
            return value

        output_root = run_dir / "output"
        if output_root.is_dir():
            candidates = [
                path.name
                for path in output_root.iterdir()
                if path.is_dir() and path.name not in {"integration", "agent2"}
            ]
            if len(candidates) == 1:
                return candidates[0]
        return "transcript"

    # ------------------------------------------------------------------
    # Request creation and real Notebook 05 execution
    # ------------------------------------------------------------------
    def _build_assessment_request(
        self,
        *,
        request: RunAgent2RetrievalRequest,
        approved_topics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        unique_references = sorted(
            {
                str(topic.get("official_reference") or "").strip()
                for topic in approved_topics
                if isinstance(topic, dict)
                and str(topic.get("official_reference") or "").strip()
            }
        )

        paper_code = {
            "Any": None,
            "Paper 1": "1",
            "Paper 2": "2",
        }[request.paper]
        paper_label = "Both papers" if request.paper == "Any" else request.paper
        programming_language = (
            None
            if request.programming_language == "Automatic"
            else request.programming_language
        )

        minimum_distinct = 1
        if request.cover_all_approved_topics and unique_references:
            minimum_distinct = min(
                len(unique_references), request.number_of_questions
            )

        # This is intentionally the same family of fields the current
        # Streamlit frontend passes into Notebook 05.  Notebook 05 remains free
        # to adapt infeasible constraints exactly as it does today.
        payload: dict[str, Any] = {
            "number_of_questions": int(request.number_of_questions),
            "target_total_marks": int(request.target_total_marks),
            "minimum_question_marks": int(request.minimum_question_marks),
            "maximum_question_marks": int(request.maximum_question_marks),
            "minimum_primary_questions": int(request.minimum_primary_questions),
            "minimum_supporting_questions": int(request.minimum_supporting_questions),
            "minimum_distinct_official_references": int(minimum_distinct),
            "cover_all_approved_topics": bool(request.cover_all_approved_topics),
            "include_code_questions": bool(request.include_code_questions),
            "include_visual_questions": bool(request.include_visual_questions),
            "paper_code": paper_code,
            "paper_filter_label": paper_label,
            "programming_language": programming_language,
        }
        return payload

    def write_assessment_request(
        self, request: RunAgent2RetrievalRequest
    ) -> tuple[Path, dict[str, Any]]:
        run_dir = self._run_dir(request.run_id)
        approved_path = self.approved_topics_path(request.run_id)
        approved_payload = self._read_json(approved_path)
        approved_topics = approved_payload.get("topics") or []
        if not isinstance(approved_topics, list):
            approved_topics = []
        approved_topics = [
            dict(item) for item in approved_topics if isinstance(item, dict)
        ]
        if not approved_topics:
            raise RuntimeError(
                "The current run has no human-approved Agent 1 -> Agent 2 topics."
            )

        assessment_request = self._build_assessment_request(
            request=request,
            approved_topics=approved_topics,
        )
        topic_names = [
            str(
                topic.get("topic")
                or topic.get("detected_topic")
                or topic.get("official_concept_name")
                or ""
            ).strip()
            for topic in approved_topics
        ]
        topic_names = [name for name in topic_names if name]

        output_path = self.assessment_request_path(request.run_id)
        payload = {
            "schema_version": "agent2-assessment-request-v1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": run_dir.name,
            "transcript": self._transcript_name(run_dir),
            "lesson_summary": (
                "The approved lesson topics are: " + ", ".join(topic_names) + "."
                if topic_names
                else "Approved Agent 1 topics are available for retrieval."
            ),
            "assessment_request": assessment_request,
            "notebook_logic_changed": False,
            "execution_method": "temporary_parameterized_copy",
            "mcp_phase": "phase8_agent2_wrapper",
        }
        if request.user_request:
            payload["user_request"] = request.user_request

        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path, payload

    def _resolve_agent2_project_root(self) -> Path:
        explicit = os.getenv("EDTECH_AGENT2_PROJECT_ROOT", "").strip()
        default_root = self.frontend_project_root.parents[1] / "Agent2"
        explicit_path = explicit or str(default_root)

        resolver = getattr(self.runner, "resolve_agent2_project_root", None)
        if resolver is None:
            root = Path(explicit_path)
            if not root.is_dir():
                raise FileNotFoundError(f"Agent 2 project root not found: {root}")
            return root.resolve()

        return Path(
            resolver(
                self.frontend_project_root,
                explicit_path=explicit_path,
            )
        ).resolve()

    def _resolve_agent2_notebook(self, agent2_project_root: Path) -> Path:
        explicit = os.getenv("EDTECH_AGENT2_NOTEBOOK05", "").strip() or None
        resolver = getattr(self.runner, "resolve_agent2_notebook", None)
        if resolver is None:
            raise RuntimeError(
                "The existing agent2_runner.py does not expose resolve_agent2_notebook."
            )
        return Path(
            resolver(
                agent2_project_root=agent2_project_root,
                frontend_project_root=self.frontend_project_root,
                explicit_path=explicit,
            )
        ).resolve()

    def execute_retrieval(self, request: RunAgent2RetrievalRequest) -> dict[str, Any]:
        run_dir = self._run_dir(request.run_id)
        approved_path = self.approved_topics_path(request.run_id)
        if not approved_path.is_file():
            raise RuntimeError(
                "Agent 2 retrieval is blocked until a human-approved "
                "approved_topics.json handoff exists."
            )

        request_path, request_payload = self.write_assessment_request(request)
        agent2_root = self._resolve_agent2_project_root()
        source_notebook = self._resolve_agent2_notebook(agent2_root)

        runner = getattr(self.runner, "run_agent2_notebook", None)
        if runner is None:
            raise RuntimeError(
                "The existing agent2_runner.py does not expose run_agent2_notebook."
            )

        result = runner(
            frontend_project_root=self.frontend_project_root,
            run_dir=run_dir,
            approved_topics_path=approved_path,
            assessment_request_path=request_path,
            agent2_project_root=agent2_root,
            source_notebook=source_notebook,
            progress_callback=None,
        )

        manifest_path = Path(getattr(result, "manifest_path", "") or "")
        package_path = Path(getattr(result, "package_path", "") or "")
        manifest = self._read_json(manifest_path) if manifest_path.is_file() else {}
        if not package_path.is_file():
            package_path = self.current_package_path(request.run_id) or package_path
        package = self._read_json(package_path) if package_path.is_file() else {}

        return {
            "assessment_request_path": str(request_path.resolve()),
            "assessment_request": request_payload,
            "agent2_project_root": str(agent2_root),
            "source_notebook": str(source_notebook),
            "manifest_path": str(manifest_path.resolve()) if manifest_path.is_file() else None,
            "package_path": str(package_path.resolve()) if package_path.is_file() else None,
            "manifest": manifest,
            "package": package,
            "question_count": self._question_count(package),
            "assessment_generated": self._assessment_generated(package),
        }


    # ------------------------------------------------------------------
    # Notebook 06 quiz generation
    # ------------------------------------------------------------------
    def _resolve_quiz_notebook(self, agent2_project_root: Path) -> Path:
        explicit = os.getenv("EDTECH_AGENT2_NOTEBOOK06", "").strip()
        candidates = []
        if explicit:
            candidates.append(Path(explicit))
        for folder in [
            agent2_project_root / "Notebooks",
            agent2_project_root / "notebooks",
            agent2_project_root,
        ]:
            candidates.extend([
                folder / "06_quiz_generation.ipynb",
                folder / "06_quiz_generation_FINAL.ipynb",
            ])
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                resolved = candidate.expanduser()
            if resolved.is_file():
                return resolved
        raise FileNotFoundError(
            "Could not locate Agent 2 Notebook 06. Expected "
            "06_quiz_generation.ipynb under Agent2/Notebooks."
        )

    def _quiz_output_dir(self, run_id: str, quiz_mode: str) -> Path:
        path = (
            self._run_dir(run_id)
            / "output"
            / "agent2_quiz"
            / str(quiz_mode)
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _quiz_request_path(self, run_id: str, quiz_mode: str) -> Path:
        path = (
            self._run_dir(run_id)
            / "output"
            / "integration"
            / f"quiz_request_{quiz_mode}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _approved_topics(self, run_id: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
        path = self.approved_topics_path(run_id)
        payload = self._read_json(path)
        topics = payload.get("topics") or []
        if not isinstance(topics, list):
            topics = []
        topics = [dict(item) for item in topics if isinstance(item, dict)]
        if not topics:
            raise RuntimeError(
                "Quiz generation is blocked until human-approved Agent 1 topics exist."
            )
        return path, payload, topics

    def _quiz_filters_from_request(
        self,
        request: GenerateAgent2CompleteQuizRequest,
        approved_topics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Reuse exactly the same filter mapping as Notebook 05.
        return self._build_assessment_request(
            request=request,
            approved_topics=approved_topics,
        )

    @staticmethod
    def _lesson_summary_from_topics(topics: list[dict[str, Any]]) -> str:
        evidence = []
        for topic in topics:
            for value in topic.get("source_chunk_texts") or []:
                text = str(value or "").strip()
                if text and text not in evidence:
                    evidence.append(text)
                if len(evidence) >= 4:
                    break
            if len(evidence) >= 4:
                break
        if evidence:
            return " ".join(evidence)
        names = [
            str(item.get("topic") or item.get("detected_topic") or "").strip()
            for item in topics
        ]
        names = [name for name in names if name]
        return (
            "The approved lesson topics are: " + ", ".join(names) + "."
            if names
            else "Approved Agent 1 topics are available for quiz generation."
        )

    def _matching_selected_csv(self, package_path: Path) -> Path:
        match = __import__("re").search(
            r"agent2_assessment_package_(\d{8}_\d{6})\.json$",
            package_path.name,
        )
        if not match:
            raise RuntimeError(
                "Could not determine Notebook 05 run timestamp from package filename."
            )
        selected = package_path.parent / f"agent2_selected_questions_{match.group(1)}.csv"
        if not selected.is_file():
            raise FileNotFoundError(
                f"Matching Notebook 05 selected-question CSV not found: {selected}"
            )
        return selected.resolve()

    def _run_quiz_notebook(
        self,
        *,
        run_id: str,
        quiz_mode: str,
        assessment_request: dict[str, Any],
        approved_topics: list[dict[str, Any]],
        lesson_summary: str,
        notebook05_package_path: Path | None = None,
        notebook05_selected_csv_path: Path | None = None,
        source_notebook: Path | None = None,
        run_generation: bool,
        review_decision: str = "pending",
        review_reason: str = "",
    ) -> dict[str, Any]:
        import subprocess
        import sys
        import time

        agent2_root = self._resolve_agent2_project_root()

        if source_notebook is None:
            source_notebook = self._resolve_quiz_notebook(agent2_root)
        else:
            source_notebook = Path(source_notebook).expanduser().resolve()
            if not source_notebook.is_file():
                raise FileNotFoundError(
                    f"Persisted Agent 2 quiz notebook was not found: {source_notebook}"
                )

        run_dir = self._run_dir(run_id)
        output_dir = self._quiz_output_dir(run_id, quiz_mode)
        execution_dir = run_dir / "executed_notebooks" / "agent2_quiz" / quiz_mode
        logs_dir = run_dir / "logs" / "agent2_quiz"
        execution_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        executed_notebook = execution_dir / "executed_06_quiz_generation.ipynb"
        log_path = logs_dir / f"06_quiz_generation_{quiz_mode}.log"
        execution_manifest_path = output_dir / "quiz_execution_manifest.json"

        env = os.environ.copy()
        env.update({
            "AGENT2_PROJECT_ROOT": str(agent2_root),
            "AGENT2_QUIZ_OUTPUT_DIR": str(output_dir),
            "AGENT2_QUIZ_MODE": quiz_mode,
            "AGENT2_AGENT1_TOPICS_JSON": json.dumps(approved_topics, ensure_ascii=False, default=str),
            "AGENT2_ASSESSMENT_REQUEST_JSON": json.dumps(assessment_request, ensure_ascii=False, default=str),
            "AGENT2_LESSON_SUMMARY": str(lesson_summary or ""),
            "AGENT2_QUIZ_USER_APPROVED_GENERATION": "1",
            "AGENT2_QUIZ_RUN_GENERATION": "1" if run_generation else "0",
            "AGENT2_QUIZ_REVIEW_DECISION": str(review_decision or "pending"),
            "AGENT2_QUIZ_REVIEW_REASON": str(review_reason or ""),
            "PYTHONUTF8": "1",
        })
        if notebook05_package_path is not None:
            env["AGENT2_NOTEBOOK05_PACKAGE_PATH"] = str(notebook05_package_path.resolve())
        else:
            env.pop("AGENT2_NOTEBOOK05_PACKAGE_PATH", None)
        if notebook05_selected_csv_path is not None:
            env["AGENT2_NOTEBOOK05_SELECTED_CSV_PATH"] = str(notebook05_selected_csv_path.resolve())
        else:
            env.pop("AGENT2_NOTEBOOK05_SELECTED_CSV_PATH", None)

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
            "Agent 2 Notebook 06 execution\n"
            + "=" * 100 + "\n"
            + f"Mode: {quiz_mode}\n"
            + f"Source notebook: {source_notebook}\n"
            + f"Output directory: {output_dir}\n"
            + f"Elapsed seconds: {elapsed:.3f}\n"
            + f"Return code: {completed.returncode}\n\nSTDOUT\n"
            + completed.stdout
            + "\n\nSTDERR\n"
            + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-40:])
            raise RuntimeError(
                "Agent 2 Notebook 06 failed. Open the quiz log for details.\n\n" + tail
            )

        final_manifest_path = output_dir / "final_quiz_manifest.json"
        if not final_manifest_path.is_file():
            raise RuntimeError(
                "Notebook 06 completed but final_quiz_manifest.json was not created."
            )
        final_manifest = self._read_json(final_manifest_path)
        execution_manifest = {
            "schema_version": "agent2-quiz-execution-v1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "quiz_mode": quiz_mode,
            "source_notebook": str(source_notebook),
            "executed_notebook": str(executed_notebook),
            "output_dir": str(output_dir),
            "final_manifest_path": str(final_manifest_path),
            "log_path": str(log_path),
            "elapsed_seconds": round(elapsed, 3),
            "review_decision": review_decision,
            "run_generation": bool(run_generation),
        }
        execution_manifest_path.write_text(
            json.dumps(execution_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "quiz_mode": quiz_mode,
            "outcome": final_manifest.get("assessment_type"),
            "release_ready": bool(final_manifest.get("release_ready")),
            "human_review_state": final_manifest.get("generated_human_review_state"),
            "final_manifest_path": str(final_manifest_path),
            "final_manifest": final_manifest,
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "executed_notebook": str(executed_notebook),
        }

    def execute_complete_quiz(self, request: GenerateAgent2CompleteQuizRequest) -> dict[str, Any]:
        _, _, topics = self._approved_topics(request.run_id)
        filters = self._quiz_filters_from_request(request, topics)

        # Resolve the currently selected 06 / 06B / 06C once and persist the
        # exact source path with this quiz request. This makes later HITL
        # review/regeneration use the same generation plan even after the
        # temporary Streamlit environment has been restored.
        agent2_root = self._resolve_agent2_project_root()
        selected_quiz_notebook = self._resolve_quiz_notebook(agent2_root)

        request_path = self._quiz_request_path(request.run_id, "complete_quiz")
        payload = {
            "schema_version": "agent2-complete-quiz-request-v1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "assessment_request": filters,
            "user_request": request.user_request,
            "source_notebook": str(selected_quiz_notebook),
        }
        request_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self._run_quiz_notebook(
            run_id=request.run_id,
            quiz_mode="complete_quiz",
            assessment_request=filters,
            approved_topics=topics,
            lesson_summary=self._lesson_summary_from_topics(topics),
            source_notebook=selected_quiz_notebook,
            run_generation=True,
        )

    def execute_missing_quiz(self, request: GenerateAgent2MissingQuizRequest) -> dict[str, Any]:
        _, _, topics = self._approved_topics(request.run_id)
        package_path = self.current_package_path(request.run_id)
        if package_path is None or not package_path.is_file():
            raise RuntimeError(
                "Generate Missing Quiz Coverage requires a current Notebook 05 assessment package."
            )
        selected_path = self._matching_selected_csv(package_path)
        package = self._read_json(package_path)
        filters = package.get("assessment_request") or {}
        if not isinstance(filters, dict):
            raise RuntimeError("Current Notebook 05 package has no valid assessment_request.")
        lesson_summary = str(package.get("lesson_summary") or "").strip()
        if not lesson_summary:
            lesson_summary = self._lesson_summary_from_topics(topics)
        agent2_root = self._resolve_agent2_project_root()
        selected_quiz_notebook = self._resolve_quiz_notebook(agent2_root)

        request_path = self._quiz_request_path(request.run_id, "fill_shortfall")
        payload = {
            "schema_version": "agent2-missing-quiz-request-v1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "notebook05_package_path": str(package_path.resolve()),
            "notebook05_selected_csv_path": str(selected_path),
            "assessment_request": filters,
            "user_request": request.user_request,
            "source_notebook": str(selected_quiz_notebook),
        }
        request_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return self._run_quiz_notebook(
            run_id=request.run_id,
            quiz_mode="fill_shortfall",
            assessment_request=filters,
            approved_topics=topics,
            lesson_summary=lesson_summary,
            notebook05_package_path=package_path,
            notebook05_selected_csv_path=selected_path,
            source_notebook=selected_quiz_notebook,
            run_generation=True,
        )

    def submit_quiz_review(self, request: SubmitAgent2QuizReviewRequest) -> dict[str, Any]:
        _, _, topics = self._approved_topics(request.run_id)
        quiz_mode = request.quiz_mode
        saved_request = self._read_json(self._quiz_request_path(request.run_id, quiz_mode))
        filters = saved_request.get("assessment_request") or {}
        if not isinstance(filters, dict) or not filters:
            raise RuntimeError(
                "No matching persisted quiz request exists for this human review."
            )

        persisted_source_raw = str(
            saved_request.get("source_notebook") or ""
        ).strip()
        persisted_source_notebook = (
            Path(persisted_source_raw).expanduser().resolve()
            if persisted_source_raw
            else None
        )

        package_path = None
        selected_path = None
        if quiz_mode == "fill_shortfall":
            raw_package = str(saved_request.get("notebook05_package_path") or "").strip()
            raw_selected = str(saved_request.get("notebook05_selected_csv_path") or "").strip()
            if not raw_package or not raw_selected:
                raise RuntimeError("Missing exact Notebook 05 lineage for hybrid review.")
            package_path = Path(raw_package)
            selected_path = Path(raw_selected)
            if not package_path.is_file() or not selected_path.is_file():
                raise RuntimeError("The persisted Notebook 05 lineage files are no longer available.")
        return self._run_quiz_notebook(
            run_id=request.run_id,
            quiz_mode=quiz_mode,
            assessment_request=filters,
            approved_topics=topics,
            lesson_summary=self._lesson_summary_from_topics(topics),
            notebook05_package_path=package_path,
            notebook05_selected_csv_path=selected_path,
            source_notebook=persisted_source_notebook,
            run_generation=False,
            review_decision=request.decision,
            review_reason=request.reason,
        )

    def get_current_quiz(self, run_id: str, quiz_mode: str) -> dict[str, Any]:
        output_dir = self._quiz_output_dir(run_id, quiz_mode)
        manifest_path = output_dir / "final_quiz_manifest.json"
        return {
            "quiz_mode": quiz_mode,
            "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
            "manifest": self._read_json(manifest_path),
            "output_dir": str(output_dir),
        }

    # ------------------------------------------------------------------
    # Current-run read path
    # ------------------------------------------------------------------
    @staticmethod
    def _question_count(package: dict[str, Any]) -> int:
        questions = package.get("questions") or []
        return len(questions) if isinstance(questions, list) else 0

    @classmethod
    def _assessment_generated(cls, package: dict[str, Any]) -> bool:
        raw = package.get("assessment_generated")
        if raw is not None:
            return bool(raw)
        return cls._question_count(package) > 0

    def current_package_path(self, run_id: str) -> Path | None:
        run_dir = self._run_dir(run_id)
        output_dir = run_dir / "output" / "agent2"
        manifest = self._read_json(output_dir / "agent2_execution_manifest.json")
        path = self._resolve_file(manifest.get("package_path"), output_dir=output_dir)
        if path is not None:
            return path

        current = self._read_json(output_dir / "agent2_current_run.json")
        path = self._resolve_file(
            current.get("assessment_package_json"), output_dir=output_dir
        )
        if path is not None:
            return path

        candidates = sorted(
            output_dir.glob("agent2_assessment_package_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if output_dir.is_dir() else []
        return candidates[0].resolve() if candidates else None

    def get_current_assessment(self, run_id: str) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        output_dir = run_dir / "output" / "agent2"
        manifest_path = output_dir / "agent2_execution_manifest.json"
        attempt_path = output_dir / "agent2_frontend_last_attempt.json"
        current_path = output_dir / "agent2_current_run.json"
        package_path = self.current_package_path(run_id)
        package = self._read_json(package_path) if package_path else {}
        questions = package.get("questions") or []
        if not isinstance(questions, list):
            questions = []

        return {
            "manifest_path": str(manifest_path.resolve()) if manifest_path.is_file() else None,
            "manifest": self._read_json(manifest_path),
            "frontend_attempt_path": str(attempt_path.resolve()) if attempt_path.is_file() else None,
            "frontend_attempt": self._read_json(attempt_path),
            "current_run_path": str(current_path.resolve()) if current_path.is_file() else None,
            "current_run": self._read_json(current_path),
            "package_path": str(package_path) if package_path else None,
            "package": package,
            "questions": questions,
            "question_count": len(questions),
            "assessment_generated": self._assessment_generated(package),
        }

    @staticmethod
    def _question_id(item: dict[str, Any]) -> str:
        question = item.get("question") or {}
        candidates = [
            item.get("question_id"),
            question.get("question_id") if isinstance(question, dict) else None,
            question.get("id") if isinstance(question, dict) else None,
            item.get("id"),
        ]
        for value in candidates:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    def _question_index(self, run_id: str) -> dict[str, dict[str, Any]]:
        assessment = self.get_current_assessment(run_id)
        return {
            self._question_id(item): dict(item)
            for item in assessment["questions"]
            if isinstance(item, dict) and self._question_id(item)
        }

    def fetch_mark_schemes(
        self, run_id: str, question_ids: list[str]
    ) -> dict[str, Any]:
        index = self._question_index(run_id)
        results: list[dict[str, Any]] = []
        missing: list[str] = []
        for raw_id in question_ids:
            question_id = str(raw_id).strip()
            item = index.get(question_id)
            if item is None:
                missing.append(question_id)
                continue
            results.append(
                {
                    "question_id": question_id,
                    "mark_scheme": item.get("mark_scheme") or {},
                    "source": "current_agent2_assessment_package",
                    "deterministic_question_id_link": True,
                }
            )
        return {
            "results": results,
            "result_count": len(results),
            "missing_question_ids": missing,
        }

    def get_rendered_question_pages(
        self, run_id: str, question_ids: list[str]
    ) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        output_dir = run_dir / "output" / "agent2"
        index = self._question_index(run_id)
        results: list[dict[str, Any]] = []
        missing: list[str] = []

        for raw_id in question_ids:
            question_id = str(raw_id).strip()
            item = index.get(question_id)
            if item is None:
                missing.append(question_id)
                continue
            question = item.get("question") or {}
            if not isinstance(question, dict):
                question = {}
            raw_paths = (
                question.get("rendered_page_images")
                or item.get("rendered_page_images")
                or []
            )
            if not isinstance(raw_paths, list):
                raw_paths = []
            paths: list[str] = []
            for raw in raw_paths:
                path = Path(str(raw))
                if not path.is_file() and not path.is_absolute():
                    path = output_dir / path
                paths.append(str(path.resolve()) if path.exists() else str(path))
            results.append(
                {
                    "question_id": question_id,
                    "rendered_page_images": paths,
                    "rendered_page_count": len(paths),
                    "already_rendered_by_existing_agent2": True,
                }
            )

        return {
            "results": results,
            "result_count": len(results),
            "missing_question_ids": missing,
            "note": (
                "Phase 8 reads page images already produced by the existing "
                "Agent 2/Notebook 07 rendering path; MCP does not rerender PDFs."
            ),
        }