from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    frontend_root = project_root / "frontend"
    sys.path.insert(0, str(frontend_root))

    from pipeline_runner import create_pipeline_run, run_pipeline

    rows: list[dict[str, object]] = []
    for index, input_path in enumerate(args.inputs, start=1):
        input_path = input_path.resolve()
        started = time.perf_counter()
        print(f"[{index}/{len(args.inputs)}] {input_path.name}", flush=True)

        row: dict[str, object] = {
            "input": str(input_path),
            "filename": input_path.name,
        }
        try:
            run = create_pipeline_run(
                project_root,
                input_path.name,
                input_path.read_bytes(),
            )
            run_pipeline(project_root=project_root, run=run)
            result_path = run.transcript_output / "03_topic_mapping.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            merged = payload.get("module3_result", {}).get("merged_topics", [])
            row.update(
                {
                    "status": "completed",
                    "job_id": run.job_id,
                    "run_dir": str(run.run_dir),
                    "result_json": str(result_path),
                    "topics": [
                        {
                            "topic": item.get("topic"),
                            "role": item.get("topic_role"),
                            "confidence": item.get("confidence"),
                            "ranking_score": item.get("ranking_score"),
                        }
                        for item in merged
                    ],
                }
            )
        except Exception as error:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

        row["seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  {row['status']} ({row['seconds']}s)", flush=True)

    completed = sum(row["status"] == "completed" for row in rows)
    print(f"Completed {completed}/{len(rows)} inputs.", flush=True)
    return 0 if completed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
