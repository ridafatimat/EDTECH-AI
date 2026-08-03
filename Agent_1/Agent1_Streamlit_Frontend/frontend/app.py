from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from pipeline_runner import create_pipeline_run, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="Agent 1 Transcript Pipeline",
    page_icon="📘",
    layout="wide",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def deep_get(data: Any, *keys: str, default: Any = None) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def download_file(path: Path, label: str) -> None:
    if path.is_file():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=(
                "application/pdf"
                if path.suffix.casefold() == ".pdf"
                else "application/json"
                if path.suffix.casefold() == ".json"
                else "text/plain"
            ),
            use_container_width=True,
            key=f"download_{path}",
        )


def display_pdf(path: Path) -> None:
    if not path.is_file():
        st.info("PDF is not available.")
        return
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    components.html(
        f'<iframe src="data:application/pdf;base64,{encoded}" '
        'width="100%" height="720" type="application/pdf"></iframe>',
        height=740,
        scrolling=True,
    )


def render_results(run_dir: Path) -> None:
    manifest = load_json(run_dir / "pipeline_manifest.json")
    transcript_name = manifest.get("transcript_name", "transcript")
    output_folder = run_dir / "output" / transcript_name

    module1_json = load_json(output_folder / "01_preprocessing.json")
    module2_json = load_json(output_folder / "02_chunking.json")
    module3_json = load_json(output_folder / "03_topic_mapping.json")

    cleaned_text = (
        (output_folder / "01_cleaned_transcript.txt").read_text(encoding="utf-8")
        if (output_folder / "01_cleaned_transcript.txt").is_file()
        else ""
    )
    chunks = module2_json.get("chunks", [])
    module3_result = module3_json.get("module3_result", {})
    merged_topics = module3_result.get("merged_topics", [])
    llm_results = module3_json.get("llm_results", [])
    unmapped_inputs = module3_json.get("unmapped_inputs", [])

    st.success(f"Pipeline completed for: {transcript_name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cleaned words", len(cleaned_text.split()))
    col2.metric("Semantic chunks", len(chunks))
    col3.metric("Official topics", len(merged_topics))
    col4.metric("LLM fallback items", len(unmapped_inputs))

    overview_tab, module1_tab, module2_tab, module3_tab, files_tab, logs_tab = st.tabs(
        [
            "Overview",
            "Module 1 — Cleaned Transcript",
            "Module 2 — Chunks",
            "Module 3 — Topics",
            "Generated Files",
            "Execution Logs",
        ]
    )

    with overview_tab:
        st.subheader("Pipeline status")
        rows = manifest.get("modules", [])
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.json(
            {
                "job_id": manifest.get("job_id"),
                "transcript": transcript_name,
                "status": manifest.get("status"),
                "output_folder": str(output_folder),
            }
        )

    with module1_tab:
        st.subheader("Final cleaned transcript")
        st.text_area(
            "Cleaned text",
            cleaned_text,
            height=420,
            label_visibility="collapsed",
        )

        stats = deep_get(module1_json, "preprocessing_result", "stats", default={}) or {}
        technical_stats = deep_get(
            module1_json,
            "technical_normalisation_result",
            "stats",
            default={},
        ) or {}

        left, right = st.columns(2)
        with left:
            st.markdown("#### Deterministic cleaning")
            st.json(stats)
        with right:
            st.markdown("#### Technical normalisation")
            st.json(technical_stats)

        unresolved = deep_get(
            module1_json,
            "technical_normalisation_result",
            "unresolved_issues",
            default=[],
        ) or []
        if unresolved:
            st.warning(f"{len(unresolved)} unresolved technical issue(s)")
            st.dataframe(pd.DataFrame(unresolved), use_container_width=True)

    with module2_tab:
        st.subheader("Semantic chunking summary")
        summary = {
            key: value
            for key, value in module2_json.items()
            if key != "chunks"
        }
        st.json(summary)

        if not chunks:
            st.warning("No chunks were found.")
        for chunk in chunks:
            title = (
                f"Chunk {chunk.get('chunk_id')} — "
                f"{chunk.get('word_count')} words — "
                f"{chunk.get('boundary_reason')}"
            )
            with st.expander(title, expanded=len(chunks) <= 3):
                st.write(chunk.get("text", ""))
                metadata = {
                    key: value
                    for key, value in chunk.items()
                    if key != "text"
                }
                st.json(metadata)

    with module3_tab:
        st.subheader("Detected official topics")
        if merged_topics:
            topic_rows = []
            for topic in merged_topics:
                topic_rows.append(
                    {
                        "Topic": topic.get("topic"),
                        "Role": topic.get("topic_role"),
                        "Official reference": topic.get("official_reference"),
                        "Confidence": topic.get("confidence"),
                        "Ranking score": topic.get("ranking_score"),
                        "Source chunks": topic.get("source_chunk_ids"),
                    }
                )
            st.dataframe(pd.DataFrame(topic_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No official topics were retained.")

        primary = [t for t in merged_topics if t.get("topic_role") == "primary"]
        supporting = [t for t in merged_topics if t.get("topic_role") == "supporting"]
        pcol, scol = st.columns(2)
        with pcol:
            st.markdown("#### Primary topics")
            for topic in primary:
                st.write(f"• {topic.get('topic')}")
        with scol:
            st.markdown("#### Supporting topics")
            for topic in supporting:
                st.write(f"• {topic.get('topic')}")

        st.markdown("#### LLM / fallback resolution")
        if llm_results:
            st.dataframe(pd.DataFrame(llm_results), use_container_width=True, hide_index=True)
        else:
            st.info("No Module 4 result was required.")

        with st.expander("Complete Module 3 JSON"):
            st.json(module3_json)

    with files_tab:
        st.subheader("Generated output files")
        files = [
            "01_cleaned_transcript.txt",
            "01_preprocessing.json",
            "01_preprocessing.pdf",
            "02_chunking.json",
            "02_chunking.pdf",
            "03_topic_mapping.json",
            "03_topics_readable.pdf",
            "04_llm_mapping.pdf",
            "05_final_topic_summary.pdf",
        ]
        columns = st.columns(3)
        for index, filename in enumerate(files):
            with columns[index % 3]:
                path = output_folder / filename
                download_file(path, f"Download {filename}")

        st.markdown("#### Preview final summary PDF")
        display_pdf(output_folder / "05_final_topic_summary.pdf")

    with logs_tab:
        logs_dir = run_dir / "logs"
        log_paths = sorted(logs_dir.glob("*.log")) if logs_dir.is_dir() else []
        if not log_paths:
            st.info("No execution logs found.")
        for log_path in log_paths:
            with st.expander(log_path.name):
                st.code(log_path.read_text(encoding="utf-8", errors="replace"))


st.title("Agent 1 — Transcript to AQA Topic Mapping")
st.caption(
    "Upload one transcript. The page runs Module 1, then Module 2, then Module 3, "
    "and displays the complete outputs without changing the notebooks' batch mode."
)

with st.sidebar:
    st.header("Environment checks")
    st.write("Project root:")
    st.code(str(PROJECT_ROOT))
    st.write("Qdrant must be running before Module 3.")
    st.write("Groq is optional unless an unresolved topic requires fallback.")

uploaded = st.file_uploader(
    "Upload a transcript",
    type=["pdf", "docx", "txt"],
    help="PDF, DOCX, and TXT are supported by Module 1.",
)

run_clicked = st.button(
    "Run Agent 1 Pipeline",
    type="primary",
    disabled=uploaded is None,
    use_container_width=True,
)

if run_clicked and uploaded is not None:
    run = create_pipeline_run(
        PROJECT_ROOT,
        uploaded.name,
        uploaded.getvalue(),
    )

    progress = st.progress(0, text="Preparing pipeline")
    status = st.status("Running notebooks", expanded=True)

    def update_progress(completed_modules: int, message: str) -> None:
        percentage = int((completed_modules / 3) * 100)
        progress.progress(min(percentage, 100), text=message)
        status.write(message)

    try:
        run_pipeline(
            project_root=PROJECT_ROOT,
            run=run,
            progress_callback=update_progress,
        )
    except Exception as exc:
        status.update(label="Pipeline failed", state="error", expanded=True)
        st.error(str(exc))
        st.info(f"Run folder preserved for debugging: {run.run_dir}")
        st.session_state["agent1_run_dir"] = str(run.run_dir)
    else:
        progress.progress(100, text="Pipeline completed")
        status.update(label="Pipeline completed", state="complete", expanded=False)
        st.session_state["agent1_run_dir"] = str(run.run_dir)

run_dir_value = st.session_state.get("agent1_run_dir")
if run_dir_value:
    run_dir = Path(run_dir_value)
    if (run_dir / "pipeline_manifest.json").is_file():
        manifest = load_json(run_dir / "pipeline_manifest.json")
        if manifest.get("status") == "completed":
            st.divider()
            render_results(run_dir)
