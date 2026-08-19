from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage,
    )
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage,
    )

GENERATION_MODEL = ""
QUIZ_MODE = ""
request: dict[str, Any] = {}


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def question_marks(question: dict[str, Any]) -> int:
    for key in ("marks", "marks_numeric", "question_marks"):
        try:
            if question.get(key) is not None:
                return int(question.get(key))
        except (TypeError, ValueError):
            pass
    nested = question.get("question")
    if isinstance(nested, dict):
        try:
            return int(nested.get("marks") or 0)
        except (TypeError, ValueError):
            pass
    return 0


def question_text(question: dict[str, Any]) -> str:
    for key in (
        "question_text", "question_text_canonical",
        "question_text_postgres", "question_text_retrieval", "text",
    ):
        value = str(question.get(key) or "").strip()
        if value:
            return value
    nested = question.get("question")
    if isinstance(nested, dict):
        return str(nested.get("text") or "").strip()
    return ""


def question_reference(question: dict[str, Any]) -> str:
    for key in (
        "official_reference", "agent1_official_reference",
        "official_reference_canonical",
    ):
        value = str(question.get(key) or "").strip()
        if value:
            return value
    topic = question.get("topic")
    if isinstance(topic, dict):
        return str(topic.get("official_reference") or "").strip()
    return ""


def normalize_visual_requirement(value: Any) -> str:
    return str(value or "none").strip().casefold() or "none"


def _is_generated_question(question: dict[str, Any]) -> bool:
    return bool(
        str(question.get("source_type") or "").strip().casefold()
        == "ai_generated_aqa_aligned"
        or str(question.get("generated_question_id") or "").strip()
    )


def _resolve_existing_path(value: Any, roots: list[Path]) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        for root in roots:
            candidate = Path(root) / path
            if candidate.is_file():
                return candidate.resolve()
    return None


def _pdf_visual_image_path(question: dict[str, Any]) -> Path | None:
    roots = [
        Path(str(question.get("_mcp_quiz_output_dir") or ".")),
        Path(str(question.get("_mcp_visual_output_dir") or ".")),
    ]
    for key in ("notebook08_visual_path", "visual_path"):
        path = _resolve_existing_path(question.get(key), roots)
        if path is not None:
            return path
    return None


def _pdf_add_visual(story: list[Any], question: dict[str, Any], *, styles: Any) -> None:
    required_type = normalize_visual_requirement(
        question.get("visual_requirement", "none")
    )
    requires_visual = bool(
        question.get("requires_visual", required_type != "none")
    )
    if required_type == "none" or not requires_visual:
        return

    visual_path = _pdf_visual_image_path(question)
    if visual_path is None or not visual_path.is_file():
        raise RuntimeError(
            "Required MCP/Notebook 08 visual is missing for "
            f"{question.get('generated_question_id') or question.get('plan_index')} "
            f"({required_type})."
        )

    image = RLImage(str(visual_path))
    max_width = A4[0] - 40 * mm
    max_height = 60 * mm if required_type == "code_block" else 95 * mm
    width = float(image.imageWidth)
    height = float(image.imageHeight)
    scale = min(
        1.0,
        max_width / max(1.0, width),
        max_height / max(1.0, height),
    )
    image.drawWidth = width * scale
    image.drawHeight = height * scale
    story.extend([Spacer(1, 4), image, Spacer(1, 6)])


def _pdf_visual_integrity_preflight(
    questions: list[dict[str, Any]], *, styles: Any
) -> dict[str, Any]:
    errors = []
    diagnostics = []
    for position, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            continue
        visual_type = normalize_visual_requirement(
            question.get("visual_requirement", "none")
        )
        requires_visual = bool(
            question.get("requires_visual", visual_type != "none")
        )
        if visual_type == "none" or not requires_visual:
            continue

        qid = str(
            question.get("generated_question_id")
            or question.get("question_id")
            or f"Q{position}"
        )
        path = _pdf_visual_image_path(question)
        if path is None or not path.is_file():
            errors.append(
                f"{qid}: required MCP/Notebook 08 visual missing ({visual_type})"
            )
        else:
            diagnostics.append({
                "question_id": qid,
                "status": "ready",
                "visual_requirement": visual_type,
                "path": str(path),
                "asset_source": "mcp_notebook08",
            })

    if errors:
        raise RuntimeError(
            "MCP final PDF visual-integrity preflight failed: " + " | ".join(errors)
        )
    return {"status": "PASS", "errors": [], "diagnostics": diagnostics}

def _pdf_clean_text(value: Any) -> str:
    """Normalize common glyphs before ReportLab rendering."""
    text_value = str(
        value
        or ""
    )

    replacements = {
        "🡨": "<-",
        "←": "<-",
        "≤": "<=",
        "≥": ">=",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "\u00a0": " ",
        "\u200b": "",
        "\u2060": "",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "•": "-",
        "×": "x",
    }

    for old, new in replacements.items():
        text_value = text_value.replace(
            old,
            new,
        )

    return text_value

def _pdf_escape(value: Any) -> str:
    return html.escape(
        _pdf_clean_text(
            value
        )
    )

def _pdf_question_marks(
    question: dict[str, Any],
) -> int:
    if str(
        question.get(
            "source_type",
            "",
        )
        or ""
    ).strip().casefold() == "official_aqa":
        try:
            return int(
                question_marks(
                    question
                )
            )
        except Exception:
            pass

    return safe_int(
        question.get(
            "marks",
            question.get(
                "marks_numeric",
                0,
            ),
        )
    )

def _pdf_question_text(
    question: dict[str, Any],
) -> str:
    if str(
        question.get(
            "source_type",
            "",
        )
        or ""
    ).strip().casefold() == "official_aqa":
        try:
            value = question_text(
                question
            )
            if str(
                value
                or ""
            ).strip():
                return str(
                    value
                )
        except Exception:
            pass

    for key in [
        "question_text",
        "question_text_canonical",
        "question_text_postgres",
        "question_text_retrieval",
        "text",
    ]:
        value = str(
            question.get(
                key,
                "",
            )
            or ""
        ).strip()

        if value:
            return value

    return "Question text unavailable."

def _pdf_question_topic(
    question: dict[str, Any],
) -> str:
    for key in [
        "topic",
        "detected_topic",
        "topic_label",
    ]:
        value = str(
            question.get(
                key,
                "",
            )
            or ""
        ).strip()

        if value:
            return value

    return "Topic"

def _pdf_question_reference(
    question: dict[str, Any],
) -> str:
    for key in [
        "official_reference",
        "agent1_official_reference",
        "official_reference_canonical",
    ]:
        value = str(
            question.get(
                key,
                "",
            )
            or ""
        ).strip()

        if value:
            return value

    try:
        return str(
            question_reference(
                question
            )
            or ""
        ).strip()
    except Exception:
        return ""

def _pdf_question_paper(
    question: dict[str, Any],
) -> str:
    explicit = str(
        question.get(
            "paper_label",
            "",
        )
        or ""
    ).strip()

    if explicit:
        return explicit

    raw = str(
        question.get(
            "paper_code",
            question.get(
                "paper",
                "",
            ),
        )
        or ""
    ).strip().casefold()

    if raw in {
        "1",
        "1a",
        "1b",
        "paper 1",
        "paper1",
        "p1",
    }:
        return "Paper 1"

    if raw in {
        "2",
        "2a",
        "2b",
        "paper 2",
        "paper2",
        "p2",
    }:
        return "Paper 2"

    reference = _pdf_question_reference(
        question
    )

    match = re.match(
        r"^3\.(\d+)",
        reference,
    )

    if match:
        section = int(
            match.group(
                1
            )
        )
        return (
            "Paper 1"
            if section in {
                1,
                2,
            }
            else "Paper 2"
        )

    return ""

def _pdf_marking_rows(
    question: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Normalize generated or official marking guidance into rows:
        {"marks": int|str, "criterion": str}
    """
    guidance = question.get(
        "marking_guidance"
    )

    rows: list[
        dict[str, Any]
    ] = []

    if isinstance(
        guidance,
        list,
    ):
        for item in guidance:
            if isinstance(
                item,
                dict,
            ):
                criterion = str(
                    item.get(
                        "criterion",
                        item.get(
                            "text",
                            "",
                        ),
                    )
                    or ""
                ).strip()

                if criterion:
                    rows.append(
                        {
                            "marks":
                                item.get(
                                    "marks",
                                    "",
                                ),
                            "criterion":
                                criterion,
                        }
                    )

            elif str(
                item
                or ""
            ).strip():
                rows.append(
                    {
                        "marks":
                            "",
                        "criterion":
                            str(
                                item
                            ).strip(),
                    }
                )

    if rows:
        return rows

    # Official-question fallback shapes.
    mark_scheme = question.get(
        "mark_scheme"
    )

    if isinstance(
        mark_scheme,
        dict,
    ):
        raw = str(
            mark_scheme.get(
                "raw_marking_guidance",
                mark_scheme.get(
                    "marking_guidance",
                    "",
                ),
            )
            or ""
        ).strip()

        if raw:
            return [
                {
                    "marks":
                        "",
                    "criterion":
                        raw,
                }
            ]

        structured = mark_scheme.get(
            "phase3_structured"
        )

        if isinstance(
            structured,
            dict,
        ):
            structured_points = structured.get(
                "marking_points",
                [],
            )

            if isinstance(
                structured_points,
                list,
            ):
                for item in structured_points:
                    if isinstance(
                        item,
                        dict,
                    ):
                        criterion = str(
                            item.get(
                                "criterion",
                                item.get(
                                    "text",
                                    item.get(
                                        "point",
                                        "",
                                    ),
                                ),
                            )
                            or ""
                        ).strip()

                        if criterion:
                            rows.append(
                                {
                                    "marks":
                                        item.get(
                                            "marks",
                                            "",
                                        ),
                                    "criterion":
                                        criterion,
                                }
                            )

                    elif str(
                        item
                        or ""
                    ).strip():
                        rows.append(
                            {
                                "marks":
                                    "",
                                "criterion":
                                    str(
                                        item
                                    ).strip(),
                            }
                        )

    if rows:
        return rows

    return [
        {
            "marks":
                "",
            "criterion":
                "No marking guidance was available in the current quiz artifact.",
        }
    ]

def _pdf_format_prose_markup(
    value: Any,
) -> str:
    text_value = _pdf_clean_text(
        value
    )

    escaped = html.escape(
        text_value
    )

    # Minimal Markdown support used by generated question text.
    escaped = re.sub(
        r"\*\*([^*]+)\*\*",
        r"<b>\1</b>",
        escaped,
    )

    escaped = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        r"<i>\1</i>",
        escaped,
    )

    escaped = re.sub(
        r"`([^`\n]+)`",
        r'<font name="Courier">\1</font>',
        escaped,
    )

    return escaped.replace(
        "\n",
        "<br/>",
    )

def _pdf_strip_fence_language(value: Any) -> str:
    """Remove a Markdown fence language tag without touching real code."""
    text_value = _pdf_clean_text(value).strip()
    if not text_value:
        return text_value

    lines = text_value.splitlines()
    if not lines:
        return text_value

    first = lines[0].strip().casefold()
    known_fence_languages = {
        "text", "plaintext", "plain", "pseudocode", "pseudo",
        "python", "py", "sql", "java", "javascript", "js",
        "c", "cpp", "c++", "csharp", "cs",
    }
    if first in known_fence_languages:
        return "\n".join(lines[1:]).strip()

    return text_value

def _pdf_add_question_text(
    story: list[Any],
    text_value: str,
    *,
    styles: Any,
) -> None:
    """
    Preserve code fences while keeping prose compact, glyph-safe and readable.
    """
    parts = re.split(
        r"```",
        _pdf_clean_text(
            text_value
        ),
    )

    for index, part in enumerate(
        parts
    ):
        part = part.strip()

        if not part:
            continue

        if index % 2 == 1:
            code_part = _pdf_strip_fence_language(
                part
            )
            if not code_part:
                continue

            escaped = html.escape(
                code_part
            ).replace(
                "\n",
                "<br/>",
            )

            story.append(
                Paragraph(
                    escaped,
                    styles[
                        "QuizPDFCode"
                    ],
                )
            )

        else:
            formatted = _pdf_format_prose_markup(
                part
            )

            formatted = formatted.replace(
                "<br/>* ",
                "<br/>&bull; ",
            ).replace(
                "<br/>- ",
                "<br/>&bull; ",
            )

            story.append(
                Paragraph(
                    formatted,
                    styles[
                        "QuizPDFBody"
                    ],
                )
            )

def build_quiz_questions_marking_scheme_pdf(
    *,
    questions: list[dict[str, Any]],
    output_path: Path,
    manifest_payload: dict[str, Any],
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_styles = getSampleStyleSheet()

    pdf_styles.add(
        ParagraphStyle(
            name="QuizPDFTitle",
            parent=pdf_styles[
                "Title"
            ],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )

    pdf_styles.add(
        ParagraphStyle(
            name="QuizPDFSubtitle",
            parent=pdf_styles[
                "Normal"
            ],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#444444"
            ),
            spaceAfter=14,
        )
    )

    pdf_styles.add(
        ParagraphStyle(
            name="QuizPDFQuestionHeading",
            parent=pdf_styles[
                "Heading2"
            ],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            spaceAfter=7,
            textColor=colors.HexColor(
                "#1f2937"
            ),
        )
    )

    pdf_styles.add(
        ParagraphStyle(
            name="QuizPDFSection",
            parent=pdf_styles[
                "Heading3"
            ],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=6,
        )
    )

    pdf_styles.add(
        ParagraphStyle(
            name="QuizPDFBody",
            parent=pdf_styles[
                "BodyText"
            ],
            fontName="Helvetica",
            fontSize=9.6,
            leading=13.2,
            spaceAfter=6,
        )
    )

    pdf_styles.add(
        ParagraphStyle(
            name="QuizPDFCode",
            parent=pdf_styles[
                "BodyText"
            ],
            fontName="Courier",
            fontSize=8.7,
            leading=11.5,
            leftIndent=8,
            rightIndent=8,
            borderColor=colors.HexColor(
                "#d1d5db"
            ),
            borderWidth=0.5,
            borderPadding=6,
            backColor=colors.HexColor(
                "#f8fafc"
            ),
            spaceBefore=5,
            spaceAfter=7,
        )
    )

    pdf_styles.add(
        ParagraphStyle(
            name="QuizPDFSmall",
            parent=pdf_styles[
                "BodyText"
            ],
            fontName="Helvetica",
            fontSize=8.6,
            leading=11,
            textColor=colors.HexColor(
                "#4b5563"
            ),
        )
    )

    pdf_styles.add(
        ParagraphStyle(
            name="QuizPDFMarkScheme",
            parent=pdf_styles[
                "BodyText"
            ],
            fontName="Helvetica",
            fontSize=8.8,
            leading=11.5,
        )
    )

    pdf_visual_integrity_preflight = (
        _pdf_visual_integrity_preflight(
            questions,
            styles=pdf_styles,
        )
    )

    pdf_document = SimpleDocTemplate(
        str(
            output_path
        ),
        pagesize=A4,
        rightMargin=16
        * mm,
        leftMargin=16
        * mm,
        topMargin=16
        * mm,
        bottomMargin=16
        * mm,
        title=(
            "Agent 2 Quiz Output - "
            "Questions and Marking Schemes"
        ),
        author="EDTech Agent 2",
    )

    story: list[
        Any
    ] = []

    generated_model = str(
        manifest_payload.get(
            "generation_model",
            GENERATION_MODEL,
        )
        or GENERATION_MODEL
    ).strip()

    quiz_mode_value = str(
        manifest_payload.get(
            "quiz_mode",
            QUIZ_MODE,
        )
        or QUIZ_MODE
    ).strip()

    total_marks = sum(
        _pdf_question_marks(
            question
        )
        for question in questions
        if isinstance(
            question,
            dict,
        )
    )

    question_count = len(
        [
            question
            for question in questions
            if isinstance(
                question,
                dict,
            )
        ]
    )

    target_marks_value = safe_int(
        manifest_payload.get(
            "target_marks",
            request.get(
                "target_total_marks",
                0,
            ),
        )
    )

    target_questions_value = safe_int(
        manifest_payload.get(
            "target_question_count",
            request.get(
                "number_of_questions",
                0,
            ),
        )
    )

    story.append(
        Paragraph(
            "Agent 2 - AI Quiz Generation Output",
            pdf_styles[
                "QuizPDFTitle"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                "Model: "
                + _pdf_escape(
                    generated_model
                )
                + " | Quiz mode: "
                + _pdf_escape(
                    quiz_mode_value
                )
                + " | Generated content: "
                + str(
                    question_count
                )
                + " questions / "
                + str(
                    total_marks
                )
                + " marks"
            ),
            pdf_styles[
                "QuizPDFSubtitle"
            ],
        )
    )

    summary_rows = [
        [
            "Target marks",
            str(
                target_marks_value
            ),
            "PDF marks",
            str(
                total_marks
            ),
        ],
        [
            "Target questions",
            str(
                target_questions_value
            ),
            "PDF questions",
            str(
                question_count
            ),
        ],
        [
            "Release ready",
            (
                "Yes"
                if manifest_payload.get(
                    "release_ready"
                )
                else "No"
            ),
            "Generation status",
            _pdf_clean_text(
                manifest_payload.get(
                    "generation_status",
                    "",
                )
            ),
        ],
    ]

    summary_table = Table(
        summary_rows,
        colWidths=[
            34
            * mm,
            28
            * mm,
            38
            * mm,
            48
            * mm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.5,
                    colors.HexColor(
                        "#cbd5e1"
                    ),
                ),
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        0,
                        -1,
                    ),
                    colors.HexColor(
                        "#f1f5f9"
                    ),
                ),
                (
                    "BACKGROUND",
                    (
                        2,
                        0,
                    ),
                    (
                        2,
                        -1,
                    ),
                    colors.HexColor(
                        "#f1f5f9"
                    ),
                ),
                (
                    "FONTNAME",
                    (
                        0,
                        0,
                    ),
                    (
                        0,
                        -1,
                    ),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (
                        2,
                        0,
                    ),
                    (
                        2,
                        -1,
                    ),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    9,
                ),
                (
                    "VALIGN",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),
                (
                    "RIGHTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    6,
                ),
                (
                    "TOPPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    5,
                ),
            ]
        )
    )

    story.extend(
        [
            summary_table,
            Spacer(
                1,
                9,
            ),
        ]
    )

    visual_errors = safe_list(
        (
            manifest_payload.get(
                "generation_validation",
                {},
            )
            or {}
        ).get(
            "visual_validation",
            {},
        ).get(
            "errors",
            [],
        )
    )

    if visual_errors:
        story.append(
            Paragraph(
                (
                    "<b>Demonstration note:</b> "
                    "Questions and marking guidance were generated. "
                    "Where an original rendered visual is unavailable, "
                    "this PDF uses the stored visual specification when "
                    "a deterministic fallback is supported. This PDF does "
                    "not override the notebook's validation or release gate."
                ),
                pdf_styles[
                    "QuizPDFSmall"
                ],
            )
        )

    story.append(
        PageBreak()
    )

    valid_questions = [
        question
        for question in questions
        if isinstance(
            question,
            dict,
        )
    ]

    for position, question in enumerate(
        valid_questions,
        start=1,
    ):
        marks = _pdf_question_marks(
            question
        )

        topic = _pdf_question_topic(
            question
        )

        reference = _pdf_question_reference(
            question
        )

        paper_label = _pdf_question_paper(
            question
        )

        role = str(
            question.get(
                "role",
                question.get(
                    "topic_role",
                    "",
                ),
            )
            or ""
        ).strip()

        assessment_pattern = str(
            question.get(
                "assessment_pattern",
                "",
            )
            or ""
        ).strip()

        generated_question_id = str(
            question.get(
                "generated_question_id",
                question.get(
                    "question_id",
                    "",
                ),
            )
            or ""
        ).strip()

        story.append(
            Paragraph(
                (
                    "Question "
                    + str(
                        position
                    )
                    + " ("
                    + str(
                        marks
                    )
                    + " mark"
                    + (
                        ""
                        if marks
                        == 1
                        else "s"
                    )
                    + ")"
                ),
                pdf_styles[
                    "QuizPDFQuestionHeading"
                ],
            )
        )

        metadata_rows = [
            [
                "Topic",
                topic,
                "AQA ref",
                reference,
            ],
            [
                "Paper",
                paper_label,
                "Role",
                role,
            ],
            [
                "Pattern",
                assessment_pattern,
                "ID",
                generated_question_id,
            ],
        ]

        metadata_table = Table(
            metadata_rows,
            colWidths=[
                18
                * mm,
                62
                * mm,
                18
                * mm,
                62
                * mm,
            ],
        )

        metadata_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.4,
                        colors.HexColor(
                            "#d1d5db"
                        ),
                    ),
                    (
                        "BACKGROUND",
                        (
                            0,
                            0,
                        ),
                        (
                            0,
                            -1,
                        ),
                        colors.HexColor(
                            "#f8fafc"
                        ),
                    ),
                    (
                        "BACKGROUND",
                        (
                            2,
                            0,
                        ),
                        (
                            2,
                            -1,
                        ),
                        colors.HexColor(
                            "#f8fafc"
                        ),
                    ),
                    (
                        "FONTNAME",
                        (
                            0,
                            0,
                        ),
                        (
                            0,
                            -1,
                        ),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTNAME",
                        (
                            2,
                            0,
                        ),
                        (
                            2,
                            -1,
                        ),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        8.3,
                    ),
                    (
                        "VALIGN",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        "TOP",
                    ),
                    (
                        "LEFTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        4,
                    ),
                    (
                        "RIGHTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        4,
                    ),
                    (
                        "TOPPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        3,
                    ),
                    (
                        "BOTTOMPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        3,
                    ),
                ]
            )
        )

        story.extend(
            [
                metadata_table,
                Spacer(
                    1,
                    7,
                ),
                Paragraph(
                    "Question",
                    pdf_styles[
                        "QuizPDFSection"
                    ],
                ),
            ]
        )

        _pdf_add_question_text(
            story,
            _pdf_question_text(
                question
            ),
            styles=pdf_styles,
        )

        _pdf_add_visual(
            story,
            question,
            styles=pdf_styles,
        )

        story.append(
            Paragraph(
                (
                    "Marking Scheme / "
                    "AI-generated Marking Guidance"
                ),
                pdf_styles[
                    "QuizPDFSection"
                ],
            )
        )

        marking_rows = [
            [
                Paragraph(
                    "<b>Mark(s)</b>",
                    pdf_styles[
                        "QuizPDFMarkScheme"
                    ],
                ),
                Paragraph(
                    "<b>Criterion</b>",
                    pdf_styles[
                        "QuizPDFMarkScheme"
                    ],
                ),
            ]
        ]

        for item in _pdf_marking_rows(
            question
        ):
            marking_rows.append(
                [
                    Paragraph(
                        _pdf_escape(
                            item.get(
                                "marks",
                                "",
                            )
                        ),
                        pdf_styles[
                            "QuizPDFMarkScheme"
                        ],
                    ),
                    Paragraph(
                        _pdf_escape(
                            item.get(
                                "criterion",
                                "",
                            )
                        ).replace(
                            "\n",
                            "<br/>",
                        ),
                        pdf_styles[
                            "QuizPDFMarkScheme"
                        ],
                    ),
                ]
            )

        marking_table = Table(
            marking_rows,
            colWidths=[
                19
                * mm,
                142
                * mm,
            ],
            repeatRows=1,
        )

        marking_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        0.5,
                        colors.HexColor(
                            "#cbd5e1"
                        ),
                    ),
                    (
                        "BACKGROUND",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            0,
                        ),
                        colors.HexColor(
                            "#e5e7eb"
                        ),
                    ),
                    (
                        "ALIGN",
                        (
                            0,
                            1,
                        ),
                        (
                            0,
                            -1,
                        ),
                        "CENTER",
                    ),
                    (
                        "VALIGN",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        "TOP",
                    ),
                    (
                        "TOPPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        5,
                    ),
                    (
                        "LEFTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        5,
                    ),
                    (
                        "RIGHTPADDING",
                        (
                            0,
                            0,
                        ),
                        (
                            -1,
                            -1,
                        ),
                        5,
                    ),
                ]
            )
        )

        story.append(
            marking_table
        )

        if position != len(
            valid_questions
        ):
            story.append(
                PageBreak()
            )

    def _footer(
        canvas: Any,
        document: Any,
    ) -> None:
        canvas.saveState()
        canvas.setFont(
            "Helvetica",
            8,
        )
        canvas.setFillColor(
            colors.HexColor(
                "#6b7280"
            )
        )
        canvas.drawString(
            16
            * mm,
            9
            * mm,
            (
                "EDTech Agent 2 - "
                "Generated Quiz Output"
            ),
        )
        canvas.drawRightString(
            A4[
                0
            ]
            - 16
            * mm,
            9
            * mm,
            (
                "Page "
                + str(
                    document.page
                )
            ),
        )
        canvas.restoreState()

    pdf_document.build(
        story,
        onFirstPage=_footer,
        onLaterPages=_footer,
    )

    return output_path

def _build_hybrid_heading_pdf(
    *,
    output_path: Path,
    title: str,
    subtitle: str = "",
    lines: list[str] | None = None,
) -> Path:
    """Create one clean A4 heading/summary page for the hybrid PDF."""
    from reportlab.pdfgen import canvas as rl_canvas

    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = rl_canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4

    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 52 * mm, _pdf_clean_text(title))

    if subtitle:
        c.setFillColor(colors.HexColor("#4b5563"))
        c.setFont("Helvetica", 11)
        c.drawCentredString(width / 2, height - 62 * mm, _pdf_clean_text(subtitle))

    y = height - 88 * mm
    for line in (lines or []):
        c.setFillColor(colors.HexColor("#1f2937"))
        c.setFont("Helvetica", 11)
        c.drawString(28 * mm, y, _pdf_clean_text(line))
        y -= 10 * mm

    c.setStrokeColor(colors.HexColor("#d1d5db"))
    c.line(24 * mm, 24 * mm, width - 24 * mm, 24 * mm)
    c.setFillColor(colors.HexColor("#6b7280"))
    c.setFont("Helvetica", 8)
    c.drawString(24 * mm, 16 * mm, "EDTech Agent 2")
    c.save()
    return output_path

def _pdf_content_start_page(
    path: Path,
    *,
    cover_markers: list[str],
) -> int:
    """Skip a generated cover page only when its text clearly matches."""
    try:
        import fitz
        with fitz.open(str(path)) as doc:
            if doc.page_count <= 1:
                return 0
            first_text = (doc[0].get_text("text") or "").casefold()
    except Exception:
        return 0

    return 1 if any(marker.casefold() in first_text for marker in cover_markers) else 0

def _merge_pdf_segments(
    segments: list[tuple[Path, int | None, int | None]],
    output_path: Path,
) -> Path:
    """Merge selected page ranges into one final PDF using PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required to build the final hybrid quiz PDF."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(output_path.stem + "__merge_tmp.pdf")
    if temp_output.exists():
        temp_output.unlink()

    merged = fitz.open()
    try:
        for input_path, from_page, to_page in segments:
            input_path = Path(input_path)
            if not input_path.is_file():
                continue
            with fitz.open(str(input_path)) as source_doc:
                if source_doc.page_count <= 0:
                    continue
                start_page = 0 if from_page is None else max(0, int(from_page))
                end_page = (
                    source_doc.page_count - 1
                    if to_page is None
                    else min(source_doc.page_count - 1, int(to_page))
                )
                if start_page <= end_page:
                    merged.insert_pdf(
                        source_doc,
                        from_page=start_page,
                        to_page=end_page,
                    )
        if merged.page_count <= 0:
            raise RuntimeError("No PDF pages were available for hybrid export.")
        merged.save(str(temp_output))
    finally:
        merged.close()

    if not temp_output.is_file() or temp_output.stat().st_size <= 0:
        raise RuntimeError("Final hybrid PDF was not created correctly.")

    if output_path.exists():
        output_path.unlink()
    temp_output.replace(output_path)
    return output_path


def _attach_runtime_roots(
    questions: list[dict[str, Any]],
    *,
    quiz_output_dir: Path,
    visual_output_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        row = deepcopy(question)
        row["_mcp_quiz_output_dir"] = str(quiz_output_dir)
        row["_mcp_visual_output_dir"] = str(visual_output_dir)
        rows.append(row)
    return rows


def _official_source_pdf(
    manifest: dict[str, Any],
    *,
    quiz_output_dir: Path,
) -> Path | None:
    source = manifest.get("source_artifacts") or {}
    if not isinstance(source, dict):
        source = {}
    for key in (
        "official_retrieval_student_pdf",
        "notebook05_student_question_paper_pdf",
        "official_retrieval_student_question_paper_pdf",
    ):
        path = _resolve_existing_path(
            source.get(key),
            [quiz_output_dir],
        )
        if path is not None:
            return path
    return None


def finalize_mcp_quiz_pdf(
    *,
    quiz_output_dir: Path,
    patched_manifest_path: Path,
    visual_results_path: Path,
) -> dict[str, Any]:
    quiz_output_dir = Path(quiz_output_dir).resolve()
    patched_manifest_path = Path(patched_manifest_path).resolve()
    visual_results_path = Path(visual_results_path).resolve()

    manifest = json.loads(
        patched_manifest_path.read_text(encoding="utf-8")
    )
    visual_results = json.loads(
        visual_results_path.read_text(encoding="utf-8")
    )

    gate = visual_results.get("visual_integrity_gate") or {}
    if str(gate.get("status") or "").strip().upper() != "PASS":
        raise RuntimeError(
            "MCP final PDF waits until all required Notebook 08 visuals pass."
        )

    visual_output_dir = visual_results_path.parent
    quiz_mode = str(manifest.get("quiz_mode") or "").strip()

    candidate = [
        q for q in (manifest.get("candidate_questions") or [])
        if isinstance(q, dict)
    ]
    accepted = [
        q for q in (manifest.get("questions") or [])
        if isinstance(q, dict)
    ]

    display_questions = accepted if accepted else candidate
    display_questions = _attach_runtime_roots(
        display_questions,
        quiz_output_dir=quiz_output_dir,
        visual_output_dir=visual_output_dir,
    )

    final_pdf = (
        visual_output_dir
        / "Agent2_Quiz_Output_Questions_and_Marking_Schemes_MCP.pdf"
    )

    if quiz_mode == "fill_shortfall":
        official_pdf = _official_source_pdf(
            manifest,
            quiz_output_dir=quiz_output_dir,
        )
        if official_pdf is None:
            raise FileNotFoundError(
                "Current Notebook 05 student PDF is missing from the manifest."
            )

        ai_questions = [q for q in display_questions if _is_generated_question(q)]
        if not ai_questions and candidate:
            ai_questions = _attach_runtime_roots(
                candidate,
                quiz_output_dir=quiz_output_dir,
                visual_output_dir=visual_output_dir,
            )

        official_marks = safe_int(manifest.get("official_total_marks", 0))
        official_count = safe_int(manifest.get("official_question_count", 0))
        ai_marks = sum(_pdf_question_marks(q) for q in ai_questions)
        ai_count = len(ai_questions)
        target_marks = safe_int(manifest.get("target_marks", 0))
        target_questions = safe_int(manifest.get("target_question_count", 0))

        cover = visual_output_dir / "__mcp_hybrid_summary.pdf"
        official_heading = visual_output_dir / "__mcp_official_heading.pdf"
        ai_heading = visual_output_dir / "__mcp_ai_heading.pdf"
        ai_extension = visual_output_dir / "__mcp_ai_extension.pdf"

        _build_hybrid_heading_pdf(
            output_path=cover,
            title="Agent 2 - Final Hybrid Quiz",
            subtitle="Official AQA questions followed by AI-generated missing coverage",
            lines=[
                f"Official retrieval: {official_marks}/{target_marks} marks | "
                f"{official_count}/{target_questions} questions",
                f"AI-generated missing coverage: +{ai_marks} marks | "
                f"+{ai_count} question(s)",
                f"Combined after approval: {official_marks + ai_marks}/{target_marks} marks | "
                f"{official_count + ai_count} questions",
                "Visual source: LangGraph -> MCP -> Notebook 08",
                f"Human review state: {manifest.get('generated_human_review_state') or 'N/A'}",
            ],
        )
        _build_hybrid_heading_pdf(
            output_path=official_heading,
            title="Official AQA Questions",
            subtitle="Retrieved by Notebook 05",
            lines=[
                f"Official questions: {official_count}",
                f"Official marks: {official_marks}",
            ],
        )

        official_start = _pdf_content_start_page(
            official_pdf,
            cover_markers=["Student Question Paper", "AQA GCSE Computer Science"],
        )

        segments = [
            (cover, 0, None),
            (official_heading, 0, None),
            (official_pdf, official_start, None),
        ]

        if ai_questions:
            _build_hybrid_heading_pdf(
                output_path=ai_heading,
                title="AI-Generated Missing Coverage",
                subtitle="AQA-aligned practice questions generated by Notebook 06",
                lines=[
                    f"AI-generated questions: {ai_count}",
                    f"AI-generated marks: {ai_marks}",
                    "Required visuals are MCP-routed Notebook 08 assets.",
                ],
            )
            build_quiz_questions_marking_scheme_pdf(
                questions=ai_questions,
                output_path=ai_extension,
                manifest_payload=manifest,
            )
            ai_start = _pdf_content_start_page(
                ai_extension,
                cover_markers=["AI Quiz Generation Output", "Generated content"],
            )
            segments.extend([
                (ai_heading, 0, None),
                (ai_extension, ai_start, None),
            ])

        _merge_pdf_segments(segments, final_pdf)

        for temp in (cover, official_heading, ai_heading, ai_extension):
            if temp.is_file():
                try:
                    temp.unlink()
                except OSError:
                    pass

    else:
        if not display_questions:
            raise RuntimeError("No quiz questions are available for MCP PDF assembly.")
        build_quiz_questions_marking_scheme_pdf(
            questions=display_questions,
            output_path=final_pdf,
            manifest_payload=manifest,
        )

    if not final_pdf.is_file() or final_pdf.stat().st_size <= 0:
        raise RuntimeError("MCP final PDF was not created correctly.")

    manifest.setdefault("output_files", {})[
        "questions_and_marking_schemes_pdf"
    ] = str(final_pdf)
    manifest.setdefault("source_artifacts", {})[
        "mcp_final_questions_and_marking_schemes_pdf"
    ] = str(final_pdf)

    manifest["mcp_final_pdf"] = {
        "status": "SAVED",
        "path": str(final_pdf),
        "visual_source": "mcp_notebook08",
        "visual_results_path": str(visual_results_path),
        "uses_notebook08_assets": True,
        "pdf_format_source": "current_notebook06_pdf_builder",
    }

    patched_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "status": "SAVED",
        "final_pdf_path": str(final_pdf),
        "uses_notebook08_assets": True,
    }
