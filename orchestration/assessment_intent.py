from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AssessmentRequestKind(str, Enum):
    GENERATE_ASSESSMENT = "generate_assessment"
    GENERATE_COMPLETE_QUIZ = "generate_complete_quiz"
    GENERATE_MISSING_QUIZ = "generate_missing_quiz"
    SHOW_ASSESSMENT = "show_assessment"
    SHOW_MARK_SCHEMES = "show_mark_schemes"
    SHOW_RENDERED_PAGES = "show_rendered_pages"
    OTHER = "other"


class AssessmentRequestIntent(BaseModel):
    """Deterministically parsed Agent 2 action + shared UI filters."""

    model_config = ConfigDict(extra="forbid")

    kind: AssessmentRequestKind = AssessmentRequestKind.OTHER
    paper: Literal["Paper 1", "Paper 2", "Any"] = "Any"
    number_of_questions: int = Field(default=5, ge=1, le=50)
    target_total_marks: int = Field(default=20, ge=1, le=500)
    minimum_question_marks: int = Field(default=1, ge=1, le=100)
    maximum_question_marks: int = Field(default=12, ge=1, le=100)
    minimum_primary_questions: int = Field(default=1, ge=0, le=50)
    minimum_supporting_questions: int = Field(default=0, ge=0, le=50)
    cover_all_approved_topics: bool = True
    include_code_questions: bool = True
    include_visual_questions: bool = True
    programming_language: Literal["Automatic", "Python"] = "Automatic"
    question_ids: list[str] = Field(default_factory=list)

    def shared_arguments(self, *, user_request: str) -> dict:
        return {
            "paper": self.paper,
            "number_of_questions": self.number_of_questions,
            "target_total_marks": self.target_total_marks,
            "minimum_question_marks": self.minimum_question_marks,
            "maximum_question_marks": self.maximum_question_marks,
            "minimum_primary_questions": self.minimum_primary_questions,
            "minimum_supporting_questions": self.minimum_supporting_questions,
            "cover_all_approved_topics": self.cover_all_approved_topics,
            "include_code_questions": self.include_code_questions,
            "include_visual_questions": self.include_visual_questions,
            "programming_language": self.programming_language,
            "user_request": user_request,
        }

    def retrieval_arguments(self, *, user_request: str) -> dict:
        return self.shared_arguments(user_request=user_request)

    def complete_quiz_arguments(self, *, user_request: str) -> dict:
        return self.shared_arguments(user_request=user_request)

    def missing_quiz_arguments(self, *, user_request: str) -> dict:
        # The exact filters are read back from the current Notebook 05
        # assessment_request artifact to prevent UI/request drift.
        return {"user_request": user_request}


_QUESTION_ID_RE = re.compile(
    r"\b(?:question\s*(?:id\s*)?|qid\s*[:#-]?\s*)?"
    r"([A-Za-z]{0,4}\d+(?:[._-][A-Za-z0-9]+)*)\b",
    re.I,
)


def _extract_question_ids(text: str) -> list[str]:
    results: list[str] = []
    for match in _QUESTION_ID_RE.finditer(text):
        token = str(match.group(1) or "").strip()
        whole = match.group(0).lower()
        if not token:
            continue
        has_alpha = any(ch.isalpha() for ch in token)
        has_question_prefix = "question" in whole or "qid" in whole
        if not (has_alpha or has_question_prefix):
            continue
        if token not in results:
            results.append(token)
    return results


def _parse_shared_filters(raw: str, *, kind: AssessmentRequestKind) -> AssessmentRequestIntent:
    text = raw.casefold()

    paper: Literal["Paper 1", "Paper 2", "Any"] = "Any"
    if re.search(r"\b(?:paper|p)\s*1\b", text):
        paper = "Paper 1"
    elif re.search(r"\b(?:paper|p)\s*2\b", text):
        paper = "Paper 2"

    number_of_questions = 5
    match = re.search(
        r"\b(\d+)\s+(?:(?:paper\s*[12]|past[- ]?paper)\s+)?(?:questions?|qs?)\b",
        text,
    )
    if match:
        number_of_questions = max(1, min(50, int(match.group(1))))

    target_total_marks = 20
    mark_matches = list(re.finditer(r"\b(\d+)\s*marks?\b", text))
    if mark_matches:
        target_total_marks = max(1, min(500, int(mark_matches[-1].group(1))))

    def _bounded_int(pattern: str, default: int, *, low: int, high: int) -> int:
        found = re.search(pattern, text)
        if not found:
            return default
        return max(low, min(high, int(found.group(1))))

    minimum_question_marks = _bounded_int(
        r"\bminimum\s+(?:question\s+)?marks(?:\s+per\s+question)?\s*(?:[:=]|is)?\s*(\d+)\b",
        1,
        low=1,
        high=100,
    )
    maximum_question_marks = _bounded_int(
        r"\bmaximum\s+(?:question\s+)?marks(?:\s+per\s+question)?\s*(?:[:=]|is)?\s*(\d+)\b",
        12,
        low=1,
        high=100,
    )
    minimum_primary_questions = _bounded_int(
        r"\bminimum\s+primary\s+questions?\s*(?:[:=]|is)?\s*(\d+)\b",
        1,
        low=0,
        high=50,
    )
    minimum_supporting_questions = _bounded_int(
        r"\bminimum\s+supporting\s+questions?\s*(?:[:=]|is)?\s*(\d+)\b",
        0,
        low=0,
        high=50,
    )

    include_code = not bool(
        re.search(r"\b(no|without|exclude)\s+(?:any\s+)?code(?:\s+questions?)?\b", text)
    )
    include_visual = not bool(
        re.search(r"\b(no|without|exclude)\s+(?:any\s+)?visual(?:\s+questions?)?\b", text)
    )
    programming_language: Literal["Automatic", "Python"] = (
        "Python" if re.search(r"\bpython\b", text) else "Automatic"
    )
    cover_all = not bool(
        re.search(r"\b(?:do\s+not|don't|dont|no\s+need\s+to)\s+cover\s+all\b", text)
    )

    return AssessmentRequestIntent(
        kind=kind,
        paper=paper,
        number_of_questions=number_of_questions,
        target_total_marks=target_total_marks,
        minimum_question_marks=minimum_question_marks,
        maximum_question_marks=maximum_question_marks,
        minimum_primary_questions=minimum_primary_questions,
        minimum_supporting_questions=minimum_supporting_questions,
        cover_all_approved_topics=cover_all,
        include_code_questions=include_code,
        include_visual_questions=include_visual,
        programming_language=programming_language,
    )


def parse_assessment_request(user_request: str) -> AssessmentRequestIntent:
    """Parse explicit official-retrieval and quiz-generation actions.

    The Streamlit buttons emit unambiguous action phrases. Generic legacy
    assessment requests remain mapped to Notebook 05 retrieval for backwards
    compatibility.
    """

    raw = " ".join(str(user_request or "").strip().split())
    text = raw.casefold()

    if re.search(r"\b(mark(?:ing)?\s*scheme|mark\s*schemes|answers?)\b", text):
        return AssessmentRequestIntent(
            kind=AssessmentRequestKind.SHOW_MARK_SCHEMES,
            question_ids=_extract_question_ids(raw),
        )

    if re.search(r"\b(rendered|source)\s+pages?\b|\bpage\s+images?\b", text):
        return AssessmentRequestIntent(
            kind=AssessmentRequestKind.SHOW_RENDERED_PAGES,
            question_ids=_extract_question_ids(raw),
        )

    show_words = bool(re.search(r"\b(show|display|view|open|see)\b", text))
    assessment_noun = bool(
        re.search(r"\b(assessment|quiz|questions?|question\s+set|test)\b", text)
    )
    if show_words and assessment_noun:
        return AssessmentRequestIntent(kind=AssessmentRequestKind.SHOW_ASSESSMENT)

    if "generate missing quiz coverage" in text or "fill quiz shortfall" in text:
        # Filters are intentionally sourced from the exact current Notebook 05
        # request artifact by the MCP adapter.
        return AssessmentRequestIntent(kind=AssessmentRequestKind.GENERATE_MISSING_QUIZ)

    if "generate complete quiz" in text or "generate full quiz" in text:
        return _parse_shared_filters(
            raw,
            kind=AssessmentRequestKind.GENERATE_COMPLETE_QUIZ,
        )

    if "retrieve official assessment" in text or "retrieve official questions" in text:
        return _parse_shared_filters(
            raw,
            kind=AssessmentRequestKind.GENERATE_ASSESSMENT,
        )

    generation_verb = bool(
        re.search(
            r"\b(generate|create|build|make|give|retrieve|find|select|prepare|produce)\b",
            text,
        )
    )
    explicit_question_count = bool(
        re.search(
            r"\b\d+\s+(?:(?:paper\s*[12]|past[- ]?paper)\s+)?(?:questions?|qs?)\b",
            text,
        )
    )
    explicit_assessment_request = generation_verb and assessment_noun
    if not (explicit_assessment_request or explicit_question_count):
        return AssessmentRequestIntent(kind=AssessmentRequestKind.OTHER)

    return _parse_shared_filters(
        raw,
        kind=AssessmentRequestKind.GENERATE_ASSESSMENT,
    )
