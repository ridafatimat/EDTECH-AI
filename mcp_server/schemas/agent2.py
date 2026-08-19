from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from .common import RunRequest


class Agent2AssessmentFilters(RunRequest):
    """Shared assessment/quiz filters from the Streamlit Agent 2 form."""

    model_config = ConfigDict(extra="forbid")

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
    user_request: str | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> "Agent2AssessmentFilters":
        if self.maximum_question_marks < self.minimum_question_marks:
            raise ValueError(
                "maximum_question_marks must be at least minimum_question_marks"
            )
        if self.minimum_primary_questions > self.number_of_questions:
            raise ValueError(
                "minimum_primary_questions cannot exceed number_of_questions"
            )
        if self.minimum_supporting_questions > self.number_of_questions:
            raise ValueError(
                "minimum_supporting_questions cannot exceed number_of_questions"
            )
        if (
            self.minimum_primary_questions + self.minimum_supporting_questions
            > self.number_of_questions
        ):
            raise ValueError(
                "primary and supporting minimums cannot exceed total questions"
            )
        return self


class RunAgent2RetrievalRequest(Agent2AssessmentFilters):
    """Run existing Notebook 05 retrieval/ranking with the shared filters."""


class GenerateAgent2CompleteQuizRequest(Agent2AssessmentFilters):
    """Run Notebook 06 complete_quiz mode without Notebook 05 retrieval."""


class GenerateAgent2MissingQuizRequest(RunRequest):
    """Fill only the shortfall from the exact current Notebook 05 request."""

    model_config = ConfigDict(extra="forbid")
    user_request: str | None = None


class SubmitAgent2QuizReviewRequest(RunRequest):
    """Human-only generated-question quality decision."""

    model_config = ConfigDict(extra="forbid")
    quiz_mode: Literal["complete_quiz", "fill_shortfall"]
    decision: Literal["approve", "regenerate", "reject"]
    reason: str = Field(min_length=1)
    reviewed_by: str = "streamlit"


class GetAgent2AssessmentRequest(RunRequest):
    model_config = ConfigDict(extra="forbid")


class GetAgent2MarkSchemesRequest(RunRequest):
    model_config = ConfigDict(extra="forbid")
    question_ids: list[str] = Field(min_length=1)


class GetAgent2RenderedPagesRequest(RunRequest):
    model_config = ConfigDict(extra="forbid")
    question_ids: list[str] = Field(min_length=1)
