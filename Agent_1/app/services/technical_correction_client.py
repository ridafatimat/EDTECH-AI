from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from dotenv import load_dotenv

from app.schemas.technical_normalisation import (
    CorrectionProposal,
    SuspiciousTechnicalSpan,
)


class TechnicalCorrectionClient(Protocol):
    """Interface for selective technical-correction providers."""

    model_name: str

    def suggest_corrections(
        self,
        *,
        sentence: str,
        issues: Sequence[SuspiciousTechnicalSpan],
    ) -> list[CorrectionProposal]:
        """
        Return exact replacement pairs only.

        Implementations must never return or apply a rewritten sentence.
        """


class TechnicalCorrectionClientError(RuntimeError):
    """Raised when the selective correction provider fails."""


@dataclass(slots=True)
class GroqTechnicalCorrectionClient:
    """
    Selective Groq client for technical spelling and ASR corrections.

    Only one affected sentence and its suspicious spans are sent to the
    model. The response is constrained to replacement pairs through a
    strict JSON schema.
    """

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "GROQ_TECH_CORRECTION_MODEL",
            "openai/gpt-oss-120b",
        )
    )

    api_key: str | None = None
    timeout_seconds: float = 45.0

    # GPT-OSS is a reasoning model. A 700-token limit can be exhausted
    # before the final JSON document is emitted, even for a short answer.
    max_completion_tokens: int = 2048

    # Retry once or twice with a larger output budget only when Groq says
    # JSON generation ended because the completion-token limit was reached.
    retry_token_budgets: tuple[int, ...] = (
        4096,
        8192,
    )

    reasoning_effort: str = "low"

    # Required because this dataclass uses slots=True.
    _client: Any = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        load_dotenv()

        self.api_key = (
            self.api_key
            or os.getenv("GROQ_API_KEY")
        )

        configured_model = os.getenv(
            "GROQ_TECH_CORRECTION_MODEL"
        )

        if configured_model:
            self.model_name = configured_model

        if not self.api_key:
            raise TechnicalCorrectionClientError(
                "GROQ_API_KEY is not configured. "
                "Add it to Agent_1/.env."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise TechnicalCorrectionClientError(
                "The Groq SDK is not installed. "
                "Run: pip install groq"
            ) from exc

        self._client = Groq(
            api_key=self.api_key,
            timeout=self.timeout_seconds,
        )

    def suggest_corrections(
        self,
        *,
        sentence: str,
        issues: Sequence[SuspiciousTechnicalSpan],
    ) -> list[CorrectionProposal]:
        """
        Request replacement pairs for one affected sentence.

        The method retries only when structured-output generation reaches
        the completion-token limit before producing valid JSON.
        """

        if not issues:
            return []

        payload = {
            "sentence": sentence,
            "suspicious_spans": [
                {
                    "issue_id": issue.issue_id,
                    "original_span": issue.original_span,
                    "candidate_terms": list(
                        issue.candidate_terms
                    ),
                    "detector_score": (
                        issue.detector_score
                    ),
                    "context_keywords": list(
                        issue.context_keywords
                    ),
                }
                for issue in issues
            ],
        }

        token_budgets = (
            self.max_completion_tokens,
            *self.retry_token_budgets,
        )

        last_error: Exception | None = None

        for attempt, token_budget in enumerate(
            token_budgets,
            start=1,
        ):
            try:
                completion = (
                    self._client
                    .chat
                    .completions
                    .create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": _SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                ),
                            },
                        ],
                        temperature=0,
                        reasoning_effort=(
                            self.reasoning_effort
                        ),
                        max_completion_tokens=(
                            token_budget
                        ),
                        response_format=(
                            _RESPONSE_FORMAT
                        ),
                    )
                )

                content = (
                    completion
                    .choices[0]
                    .message
                    .content
                )

                if not content:
                    return []

                try:
                    response = json.loads(
                        content
                    )
                except json.JSONDecodeError as exc:
                    raise TechnicalCorrectionClientError(
                        "Groq returned invalid JSON."
                    ) from exc

                return _parse_proposals(
                    response=response,
                    allowed_issues=issues,
                )

            except Exception as exc:
                last_error = exc

                should_retry = (
                    attempt < len(token_budgets)
                    and _is_completion_limit_json_error(
                        exc
                    )
                )

                if should_retry:
                    continue

                raise TechnicalCorrectionClientError(
                    _build_provider_error_message(
                        error=exc,
                        attempt=attempt,
                        token_budget=token_budget,
                    )
                ) from exc

        raise TechnicalCorrectionClientError(
            "Groq technical-correction request failed "
            "after all structured-output retries."
        ) from last_error


def _is_completion_limit_json_error(
    error: Exception,
) -> bool:
    """
    Return True only for Groq structured-output failures caused by the
    completion-token budget ending before valid JSON was produced.
    """

    message = str(error).casefold()

    return (
        "json_validate_failed" in message
        and (
            "max completion tokens reached"
            in message
            or "completion tokens reached"
            in message
        )
    )


def _build_provider_error_message(
    *,
    error: Exception,
    attempt: int,
    token_budget: int,
) -> str:
    message = str(error)

    # Keep the useful provider explanation without exposing credentials.
    if len(message) > 700:
        message = message[:700] + "..."

    return (
        "Groq technical-correction request failed "
        f"on attempt {attempt} with "
        f"max_completion_tokens={token_budget}. "
        f"Provider message: {message}"
    )


def _parse_proposals(
    *,
    response: dict[str, Any],
    allowed_issues: Sequence[SuspiciousTechnicalSpan],
) -> list[CorrectionProposal]:
    """
    Convert the structured response into proposal objects.

    Full safety validation is performed later by
    TechnicalCorrectionValidator.
    """

    allowed_by_id = {
        issue.issue_id: issue
        for issue in allowed_issues
    }

    raw_corrections = response.get(
        "corrections",
        [],
    )

    if not isinstance(
        raw_corrections,
        list,
    ):
        raise TechnicalCorrectionClientError(
            "Groq response field 'corrections' must be a list."
        )

    proposals: list[CorrectionProposal] = []
    seen_issue_ids: set[str] = set()

    for item in raw_corrections:
        if not isinstance(
            item,
            dict,
        ):
            continue

        issue_id = item.get(
            "issue_id"
        )

        if not isinstance(
            issue_id,
            str,
        ):
            continue

        if issue_id in seen_issue_ids:
            continue

        issue = allowed_by_id.get(
            issue_id
        )

        if issue is None:
            continue

        original = item.get(
            "original"
        )

        # The model must reproduce the exact detected span.
        if original != issue.original_span:
            continue

        replacement = item.get(
            "replacement"
        )

        if not isinstance(
            replacement,
            str,
        ):
            continue

        try:
            confidence = float(
                item.get(
                    "confidence",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        correction_type = item.get(
            "correction_type",
            "technical_asr",
        )

        reason = item.get(
            "reason",
            "",
        )

        if correction_type not in {
            "technical_spelling",
            "technical_asr",
        }:
            continue

        if not isinstance(
            reason,
            str,
        ):
            reason = ""

        proposals.append(
            CorrectionProposal(
                issue_id=issue_id,
                original=original,
                replacement=replacement.strip(),
                confidence=max(
                    0.0,
                    min(
                        confidence,
                        1.0,
                    ),
                ),
                correction_type=(
                    correction_type
                ),
                reason=reason.strip(),
            )
        )

        seen_issue_ids.add(
            issue_id
        )

    return proposals


_SYSTEM_PROMPT = """
Correct only a genuinely corrupted Computer Science spelling or
speech-recognition span.

Hard rules:
- Return replacement pairs only; never return a rewritten sentence.
- Never change text outside a supplied suspicious span.
- Do not change singular to plural or plural to singular.
- Do not change verb tense or grammatical form.
- Do not replace wording that is already valid in the sentence.
- Do not infer a narrower or more specific technical topic.
- Do not reinterpret connector/control words such as and, or, one, same,
  while, for, if, or not.
- Treat variable names in pseudocode as valid, including expressions such as
  "while low less than or equal high".
- Preserve a span whenever the intended term is uncertain.
- Use the exact issue_id and exact original_span supplied by the user.
- Keep each reason under 12 words.
- Return an empty corrections list when no safe correction exists.

Safe examples:
- "wild loop" -> "while loop"
- "algoritm" -> "algorithm"

Do not change:
- "while loops"
- "function returns value"
- "same algorithm"
- "while low less than high"
- "syntax or name error"
- "if condition used OR, loop might continue"

Python performs final validation and exact replacement.
""".strip()


_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": (
            "technical_span_corrections"
        ),
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "corrections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "issue_id": {
                                "type": "string"
                            },
                            "original": {
                                "type": "string"
                            },
                            "replacement": {
                                "type": "string"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "correction_type": {
                                "type": "string",
                                "enum": [
                                    "technical_spelling",
                                    "technical_asr"
                                ]
                            },
                            "reason": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "issue_id",
                            "original",
                            "replacement",
                            "confidence",
                            "correction_type",
                            "reason"
                        ]
                    }
                }
            },
            "required": [
                "corrections"
            ]
        }
    }
}