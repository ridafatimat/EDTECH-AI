from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from dotenv import load_dotenv
from groq import Groq


SYSTEM_PROMPT = """
You are a conservative validation component inside an educational topic-mapping
system.

Your task is NOT to decide whether two passages are merely semantically similar.

Your task is to decide whether a previously HUMAN-APPROVED topic-edit rationale
still applies to new lesson evidence.

You will receive:
- an edit action;
- the source topic/role, when applicable;
- the target topic/role, when applicable;
- the reviewer's original reason;
- the original reviewed evidence;
- new current lesson evidence.

The key question is:

    Does the same human rationale still justify applying the same edit to the
    current evidence?

Rules:
1. Prefer abstention over an unsafe automatic edit.
2. "uncertain" is correct whenever the evidence is insufficient or mixed.
3. Do not treat vocabulary overlap as proof of the same teaching context.
4. Distinguish incidental use/mention from independent teaching.
5. For remove_topic:
   - compatible only when the topic is again incidental/non-independent in the
     same material sense captured by the reviewer reason;
   - if the topic is independently taught, return incompatible.
6. For add_topic:
   - compatible only when the current evidence independently teaches the same
     missed topic for substantially the same reason.
7. For change_role:
   - compatible only when the same lesson-emphasis rationale still applies.
8. For replace_topic:
   - compatible only when the same conceptual misclassification is present.
9. If the original reviewer reason no longer applies, return incompatible.
10. Return JSON only.

Required JSON object:
{
  "decision": "compatible | incompatible | uncertain",
  "rationale_still_applies": true,
  "same_teaching_context": true,
  "independent_teaching_detected": false,
  "confidence": 0.0,
  "explanation": "brief evidence-based explanation"
}

For independent_teaching_detected:
- true = the source/target topic is independently taught in the current evidence;
- false = it is only incidental/supporting in the relevant sense;
- null = not applicable or cannot be determined.

Confidence must reflect certainty in THIS compatibility judgment, not general
confidence in the topic name.
""".strip()


@dataclass(slots=True)
class GroqReviewerReasonProvider:
    """
    Real Groq adapter for Step 4.7 reviewer-reason diagnostics.

    FIX:
    Because this dataclass uses slots=True, every instance attribute assigned
    later must be declared as a dataclass field. The Groq client is therefore
    explicitly declared with init=False.
    """

    model_name: str = field(
        default_factory=lambda: (
            os.getenv("GROQ_EDIT_MEMORY_VALIDATOR_MODEL")
            or os.getenv("GROQ_MODEL")
            or "openai/gpt-oss-20b"
        )
    )

    api_key: str | None = None
    max_completion_tokens: int = 700
    reasoning_effort: str = "low"

    # Required when slots=True. This is the only functional fix.
    client: Groq = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        load_dotenv()

        if self.api_key is None:
            self.api_key = os.getenv(
                "GROQ_API_KEY",
                "",
            ).strip()

        self.model_name = str(
            self.model_name
        ).strip()

        if not self.api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is missing from Agent_1/.env."
            )

        if not self.model_name:
            raise EnvironmentError(
                "No Groq model is configured. Set GROQ_MODEL or "
                "GROQ_EDIT_MEMORY_VALIDATOR_MODEL."
            )

        self.client = Groq(
            api_key=self.api_key
        )

    @staticmethod
    def _clean_optional(value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = " ".join(
            str(value).strip().split()
        )

        return cleaned or None

    def _build_payload(
        self,
        *,
        edit_action: str,
        source_topic: str | None,
        source_role: str | None,
        target_topic: str | None,
        target_role: str | None,
        reviewer_reason: str,
        stored_evidence: str,
        current_evidence: str,
    ) -> dict[str, Any]:
        return {
            "edit_action": str(
                edit_action
            ).strip(),
            "source_topic": self._clean_optional(
                source_topic
            ),
            "source_role": self._clean_optional(
                source_role
            ),
            "target_topic": self._clean_optional(
                target_topic
            ),
            "target_role": self._clean_optional(
                target_role
            ),
            "reviewer_reason": " ".join(
                str(reviewer_reason).strip().split()
            ),
            "stored_reviewed_evidence": " ".join(
                str(stored_evidence).strip().split()
            ),
            "current_lesson_evidence": " ".join(
                str(current_evidence).strip().split()
            ),
        }

    @staticmethod
    def _validate_raw_payload(
        raw: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError(
                "Groq reviewer-reason response must be a JSON object."
            )

        decision = str(
            raw.get("decision", "")
        ).strip().casefold()

        if decision not in {
            "compatible",
            "incompatible",
            "uncertain",
        }:
            raise ValueError(
                "Groq returned an invalid reviewer-reason decision."
            )

        rationale = raw.get(
            "rationale_still_applies"
        )
        same_context = raw.get(
            "same_teaching_context"
        )
        independent = raw.get(
            "independent_teaching_detected"
        )

        if not isinstance(rationale, bool):
            raise ValueError(
                "rationale_still_applies must be boolean."
            )

        if not isinstance(same_context, bool):
            raise ValueError(
                "same_teaching_context must be boolean."
            )

        if (
            independent is not None
            and not isinstance(independent, bool)
        ):
            raise ValueError(
                "independent_teaching_detected must be boolean or null."
            )

        try:
            confidence = float(
                raw.get("confidence")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Groq confidence must be numeric."
            ) from exc

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Groq confidence must be between 0 and 1."
            )

        explanation = " ".join(
            str(
                raw.get(
                    "explanation",
                    "",
                )
            ).strip().split()
        )

        if not explanation:
            raise ValueError(
                "Groq explanation must not be empty."
            )

        return {
            "decision": decision,
            "rationale_still_applies": rationale,
            "same_teaching_context": same_context,
            "independent_teaching_detected": independent,
            "confidence": confidence,
            "explanation": explanation,
        }

    def _call_groq(
        self,
        user_payload: dict[str, Any],
    ) -> dict[str, Any]:
        common_kwargs = {
            "model": self.model_name,
            "temperature": 0,
            "max_completion_tokens": int(
                self.max_completion_tokens
            ),
            "response_format": {
                "type": "json_object",
            },
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        user_payload,
                        indent=2,
                        ensure_ascii=False,
                    ),
                },
            ],
        }

        try:
            response = (
                self.client
                .chat
                .completions
                .create(
                    reasoning_effort=self.reasoning_effort,
                    **common_kwargs,
                )
            )
        except Exception as first_error:
            message = str(
                first_error
            ).casefold()

            optional_parameter_problem = (
                "reasoning_effort" in message
                or "reasoning effort" in message
                or "unsupported" in message
                or "unknown field" in message
            )

            if not optional_parameter_problem:
                raise

            response = (
                self.client
                .chat
                .completions
                .create(
                    **common_kwargs,
                )
            )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "Groq returned an empty reviewer-reason response."
            )

        try:
            parsed = json.loads(
                content
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Groq reviewer-reason response was not valid JSON."
            ) from exc

        return self._validate_raw_payload(
            parsed
        )

    def validate_edit_context(
        self,
        *,
        edit_action: str,
        source_topic: str | None,
        source_role: str | None,
        target_topic: str | None,
        target_role: str | None,
        reviewer_reason: str,
        stored_evidence: str,
        current_evidence: str,
    ) -> Mapping[str, Any]:
        payload = self._build_payload(
            edit_action=edit_action,
            source_topic=source_topic,
            source_role=source_role,
            target_topic=target_topic,
            target_role=target_role,
            reviewer_reason=reviewer_reason,
            stored_evidence=stored_evidence,
            current_evidence=current_evidence,
        )

        return self._call_groq(
            payload
        )