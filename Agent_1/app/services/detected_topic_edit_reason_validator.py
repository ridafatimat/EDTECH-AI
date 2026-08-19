from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Any


VALID_DECISIONS = frozenset(
    {
        "compatible",
        "incompatible",
        "uncertain",
    }
)


class ContextReasoningProviderProtocol(Protocol):
    """
    Provider boundary for contextual reasoning.

    Step 4.6 deliberately does not import Groq, OpenAI, Qdrant, or any
    project-specific LLM client. A later real-provider diagnostic can adapt
    the existing Agent 1 Groq service to this interface without changing
    the validator itself.
    """

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
        ...


@dataclass(frozen=True, slots=True)
class ReviewerReasonValidationRequest:
    """
    Inputs needed to decide whether the human review rationale still applies.

    This is intentionally independent of Streamlit and Module 3 models.
    """

    edit_action: str

    source_topic: str | None
    source_role: str | None

    target_topic: str | None
    target_role: str | None

    reviewer_reason: str

    stored_evidence: str
    current_evidence: str


@dataclass(frozen=True, slots=True)
class ReviewerReasonValidationResult:
    """
    Final conservative validation decision.

    Only `compatible` may be eligible for later edit-memory reuse.

    `incompatible` and `uncertain` must both result in:
        leave fresh Module 3 output unchanged
    """

    decision: str

    rationale_still_applies: bool
    same_teaching_context: bool
    independent_teaching_detected: bool | None

    confidence: float
    explanation: str

    safe_for_automatic_reuse: bool


@dataclass(frozen=True, slots=True)
class ReviewerReasonValidatorConfig:
    """
    Precision-first policy.

    The numeric confidence floor is NOT an embedding similarity threshold.
    It is only a final LLM/provider self-confidence guard.

    It remains deliberately high. Step 4.6 does not tune it from transcript
    examples and does not write it into any existing Agent 1 configuration.
    """

    minimum_provider_confidence: float = 0.90


class ReviewerReasonContextValidator:
    """
    Generic rationale-based validator for final-topic edit memories.

    Core question:
        Does the original human review reason still apply to the current
        transcript evidence?

    The production validator contains no topic-specific rules.

    Safety:
    - no database writes;
    - no Module 3 changes;
    - no automatic edit application;
    - no embedding threshold changes;
    - no Qdrant/Groq dependency in this class;
    - uncertain always abstains.
    """

    def __init__(
        self,
        *,
        provider: ContextReasoningProviderProtocol,
        config: ReviewerReasonValidatorConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or ReviewerReasonValidatorConfig()

    @staticmethod
    def _clean_text(
        value: str | None,
    ) -> str:
        return " ".join(
            str(value or "").strip().split()
        )

    @classmethod
    def _validate_request(
        cls,
        request: ReviewerReasonValidationRequest,
    ) -> ReviewerReasonValidationRequest:
        edit_action = cls._clean_text(
            request.edit_action
        ).casefold()

        if edit_action not in {
            "remove_topic",
            "add_topic",
            "replace_topic",
            "change_role",
        }:
            raise ValueError(
                f"Unsupported edit action: {request.edit_action!r}"
            )

        reviewer_reason = cls._clean_text(
            request.reviewer_reason
        )
        stored_evidence = cls._clean_text(
            request.stored_evidence
        )
        current_evidence = cls._clean_text(
            request.current_evidence
        )

        if not reviewer_reason:
            raise ValueError(
                "Reviewer reason is required for contextual validation."
            )

        if not stored_evidence:
            raise ValueError(
                "Stored reviewed evidence is required."
            )

        if not current_evidence:
            raise ValueError(
                "Current transcript evidence is required."
            )

        return ReviewerReasonValidationRequest(
            edit_action=edit_action,
            source_topic=cls._clean_text(
                request.source_topic
            ) or None,
            source_role=cls._clean_text(
                request.source_role
            ).casefold() or None,
            target_topic=cls._clean_text(
                request.target_topic
            ) or None,
            target_role=cls._clean_text(
                request.target_role
            ).casefold() or None,
            reviewer_reason=reviewer_reason,
            stored_evidence=stored_evidence,
            current_evidence=current_evidence,
        )

    @staticmethod
    def _bool_field(
        payload: Mapping[str, Any],
        field_name: str,
        *,
        allow_none: bool = False,
    ) -> bool | None:
        value = payload.get(field_name)

        if allow_none and value is None:
            return None

        if isinstance(value, bool):
            return value

        raise ValueError(
            f"Provider field {field_name!r} must be a boolean"
            + (" or null." if allow_none else ".")
        )

    @staticmethod
    def _confidence_field(
        payload: Mapping[str, Any],
    ) -> float:
        raw = payload.get("confidence")

        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Provider confidence must be numeric."
            ) from exc

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Provider confidence must be between 0 and 1."
            )

        return value

    def validate(
        self,
        request: ReviewerReasonValidationRequest,
    ) -> ReviewerReasonValidationResult:
        request = self._validate_request(request)

        raw = self.provider.validate_edit_context(
            edit_action=request.edit_action,
            source_topic=request.source_topic,
            source_role=request.source_role,
            target_topic=request.target_topic,
            target_role=request.target_role,
            reviewer_reason=request.reviewer_reason,
            stored_evidence=request.stored_evidence,
            current_evidence=request.current_evidence,
        )

        decision = self._clean_text(
            raw.get("decision")
        ).casefold()

        if decision not in VALID_DECISIONS:
            raise ValueError(
                "Provider decision must be one of: "
                "compatible, incompatible, uncertain."
            )

        rationale_still_applies = self._bool_field(
            raw,
            "rationale_still_applies",
        )

        same_teaching_context = self._bool_field(
            raw,
            "same_teaching_context",
        )

        independent_teaching_detected = self._bool_field(
            raw,
            "independent_teaching_detected",
            allow_none=True,
        )

        confidence = self._confidence_field(raw)

        explanation = self._clean_text(
            raw.get("explanation")
        )

        if not explanation:
            raise ValueError(
                "Provider must return a non-empty explanation."
            )

        # --------------------------------------------------------------
        # Conservative policy
        # --------------------------------------------------------------
        # An edit is safe for future automatic reuse only if ALL of these
        # independently agree:
        #
        # 1. provider explicitly says compatible;
        # 2. human rationale still applies;
        # 3. teaching context is materially the same;
        # 4. provider confidence is high;
        # 5. for removal edits, the topic is NOT independently taught.
        #
        # Any disagreement or uncertainty means abstain.
        # --------------------------------------------------------------

        safe = (
            decision == "compatible"
            and rationale_still_applies is True
            and same_teaching_context is True
            and confidence
            >= self.config.minimum_provider_confidence
        )

        if (
            request.edit_action == "remove_topic"
            and independent_teaching_detected is not False
        ):
            safe = False

        # Provider contradictions always force abstention.
        if decision == "compatible":
            if not rationale_still_applies:
                safe = False
            if not same_teaching_context:
                safe = False

        if decision in {
            "incompatible",
            "uncertain",
        }:
            safe = False

        return ReviewerReasonValidationResult(
            decision=decision,
            rationale_still_applies=bool(
                rationale_still_applies
            ),
            same_teaching_context=bool(
                same_teaching_context
            ),
            independent_teaching_detected=(
                independent_teaching_detected
            ),
            confidence=confidence,
            explanation=explanation,
            safe_for_automatic_reuse=bool(safe),
        )