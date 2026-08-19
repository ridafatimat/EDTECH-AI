from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Protocol, Sequence


class TextEmbedderProtocol(Protocol):
    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        ...


VALID_CONTEXT_DECISIONS = frozenset(
    {
        "strong_match",
        "strong_mismatch",
        "ambiguous",
    }
)


@dataclass(frozen=True, slots=True)
class ContextComparisonConfig:
    """
    Precision-first contextual gate for reviewer-approved final-topic edits.

    This is deliberately separate from Module 3 scoring/ranking thresholds.
    It does not change topic extraction, Qdrant, syllabus mapping, or the
    existing rough-topic memory layer.

    The gate is intentionally conservative:
    - exact/near-identical reviewed evidence can be reused deterministically;
    - broader automatic reuse needs agreement between stored evidence,
      reviewer reason, semantic similarity, and lexical context;
    - clearly unrelated contexts are rejected deterministically;
    - everything in the middle is marked ambiguous for the existing LLM
      rationale validator.

    Defaults can be overridden through DETECTED_TOPIC_EDIT_CONTEXT_* env vars
    without modifying Module 3 logic.
    """

    minimum_evidence_characters: int = 40

    # Near-identical evidence path. This is deliberately stricter than the
    # normal rough-topic memory thresholds because final-topic edits can remove
    # or alter an otherwise valid Module 3 topic.
    near_identical_evidence_similarity: float = 0.97
    near_identical_token_containment: float = 0.70
    near_identical_length_ratio_floor: float = 0.75

    # Multi-signal strong-context path. These values are NOT Module 3
    # thresholds and do not affect fresh topic detection.
    standard_evidence_similarity: float = 0.92
    role_change_evidence_similarity: float = 0.94
    add_topic_evidence_similarity: float = 0.95
    reviewer_reason_similarity: float = 0.70
    combined_similarity: float = 0.88
    minimum_token_containment: float = 0.35
    context_length_ratio_floor: float = 0.65

    # Only very clearly unrelated contexts bypass the LLM as a deterministic
    # mismatch. Borderline cases remain ambiguous/fail-closed.
    mismatch_evidence_ceiling: float = 0.20
    mismatch_reason_ceiling: float = 0.20
    mismatch_token_containment_ceiling: float = 0.08

    reviewer_reason_weight: float = 0.25

    @classmethod
    def from_environment(cls) -> "ContextComparisonConfig":
        defaults = cls()

        def _float(name: str, default: float) -> float:
            return float(os.getenv(name, str(default)))

        def _int(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))

        return cls(
            minimum_evidence_characters=_int(
                "DETECTED_TOPIC_EDIT_CONTEXT_MIN_CHARS",
                defaults.minimum_evidence_characters,
            ),
            near_identical_evidence_similarity=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_NEAR_IDENTICAL_EVIDENCE",
                defaults.near_identical_evidence_similarity,
            ),
            near_identical_token_containment=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_NEAR_IDENTICAL_TOKEN",
                defaults.near_identical_token_containment,
            ),
            near_identical_length_ratio_floor=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_NEAR_IDENTICAL_LENGTH_FLOOR",
                defaults.near_identical_length_ratio_floor,
            ),
            standard_evidence_similarity=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_STANDARD_EVIDENCE",
                defaults.standard_evidence_similarity,
            ),
            role_change_evidence_similarity=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_ROLE_EVIDENCE",
                defaults.role_change_evidence_similarity,
            ),
            add_topic_evidence_similarity=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_ADD_EVIDENCE",
                defaults.add_topic_evidence_similarity,
            ),
            reviewer_reason_similarity=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_REASON",
                defaults.reviewer_reason_similarity,
            ),
            combined_similarity=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_COMBINED",
                defaults.combined_similarity,
            ),
            minimum_token_containment=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_TOKEN",
                defaults.minimum_token_containment,
            ),
            context_length_ratio_floor=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_LENGTH_FLOOR",
                defaults.context_length_ratio_floor,
            ),
            mismatch_evidence_ceiling=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_MISMATCH_EVIDENCE",
                defaults.mismatch_evidence_ceiling,
            ),
            mismatch_reason_ceiling=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_MISMATCH_REASON",
                defaults.mismatch_reason_ceiling,
            ),
            mismatch_token_containment_ceiling=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_MISMATCH_TOKEN",
                defaults.mismatch_token_containment_ceiling,
            ),
            reviewer_reason_weight=_float(
                "DETECTED_TOPIC_EDIT_CONTEXT_REASON_WEIGHT",
                defaults.reviewer_reason_weight,
            ),
        )

    def __post_init__(self) -> None:
        bounded = (
            "near_identical_evidence_similarity",
            "near_identical_token_containment",
            "near_identical_length_ratio_floor",
            "standard_evidence_similarity",
            "role_change_evidence_similarity",
            "add_topic_evidence_similarity",
            "reviewer_reason_similarity",
            "combined_similarity",
            "minimum_token_containment",
            "context_length_ratio_floor",
            "mismatch_evidence_ceiling",
            "mismatch_reason_ceiling",
            "mismatch_token_containment_ceiling",
            "reviewer_reason_weight",
        )

        for name in bounded:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if self.minimum_evidence_characters < 1:
            raise ValueError("minimum_evidence_characters must be positive.")


@dataclass(frozen=True, slots=True)
class ContextComparisonResult:
    decision: str
    evidence_similarity: float | None
    reviewer_reason_similarity: float | None
    combined_similarity: float | None
    token_containment: float | None
    length_ratio: float | None
    confidence: float
    explanation: str

    @property
    def is_strong_match(self) -> bool:
        return self.decision == "strong_match"

    @property
    def is_strong_mismatch(self) -> bool:
        return self.decision == "strong_mismatch"

    @property
    def is_ambiguous(self) -> bool:
        return self.decision == "ambiguous"


class DetectedTopicEditContextualComparator:
    """
    Compare CURRENT transcript evidence against BOTH pieces of human memory:

        stored reviewed evidence + reviewer correction reason

    This mirrors the philosophy of Agent 1's rough-topic memory layer while
    remaining deliberately stricter for final-topic edits.

    It never applies an edit, never writes PostgreSQL, and never changes Module
    3 output. It only returns one of three outcomes:

        strong_match    -> safe enough for deterministic reuse
        strong_mismatch -> clearly unrelated; do not reuse
        ambiguous       -> ask the existing LLM reason validator
    """

    def __init__(
        self,
        *,
        embedder: TextEmbedderProtocol,
        config: ContextComparisonConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.config = config or ContextComparisonConfig.from_environment()

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        return " ".join(str(value or "").strip().casefold().split())

    @classmethod
    def _text_hash(cls, value: str) -> str:
        normalized = cls._normalize_text(value)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", value.casefold())
            if len(token) >= 3
        }

    @classmethod
    def _token_containment(cls, left: str, right: str) -> float:
        left_tokens = cls._tokens(left)
        right_tokens = cls._tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        denominator = min(len(left_tokens), len(right_tokens))
        return len(left_tokens & right_tokens) / denominator

    @staticmethod
    def _length_ratio(left: str, right: str) -> float:
        left_len = max(1, len(left))
        right_len = max(1, len(right))
        longer = max(left_len, right_len)
        shorter = min(left_len, right_len)
        return shorter / longer

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding vectors must have equal dimensions.")
        if not left:
            return 0.0

        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
        right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (left_norm * right_norm)))

    def _evidence_threshold(self, edit_action: str) -> float:
        if edit_action == "add_topic":
            return self.config.add_topic_evidence_similarity
        if edit_action == "change_role":
            return self.config.role_change_evidence_similarity
        return self.config.standard_evidence_similarity

    def compare(
        self,
        *,
        edit_action: str,
        stored_evidence: str,
        reviewer_reason: str,
        current_evidence: str,
    ) -> ContextComparisonResult:
        action = self._normalize_text(edit_action)
        stored = self._normalize_text(stored_evidence)
        reason = self._normalize_text(reviewer_reason)
        current = self._normalize_text(current_evidence)

        if action not in {
            "remove_topic",
            "add_topic",
            "replace_topic",
            "change_role",
        }:
            return ContextComparisonResult(
                decision="ambiguous",
                evidence_similarity=None,
                reviewer_reason_similarity=None,
                combined_similarity=None,
                token_containment=None,
                length_ratio=None,
                confidence=0.0,
                explanation="Unsupported edit action; contextual gate abstained.",
            )

        if (
            len(stored) < self.config.minimum_evidence_characters
            or len(current) < self.config.minimum_evidence_characters
            or not reason
        ):
            return ContextComparisonResult(
                decision="ambiguous",
                evidence_similarity=None,
                reviewer_reason_similarity=None,
                combined_similarity=None,
                token_containment=None,
                length_ratio=None,
                confidence=0.0,
                explanation=(
                    "Evidence/reviewer reason is insufficient for a deterministic "
                    "context decision; defer to the rationale validator."
                ),
            )

        # Exact reviewed evidence is the safest deterministic reuse path.
        if self._text_hash(stored) == self._text_hash(current):
            return ContextComparisonResult(
                decision="strong_match",
                evidence_similarity=1.0,
                reviewer_reason_similarity=None,
                combined_similarity=1.0,
                token_containment=1.0,
                length_ratio=1.0,
                confidence=1.0,
                explanation=(
                    "Context-first reuse: current evidence exactly matches the "
                    "human-reviewed evidence for this edit."
                ),
            )

        vectors = self.embedder.embed_texts([current, stored, reason])
        evidence_similarity = self._cosine(vectors[0], vectors[1])
        reason_similarity = self._cosine(vectors[0], vectors[2])
        reason_weight = self.config.reviewer_reason_weight
        combined_similarity = (
            (1.0 - reason_weight) * evidence_similarity
            + reason_weight * reason_similarity
        )
        token_containment = self._token_containment(current, stored)
        length_ratio = self._length_ratio(current, stored)

        near_identical = (
            evidence_similarity
            >= self.config.near_identical_evidence_similarity
            and token_containment
            >= self.config.near_identical_token_containment
            and length_ratio
            >= self.config.near_identical_length_ratio_floor
        )

        strong_context = (
            evidence_similarity >= self._evidence_threshold(action)
            and reason_similarity >= self.config.reviewer_reason_similarity
            and combined_similarity >= self.config.combined_similarity
            and token_containment >= self.config.minimum_token_containment
            and length_ratio >= self.config.context_length_ratio_floor
        )

        if near_identical or strong_context:
            mode = "near-identical" if near_identical else "multi-signal"
            confidence = max(
                0.0,
                min(1.0, combined_similarity),
            )
            return ContextComparisonResult(
                decision="strong_match",
                evidence_similarity=evidence_similarity,
                reviewer_reason_similarity=reason_similarity,
                combined_similarity=combined_similarity,
                token_containment=token_containment,
                length_ratio=length_ratio,
                confidence=confidence,
                explanation=(
                    f"Context-first reuse ({mode}): stored evidence and reviewer "
                    "reason strongly agree with current transcript evidence."
                ),
            )

        clear_mismatch = (
            evidence_similarity <= self.config.mismatch_evidence_ceiling
            and reason_similarity <= self.config.mismatch_reason_ceiling
            and token_containment
            <= self.config.mismatch_token_containment_ceiling
        )

        if clear_mismatch:
            confidence = max(
                0.0,
                min(
                    1.0,
                    1.0 - max(evidence_similarity, reason_similarity),
                ),
            )
            return ContextComparisonResult(
                decision="strong_mismatch",
                evidence_similarity=evidence_similarity,
                reviewer_reason_similarity=reason_similarity,
                combined_similarity=combined_similarity,
                token_containment=token_containment,
                length_ratio=length_ratio,
                confidence=confidence,
                explanation=(
                    "Context-first rejection: current evidence is clearly "
                    "unrelated to both the stored reviewed evidence and the "
                    "human correction rationale."
                ),
            )

        return ContextComparisonResult(
            decision="ambiguous",
            evidence_similarity=evidence_similarity,
            reviewer_reason_similarity=reason_similarity,
            combined_similarity=combined_similarity,
            token_containment=token_containment,
            length_ratio=length_ratio,
            confidence=max(0.0, min(1.0, combined_similarity)),
            explanation=(
                "Context comparison is not strong enough to reuse or reject the "
                "human edit deterministically; defer to the existing LLM "
                "rationale validator."
            ),
        )
