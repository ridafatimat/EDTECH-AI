from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from app.schemas.technical_normalisation import (
    CorrectionProposal,
    SuspiciousTechnicalSpan,
)
from app.services.technical_vocabulary import (
    TechnicalVocabulary,
    are_simple_inflectional_variants,
    normalise_lookup_key,
    tokenise_lookup_key,
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    accepted: bool
    reason: str


class TechnicalCorrectionValidator:
    """
    Validate one exact replacement pair before transcript modification.

    Precision safeguards block:
    - valid canonical/grammatical phrases;
    - singular/plural or verb-form-only changes;
    - punctuation-crossing spans;
    - changes to ordinary control/connector words;
    - multi-token reinterpretations;
    - control-flow identifiers such as "while low less than high".
    """

    _SAFE_REPLACEMENT_PATTERN = re.compile(
        r"^[A-Za-z0-9_+\-./<>=\[\]() ]+$"
    )

    _COMPARISON_AFTER_IDENTIFIER = re.compile(
        r"^\s*(?:"
        r"less\s+than|"
        r"greater\s+than|"
        r"equal(?:s|\s+to)?|"
        r"is\s+(?:less|greater|equal)|"
        r"<=|>=|==|!=|<|>"
        r")",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        vocabulary: TechnicalVocabulary,
        minimum_confidence: float = 0.82,
        minimum_changed_token_similarity: float = 0.62,
        max_replacement_words: int = 6,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError(
                "minimum_confidence must be between 0 and 1."
            )

        if not 0 <= minimum_changed_token_similarity <= 1:
            raise ValueError(
                "minimum_changed_token_similarity must be between 0 and 1."
            )

        self._vocabulary = vocabulary
        self._minimum_confidence = minimum_confidence
        self._minimum_changed_token_similarity = (
            minimum_changed_token_similarity
        )
        self._max_replacement_words = max_replacement_words

    def validate(
        self,
        *,
        issue: SuspiciousTechnicalSpan,
        proposal: CorrectionProposal,
        current_text: str,
    ) -> ValidationResult:
        if proposal.issue_id != issue.issue_id:
            return ValidationResult(
                False,
                "Proposal issue_id does not match the detected issue.",
            )

        if proposal.original != issue.original_span:
            return ValidationResult(
                False,
                "Proposal original is not the exact detected span.",
            )

        actual = current_text[
            issue.start:issue.end
        ]

        if actual != issue.original_span:
            return ValidationResult(
                False,
                "The source text changed after detection.",
            )

        replacement = proposal.replacement.strip()

        if not replacement:
            return ValidationResult(
                False,
                "Replacement is empty.",
            )

        if proposal.confidence < self._minimum_confidence:
            return ValidationResult(
                False,
                "Confidence is below the acceptance threshold.",
            )

        if "\n" in replacement or "\r" in replacement:
            return ValidationResult(
                False,
                "Replacement contains a line break.",
            )

        if len(replacement.split()) > self._max_replacement_words:
            return ValidationResult(
                False,
                "Replacement is too long for one technical span.",
            )

        if not self._SAFE_REPLACEMENT_PATTERN.fullmatch(
            replacement
        ):
            return ValidationResult(
                False,
                "Replacement contains unsupported characters.",
            )

        original_key = normalise_lookup_key(
            proposal.original
        )
        replacement_key = normalise_lookup_key(
            replacement
        )

        if not original_key or not replacement_key:
            return ValidationResult(
                False,
                "Original or replacement has no lexical content.",
            )

        if original_key == replacement_key:
            return ValidationResult(
                False,
                "Replacement only changes formatting or punctuation.",
            )

        if self._vocabulary.is_valid_surface_form(
            proposal.original
        ):
            return ValidationResult(
                False,
                "Original span is already valid technical wording.",
            )

        if not _has_plain_word_separators(
            proposal.original
        ):
            return ValidationResult(
                False,
                "Detected span crosses punctuation.",
            )

        original_tokens = tokenise_lookup_key(
            proposal.original
        )
        replacement_tokens = tokenise_lookup_key(
            replacement
        )

        if len(original_tokens) != len(replacement_tokens):
            return ValidationResult(
                False,
                "Replacement changes the number of lexical tokens.",
            )

        mismatch_indices: list[int] = []
        grammatical_only = True

        for index, (original, corrected) in enumerate(
            zip(original_tokens, replacement_tokens)
        ):
            if original == corrected:
                continue

            mismatch_indices.append(index)

            if not are_simple_inflectional_variants(
                original,
                corrected,
            ):
                grammatical_only = False

        if not mismatch_indices:
            return ValidationResult(
                False,
                "Replacement does not change a lexical token.",
            )

        if grammatical_only:
            return ValidationResult(
                False,
                "Replacement only changes plurality or verb form.",
            )

        if len(mismatch_indices) != 1:
            return ValidationResult(
                False,
                "Replacement reinterprets more than one token.",
            )

        mismatch_index = mismatch_indices[0]
        original_token = original_tokens[mismatch_index]
        corrected_token = replacement_tokens[mismatch_index]

        if (
            original_token
            in self._vocabulary.protected_tokens
        ):
            return ValidationResult(
                False,
                "Replacement changes a valid grammatical/control word.",
            )

        similarity = SequenceMatcher(
            None,
            original_token,
            corrected_token,
        ).ratio()

        if similarity < self._minimum_changed_token_similarity:
            return ValidationResult(
                False,
                "Changed token is not close enough to be a safe ASR or spelling correction.",
            )

        edit_distance = _levenshtein_distance(
            original_token,
            corrected_token,
        )

        maximum_distance = (
            1
            if max(
                len(original_token),
                len(corrected_token),
            ) <= 7
            and len(original_tokens) == 1
            else 2
        )

        if edit_distance > maximum_distance:
            return ValidationResult(
                False,
                "Changed token requires too many character edits.",
            )

        if self._looks_like_control_flow_identifier(
            issue=issue,
            original_tokens=original_tokens,
            replacement_tokens=replacement_tokens,
        ):
            return ValidationResult(
                False,
                "Original span is a valid control-flow expression using an identifier.",
            )

        if not self._is_supported_replacement(
            issue=issue,
            replacement=replacement,
        ):
            return ValidationResult(
                False,
                "Replacement is not supported by the controlled technical vocabulary or detector candidates.",
            )

        return ValidationResult(
            True,
            "Validated one-token technical ASR/spelling correction.",
        )

    def _looks_like_control_flow_identifier(
        self,
        *,
        issue: SuspiciousTechnicalSpan,
        original_tokens: tuple[str, ...],
        replacement_tokens: tuple[str, ...],
    ) -> bool:
        if len(original_tokens) != 2:
            return False

        if original_tokens[0] not in {
            "while",
            "if",
        }:
            return False

        if replacement_tokens[0] != original_tokens[0]:
            return False

        if replacement_tokens[1] not in {
            "loop",
            "statement",
            "condition",
        }:
            return False

        local_end = issue.end - issue.sentence_start
        remainder = issue.sentence_text[local_end:]

        return bool(
            self._COMPARISON_AFTER_IDENTIFIER.match(
                remainder
            )
        )

    def _is_supported_replacement(
        self,
        *,
        issue: SuspiciousTechnicalSpan,
        replacement: str,
    ) -> bool:
        replacement_key = normalise_lookup_key(
            replacement
        )

        if (
            replacement_key
            in self._vocabulary.canonical_term_keys
        ):
            return True

        candidate_keys = {
            normalise_lookup_key(candidate)
            for candidate in issue.candidate_terms
        }

        if replacement_key in candidate_keys:
            return True

        # Future deterministic code-like corrections remain possible while
        # arbitrary prose stays blocked.
        return bool(
            re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*"
                r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
                replacement,
            )
        )


def ensure_non_overlapping(
    accepted: Iterable[
        tuple[
            SuspiciousTechnicalSpan,
            CorrectionProposal,
        ]
    ],
) -> list[
    tuple[
        SuspiciousTechnicalSpan,
        CorrectionProposal,
    ]
]:
    ordered = sorted(
        accepted,
        key=lambda pair: pair[0].start,
    )

    result: list[
        tuple[
            SuspiciousTechnicalSpan,
            CorrectionProposal,
        ]
    ] = []
    previous_end = -1

    for issue, proposal in ordered:
        if issue.start < previous_end:
            continue

        result.append((issue, proposal))
        previous_end = issue.end

    return result


def _has_plain_word_separators(value: str) -> bool:
    matches = list(
        re.finditer(
            r"[A-Za-z][A-Za-z0-9_'’-]*",
            value,
        )
    )

    if len(matches) <= 1:
        return True

    for left, right in zip(matches, matches[1:]):
        separator = value[left.end():right.start()]

        if not re.fullmatch(r"[ \t]+", separator):
            return False

    return True


def _levenshtein_distance(
    left: str,
    right: str,
) -> int:
    if left == right:
        return 0

    if not left:
        return len(right)

    if not right:
        return len(left)

    previous = list(range(len(right) + 1))

    for left_index, left_character in enumerate(
        left,
        start=1,
    ):
        current = [left_index]

        for right_index, right_character in enumerate(
            right,
            start=1,
        ):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = (
                previous[right_index - 1]
                + (left_character != right_character)
            )
            current.append(
                min(insertion, deletion, substitution)
            )

        previous = current

    return previous[-1]