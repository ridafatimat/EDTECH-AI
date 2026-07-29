from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Match, Pattern

from app.schemas.technical_normalisation import (
    AppliedTechnicalCorrection,
)


Replacement = str | Callable[[Match[str]], str]


@dataclass(frozen=True, slots=True)
class SpokenCodeRule:
    rule_id: str
    pattern: Pattern[str]
    replacement: Replacement
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class SpokenCodeResult:
    text: str
    corrections: tuple[AppliedTechnicalCorrection, ...]


class SpokenCodeNormaliser:
    """
    Deterministically convert only unambiguous spoken code notation.

    This service does not correct ASR wording. For example, it converts
    "array dot length" to "array.length", but it does not decide whether
    "wild loop" means "while loop".
    """

    _RULES = (
        SpokenCodeRule(
            rule_id="spoken_member_access",
            pattern=re.compile(
                r"\b([A-Za-z_][A-Za-z0-9_]*)\s+dot\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)\b",
                re.IGNORECASE,
            ),
            replacement=lambda match: (
                f"{match.group(1)}.{match.group(2)}"
            ),
            confidence=0.99,
            reason=(
                "Converted unambiguous spoken member access to "
                "code notation."
            ),
        ),
    )

    def normalise(self, text: str) -> SpokenCodeResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        working = text
        corrections: list[AppliedTechnicalCorrection] = []

        for rule in self._RULES:
            working, current = self._apply_rule(
                text=working,
                rule=rule,
            )
            corrections.extend(current)

        return SpokenCodeResult(
            text=working,
            corrections=tuple(corrections),
        )

    @staticmethod
    def _apply_rule(
        *,
        text: str,
        rule: SpokenCodeRule,
    ) -> tuple[str, list[AppliedTechnicalCorrection]]:
        corrections: list[AppliedTechnicalCorrection] = []

        def replace(match: Match[str]) -> str:
            replacement = (
                rule.replacement(match)
                if callable(rule.replacement)
                else rule.replacement
            )

            corrections.append(
                AppliedTechnicalCorrection(
                    issue_id=(
                        f"{rule.rule_id}:"
                        f"{match.start()}:{match.end()}"
                    ),
                    original=match.group(0),
                    replacement=replacement,
                    start=match.start(),
                    end=match.end(),
                    source="spoken_code_rule",
                    confidence=rule.confidence,
                    correction_type="spoken_code",
                    reason=rule.reason,
                    sentence_text=_sentence_around(
                        text=text,
                        start=match.start(),
                        end=match.end(),
                    ),
                )
            )

            return replacement

        return rule.pattern.sub(replace, text), corrections


def _sentence_around(
    *,
    text: str,
    start: int,
    end: int,
) -> str:
    left = max(
        text.rfind(".", 0, start),
        text.rfind("?", 0, start),
        text.rfind("!", 0, start),
        text.rfind("\n", 0, start),
    )

    right_positions = [
        position
        for position in (
            text.find(".", end),
            text.find("?", end),
            text.find("!", end),
            text.find("\n", end),
        )
        if position != -1
    ]

    sentence_start = left + 1 if left != -1 else 0
    sentence_end = (
        min(right_positions) + 1
        if right_positions
        else len(text)
    )

    return text[sentence_start:sentence_end].strip()