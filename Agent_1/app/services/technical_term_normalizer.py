from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Iterable, Match, Pattern

from app.schemas.technical_normalisation import (
    TechnicalCorrection,
    TechnicalNormalisationResult,
)


Replacement = str | Callable[[Match[str]], str]


@dataclass(frozen=True, slots=True)
class NormalisationRule:
    """A conservative, context-aware ASR correction rule."""

    rule_id: str
    pattern: Pattern[str]
    replacement: Replacement
    confidence: float
    reason: str
    context_any: tuple[str, ...] = ()
    context_all: tuple[str, ...] = ()
    negative_context_any: tuple[str, ...] = ()


class TechnicalTermNormaliser:
    """
    Module 1B: deterministic technical terminology normalisation.

    Design rules:
    - No LLM.
    - Never summarise or rewrite sentences.
    - Correct only high-confidence ASR/technical terminology errors.
    - Require context for ambiguous corrections.
    - Keep an audit trail for every change.
    - Be idempotent: normalising twice should not change the text again.
    """

    _GENERAL_TECHNICAL_CONTEXT = (
        "algorithm",
        "array",
        "binary",
        "boolean",
        "cache",
        "class",
        "code",
        "compiler",
        "condition",
        "database",
        "execute",
        "function",
        "index",
        "instruction",
        "iteration",
        "loop",
        "memory",
        "parameter",
        "procedure",
        "program",
        "query",
        "record",
        "search",
        "statement",
        "table",
        "variable",
    )

    _DISTINCTIVE_FUZZY_TERMS = {
        "algorithm",
        "pseudocode",
        "hexadecimal",
        "subroutine",
        "iteration",
        "interpreter",
        "compiler",
        "bytecode",
    }

    _AMBIGUOUS_FUZZY_TERMS = {
        "array",
        "bit",
        "byte",
        "cache",
        "character",
        "field",
        "function",
        "integer",
        "procedure",
        "record",
        "string",
    }

    _DEFAULT_CANONICAL_TERMS = (
        "algorithm",
        "array",
        "binary",
        "Boolean",
        "bubble",
        "byte",
        "bytecode",
        "cache",
        "character",
        "compiler",
        "denary",
        "function",
        "hexadecimal",
        "integer",
        "interpreter",
        "iteration",
        "parameter",
        "procedure",
        "pseudocode",
        "record",
        "selection",
        "sequence",
        "string",
        "subroutine",
    )

    def __init__(
        self,
        *,
        canonical_terms: Iterable[str] | None = None,
        enable_fuzzy_spelling: bool = True,
        fuzzy_threshold: float = 0.92,
        fuzzy_margin: float = 0.05,
    ) -> None:
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be between 0 and 1.")

        if not 0.0 <= fuzzy_margin <= 1.0:
            raise ValueError("fuzzy_margin must be between 0 and 1.")

        supplied_terms = tuple(canonical_terms or ())
        combined_terms = self._DEFAULT_CANONICAL_TERMS + supplied_terms

        self._canonical_terms = self._deduplicate_terms(combined_terms)
        self._enable_fuzzy_spelling = enable_fuzzy_spelling
        self._fuzzy_threshold = fuzzy_threshold
        self._fuzzy_margin = fuzzy_margin
        self._rules = self._build_rules()

    def normalise(self, text: str) -> TechnicalNormalisationResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text.strip():
            return TechnicalNormalisationResult(
                original_text=text,
                normalised_text=text,
            )

        working = text
        corrections: list[TechnicalCorrection] = []

        for rule in self._rules:
            working, rule_corrections = self._apply_rule(
                text=working,
                rule=rule,
            )
            corrections.extend(rule_corrections)

        if self._enable_fuzzy_spelling:
            working, fuzzy_corrections = self._normalise_fuzzy_spellings(
                working
            )
            corrections.extend(fuzzy_corrections)

        return TechnicalNormalisationResult(
            original_text=text,
            normalised_text=working,
            corrections=corrections,
        )

    @staticmethod
    def _deduplicate_terms(terms: Iterable[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []

        for term in terms:
            cleaned = term.strip()
            key = cleaned.casefold()

            if not cleaned or key in seen:
                continue

            seen.add(key)
            result.append(cleaned)

        return tuple(result)

    @staticmethod
    def _build_rules() -> tuple[NormalisationRule, ...]:
        programming_context = (
            "array",
            "condition",
            "execute",
            "index",
            "less than",
            "greater than",
            "program",
            "statement",
            "variable",
        )

        return (
            NormalisationRule(
                rule_id="asr_wild_loop",
                pattern=re.compile(r"\bwild\s+loop\b", re.IGNORECASE),
                replacement="while loop",
                confidence=0.97,
                reason=(
                    "Likely ASR confusion between 'wild loop' and "
                    "'while loop' in programming context."
                ),
                context_any=programming_context,
                negative_context_any=(
                    "roller coaster",
                    "ride",
                    "story",
                    "music",
                    "wildlife",
                ),
            ),
            NormalisationRule(
                rule_id="asr_wild_case",
                pattern=re.compile(r"\bwild\s+case\b", re.IGNORECASE),
                replacement="while K is",
                confidence=0.93,
                reason=(
                    "Likely ASR rendering of 'while K is' inside an "
                    "array/condition explanation."
                ),
                context_any=(
                    "array",
                    "length",
                    "limit",
                    "less than",
                    "greater than",
                ),
                context_all=("array",),
            ),
            NormalisationRule(
                rule_id="spoken_dot_length",
                pattern=re.compile(
                    r"\b([A-Za-z_][A-Za-z0-9_]*)\s+dot\s+length\b",
                    re.IGNORECASE,
                ),
                replacement=lambda match: f"{match.group(1)}.length",
                confidence=0.99,
                reason=(
                    "Normalised a spoken property-access phrase to "
                    "code-style '.length'."
                ),
                context_any=(
                    "array",
                    "index",
                    "loop",
                    "less than",
                    "greater than",
                ),
            ),
            NormalisationRule(
                rule_id="asr_cash_memory",
                pattern=re.compile(r"\bcash\s+memory\b", re.IGNORECASE),
                replacement="cache memory",
                confidence=0.97,
                reason=(
                    "Likely ASR confusion between 'cash memory' and "
                    "'cache memory'."
                ),
                context_any=(
                    "cpu",
                    "processor",
                    "computer",
                    "memory",
                    "storage",
                ),
            ),
            NormalisationRule(
                rule_id="asr_sequel_sql",
                pattern=re.compile(r"\bsequel\b", re.IGNORECASE),
                replacement="SQL",
                confidence=0.96,
                reason=(
                    "Likely spoken/ASR form of SQL in a database context."
                ),
                context_any=(
                    "database",
                    "query",
                    "table",
                    "select",
                    "insert",
                    "update",
                    "delete",
                ),
            ),
            NormalisationRule(
                rule_id="asr_bite_code",
                pattern=re.compile(r"\bbite\s+code\b", re.IGNORECASE),
                replacement="bytecode",
                confidence=0.97,
                reason=(
                    "Likely ASR confusion between 'bite code' and "
                    "'bytecode'."
                ),
                context_any=(
                    "compiler",
                    "interpreter",
                    "java",
                    "python",
                    "virtual machine",
                    "instruction",
                ),
            ),
            NormalisationRule(
                rule_id="missing_or_less_equal",
                pattern=re.compile(
                    r"\bless\s+than\s+equal\s+to\b",
                    re.IGNORECASE,
                ),
                replacement="less than or equal to",
                confidence=0.99,
                reason="Restored the standard relational-operator phrase.",
                context_any=(
                    "condition",
                    "if",
                    "while",
                    "array",
                    "value",
                    "variable",
                    "index",
                ),
            ),
            NormalisationRule(
                rule_id="missing_or_greater_equal",
                pattern=re.compile(
                    r"\bgreater\s+than\s+equal\s+to\b",
                    re.IGNORECASE,
                ),
                replacement="greater than or equal to",
                confidence=0.99,
                reason="Restored the standard relational-operator phrase.",
                context_any=(
                    "condition",
                    "if",
                    "while",
                    "array",
                    "value",
                    "variable",
                    "index",
                ),
            ),
        )

    def _apply_rule(
        self,
        *,
        text: str,
        rule: NormalisationRule,
    ) -> tuple[str, list[TechnicalCorrection]]:
        corrections: list[TechnicalCorrection] = []

        def replacement_callback(match: Match[str]) -> str:
            context = self._context_window(
                text=text,
                start=match.start(),
                end=match.end(),
            )

            if not self._context_matches(
                context=context,
                rule=rule,
            ):
                return match.group(0)

            if callable(rule.replacement):
                replacement = rule.replacement(match)
            else:
                replacement = rule.replacement

            replacement = self._preserve_initial_case(
                original=match.group(0),
                replacement=replacement,
            )

            if replacement == match.group(0):
                return match.group(0)

            corrections.append(
                TechnicalCorrection(
                    original=match.group(0),
                    normalised=replacement,
                    rule_id=rule.rule_id,
                    confidence=rule.confidence,
                    reason=rule.reason,
                    context=context.strip(),
                )
            )

            return replacement

        updated = rule.pattern.sub(replacement_callback, text)
        return updated, corrections

    @staticmethod
    def _context_matches(
        *,
        context: str,
        rule: NormalisationRule,
    ) -> bool:
        lowered = context.casefold()

        if any(
            negative.casefold() in lowered
            for negative in rule.negative_context_any
        ):
            return False

        if rule.context_all and not all(
            required.casefold() in lowered
            for required in rule.context_all
        ):
            return False

        if rule.context_any and not any(
            required.casefold() in lowered
            for required in rule.context_any
        ):
            return False

        return True

    @staticmethod
    def _context_window(
        *,
        text: str,
        start: int,
        end: int,
        radius: int = 180,
    ) -> str:
        left = max(
            text.rfind(".", 0, start),
            text.rfind("?", 0, start),
            text.rfind("!", 0, start),
            text.rfind("\n", 0, start),
        )

        right_candidates = [
            position
            for position in (
                text.find(".", end),
                text.find("?", end),
                text.find("!", end),
                text.find("\n", end),
            )
            if position != -1
        ]

        sentence_start = left + 1 if left != -1 else max(0, start - radius)
        sentence_end = (
            min(right_candidates) + 1
            if right_candidates
            else min(len(text), end + radius)
        )

        return text[sentence_start:sentence_end]

    @staticmethod
    def _preserve_initial_case(
        *,
        original: str,
        replacement: str,
    ) -> str:
        if not original:
            return replacement

        if original[0].isupper() and replacement:
            return replacement[0].upper() + replacement[1:]

        return replacement

    def _normalise_fuzzy_spellings(
        self,
        text: str,
    ) -> tuple[str, list[TechnicalCorrection]]:
        token_pattern = re.compile(r"\b[A-Za-z][A-Za-z'-]{4,}\b")
        matches = list(token_pattern.finditer(text))
        replacements: list[tuple[int, int, str, TechnicalCorrection]] = []

        one_word_terms = [
            term for term in self._canonical_terms if " " not in term
        ]
        canonical_by_lower = {
            term.casefold(): term for term in one_word_terms
        }

        for match in matches:
            token = match.group(0)
            token_lower = token.casefold()

            if token_lower in canonical_by_lower:
                continue

            ranked = sorted(
                (
                    (
                        SequenceMatcher(
                            None,
                            token_lower,
                            canonical.casefold(),
                        ).ratio(),
                        canonical,
                    )
                    for canonical in one_word_terms
                ),
                reverse=True,
            )

            if not ranked:
                continue

            best_score, best_term = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0

            if best_score < self._fuzzy_threshold:
                continue

            if best_score - second_score < self._fuzzy_margin:
                continue

            context = self._context_window(
                text=text,
                start=match.start(),
                end=match.end(),
            )
            context_lower = context.casefold()
            best_lower = best_term.casefold()

            requires_context = best_lower in self._AMBIGUOUS_FUZZY_TERMS
            is_distinctive = best_lower in self._DISTINCTIVE_FUZZY_TERMS
            has_context = any(
                anchor in context_lower
                for anchor in self._GENERAL_TECHNICAL_CONTEXT
            )

            if requires_context and not has_context:
                continue

            if not requires_context and not is_distinctive and not has_context:
                continue

            replacement = self._preserve_initial_case(
                original=token,
                replacement=best_term,
            )

            correction = TechnicalCorrection(
                original=token,
                normalised=replacement,
                rule_id="conservative_fuzzy_spelling",
                confidence=round(best_score, 4),
                reason=(
                    "High-similarity spelling correction against the "
                    "controlled technical glossary."
                ),
                context=context.strip(),
            )

            replacements.append(
                (match.start(), match.end(), replacement, correction)
            )

        if not replacements:
            return text, []

        working = text
        for start, end, replacement, _ in reversed(replacements):
            working = working[:start] + replacement + working[end:]

        corrections = [
            correction for _, _, _, correction in replacements
        ]
        return working, corrections