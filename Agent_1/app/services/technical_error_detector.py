from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Match

from app.schemas.technical_normalisation import (
    SuspiciousTechnicalSpan,
)
from app.services.technical_vocabulary import (
    TechnicalVocabulary,
    are_simple_inflectional_variants,
    normalise_lookup_key,
    tokenise_lookup_key,
)


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    sentence_index: int
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _CandidateMatch:
    term: str
    score: float
    mismatch_index: int
    original_token: str
    canonical_token: str


class TechnicalErrorDetector:
    """
    Precision-first detector for likely technical spelling or ASR errors.

    A span is flagged only when all of the following hold:
    - the sentence contains enough Computer Science context;
    - tokens are contiguous and do not cross punctuation;
    - the span is not already a valid canonical/grammatical surface form;
    - exactly one token is plausibly corrupted;
    - the corrupted token is character-close to the canonical token;
    - the span is not a normal control-flow expression such as
      "while low less than or equal high".

    The detector never changes text. It only produces small candidate spans.
    """

    _TOKEN_PATTERN = re.compile(
        r"[A-Za-z][A-Za-z0-9_'’-]*"
    )

    _SENTENCE_PATTERN = re.compile(
        r".+?(?:[.!?](?=\s|$)|\n+|$)",
        re.DOTALL,
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
        single_word_threshold: float = 0.86,
        phrase_threshold: float = 0.78,
        mismatch_token_threshold: float = 0.62,
        candidate_margin: float = 0.04,
        max_ngram_words: int = 4,
    ) -> None:
        for name, value in (
            ("single_word_threshold", single_word_threshold),
            ("phrase_threshold", phrase_threshold),
            ("mismatch_token_threshold", mismatch_token_threshold),
            ("candidate_margin", candidate_margin),
        ):
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if max_ngram_words < 1:
            raise ValueError(
                "max_ngram_words must be at least 1."
            )

        self._vocabulary = vocabulary
        self._single_word_threshold = single_word_threshold
        self._phrase_threshold = phrase_threshold
        self._mismatch_token_threshold = (
            mismatch_token_threshold
        )
        self._candidate_margin = candidate_margin
        self._max_ngram_words = max_ngram_words
        self._canonical_records = self._build_canonical_records(
            vocabulary.canonical_terms
        )

    def detect(
        self,
        text: str,
    ) -> list[SuspiciousTechnicalSpan]:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        issues: list[SuspiciousTechnicalSpan] = []

        for sentence in self._split_sentences(text):
            context_keywords = self._technical_context_keywords(
                sentence.text
            )

            if not self._has_enough_technical_context(
                context_keywords
            ):
                continue

            issues.extend(
                self._detect_in_sentence(
                    sentence=sentence,
                    context_keywords=context_keywords,
                )
            )

        return self._remove_overlapping_issues(issues)

    @staticmethod
    def _build_canonical_records(
        terms: Iterable[str],
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        records: list[tuple[str, tuple[str, ...]]] = []

        for term in terms:
            tokens = tokenise_lookup_key(term)

            if tokens:
                records.append((term, tokens))

        return tuple(records)

    def _detect_in_sentence(
        self,
        *,
        sentence: SentenceSpan,
        context_keywords: tuple[str, ...],
    ) -> list[SuspiciousTechnicalSpan]:
        tokens = list(
            self._TOKEN_PATTERN.finditer(sentence.text)
        )
        issues: list[SuspiciousTechnicalSpan] = []

        for start_index in range(len(tokens)):
            maximum_size = min(
                self._max_ngram_words,
                len(tokens) - start_index,
            )

            for size in range(1, maximum_size + 1):
                selected = tokens[
                    start_index:start_index + size
                ]

                if (
                    size > 1
                    and not self._has_plain_internal_separators(
                        sentence.text,
                        selected,
                    )
                ):
                    # Never create "OR, loop" or any other candidate that
                    # crosses punctuation.
                    continue

                original = sentence.text[
                    selected[0].start():selected[-1].end()
                ]
                original_key = normalise_lookup_key(original)

                if not original_key:
                    continue

                if self._vocabulary.is_valid_surface_form(
                    original
                ):
                    continue

                matches = self._rank_candidates(
                    original_key=original_key,
                    word_count=size,
                )

                if not matches:
                    continue

                best = matches[0]
                second_score = (
                    matches[1].score
                    if len(matches) > 1
                    else 0.0
                )

                threshold = (
                    self._single_word_threshold
                    if size == 1
                    else self._phrase_threshold
                )

                if best.score < threshold:
                    continue

                if (
                    best.score - second_score
                    < self._candidate_margin
                ):
                    continue

                if self._looks_like_valid_control_expression(
                    sentence_text=sentence.text,
                    selected=selected,
                    original_tokens=tokenise_lookup_key(
                        original
                    ),
                    candidate_tokens=tokenise_lookup_key(
                        best.term
                    ),
                ):
                    continue

                absolute_start = (
                    sentence.start + selected[0].start()
                )
                absolute_end = (
                    sentence.start + selected[-1].end()
                )

                issue_id = (
                    f"s{sentence.sentence_index}:"
                    f"{absolute_start}:{absolute_end}"
                )

                issues.append(
                    SuspiciousTechnicalSpan(
                        issue_id=issue_id,
                        sentence_index=sentence.sentence_index,
                        sentence_text=sentence.text.strip(),
                        sentence_start=sentence.start,
                        start=absolute_start,
                        end=absolute_end,
                        original_span=original,
                        detector_score=round(best.score, 4),
                        reason=(
                            "Exactly one token is close to controlled "
                            f"technical term {best.term!r}: "
                            f"{best.original_token!r} -> "
                            f"{best.canonical_token!r}."
                        ),
                        candidate_terms=tuple(
                            match.term
                            for match in matches[:3]
                        ),
                        context_keywords=context_keywords,
                    )
                )

        return issues

    def _rank_candidates(
        self,
        *,
        original_key: str,
        word_count: int,
    ) -> list[_CandidateMatch]:
        original_tokens = tuple(
            original_key.split()
        )
        matches: list[_CandidateMatch] = []

        for term, canonical_tokens in self._canonical_records:
            if len(canonical_tokens) != word_count:
                continue

            match = self._score_candidate(
                term=term,
                original_tokens=original_tokens,
                canonical_tokens=canonical_tokens,
            )

            if match is not None:
                matches.append(match)

        matches.sort(
            key=lambda item: item.score,
            reverse=True,
        )
        return matches

    def _score_candidate(
        self,
        *,
        term: str,
        original_tokens: tuple[str, ...],
        canonical_tokens: tuple[str, ...],
    ) -> _CandidateMatch | None:
        mismatch_indices: list[int] = []

        for index, (original, canonical) in enumerate(
            zip(original_tokens, canonical_tokens)
        ):
            if original == canonical:
                continue

            if are_simple_inflectional_variants(
                original,
                canonical,
            ):
                continue

            mismatch_indices.append(index)

        # Precision rule: only one token may be corrupted. Multi-token
        # reinterpretations belong in unresolved/manual review, not automatic
        # normalisation.
        if len(mismatch_indices) != 1:
            return None

        mismatch_index = mismatch_indices[0]
        original_token = original_tokens[mismatch_index]
        canonical_token = canonical_tokens[mismatch_index]

        if (
            original_token
            in self._vocabulary.protected_tokens
        ):
            return None

        if min(
            len(original_token),
            len(canonical_token),
        ) < 3:
            return None

        token_similarity = SequenceMatcher(
            None,
            original_token,
            canonical_token,
        ).ratio()

        edit_distance = _levenshtein_distance(
            original_token,
            canonical_token,
        )

        maximum_distance = (
            1
            if max(
                len(original_token),
                len(canonical_token),
            ) <= 7
            else 2
        )

        # Phrases can tolerate a two-edit ASR corruption such as
        # "wild loop" -> "while loop". Single words remain stricter.
        if len(original_tokens) > 1:
            maximum_distance = 2

        if edit_distance > maximum_distance:
            return None

        required_token_similarity = (
            self._single_word_threshold
            if len(original_tokens) == 1
            else self._mismatch_token_threshold
        )

        if token_similarity < required_token_similarity:
            return None

        positional_scores = [
            SequenceMatcher(
                None,
                original,
                canonical,
            ).ratio()
            for original, canonical in zip(
                original_tokens,
                canonical_tokens,
            )
        ]

        overall_score = sum(positional_scores) / len(
            positional_scores
        )

        return _CandidateMatch(
            term=term,
            score=overall_score,
            mismatch_index=mismatch_index,
            original_token=original_token,
            canonical_token=canonical_token,
        )

    @staticmethod
    def _has_plain_internal_separators(
        sentence_text: str,
        selected: list[Match[str]],
    ) -> bool:
        for left, right in zip(
            selected,
            selected[1:],
        ):
            separator = sentence_text[
                left.end():right.start()
            ]

            if not re.fullmatch(r"[ \t]+", separator):
                return False

        return True

    def _looks_like_valid_control_expression(
        self,
        *,
        sentence_text: str,
        selected: list[Match[str]],
        original_tokens: tuple[str, ...],
        candidate_tokens: tuple[str, ...],
    ) -> bool:
        """
        Protect variable names in pseudocode conditions.

        Example that must remain unchanged:
        "while low less than or equal high"
        """

        if len(original_tokens) != 2:
            return False

        if original_tokens[0] not in {
            "while",
            "if",
        }:
            return False

        if candidate_tokens[0] != original_tokens[0]:
            return False

        if candidate_tokens[1] not in {
            "loop",
            "statement",
            "condition",
        }:
            return False

        remainder = sentence_text[
            selected[-1].end():
        ]

        return bool(
            self._COMPARISON_AFTER_IDENTIFIER.match(
                remainder
            )
        )

    def _technical_context_keywords(
        self,
        sentence: str,
    ) -> tuple[str, ...]:
        sentence_key = normalise_lookup_key(sentence)
        words = set(sentence_key.split())

        anchors = [
            word
            for word in sorted(words)
            if word in self._vocabulary.context_anchors
        ]

        if re.search(
            r"\b[A-Za-z_][A-Za-z0-9_]*\."
            r"[A-Za-z_][A-Za-z0-9_]*\b",
            sentence,
        ):
            anchors.append("member_access")

        if re.search(
            r"(==|!=|<=|>=|\[[^\]]*\]|\bif\b|\bwhile\b)",
            sentence,
            re.IGNORECASE,
        ):
            anchors.append("code_syntax")

        return tuple(dict.fromkeys(anchors))

    def _has_enough_technical_context(
        self,
        keywords: tuple[str, ...],
    ) -> bool:
        if not keywords:
            return False

        strong = [
            keyword
            for keyword in keywords
            if (
                keyword
                not in self._vocabulary.weak_context_anchors
                or keyword in {
                    "member_access",
                    "code_syntax",
                }
            )
        ]

        if strong:
            return True

        return len(set(keywords)) >= 2

    def _split_sentences(
        self,
        text: str,
    ) -> list[SentenceSpan]:
        sentences: list[SentenceSpan] = []

        for index, match in enumerate(
            self._SENTENCE_PATTERN.finditer(text)
        ):
            raw = match.group(0)

            if not raw.strip():
                continue

            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            start = match.start() + leading
            end = match.start() + trailing

            sentences.append(
                SentenceSpan(
                    sentence_index=index,
                    text=text[start:end],
                    start=start,
                    end=end,
                )
            )

        return sentences

    @staticmethod
    def _remove_overlapping_issues(
        issues: list[SuspiciousTechnicalSpan],
    ) -> list[SuspiciousTechnicalSpan]:
        # Prefer longer spans, then stronger detector scores.
        ordered = sorted(
            issues,
            key=lambda issue: (
                -(issue.end - issue.start),
                -issue.detector_score,
                issue.start,
            ),
        )

        selected: list[SuspiciousTechnicalSpan] = []

        for issue in ordered:
            overlaps = any(
                issue.start < current.end
                and current.start < issue.end
                for current in selected
            )

            if not overlaps:
                selected.append(issue)

        return sorted(
            selected,
            key=lambda issue: issue.start,
        )


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