from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from app.schemas.topic import RawTopicCandidate
from app.services.cs_concept_catalog import CS_CONCEPTS, CSConcept
from app.services.embedding_service import (
    CHUNKING_EMBEDDING_MODEL,
    embed_texts,
)


EmbeddingFunction = Callable[[Sequence[str], str, int], np.ndarray]


@dataclass(frozen=True)
class TopicExtractionConfig:
    """
    Configuration for official AQA topic candidate extraction.

    The salience rules are generic. They do not contain transcript-specific
    words. Their purpose is to stop an isolated ordinary word such as
    "integer", "function" or "bit" from becoming a lesson topic unless the
    surrounding semantic or repeated evidence supports it.
    """

    semantic_unit_words: int = 60

    raw_candidate_floor: float = 0.30
    semantic_only_threshold: float = 0.50

    # A topic supported only by one-word aliases requires stronger context.
    single_word_semantic_floor: float = 0.45
    single_word_min_evidence_sentences: int = 2
    single_word_min_distinct_aliases: int = 2

    max_raw_candidates: int = 12
    max_final_candidates: int = 6
    max_evidence_per_candidate: int = 3

    suppress_redundant_parents: bool = True
    parent_suppression_margin: float = 0.08

    # Suppress two labels that are driven by effectively the same sentence
    # and the same technical phrase.
    duplicate_evidence_overlap: float = 0.80
    duplicate_alias_token_overlap: float = 0.50

    embedding_model: str = CHUNKING_EMBEDDING_MODEL

    def __post_init__(self) -> None:
        if self.semantic_unit_words <= 0:
            raise ValueError("semantic_unit_words must be positive.")

        for value_name in (
            "raw_candidate_floor",
            "semantic_only_threshold",
            "single_word_semantic_floor",
            "parent_suppression_margin",
            "duplicate_evidence_overlap",
            "duplicate_alias_token_overlap",
        ):
            value = getattr(self, value_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{value_name} must be between 0 and 1.")

        if self.max_raw_candidates < 1 or self.max_final_candidates < 1:
            raise ValueError("Candidate limits must be at least 1.")


@dataclass(frozen=True)
class KeywordEvidence:
    score: float
    matched_aliases: list[str]
    evidence_sentences: list[str]
    total_hits: int
    excluded_hits: int
    single_word_alias_only: bool


class TopicCandidateExtractor:
    """
    Extract official AQA topic candidates from one transcript chunk.

    Evidence sources:
    - exact transcript-friendly catalogue aliases
    - MiniLM semantic similarity
    - evidence repetition/diversity (topic salience)
    """

    def __init__(
        self,
        config: TopicExtractionConfig | None = None,
        embedding_function: EmbeddingFunction = embed_texts,
    ) -> None:
        self.config = config or TopicExtractionConfig()
        self._embedding_function = embedding_function
        self._concept_embeddings: np.ndarray | None = None

    def extract(
        self,
        chunk_id: int,
        text: str,
    ) -> list[RawTopicCandidate]:
        if chunk_id < 1:
            raise ValueError("chunk_id must be at least 1.")

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        semantic_units = self._build_semantic_units(sentences) or [text]

        unit_embeddings = self._embedding_function(
            semantic_units,
            self.config.embedding_model,
            32,
        )
        concept_embeddings = self._get_concept_embeddings()

        if unit_embeddings.size == 0 or concept_embeddings.size == 0:
            return []

        similarity_matrix = unit_embeddings @ concept_embeddings.T
        raw_candidates: list[RawTopicCandidate] = []

        for concept_index, concept in enumerate(CS_CONCEPTS):
            similarities = similarity_matrix[:, concept_index]
            best_unit_index = int(np.argmax(similarities))
            semantic_score = float(similarities[best_unit_index])

            keyword = self._keyword_evidence(
                concept=concept,
                sentences=sentences,
                full_text=text,
            )

            has_keyword_support = keyword.score > 0.0
            # A known longer compound term can contain a shorter alias.
            # When every lexical hit was blocked by catalogue exclusions,
            # semantic similarity alone must not remap that compound back to
            # the shorter official concept.
            has_semantic_only_support = (
                semantic_score >= self.config.semantic_only_threshold
                and keyword.excluded_hits == 0
            )

            if not (has_keyword_support or has_semantic_only_support):
                continue

            if has_keyword_support and not self._passes_salience_gate(
                keyword=keyword,
                semantic_score=semantic_score,
            ):
                continue

            salience_score = self._calculate_salience_score(
                keyword=keyword,
                semantic_score=semantic_score,
            )

            confidence = self._calculate_confidence(
                keyword_score=keyword.score,
                semantic_score=semantic_score,
                single_word_alias_only=keyword.single_word_alias_only,
                salience_score=salience_score,
            )

            if confidence < self.config.raw_candidate_floor:
                continue

            semantic_evidence = semantic_units[best_unit_index].strip()
            evidence = (
                keyword.evidence_sentences
                if keyword.evidence_sentences
                else [semantic_evidence]
            )
            evidence = self._unique_strings(evidence)[
                : self.config.max_evidence_per_candidate
            ]

            if keyword.score > 0.0 and semantic_score >= 0.20:
                method = "keyword_embedding"
            elif keyword.score > 0.0:
                method = "keyword"
            else:
                method = "embedding"

            raw_candidates.append(
                RawTopicCandidate(
                    concept_id=concept.concept_id,
                    topic=concept.label,
                    domain=concept.domain,
                    official_reference=concept.official_reference,
                    chapter_reference=concept.chapter_reference,
                    official_title=concept.official_title,
                    paper=concept.paper,
                    source_pages=list(concept.source_pages),
                    confidence=round(confidence, 4),
                    keyword_score=round(keyword.score, 4),
                    semantic_score=round(semantic_score, 4),
                    salience_score=round(salience_score, 4),
                    extraction_method=method,
                    matched_aliases=keyword.matched_aliases,
                    total_alias_hits=keyword.total_hits,
                    evidence_sentence_count=len(keyword.evidence_sentences),
                    single_word_alias_only=keyword.single_word_alias_only,
                    evidence=evidence,
                    parent_concept_id=concept.parent_concept_id,
                )
            )

        raw_candidates.sort(
            key=lambda candidate: (
                candidate.confidence,
                candidate.salience_score,
                candidate.semantic_score,
                candidate.keyword_score,
            ),
            reverse=True,
        )

        raw_candidates = raw_candidates[: self.config.max_raw_candidates]

        if self.config.suppress_redundant_parents:
            raw_candidates = self._suppress_redundant_candidates(
                raw_candidates
            )

        return raw_candidates[: self.config.max_final_candidates]

    def _get_concept_embeddings(self) -> np.ndarray:
        if self._concept_embeddings is not None:
            return self._concept_embeddings

        concept_texts = [concept.embedding_text for concept in CS_CONCEPTS]
        self._concept_embeddings = self._embedding_function(
            concept_texts,
            self.config.embedding_model,
            32,
        )
        return self._concept_embeddings

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def _build_semantic_units(self, sentences: list[str]) -> list[str]:
        units: list[str] = []
        buffer: list[str] = []
        buffer_words = 0

        for sentence in sentences:
            buffer.append(sentence)
            buffer_words += self._word_count(sentence)

            if buffer_words >= self.config.semantic_unit_words:
                units.append(" ".join(buffer))
                buffer = []
                buffer_words = 0

        if buffer:
            units.append(" ".join(buffer))

        return units

    def _keyword_evidence(
        self,
        concept: CSConcept,
        sentences: list[str],
        full_text: str,
    ) -> KeywordEvidence:
        matched_aliases: list[str] = []
        evidence: list[str] = []
        alias_weights: list[float] = []
        total_hits = 0
        excluded_hits = 0
        matched_word_counts: list[int] = []

        for alias in concept.aliases:
            normalized_alias = self._normalize_for_match(alias)
            if not normalized_alias:
                continue

            pattern = re.compile(
                r"(?<!\w)"
                + re.escape(normalized_alias).replace(r"\ ", r"\s+")
                + r"(?!\w)",
                re.IGNORECASE,
            )

            valid_alias_hits = 0

            for sentence in sentences:
                normalized_sentence = self._normalize_for_match(sentence)
                blocked_ranges = self._excluded_ranges(
                    text=normalized_sentence,
                    excluded_phrases=concept.excluded_phrases,
                )

                sentence_valid_hits = 0

                for match in pattern.finditer(normalized_sentence):
                    if self._overlaps_any(match.span(), blocked_ranges):
                        excluded_hits += 1
                        continue

                    sentence_valid_hits += 1

                if sentence_valid_hits:
                    valid_alias_hits += sentence_valid_hits
                    evidence.append(sentence.strip())

            if not valid_alias_hits:
                continue

            total_hits += valid_alias_hits
            matched_aliases.append(alias)

            alias_word_count = len(normalized_alias.split())
            matched_word_counts.append(alias_word_count)

            if alias_word_count == 1:
                weight = 0.50
            elif alias_word_count == 2:
                weight = 0.72
            else:
                weight = 0.82

            alias_weights.append(weight)

        # Flexible patterns support natural classroom phrasing such as
        # passive voice, inserted variable names and ASR variation. The
        # mechanism is catalogue-driven and can be reused by any concept.
        for flexible_pattern in concept.match_patterns:
            pattern = re.compile(
                flexible_pattern.regex,
                re.IGNORECASE,
            )

            pattern_hits = 0

            for sentence in sentences:
                normalized_sentence = self._normalize_for_match(sentence)
                blocked_ranges = self._excluded_ranges(
                    text=normalized_sentence,
                    excluded_phrases=concept.excluded_phrases,
                )

                sentence_hits = 0

                for match in pattern.finditer(normalized_sentence):
                    if self._overlaps_any(match.span(), blocked_ranges):
                        excluded_hits += 1
                        continue

                    sentence_hits += 1

                if sentence_hits:
                    pattern_hits += sentence_hits
                    evidence.append(sentence.strip())

            if not pattern_hits:
                continue

            total_hits += pattern_hits
            matched_aliases.append(flexible_pattern.label)
            matched_word_counts.append(
                max(2, len(flexible_pattern.label.split()))
            )
            alias_weights.append(flexible_pattern.weight)

        if not alias_weights:
            return KeywordEvidence(
                score=0.0,
                matched_aliases=[],
                evidence_sentences=[],
                total_hits=0,
                excluded_hits=excluded_hits,
                single_word_alias_only=False,
            )

        evidence = self._unique_strings(evidence)
        matched_aliases = self._unique_strings(matched_aliases)

        strongest_alias = max(alias_weights)
        distinct_bonus = 0.08 * max(0, len(matched_aliases) - 1)
        evidence_bonus = 0.05 * min(3, max(0, len(evidence) - 1))
        repetition_bonus = 0.02 * min(4, max(0, total_hits - 1))

        keyword_score = min(
            1.0,
            strongest_alias
            + distinct_bonus
            + evidence_bonus
            + repetition_bonus,
        )

        return KeywordEvidence(
            score=keyword_score,
            matched_aliases=matched_aliases,
            evidence_sentences=evidence,
            total_hits=total_hits,
            excluded_hits=excluded_hits,
            single_word_alias_only=all(
                word_count == 1 for word_count in matched_word_counts
            ),
        )

    @classmethod
    def _excluded_ranges(
        cls,
        text: str,
        excluded_phrases: tuple[str, ...],
    ) -> list[tuple[int, int]]:
        """
        Return character ranges occupied by longer confusable phrases.

        The rule is catalogue-driven rather than transcript-specific. Any
        concept can declare compound phrases which must not be consumed by a
        shorter alias.
        """

        ranges: list[tuple[int, int]] = []

        for phrase in excluded_phrases:
            normalized_phrase = cls._normalize_for_match(phrase)
            if not normalized_phrase:
                continue

            pattern = re.compile(
                r"(?<!\w)"
                + re.escape(normalized_phrase).replace(r"\ ", r"\s+")
                + r"(?!\w)",
                re.IGNORECASE,
            )

            ranges.extend(
                match.span()
                for match in pattern.finditer(text)
            )

        return ranges

    @staticmethod
    def _overlaps_any(
        span: tuple[int, int],
        blocked_ranges: list[tuple[int, int]],
    ) -> bool:
        start, end = span

        return any(
            start < blocked_end and end > blocked_start
            for blocked_start, blocked_end in blocked_ranges
        )

    def _passes_salience_gate(
        self,
        keyword: KeywordEvidence,
        semantic_score: float,
    ) -> bool:
        if not keyword.single_word_alias_only:
            return True

        return any(
            (
                semantic_score >= self.config.single_word_semantic_floor,
                len(keyword.evidence_sentences)
                >= self.config.single_word_min_evidence_sentences,
                len(keyword.matched_aliases)
                >= self.config.single_word_min_distinct_aliases,
            )
        )

    @staticmethod
    def _calculate_salience_score(
        keyword: KeywordEvidence,
        semantic_score: float,
    ) -> float:
        semantic_component = max(0.0, min(1.0, semantic_score))
        evidence_component = min(1.0, len(keyword.evidence_sentences) / 3.0)
        alias_component = min(1.0, len(keyword.matched_aliases) / 2.0)

        return min(
            1.0,
            0.55 * semantic_component
            + 0.25 * evidence_component
            + 0.20 * alias_component,
        )

    @staticmethod
    def _calculate_confidence(
        keyword_score: float,
        semantic_score: float,
        single_word_alias_only: bool,
        salience_score: float,
    ) -> float:
        semantic_component = max(0.0, min(1.0, semantic_score))

        combined = 0.58 * semantic_component + 0.42 * keyword_score

        # Multiword technical phrases are stronger direct evidence than an
        # isolated one-word match. Single-word matches therefore cannot win
        # through the keyword path alone.
        keyword_multiplier = 0.72 if single_word_alias_only else 0.88
        keyword_supported = keyword_multiplier * keyword_score
        semantic_only = 0.82 * semantic_component

        base_confidence = max(combined, keyword_supported, semantic_only)

        # Salience has a bounded effect: it can reduce an incidental mention,
        # but it cannot manufacture confidence without real evidence.
        adjusted = base_confidence * (0.85 + 0.15 * salience_score)

        return max(0.0, min(1.0, adjusted))

    def _suppress_redundant_candidates(
        self,
        candidates: list[RawTopicCandidate],
    ) -> list[RawTopicCandidate]:
        kept: list[RawTopicCandidate] = []

        for candidate in candidates:
            should_skip = False

            for existing in kept:
                if self._is_redundant_pair(candidate, existing):
                    should_skip = True
                    break

            if not should_skip:
                kept.append(candidate)

        return kept

    def _is_redundant_pair(
        self,
        candidate: RawTopicCandidate,
        existing: RawTopicCandidate,
    ) -> bool:
        # Parent/child suppression.
        if candidate.parent_concept_id == existing.concept_id:
            return candidate.confidence <= (
                existing.confidence + self.config.parent_suppression_margin
            )

        if existing.parent_concept_id == candidate.concept_id:
            return candidate.confidence <= existing.confidence

        evidence_overlap = self._jaccard(
            self._normalized_evidence(candidate.evidence),
            self._normalized_evidence(existing.evidence),
        )

        alias_overlap = self._jaccard(
            self._alias_tokens(candidate.matched_aliases),
            self._alias_tokens(existing.matched_aliases),
        )

        same_reference = (
            candidate.official_reference == existing.official_reference
        )

        return (
            evidence_overlap >= self.config.duplicate_evidence_overlap
            and alias_overlap >= self.config.duplicate_alias_token_overlap
            and (same_reference or self._topic_token_overlap(candidate, existing) >= 0.50)
        )

    def _topic_token_overlap(
        self,
        first: RawTopicCandidate,
        second: RawTopicCandidate,
    ) -> float:
        return self._jaccard(
            self._stemmed_tokens(first.topic),
            self._stemmed_tokens(second.topic),
        )

    @classmethod
    def _alias_tokens(cls, aliases: list[str]) -> set[str]:
        tokens: set[str] = set()
        for alias in aliases:
            tokens.update(cls._stemmed_tokens(alias))
        return tokens

    @classmethod
    def _normalized_evidence(cls, evidence: list[str]) -> set[str]:
        return {
            cls._normalize_for_match(value)
            for value in evidence
            if cls._normalize_for_match(value)
        }

    @classmethod
    def _stemmed_tokens(cls, text: str) -> set[str]:
        tokens = cls._normalize_for_match(text).split()
        return {cls._light_stem(token) for token in tokens if len(token) > 2}

    @staticmethod
    def _light_stem(token: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 3:
                return token[: -len(suffix)]
        return token

    @staticmethod
    def _jaccard(first: set[str], second: set[str]) -> float:
        if not first or not second:
            return 0.0
        return len(first & second) / len(first | second)

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\S+", text))

    @staticmethod
    def _unique_strings(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = re.sub(r"\s+", " ", value).strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(value.strip())

        return unique