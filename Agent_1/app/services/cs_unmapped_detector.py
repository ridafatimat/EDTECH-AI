from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from app.schemas.topic import TopicCandidate, UnmappedCSSignal
from app.services.embedding_service import (
    CHUNKING_EMBEDDING_MODEL,
    embed_texts,
)


EmbeddingFunction = Callable[[Sequence[str], str, int], np.ndarray]


@dataclass(frozen=True)
class CSDomain:
    name: str
    description: str


@dataclass(frozen=True)
class UnmappedConceptFamily:
    """
    A generic CS concept family which is useful for detecting syllabus gaps.

    These entries are not official AQA mappings. They only provide a rough
    label when technical content remains outside the official catalogue.
    """

    rough_topic: str
    domain: str
    description: str
    aliases: tuple[str, ...]
    minimum_distinct_aliases: int = 1
    minimum_total_hits: int = 1


# Broad descriptions used only for residual semantic detection.
CS_DOMAINS: tuple[CSDomain, ...] = (
    CSDomain(
        name="Programming and software development",
        description=(
            "Programming source code, variables, control flow, methods, "
            "classes, objects, constructors, attributes, access control, "
            "data structures and software behaviour."
        ),
    ),
    CSDomain(
        name="Algorithms and computational thinking",
        description=(
            "Algorithms, problem solving, tracing execution, searching, "
            "sorting, decomposition, abstraction and efficiency."
        ),
    ),
    CSDomain(
        name="Data representation",
        description=(
            "Binary, hexadecimal, bits, bytes, text encoding, images, "
            "sound, file size and compression."
        ),
    ),
    CSDomain(
        name="Computer systems",
        description=(
            "Computer hardware, software, CPU architecture, memory, "
            "storage, operating systems and translators."
        ),
    ),
    CSDomain(
        name="Networks and cyber security",
        description=(
            "Computer networks, protocols, network devices, security, "
            "attacks, authentication, encryption and protective controls."
        ),
    ),
    CSDomain(
        name="Databases and data management",
        description=(
            "Relational databases, tables, records, fields, keys, SQL and "
            "database queries."
        ),
    ),
)


# Generic, transcript-independent families commonly encountered in lessons
# but not represented as explicit AQA 8525 catalogue topics.
UNMAPPED_CONCEPT_FAMILIES: tuple[UnmappedConceptFamily, ...] = (
    UnmappedConceptFamily(
        rough_topic="Dynamic arrays and list collections",
        domain="Programming and software development",
        description=(
            "Resizable sequence data structures such as ArrayList or a "
            "dynamic array whose size can change while a program runs."
        ),
        aliases=(
            "array list",
            "array lists",
            "arraylist",
            "arraylists",
            "dynamic array",
            "resizable array",
            "resizable list",
        ),
    ),
    UnmappedConceptFamily(
        rough_topic="Object construction and initialisation",
        domain="Programming and software development",
        description=(
            "Constructors create or initialise objects and set their "
            "initial attributes or state."
        ),
        aliases=(
            "constructor",
            "constructors",
            "object constructor",
            "create an object",
            "initialise an object",
            "initialize an object",
            "object initialisation",
            "object initialization",
            "class instance",
        ),
    ),
    UnmappedConceptFamily(
        rough_topic="Encapsulation and access modifiers",
        domain="Programming and software development",
        description=(
            "Private and public members, class attributes and controlled "
            "access to an object's internal state."
        ),
        aliases=(
            "private attribute",
            "private attributes",
            "public attribute",
            "public attributes",
            "private and public",
            "access modifier",
            "access modifiers",
            "outside the class",
            "encapsulation",
            "controlled access",
        ),
    ),
    UnmappedConceptFamily(
        rough_topic="Inheritance and polymorphism",
        domain="Programming and software development",
        description=(
            "Object-oriented inheritance, subclasses, method overriding "
            "and polymorphic behaviour."
        ),
        aliases=(
            "inheritance",
            "subclass",
            "superclass",
            "base class",
            "derived class",
            "method overriding",
            "polymorphism",
        ),
    ),
    UnmappedConceptFamily(
        rough_topic="Higher-dimensional arrays",
        domain="Programming and software development",
        description=(
            "Arrays with three or more dimensions, including 3D, 4D and "
            "higher-dimensional indexing and visualisation."
        ),
        aliases=(
            "three dimensional array",
            "3d array",
            "four dimensional array",
            "4d array",
            "five dimensional array",
            "5d array",
            "higher dimensional array",
            "multidimensional array beyond two dimensions",
        ),
        minimum_distinct_aliases=1,
        minimum_total_hits=1,
    ),
    UnmappedConceptFamily(
        rough_topic="Time complexity",
        domain="Algorithms and computational thinking",
        description=(
            "How an algorithm's running time grows as the input size grows, "
            "including constant, linear, polynomial and exponential time."
        ),
        aliases=(
            "time complexity",
            "constant time",
            "linear time",
            "polynomial time",
            "exponential time",
            "running time grows",
        ),
        minimum_distinct_aliases=1,
        minimum_total_hits=1,
    ),
    UnmappedConceptFamily(
        rough_topic="Space complexity",
        domain="Algorithms and computational thinking",
        description=(
            "How much memory or other storage an algorithm requires as its "
            "input size changes."
        ),
        aliases=(
            "space complexity",
            "memory complexity",
            "memory required",
            "amount of memory",
            "space efficient",
        ),
        minimum_distinct_aliases=1,
        minimum_total_hits=1,
    ),
    UnmappedConceptFamily(
        rough_topic="Big O notation",
        domain="Algorithms and computational thinking",
        description=(
            "Big O notation classifies how computational cost grows with "
            "input size."
        ),
        aliases=(
            "big o notation",
            "big o",
            "o of n",
            "o n",
            "complexity notation",
        ),
        minimum_distinct_aliases=1,
        minimum_total_hits=1,
    ),
    UnmappedConceptFamily(
        rough_topic="Tractable and intractable problems",
        domain="Algorithms and computational thinking",
        description=(
            "Problems classified by whether algorithms can solve them in a "
            "practical amount of time, including polynomial and worse-than-"
            "polynomial growth."
        ),
        aliases=(
            "tractable problem",
            "tractable problems",
            "intractable problem",
            "intractable problems",
            "reasonable amount of time",
            "polynomial time or better",
        ),
        minimum_distinct_aliases=1,
        minimum_total_hits=1,
    ),
)


@dataclass(frozen=True)
class UnmappedDetectionConfig:
    sentence_similarity_threshold: float = 0.48
    strong_sentence_threshold: float = 0.60
    minimum_evidence_sentences: int = 2
    max_signals: int = 4
    embedding_model: str = CHUNKING_EMBEDDING_MODEL


class CSUnmappedDetector:
    """
    Detect CS content which is not explained by retained official topics.

    Detection has two generic routes:
    1. lexical families for recognisable off-syllabus concepts;
    2. broad semantic residual detection for unknown CS material.

    The detector never fabricates an official AQA reference.
    """

    def __init__(
        self,
        config: UnmappedDetectionConfig | None = None,
        embedding_function: EmbeddingFunction = embed_texts,
    ) -> None:
        self.config = config or UnmappedDetectionConfig()
        self._embedding_function = embedding_function
        self._domain_embeddings: np.ndarray | None = None
        self._family_embeddings: np.ndarray | None = None

    def detect(
        self,
        text: str,
        official_candidates: list[TopicCandidate],
    ) -> list[UnmappedCSSignal]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        lexical_signals = self._detect_known_families(
            sentences=sentences,
            official_candidates=official_candidates,
        )

        semantic_signals = self._detect_semantic_residual(
            sentences=sentences,
            official_candidates=official_candidates,
        )

        # A specific lexical family already gives a clearer rough topic than
        # the generic semantic residual for the same sentence. Keeping both
        # would create duplicate evidence and could trigger an unnecessary
        # LLM fallback.
        specific_evidence = {
            self._normalize(signal.evidence)
            for signal in lexical_signals
        }
        semantic_signals = [
            signal
            for signal in semantic_signals
            if self._normalize(signal.evidence) not in specific_evidence
        ]

        combined = lexical_signals + semantic_signals
        combined.sort(key=lambda signal: signal.score, reverse=True)

        return self._deduplicate(combined)[: self.config.max_signals]

    def _detect_known_families(
        self,
        sentences: list[str],
        official_candidates: list[TopicCandidate],
    ) -> list[UnmappedCSSignal]:
        official_terms = self._official_terms(official_candidates)
        family_embeddings = self._get_family_embeddings()
        signals: list[UnmappedCSSignal] = []

        for family_index, family in enumerate(UNMAPPED_CONCEPT_FAMILIES):
            matched_aliases: list[str] = []
            evidence_sentences: list[str] = []
            total_alias_hits = 0

            for sentence in sentences:
                normalized_sentence = self._normalize(sentence)
                sentence_aliases = [
                    alias
                    for alias in family.aliases
                    if self._contains_phrase(
                        normalized_sentence,
                        self._normalize(alias),
                    )
                ]

                if not sentence_aliases:
                    continue

                matched_aliases.extend(sentence_aliases)
                total_alias_hits += len(sentence_aliases)
                evidence_sentences.append(sentence)

            matched_aliases = self._unique_strings(matched_aliases)
            evidence_sentences = self._unique_strings(evidence_sentences)

            if (
                len(matched_aliases) < family.minimum_distinct_aliases
                or total_alias_hits < family.minimum_total_hits
            ):
                continue

            # Do not duplicate an official candidate that already uses the
            # same specific terminology. Partial overlap such as array versus
            # ArrayList is deliberately not treated as coverage.
            if self._family_is_officially_covered(
                family=family,
                matched_aliases=matched_aliases,
                official_terms=official_terms,
            ):
                continue

            evidence = evidence_sentences[0]
            evidence_embedding = self._embedding_function(
                [evidence],
                self.config.embedding_model,
                32,
            )
            semantic_score = float(
                evidence_embedding[0] @ family_embeddings[family_index]
            )

            longest_alias_words = max(
                len(self._normalize(alias).split())
                for alias in matched_aliases
            )

            lexical_base = 0.62 if longest_alias_words >= 2 else 0.56
            diversity_bonus = 0.05 * min(3, len(matched_aliases) - 1)
            evidence_bonus = 0.04 * min(2, len(evidence_sentences) - 1)

            score = min(
                0.95,
                lexical_base
                + diversity_bonus
                + evidence_bonus
                + 0.18 * max(0.0, semantic_score),
            )

            method = (
                "lexical_semantic"
                if semantic_score >= self.config.sentence_similarity_threshold
                else "lexical"
            )

            signals.append(
                UnmappedCSSignal(
                    rough_topic=family.rough_topic,
                    domain=family.domain,
                    score=round(score, 4),
                    evidence=evidence.strip(),
                    matched_aliases=matched_aliases,
                    detection_method=method,
                )
            )

        return signals

    def _detect_semantic_residual(
        self,
        sentences: list[str],
        official_candidates: list[TopicCandidate],
    ) -> list[UnmappedCSSignal]:
        covered = {
            self._normalize(sentence)
            for candidate in official_candidates
            for sentence in candidate.evidence
            if self._normalize(sentence)
        }

        uncovered_sentences = [
            sentence
            for sentence in sentences
            if self._normalize(sentence) not in covered
        ]

        if not uncovered_sentences:
            return []

        sentence_embeddings = self._embedding_function(
            uncovered_sentences,
            self.config.embedding_model,
            32,
        )
        domain_embeddings = self._get_domain_embeddings()

        similarities = sentence_embeddings @ domain_embeddings.T
        candidates: list[UnmappedCSSignal] = []

        for sentence_index, sentence in enumerate(uncovered_sentences):
            row = similarities[sentence_index]
            domain_index = int(np.argmax(row))
            score = float(row[domain_index])

            if score < self.config.sentence_similarity_threshold:
                continue

            candidates.append(
                UnmappedCSSignal(
                    rough_topic="Unmapped Computer Science content",
                    domain=CS_DOMAINS[domain_index].name,
                    score=round(score, 4),
                    evidence=sentence.strip(),
                    matched_aliases=[],
                    detection_method="semantic",
                )
            )

        candidates.sort(key=lambda signal: signal.score, reverse=True)
        candidates = self._deduplicate(candidates)

        strong_exists = any(
            signal.score >= self.config.strong_sentence_threshold
            for signal in candidates
        )

        if (
            len(candidates) < self.config.minimum_evidence_sentences
            and not strong_exists
        ):
            return []

        return candidates

    def _get_domain_embeddings(self) -> np.ndarray:
        if self._domain_embeddings is None:
            self._domain_embeddings = self._embedding_function(
                [domain.description for domain in CS_DOMAINS],
                self.config.embedding_model,
                32,
            )
        return self._domain_embeddings

    def _get_family_embeddings(self) -> np.ndarray:
        if self._family_embeddings is None:
            self._family_embeddings = self._embedding_function(
                [family.description for family in UNMAPPED_CONCEPT_FAMILIES],
                self.config.embedding_model,
                32,
            )
        return self._family_embeddings

    @classmethod
    def _official_terms(
        cls,
        official_candidates: list[TopicCandidate],
    ) -> set[str]:
        terms: set[str] = set()

        for candidate in official_candidates:
            terms.add(cls._normalize(candidate.topic))
            terms.add(cls._normalize(candidate.official_title))
            terms.update(
                cls._normalize(alias)
                for alias in candidate.matched_aliases
            )

        return {term for term in terms if term}

    @classmethod
    def _family_is_officially_covered(
        cls,
        family: UnmappedConceptFamily,
        matched_aliases: list[str],
        official_terms: set[str],
    ) -> bool:
        family_terms = {
            cls._normalize(family.rough_topic),
            *(
                cls._normalize(alias)
                for alias in matched_aliases
            ),
        }

        return any(
            family_term == official_term
            for family_term in family_terms
            for official_term in official_terms
            if family_term and official_term
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        if not phrase:
            return False

        pattern = re.compile(
            r"(?<!\w)"
            + re.escape(phrase).replace(r"\ ", r"\s+")
            + r"(?!\w)",
            re.IGNORECASE,
        )
        return bool(pattern.search(text))

    @classmethod
    def _deduplicate(
        cls,
        signals: list[UnmappedCSSignal],
    ) -> list[UnmappedCSSignal]:
        output: list[UnmappedCSSignal] = []
        seen: set[tuple[str, str]] = set()

        for signal in signals:
            key = (
                cls._normalize(signal.rough_topic),
                cls._normalize(signal.evidence),
            )
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            output.append(signal)

        return output

    @staticmethod
    def _unique_strings(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = " ".join(value.lower().split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(value.strip())

        return unique