from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from app.schemas.topic import TopicCandidate


@dataclass(frozen=True)
class EvidenceQualityConfig:
    """
    Generic evidence-quality rules used after confidence filtering.

    The evaluator:

    1. separates a brief mention from actual teaching;
    2. enforces catalogue-driven ambiguous-alias context;
    3. prevents generic multi-topic recaps from becoming strong evidence;
    4. reduces candidates whose evidence is owned more strongly by another
       candidate in the same chunk.
    """

    mention_only_max_score: float = 0.34
    definition_score: float = 0.50
    explanation_score: float = 0.68
    worked_example_score: float = 0.84
    sustained_teaching_score: float = 0.95

    isolated_mention_reject: bool = True
    ambiguous_alias_requires_context: bool = True

    # A sentence must be shared by at least this many official candidates
    # before it can be treated as a generic multi-topic recap.
    recap_minimum_shared_topics: int = 3
    recap_quality_penalty: float = 0.18

    # A brief comparison-only reference must not become a lesson topic.
    # More substantial comparisons are retained because they may genuinely
    # teach both concepts (for example, a dedicated merge-vs-bubble section).
    comparison_only_max_alias_hits: int = 2
    comparison_quality_penalty: float = 0.16
    comparison_window_words: int = 16

    evidence_overlap_threshold: float = 0.50
    competition_margin: float = 0.12
    maximum_shared_evidence_penalty: float = 0.24

    minimum_adjusted_score: float = 0.46

    def __post_init__(self) -> None:
        probability_fields = (
            "mention_only_max_score",
            "definition_score",
            "explanation_score",
            "worked_example_score",
            "sustained_teaching_score",
            "recap_quality_penalty",
            "comparison_quality_penalty",
            "evidence_overlap_threshold",
            "competition_margin",
            "maximum_shared_evidence_penalty",
            "minimum_adjusted_score",
        )

        for field_name in probability_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )

        if self.recap_minimum_shared_topics < 2:
            raise ValueError(
                "recap_minimum_shared_topics must be at least 2."
            )

        if self.comparison_only_max_alias_hits < 1:
            raise ValueError(
                "comparison_only_max_alias_hits must be at least 1."
            )

        if self.comparison_window_words < 4:
            raise ValueError(
                "comparison_window_words must be at least 4."
            )


@dataclass(frozen=True)
class EvidenceQualityResult:
    retained: list[TopicCandidate]
    rejected: list[TopicCandidate]


@dataclass(frozen=True)
class ComparisonEvidenceProfile:
    total_hits: int
    comparison_hits: int
    independent_hits: int
    comparison_only: bool


class EvidenceQualityEvaluator:
    """
    Evaluate retained official candidates without changing topic discovery.

    Candidate extraction answers:
        "Which official topics may be present?"

    This evaluator answers:
        "Is the evidence deep and specific enough to keep the topic?"
    """

    _MENTION_PATTERNS = (
        r"\b(?:also|briefly|just)\s+(?:mention|mentioned|talked about)\b",
        r"\bby the way\b",
        r"\bin (?:the|our) syllabus\b",
        r"\bwe (?:also )?have\b",
        r"\bwe will (?:cover|do|study|look at)\b",
        r"\bnext (?:topic|chapter|lesson)\b",
        r"\bmove on to\b",
        r"\bremember (?:that )?we (?:did|covered|talked about)\b",
        r"\bwe are not (?:covering|doing|studying)\b",
    )

    _RECAP_PATTERNS = (
        r"\b(?:today|this lesson|last lesson|previously)\b.{0,80}"
        r"\b(?:covered|revised|reviewed|looked at|studied|discussed)\b",
        r"\bwe (?:have )?(?:covered|revised|reviewed|looked at|studied|discussed)\b",
        r"\btopics? (?:were|included|covered)\b",
        r"\bto recap\b",
        r"\bin summary\b",
        r"\bquick recap\b",
        r"\bwe learned about\b",
    )

    _DEFINITION_PATTERNS = (
        r"\b(?:is|are) (?:a|an|the)\b",
        r"\bmeans\b",
        r"\brefers to\b",
        r"\bdefined as\b",
        r"\bwe call\b",
    )

    _EXPLANATION_PATTERNS = (
        r"\bbecause\b",
        r"\btherefore\b",
        r"\bso that\b",
        r"\bworks by\b",
        r"\bthe reason\b",
        r"\bthis means\b",
        r"\bif\b.{0,100}\bthen\b",
        r"\bwhy\b",
        r"\bhow\b",
        r"\bcauses?\b",
        r"\ballows?\b",
        r"\bprevents?\b",
    )

    _WORKED_EXAMPLE_PATTERNS = (
        r"\bfor example\b",
        r"\bfor instance\b",
        r"\blet'?s (?:say|take|try|trace|calculate|work)\b",
        r"\bsuppose\b",
        r"\bgiven\b",
        r"\bstep by step\b",
        r"\btrace (?:this|the|through)\b",
        r"\bcalculate\b",
        r"\bwork(?:ed)? example\b",
    )

    _PRACTICE_PATTERNS = (
        r"\bwhat (?:is|happens|would|will|do you)\b",
        r"\bwhy (?:is|does|would|do you)\b",
        r"\bhow (?:many|does|would|do you)\b",
        r"\btry (?:this|it|the next|another)\b",
        r"\byour answer\b",
        r"\bcorrect answer\b",
        r"\bquestion\b",
        r"\bpractice\b",
        r"\bexercise\b",
        r"\bexam\b",
    )


    _COMPARISON_PATTERNS = (
        r"\bunlike\b",
        r"\bcompared? (?:with|to)\b",
        r"\bin comparison (?:with|to)\b",
        r"\bin contrast (?:with|to)\b",
        r"\bas opposed to\b",
        r"\bwhereas\b",
        r"\bon the other hand\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\banother type of\b",
        r"\bdifferent from\b",
    )

    def __init__(
        self,
        config: EvidenceQualityConfig | None = None,
    ) -> None:
        self.config = config or EvidenceQualityConfig()

    def evaluate(
        self,
        *,
        text: str,
        candidates: list[TopicCandidate],
    ) -> EvidenceQualityResult:
        if not candidates:
            return EvidenceQualityResult(
                retained=[],
                rejected=[],
            )

        shared_recap_evidence = self._shared_recap_evidence(
            candidates
        )

        assessed = [
            self._assess_candidate(
                text=text,
                candidate=candidate,
                shared_recap_evidence=shared_recap_evidence,
            )
            for candidate in candidates
        ]

        retained: list[TopicCandidate] = []
        rejected: list[TopicCandidate] = []

        for candidate in assessed:
            rejection_reason = self._initial_rejection_reason(
                candidate
            )

            if rejection_reason is not None:
                rejected.append(
                    self._reject(
                        candidate,
                        rejection_reason,
                    )
                )
            else:
                retained.append(candidate)

        retained, competition_rejected = (
            self._apply_candidate_competition(
                retained
            )
        )
        rejected.extend(competition_rejected)

        retained.sort(
            key=lambda candidate: (
                candidate.cs_relevance_score,
                candidate.evidence_quality_score,
                candidate.teaching_depth_level,
                candidate.semantic_score,
            ),
            reverse=True,
        )

        rejected.sort(
            key=lambda candidate: (
                candidate.cs_relevance_score,
                candidate.evidence_quality_score,
            ),
            reverse=True,
        )

        return EvidenceQualityResult(
            retained=retained,
            rejected=rejected,
        )

    def _assess_candidate(
        self,
        *,
        text: str,
        candidate: TopicCandidate,
        shared_recap_evidence: set[str],
    ) -> TopicCandidate:
        evidence = candidate.evidence or [text]

        comparison_profile = self._comparison_profile(
            text=text,
            candidate=candidate,
        )

        recap_evidence = [
            sentence
            for sentence in evidence
            if self._normalize(sentence) in shared_recap_evidence
        ]

        substantive_evidence = [
            sentence
            for sentence in evidence
            if self._normalize(sentence) not in shared_recap_evidence
        ]

        recap_only = (
            bool(recap_evidence)
            and not substantive_evidence
        )

        evidence_for_depth = (
            substantive_evidence
            if substantive_evidence
            else recap_evidence
        )

        depth_levels = [
            (
                0
                if self._normalize(sentence) in shared_recap_evidence
                else self._sentence_depth(sentence)
            )
            for sentence in evidence_for_depth
        ]

        depth_level = max(depth_levels, default=0)

        non_mention_count = sum(
            1
            for level in depth_levels
            if level >= 1
        )
        explanatory_count = sum(
            1
            for level in depth_levels
            if level >= 2
        )

        # Only substantive evidence can upgrade a topic to sustained teaching.
        if (
            len(substantive_evidence) >= 3
            and explanatory_count >= 2
        ):
            depth_level = 4
        elif (
            len(substantive_evidence) >= 2
            and non_mention_count >= 2
            and depth_level >= 2
        ):
            depth_level = 4

        depth_label = self._depth_label(
            depth_level
        )
        depth_score = self._depth_score(
            depth_level
        )

        required_context_met = (
            len(candidate.matched_context_terms)
            >= candidate.minimum_context_hits
        )
        context_collision = bool(
            candidate.matched_conflicting_context_terms
            and candidate.ambiguous_alias_only
            and not required_context_met
        )

        context_component = (
            1.0
            if required_context_met
            else (
                0.20
                if context_collision
                else (0.35 if candidate.ambiguous_alias_only else 0.70)
            )
        )

        semantic_component = max(
            0.0,
            min(
                1.0,
                candidate.semantic_score,
            ),
        )

        quality_score = (
            0.45 * depth_score
            + 0.25 * candidate.salience_score
            + 0.20 * semantic_component
            + 0.10 * context_component
        )

        if depth_level == 0:
            quality_score = min(
                quality_score,
                self.config.mention_only_max_score,
            )

        if recap_evidence:
            recap_ratio = (
                len(recap_evidence)
                / max(1, len(evidence))
            )
            quality_score = max(
                0.0,
                quality_score
                - (
                    self.config.recap_quality_penalty
                    * recap_ratio
                ),
            )

        if comparison_profile.comparison_only:
            quality_score = max(
                0.0,
                quality_score
                - self.config.comparison_quality_penalty,
            )

        adjusted_relevance = (
            0.72 * candidate.cs_relevance_score
            + 0.28 * quality_score
        )

        notes = list(
            candidate.evidence_quality_notes
        )
        notes.append(
            f"Teaching depth: {depth_label} ({depth_level})."
        )

        if candidate.matched_context_terms:
            notes.append(
                "Supporting context: "
                + ", ".join(candidate.matched_context_terms)
                + "."
            )

        if candidate.matched_conflicting_context_terms:
            notes.append(
                "Conflicting context: "
                + ", ".join(
                    candidate.matched_conflicting_context_terms
                )
                + "."
            )

        if recap_evidence:
            notes.append(
                f"Generic recap evidence detected: "
                f"{len(recap_evidence)} sentence(s)."
            )

        if comparison_profile.comparison_hits:
            notes.append(
                "Comparison evidence detected: "
                f"{comparison_profile.comparison_hits} hit(s); "
                f"independent evidence: "
                f"{comparison_profile.independent_hits} hit(s)."
            )

        if comparison_profile.comparison_only:
            notes.append(
                "Candidate is supported only by a brief comparison or "
                "contrast reference."
            )

        return candidate.model_copy(
            update={
                "teaching_depth_level": depth_level,
                "teaching_depth_label": depth_label,
                "evidence_quality_score": round(
                    quality_score,
                    4,
                ),
                "cs_relevance_score": round(
                    adjusted_relevance,
                    4,
                ),
                "recap_evidence_only": recap_only,
                "recap_evidence_count": len(recap_evidence),
                "substantive_evidence_count": len(
                    substantive_evidence
                ),
                "context_collision": context_collision,
                "comparison_evidence_only": (
                    comparison_profile.comparison_only
                ),
                "comparison_evidence_count": (
                    comparison_profile.comparison_hits
                ),
                "independent_evidence_count": (
                    comparison_profile.independent_hits
                ),
                "evidence_quality_notes": notes,
            }
        )

    def _shared_recap_evidence(
        self,
        candidates: list[TopicCandidate],
    ) -> set[str]:
        evidence_topics: dict[str, set[str]] = defaultdict(set)
        original_evidence: dict[str, str] = {}

        for candidate in candidates:
            for sentence in candidate.evidence:
                normalized = self._normalize(sentence)
                if not normalized:
                    continue

                evidence_topics[normalized].add(
                    candidate.concept_id
                )
                original_evidence.setdefault(
                    normalized,
                    sentence,
                )

        shared_recap: set[str] = set()

        for normalized, concept_ids in evidence_topics.items():
            if (
                len(concept_ids)
                < self.config.recap_minimum_shared_topics
            ):
                continue

            original = original_evidence[normalized]
            if self._matches_any(
                self._normalize(original),
                self._RECAP_PATTERNS,
            ):
                shared_recap.add(normalized)

        return shared_recap

    def _initial_rejection_reason(
        self,
        candidate: TopicCandidate,
    ) -> str | None:
        if candidate.context_collision:
            return (
                "Rejected because ambiguous aliases were found in a "
                "conflicting conceptual context without enough "
                "catalogue-defined supporting context."
            )

        if (
            self.config.ambiguous_alias_requires_context
            and candidate.ambiguous_alias_only
            and not candidate.matched_context_terms
        ):
            return (
                "Rejected because the candidate is supported only by an "
                "ambiguous alias without catalogue-defined context."
            )

        if candidate.recap_evidence_only:
            return (
                "Rejected because the candidate is supported only by a "
                "generic multi-topic recap rather than topic-specific "
                "teaching evidence."
            )

        if candidate.comparison_evidence_only:
            return (
                "Rejected because the candidate is supported only by a "
                "brief comparison or contrast reference, without "
                "independent teaching evidence."
            )

        if (
            self.config.isolated_mention_reject
            and candidate.teaching_depth_level == 0
        ):
            return (
                "Rejected because the evidence is an isolated mention "
                "rather than definition, explanation, example or practice."
            )

        if (
            candidate.cs_relevance_score
            < self.config.minimum_adjusted_score
        ):
            return (
                "Rejected because evidence quality reduced the adjusted "
                "candidate score below the retention threshold."
            )

        return None

    def _apply_candidate_competition(
        self,
        candidates: list[TopicCandidate],
    ) -> tuple[
        list[TopicCandidate],
        list[TopicCandidate],
    ]:
        ordered = sorted(
            candidates,
            key=self._ownership_score,
            reverse=True,
        )

        kept: list[TopicCandidate] = []
        rejected: list[TopicCandidate] = []

        for candidate in ordered:
            competitor = self._stronger_competitor(
                candidate=candidate,
                kept=kept,
            )

            if competitor is None:
                kept.append(candidate)
                continue

            overlap = self._evidence_overlap(
                candidate,
                competitor,
            )

            penalty = min(
                self.config.maximum_shared_evidence_penalty,
                overlap
                * self.config.maximum_shared_evidence_penalty,
            )

            adjusted_score = max(
                0.0,
                candidate.cs_relevance_score - penalty,
            )

            notes = list(
                candidate.evidence_quality_notes
            )
            notes.append(
                "Shared evidence is owned more strongly by "
                f"'{competitor.topic}'."
            )

            updated = candidate.model_copy(
                update={
                    "shared_evidence_penalty": round(
                        penalty,
                        4,
                    ),
                    "cs_relevance_score": round(
                        adjusted_score,
                        4,
                    ),
                    "evidence_quality_notes": notes,
                }
            )

            if (
                adjusted_score
                < self.config.minimum_adjusted_score
            ):
                rejected.append(
                    self._reject(
                        updated,
                        "Rejected after candidate competition because "
                        "another topic explained the shared evidence more "
                        "strongly.",
                    )
                )
            else:
                kept.append(updated)

        return kept, rejected

    def _stronger_competitor(
        self,
        *,
        candidate: TopicCandidate,
        kept: list[TopicCandidate],
    ) -> TopicCandidate | None:
        candidate_ownership = self._ownership_score(
            candidate
        )

        for existing in kept:
            overlap = self._evidence_overlap(
                candidate,
                existing,
            )

            if (
                overlap
                < self.config.evidence_overlap_threshold
            ):
                continue

            existing_ownership = (
                self._ownership_score(
                    existing
                )
            )

            candidate_is_contextually_weaker = any(
                (
                    candidate.ambiguous_alias_only
                    and not existing.ambiguous_alias_only,
                    candidate.teaching_depth_level
                    < existing.teaching_depth_level,
                    (
                        not candidate.matched_context_terms
                        and bool(existing.matched_context_terms)
                    ),
                )
            )

            if (
                candidate_is_contextually_weaker
                and existing_ownership
                >= (
                    candidate_ownership
                    + self.config.competition_margin
                )
            ):
                return existing

        return None

    @staticmethod
    def _ownership_score(
        candidate: TopicCandidate,
    ) -> float:
        context_score = min(
            1.0,
            len(candidate.matched_context_terms)
            / 2.0,
        )

        ambiguity_penalty = (
            0.10
            if candidate.ambiguous_alias_only
            else 0.0
        )

        recap_penalty = (
            0.12
            if candidate.recap_evidence_count > 0
            else 0.0
        )

        comparison_penalty = (
            0.14
            if candidate.comparison_evidence_only
            else 0.0
        )

        context_collision_penalty = (
            0.18 if candidate.context_collision else 0.0
        )

        return (
            0.32 * candidate.cs_relevance_score
            + 0.28 * candidate.evidence_quality_score
            + 0.20 * candidate.salience_score
            + 0.12 * max(
                0.0,
                min(
                    1.0,
                    candidate.semantic_score,
                ),
            )
            + 0.08 * context_score
            - ambiguity_penalty
            - recap_penalty
            - comparison_penalty
            - context_collision_penalty
        )

    @classmethod
    def _evidence_overlap(
        cls,
        first: TopicCandidate,
        second: TopicCandidate,
    ) -> float:
        first_evidence = {
            cls._normalize(value)
            for value in first.evidence
            if cls._normalize(value)
        }
        second_evidence = {
            cls._normalize(value)
            for value in second.evidence
            if cls._normalize(value)
        }

        if not first_evidence or not second_evidence:
            return 0.0

        return (
            len(
                first_evidence
                & second_evidence
            )
            / len(
                first_evidence
                | second_evidence
            )
        )


    def _comparison_profile(
        self,
        *,
        text: str,
        candidate: TopicCandidate,
    ) -> ComparisonEvidenceProfile:
        """
        Distinguish a brief contrast reference from independent teaching.

        Example rejected:
            "Linear search works on unsorted data, unlike binary search."

        Example retained:
            A dedicated comparison section repeatedly explains merge sort,
            its behaviour, speed, suitability and memory use.

        The rule is generic: it uses catalogue aliases and comparison
        language, never transcript-specific concept pairs.
        """

        normalized_text = self._normalize(text)
        words = normalized_text.split()

        aliases = self._comparison_aliases(candidate)
        if not aliases or not words:
            return ComparisonEvidenceProfile(
                total_hits=0,
                comparison_hits=0,
                independent_hits=0,
                comparison_only=False,
            )

        hit_spans: list[tuple[int, int]] = []

        for alias in aliases:
            alias_words = self._normalize(alias).split()
            if not alias_words:
                continue

            width = len(alias_words)

            for index in range(0, len(words) - width + 1):
                if words[index:index + width] == alias_words:
                    hit_spans.append((index, index + width))

        # De-duplicate the same occurrence matched by overlapping aliases.
        hit_spans = sorted(set(hit_spans))

        comparison_hits = 0

        for start, end in hit_spans:
            left = max(
                0,
                start - self.config.comparison_window_words,
            )
            right = min(
                len(words),
                end + self.config.comparison_window_words,
            )
            window = " ".join(words[left:right])

            if self._matches_any(
                window,
                self._COMPARISON_PATTERNS,
            ):
                comparison_hits += 1

        total_hits = len(hit_spans)
        independent_hits = max(
            0,
            total_hits - comparison_hits,
        )

        comparison_only = (
            total_hits > 0
            and comparison_hits == total_hits
            and independent_hits == 0
            and total_hits
            <= self.config.comparison_only_max_alias_hits
        )

        return ComparisonEvidenceProfile(
            total_hits=total_hits,
            comparison_hits=comparison_hits,
            independent_hits=independent_hits,
            comparison_only=comparison_only,
        )

    @classmethod
    def _comparison_aliases(
        cls,
        candidate: TopicCandidate,
    ) -> list[str]:
        aliases = [
            *candidate.matched_aliases,
            candidate.topic,
        ]

        unique: list[str] = []
        seen: set[str] = set()

        for alias in aliases:
            normalized = cls._normalize(alias)

            if (
                not normalized
                or normalized in seen
                or len(normalized.split()) < 2
            ):
                continue

            seen.add(normalized)
            unique.append(alias)

        return unique

    def _sentence_depth(
        self,
        sentence: str,
    ) -> int:
        normalized = self._normalize(
            sentence
        )

        if not normalized:
            return 0

        mention = self._matches_any(
            normalized,
            self._MENTION_PATTERNS,
        )
        definition = self._matches_any(
            normalized,
            self._DEFINITION_PATTERNS,
        )
        explanation = self._matches_any(
            normalized,
            self._EXPLANATION_PATTERNS,
        )
        worked_example = self._matches_any(
            normalized,
            self._WORKED_EXAMPLE_PATTERNS,
        )
        practice = self._matches_any(
            normalized,
            self._PRACTICE_PATTERNS,
        )

        if worked_example and practice:
            return 4
        if practice and explanation:
            return 4
        if worked_example:
            return 3
        if explanation:
            return 2
        if definition:
            return 1
        if mention:
            return 0

        return 1

    def _depth_score(
        self,
        level: int,
    ) -> float:
        return {
            0: self.config.mention_only_max_score,
            1: self.config.definition_score,
            2: self.config.explanation_score,
            3: self.config.worked_example_score,
            4: self.config.sustained_teaching_score,
        }[level]

    @staticmethod
    def _depth_label(
        level: int,
    ) -> str:
        return {
            0: "mention_only",
            1: "definition",
            2: "explanation",
            3: "worked_example",
            4: "sustained_teaching",
        }[level]

    @staticmethod
    def _matches_any(
        text: str,
        patterns: tuple[str, ...],
    ) -> bool:
        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            is not None
            for pattern in patterns
        )

    @staticmethod
    def _reject(
        candidate: TopicCandidate,
        reason: str,
    ) -> TopicCandidate:
        notes = list(
            candidate.evidence_quality_notes
        )
        notes.append(reason)

        return candidate.model_copy(
            update={
                "cs_relevant": False,
                "evidence_quality_notes": notes,
            }
        )

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        text = re.sub(
            r"[^a-z0-9]+",
            " ",
            text.lower(),
        )
        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()