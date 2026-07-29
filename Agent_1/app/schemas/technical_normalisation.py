from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


CorrectionSource = Literal[
    "spoken_code_rule",
    "memory",
    "llm",
]

MemoryStatus = Literal[
    "candidate",
    "approved",
    "rejected",
    "disabled",
]


@dataclass(frozen=True, slots=True)
class SuspiciousTechnicalSpan:
    """
    A small span that may contain a technical spelling or ASR error.

    Offsets are absolute offsets in the text after deterministic spoken-code
    normalisation has been applied.
    """

    issue_id: str
    sentence_index: int
    sentence_text: str
    sentence_start: int
    start: int
    end: int
    original_span: str
    detector_score: float
    reason: str
    candidate_terms: tuple[str, ...] = ()
    context_keywords: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CorrectionProposal:
    """A replacement pair returned by memory or the selective LLM."""

    issue_id: str
    original: str
    replacement: str
    confidence: float
    correction_type: str
    reason: str


@dataclass(frozen=True, slots=True)
class AppliedTechnicalCorrection:
    """One exact span replacement applied to the transcript."""

    issue_id: str
    original: str
    replacement: str
    start: int
    end: int
    source: CorrectionSource
    confidence: float
    correction_type: str
    reason: str
    sentence_text: str
    memory_id: int | None = None
    memory_status: MemoryStatus | None = None


@dataclass(slots=True)
class TechnicalNormalisationStats:
    spoken_code_corrections: int = 0
    suspicious_spans_detected: int = 0
    memory_hits: int = 0
    llm_calls: int = 0
    llm_proposals: int = 0
    accepted_llm_corrections: int = 0
    rejected_proposals: int = 0
    stored_memory_records: int = 0


@dataclass(slots=True)
class TechnicalNormalisationResult:
    original_text: str
    normalised_text: str
    corrections: list[AppliedTechnicalCorrection] = field(
        default_factory=list
    )
    unresolved_issues: list[SuspiciousTechnicalSpan] = field(
        default_factory=list
    )
    stats: TechnicalNormalisationStats = field(
        default_factory=TechnicalNormalisationStats
    )

    @property
    def changed(self) -> bool:
        return self.original_text != self.normalised_text

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "normalised_text": self.normalised_text,
            "changed": self.changed,
            "corrections": [
                asdict(correction)
                for correction in self.corrections
            ],
            "unresolved_issues": [
                asdict(issue)
                for issue in self.unresolved_issues
            ],
            "stats": asdict(self.stats),
        }