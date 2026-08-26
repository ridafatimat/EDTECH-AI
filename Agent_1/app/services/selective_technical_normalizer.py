from __future__ import annotations

from collections import defaultdict
from typing import Protocol, Sequence, runtime_checkable

from app.schemas.technical_normalisation import (
    AppliedTechnicalCorrection,
    CorrectionProposal,
    MemoryStatus,
    SuspiciousTechnicalSpan,
    TechnicalNormalisationResult,
    TechnicalNormalisationStats,
)
from app.services.correction_validator import (
    TechnicalCorrectionValidator,
    ensure_non_overlapping,
)
from app.services.spoken_code_normalizer import (
    SpokenCodeNormaliser,
)
from app.services.technical_correction_client import (
    TechnicalCorrectionClient,
)
from app.services.technical_error_detector import (
    TechnicalErrorDetector,
)
from app.services.technical_vocabulary import (
    TechnicalVocabulary,
    build_technical_vocabulary,
)


@runtime_checkable
class CorrectionMemoryRecord(Protocol):
    id: int
    corrected_phrase: str
    confidence: float
    status: MemoryStatus


@runtime_checkable
class TechnicalCorrectionRepository(Protocol):
    """Interface implemented by the PostgreSQL correction repository."""

    def find_approved(
        self,
        *,
        original_phrase: str,
        context_keywords: Sequence[str],
    ) -> CorrectionMemoryRecord | None:
        ...

    def store_or_update(
        self,
        *,
        original_phrase: str,
        corrected_phrase: str,
        context_keywords: Sequence[str],
        confidence: float,
        status: MemoryStatus,
        source_model: str | None,
    ) -> CorrectionMemoryRecord:
        ...

    def mark_applied(self, record_id: int) -> None:
        ...


class SelectiveTechnicalNormaliser:
    """
    Selective technical terminology normalisation.

    Flow:
    1. Apply unambiguous spoken-code rules.
    2. Detect precision-filtered suspicious spans.
    3. Use approved PostgreSQL memory when context matches.
    4. Send only unresolved affected sentences/spans to the LLM.
    5. Validate each exact replacement pair.
    6. Store every first-time LLM correction as ``candidate``.
    7. Apply only non-overlapping validated spans.

    No first-time LLM correction is automatically approved. Rejected or
    disabled database records are never applied, even if the LLM proposes
    the same mapping again.
    """

    def __init__(
        self,
        *,
        repository: TechnicalCorrectionRepository,
        correction_client: TechnicalCorrectionClient | None,
        vocabulary: TechnicalVocabulary | None = None,
        detector: TechnicalErrorDetector | None = None,
        validator: TechnicalCorrectionValidator | None = None,
        spoken_code_normaliser: SpokenCodeNormaliser | None = None,
        max_llm_sentences: int = 20,
        # Kept only for backward compatibility with older callers. It is
        # intentionally ignored because first-time LLM output is never
        # auto-approved anymore.
        auto_approve_confidence: float | None = None,
    ) -> None:
        del auto_approve_confidence

        if max_llm_sentences < 0:
            raise ValueError(
                "max_llm_sentences cannot be negative."
            )

        self.vocabulary = (
            vocabulary
            or build_technical_vocabulary()
        )
        self.repository = repository
        self.correction_client = correction_client
        self.detector = (
            detector
            or TechnicalErrorDetector(
                vocabulary=self.vocabulary
            )
        )
        self.validator = (
            validator
            or TechnicalCorrectionValidator(
                vocabulary=self.vocabulary
            )
        )
        self.spoken_code_normaliser = (
            spoken_code_normaliser
            or SpokenCodeNormaliser()
        )
        self.max_llm_sentences = max_llm_sentences

    def normalise(
        self,
        cleaned_text: str,
    ) -> TechnicalNormalisationResult:
        if not isinstance(cleaned_text, str):
            raise TypeError(
                "cleaned_text must be a string."
            )

        spoken_result = (
            self.spoken_code_normaliser
            .normalise(cleaned_text)
        )
        working_text = spoken_result.text

        stats = TechnicalNormalisationStats(
            spoken_code_corrections=len(
                spoken_result.corrections
            )
        )

        issues = self.detector.detect(
            working_text
        )
        stats.suspicious_spans_detected = len(
            issues
        )

        accepted: list[_AcceptedCorrection] = []
        unresolved_after_memory: list[
            SuspiciousTechnicalSpan
        ] = []

        # Stage 1: approved PostgreSQL correction-memory lookup.
        for issue in issues:
            record = self.repository.find_approved(
                original_phrase=issue.original_span,
                context_keywords=issue.context_keywords,
            )

            if record is None:
                unresolved_after_memory.append(issue)
                continue

            proposal = CorrectionProposal(
                issue_id=issue.issue_id,
                original=issue.original_span,
                replacement=record.corrected_phrase,
                confidence=record.confidence,
                correction_type=(
                    "stored_technical_correction"
                ),
                reason=(
                    "Approved context-matched correction memory hit."
                ),
            )

            validation = self.validator.validate(
                issue=issue,
                proposal=proposal,
                current_text=working_text,
            )

            if not validation.accepted:
                unresolved_after_memory.append(issue)
                stats.rejected_proposals += 1
                continue

            accepted.append(
                _AcceptedCorrection(
                    issue=issue,
                    proposal=proposal,
                    source="memory",
                    memory_id=record.id,
                    memory_status=record.status,
                )
            )
            stats.memory_hits += 1

        # Stage 2: LLM only for unresolved affected sentences.
        unresolved_final: list[
            SuspiciousTechnicalSpan
        ] = []

        groups = _group_issues_by_sentence(
            unresolved_after_memory
        )

        for group_index, sentence_issues in enumerate(
            groups
        ):
            if self.correction_client is None:
                unresolved_final.extend(sentence_issues)
                continue

            if group_index >= self.max_llm_sentences:
                unresolved_final.extend(sentence_issues)
                continue

            stats.llm_calls += 1

            proposals = (
                self.correction_client
                .suggest_corrections(
                    sentence=(
                        sentence_issues[0].sentence_text
                    ),
                    issues=sentence_issues,
                )
            )
            stats.llm_proposals += len(proposals)

            proposals_by_issue = {
                proposal.issue_id: proposal
                for proposal in proposals
            }

            for issue in sentence_issues:
                proposal = proposals_by_issue.get(
                    issue.issue_id
                )

                if proposal is None:
                    unresolved_final.append(issue)
                    continue

                validation = self.validator.validate(
                    issue=issue,
                    proposal=proposal,
                    current_text=working_text,
                )

                if not validation.accepted:
                    unresolved_final.append(issue)
                    stats.rejected_proposals += 1
                    continue

                # Critical safety change: every first-time LLM result is a
                # candidate. Manual review is required before future memory
                # reuse can occur.
                record = self.repository.store_or_update(
                    original_phrase=proposal.original,
                    corrected_phrase=proposal.replacement,
                    context_keywords=issue.context_keywords,
                    confidence=proposal.confidence,
                    status="candidate",
                    source_model=(
                        self.correction_client.model_name
                    ),
                )
                stats.stored_memory_records += 1

                # The repository preserves rejected/disabled states during
                # upsert. Never apply a correction that reviewers blocked.
                if record.status in {
                    "rejected",
                    "disabled",
                }:
                    unresolved_final.append(issue)
                    stats.rejected_proposals += 1
                    continue

                accepted.append(
                    _AcceptedCorrection(
                        issue=issue,
                        proposal=proposal,
                        source="llm",
                        memory_id=record.id,
                        memory_status=record.status,
                    )
                )
                stats.accepted_llm_corrections += 1

        safe_pairs = ensure_non_overlapping(
            (
                item.issue,
                item.proposal,
            )
            for item in accepted
        )
        safe_keys = {
            (
                issue.issue_id,
                proposal.replacement,
            )
            for issue, proposal in safe_pairs
        }

        safe_accepted = [
            item
            for item in accepted
            if (
                item.issue.issue_id,
                item.proposal.replacement,
            ) in safe_keys
        ]

        unsafe_ids = {
            item.issue.issue_id
            for item in accepted
            if item not in safe_accepted
        }

        unresolved_final.extend(
            item.issue
            for item in accepted
            if item.issue.issue_id in unsafe_ids
        )

        applied: list[
            AppliedTechnicalCorrection
        ] = []

        # Right-to-left replacement keeps all original detected offsets valid.
        for item in sorted(
            safe_accepted,
            key=lambda value: value.issue.start,
            reverse=True,
        ):
            issue = item.issue
            proposal = item.proposal

            working_text = (
                working_text[:issue.start]
                + proposal.replacement
                + working_text[issue.end:]
            )

            applied.append(
                AppliedTechnicalCorrection(
                    issue_id=issue.issue_id,
                    original=proposal.original,
                    replacement=proposal.replacement,
                    start=issue.start,
                    end=issue.end,
                    source=item.source,
                    confidence=proposal.confidence,
                    correction_type=(
                        proposal.correction_type
                    ),
                    reason=proposal.reason,
                    sentence_text=issue.sentence_text,
                    memory_id=item.memory_id,
                    memory_status=item.memory_status,
                )
            )

            self.repository.mark_applied(
                item.memory_id
            )

        applied.reverse()

        return TechnicalNormalisationResult(
            original_text=cleaned_text,
            normalised_text=working_text,
            corrections=(
                list(spoken_result.corrections)
                + applied
            ),
            unresolved_issues=sorted(
                {
                    issue.issue_id: issue
                    for issue in unresolved_final
                }.values(),
                key=lambda issue: issue.start,
            ),
            stats=stats,
        )


class _AcceptedCorrection:
    __slots__ = (
        "issue",
        "proposal",
        "source",
        "memory_id",
        "memory_status",
    )

    def __init__(
        self,
        *,
        issue: SuspiciousTechnicalSpan,
        proposal: CorrectionProposal,
        source: str,
        memory_id: int,
        memory_status: MemoryStatus,
    ) -> None:
        self.issue = issue
        self.proposal = proposal
        self.source = source
        self.memory_id = memory_id
        self.memory_status = memory_status


def _group_issues_by_sentence(
    issues: Sequence[SuspiciousTechnicalSpan],
) -> list[list[SuspiciousTechnicalSpan]]:
    grouped: dict[
        int,
        list[SuspiciousTechnicalSpan],
    ] = defaultdict(list)

    for issue in issues:
        grouped[issue.sentence_index].append(
            issue
        )

    return [
        sorted(
            sentence_issues,
            key=lambda issue: issue.start,
        )
        for _, sentence_issues in sorted(
            grouped.items()
        )
    ]