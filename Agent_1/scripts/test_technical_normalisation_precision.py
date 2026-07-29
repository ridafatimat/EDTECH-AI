from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.schemas.technical_normalisation import (
    CorrectionProposal,
    MemoryStatus,
    SuspiciousTechnicalSpan,
)
from app.services.correction_validator import (
    TechnicalCorrectionValidator,
)
from app.services.selective_technical_normalizer import (
    SelectiveTechnicalNormaliser,
)
from app.services.technical_error_detector import (
    TechnicalErrorDetector,
)
from app.services.technical_vocabulary import (
    build_technical_vocabulary,
)


@dataclass(slots=True)
class FakeRecord:
    id: int
    corrected_phrase: str
    confidence: float
    status: MemoryStatus


class FakeRepository:
    def __init__(self) -> None:
        self.records: dict[str, FakeRecord] = {}
        self.next_id = 1
        self.applied_ids: list[int] = []

    def find_approved(
        self,
        *,
        original_phrase: str,
        context_keywords: Sequence[str],
    ) -> FakeRecord | None:
        del context_keywords
        record = self.records.get(
            original_phrase.casefold()
        )

        if record is not None and record.status == "approved":
            return record

        return None

    def store_or_update(
        self,
        *,
        original_phrase: str,
        corrected_phrase: str,
        context_keywords: Sequence[str],
        confidence: float,
        status: MemoryStatus,
        source_model: str | None,
    ) -> FakeRecord:
        del context_keywords, source_model
        key = original_phrase.casefold()
        existing = self.records.get(key)

        if existing is not None:
            return existing

        record = FakeRecord(
            id=self.next_id,
            corrected_phrase=corrected_phrase,
            confidence=confidence,
            status=status,
        )
        self.next_id += 1
        self.records[key] = record
        return record

    def mark_applied(self, record_id: int) -> None:
        self.applied_ids.append(record_id)


class FakeCorrectionClient:
    model_name = "fake-test-model"

    def __init__(self) -> None:
        self.call_count = 0

    def suggest_corrections(
        self,
        *,
        sentence: str,
        issues: Sequence[SuspiciousTechnicalSpan],
    ) -> list[CorrectionProposal]:
        del sentence
        self.call_count += 1
        replacements = {
            "wild loop": "while loop",
            "algoritm": "algorithm",
            "array lenght": "array length",
        }
        proposals: list[CorrectionProposal] = []

        for issue in issues:
            replacement = replacements.get(
                issue.original_span.casefold()
            )

            if replacement is None:
                continue

            proposals.append(
                CorrectionProposal(
                    issue_id=issue.issue_id,
                    original=issue.original_span,
                    replacement=replacement,
                    confidence=0.97,
                    correction_type="technical_asr",
                    reason="Clear one-token technical corruption.",
                )
            )

        return proposals


def _make_issue(
    text: str,
    original: str,
    candidate: str,
) -> SuspiciousTechnicalSpan:
    start = text.index(original)
    return SuspiciousTechnicalSpan(
        issue_id=f"test:{start}:{start + len(original)}",
        sentence_index=0,
        sentence_text=text,
        sentence_start=0,
        start=start,
        end=start + len(original),
        original_span=original,
        detector_score=0.95,
        reason="Test issue.",
        candidate_terms=(candidate,),
        context_keywords=("code_syntax", "loop"),
    )


def main() -> None:
    vocabulary = build_technical_vocabulary()
    detector = TechnicalErrorDetector(
        vocabulary=vocabulary
    )
    validator = TechnicalCorrectionValidator(
        vocabulary=vocabulary
    )

    clean_text = (
        "for i from one while i less than array length. "
        "let's do another array same algorithm. "
        "when tracing while loops watch initialization. "
        "function returns value and procedure performs task. "
        "why use subroutines for decomposition and reuse. "
        "while low less than or equal high compare target. "
        "if condition used OR, loop might continue. "
        "syntax or name error depending language."
    )

    clean_issues = detector.detect(clean_text)
    assert clean_issues == [], [
        issue.original_span
        for issue in clean_issues
    ]

    corrupted_text = (
        "The wild loop checks the array index. "
        "The algoritm uses the array lenght."
    )
    corrupted_spans = {
        issue.original_span.casefold()
        for issue in detector.detect(corrupted_text)
    }

    assert "wild loop" in corrupted_spans
    assert "algoritm" in corrupted_spans
    assert "array lenght" in corrupted_spans

    bad_cases = (
        (
            "when tracing while loops watch initialization",
            "while loops",
            "while loop",
        ),
        (
            "function returns value and procedure performs task",
            "returns value",
            "return value",
        ),
        (
            "let's do another array same algorithm",
            "same algorithm",
            "sorting algorithm",
        ),
        (
            "while low less than or equal high",
            "while low",
            "while loop",
        ),
        (
            "syntax or name error depending language",
            "syntax or",
            "syntax error",
        ),
        (
            "if condition used OR, loop might continue",
            "OR, loop",
            "for loop",
        ),
    )

    for text, original, replacement in bad_cases:
        issue = _make_issue(text, original, replacement)
        result = validator.validate(
            issue=issue,
            proposal=CorrectionProposal(
                issue_id=issue.issue_id,
                original=original,
                replacement=replacement,
                confidence=0.99,
                correction_type="technical_asr",
                reason="Unsafe test proposal.",
            ),
            current_text=text,
        )
        assert not result.accepted, (
            original,
            replacement,
            result.reason,
        )

    repository = FakeRepository()
    client = FakeCorrectionClient()
    normaliser = SelectiveTechnicalNormaliser(
        repository=repository,
        correction_client=client,
        vocabulary=vocabulary,
    )

    first = normaliser.normalise(corrupted_text)
    assert first.normalised_text == (
        "The while loop checks the array index. "
        "The algorithm uses the array length."
    )
    assert first.stats.accepted_llm_corrections == 3
    assert all(
        record.status == "candidate"
        for record in repository.records.values()
    )

    # Candidate memory is deliberately not reused automatically.
    calls_after_first = client.call_count
    second = normaliser.normalise(corrupted_text)
    assert second.stats.memory_hits == 0
    assert client.call_count > calls_after_first

    # Once manually approved, the correction can be reused from memory.
    for record in repository.records.values():
        record.status = "approved"

    calls_before_approved_run = client.call_count
    third = normaliser.normalise(corrupted_text)
    assert third.stats.memory_hits == 3
    assert third.stats.llm_calls == 0
    assert client.call_count == calls_before_approved_run

    # A reviewer-rejected record must never be applied again.
    repository.records["wild loop"].status = "rejected"
    rejected_run = normaliser.normalise(
        "The wild loop checks the array index."
    )
    assert rejected_run.normalised_text == (
        "The wild loop checks the array index."
    )

    print(
        "ALL TECHNICAL NORMALISATION PRECISION TESTS PASSED"
    )


if __name__ == "__main__":
    main()