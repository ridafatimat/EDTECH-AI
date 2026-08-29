from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.schemas.technical_normalisation import (
    CorrectionProposal,
    MemoryStatus,
    SuspiciousTechnicalSpan,
)
from app.services.selective_technical_normalizer import (
    SelectiveTechnicalNormaliser,
)


@dataclass(slots=True)
class FakeRecord:
    id: int
    corrected_phrase: str
    confidence: float
    status: MemoryStatus


class FakeRepository:
    """
    In-memory repository used only for unit testing.

    New LLM corrections are stored as candidates. The test explicitly
    simulates human approval before checking memory reuse.
    """

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
        record = self.records.get(
            original_phrase.casefold()
        )

        if (
            record is not None
            and record.status == "approved"
        ):
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

    def mark_applied(
        self,
        record_id: int,
    ) -> None:
        self.applied_ids.append(
            record_id
        )

    def approve_all_candidates(self) -> None:
        """
        Simulate manual review and approval.

        Production code performs this through the PostgreSQL review
        workflow. Unit tests do it directly in memory.
        """

        for record in self.records.values():
            if record.status == "candidate":
                record.status = "approved"


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
        self.call_count += 1
        proposals: list[CorrectionProposal] = []

        for issue in issues:
            key = issue.original_span.casefold()

            if key == "wild loop":
                proposals.append(
                    CorrectionProposal(
                        issue_id=issue.issue_id,
                        original=issue.original_span,
                        replacement="while loop",
                        confidence=0.97,
                        correction_type="technical_asr",
                        reason=(
                            "Likely ASR confusion in "
                            "programming context."
                        ),
                    )
                )

            elif key == "algorithim":
                proposals.append(
                    CorrectionProposal(
                        issue_id=issue.issue_id,
                        original=issue.original_span,
                        replacement="algorithm",
                        confidence=0.98,
                        correction_type="technical_spelling",
                        reason=(
                            "Technical spelling correction."
                        ),
                    )
                )

        return proposals


def main() -> None:
    repository = FakeRepository()
    client = FakeCorrectionClient()

    normaliser = SelectiveTechnicalNormaliser(
        repository=repository,
        correction_client=client,
    )

    original = (
        "It says the wild loop checks whether K is less "
        "than array dot length. "
        "The algorithim then updates the index."
    )

    expected = (
        "It says the while loop checks whether K is less "
        "than array.length. "
        "The algorithm then updates the index."
    )

    # -------------------------------------------------------------
    # First run:
    # - deterministic spoken-code correction is applied;
    # - LLM corrections are applied to the current transcript;
    # - new corrections are stored as candidates.
    # -------------------------------------------------------------

    first = normaliser.normalise(
        original
    )

    assert first.normalised_text == expected
    assert first.stats.spoken_code_corrections == 1
    assert first.stats.accepted_llm_corrections == 2
    assert first.stats.memory_hits == 0
    assert first.stats.llm_calls == 2
    assert client.call_count == 2

    assert set(repository.records) == {
        "wild loop",
        "algorithim",
    }

    assert all(
        record.status == "candidate"
        for record in repository.records.values()
    )

    # mark_applied tracks every correction that was actually written to
    # the transcript, including first-run candidate-backed LLM changes.
    assert len(repository.applied_ids) == 2

    reconstructed = (
        first.normalised_text
        .replace(
            "while loop",
            "wild loop",
        )
        .replace(
            "array.length",
            "array dot length",
        )
        .replace(
            "algorithm",
            "algorithim",
        )
    )

    assert reconstructed == original

    # -------------------------------------------------------------
    # Simulate manual approval before future memory reuse.
    # -------------------------------------------------------------

    repository.approve_all_candidates()

    assert all(
        record.status == "approved"
        for record in repository.records.values()
    )

    applied_before_memory_run = len(
        repository.applied_ids
    )

    # -------------------------------------------------------------
    # Second run:
    # - both approved corrections come from memory;
    # - no LLM call is required;
    # - two additional applications are recorded.
    # -------------------------------------------------------------

    second = normaliser.normalise(
        original
    )

    assert second.normalised_text == expected
    assert second.stats.spoken_code_corrections == 1
    assert second.stats.memory_hits == 2
    assert second.stats.llm_calls == 0
    assert second.stats.accepted_llm_corrections == 0
    assert client.call_count == 2

    assert (
        len(repository.applied_ids)
        - applied_before_memory_run
        == 2
    )

    # -------------------------------------------------------------
    # Ordinary English must not be treated as a technical correction.
    # -------------------------------------------------------------

    ordinary = (
        "The roller coaster had a wild loop near the end."
    )

    ordinary_result = normaliser.normalise(
        ordinary
    )

    assert ordinary_result.normalised_text == ordinary
    assert ordinary_result.stats.suspicious_spans_detected == 0
    assert ordinary_result.stats.llm_calls == 0
    assert ordinary_result.stats.memory_hits == 0
    assert client.call_count == 2

    print(
        "ALL SELECTIVE TECHNICAL NORMALISATION "
        "UNIT TESTS PASSED"
    )


if __name__ == "__main__":
    main()