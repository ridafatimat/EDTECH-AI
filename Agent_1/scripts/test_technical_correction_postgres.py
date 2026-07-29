from __future__ import annotations

from uuid import uuid4

from app.db.models.technical_correction import (
    TechnicalCorrection,
)
from app.db.repositories.technical_correction_repository import (
    PostgreSQLTechnicalCorrectionRepository,
)
from app.db.session import get_engine, session_scope


def main() -> None:
    TechnicalCorrection.__table__.create(
        bind=get_engine(),
        checkfirst=True,
    )

    unique_suffix = uuid4().hex[:10]
    original = f"wild loop integration {unique_suffix}"

    with session_scope() as session:
        repository = (
            PostgreSQLTechnicalCorrectionRepository(
                session
            )
        )

        stored = repository.store_or_update(
            original_phrase=original,
            corrected_phrase="while loop",
            context_keywords=[
                "array",
                "condition",
                "index",
            ],
            confidence=0.97,
            status="approved",
            source_model="integration-test",
        )

        found = repository.find_approved(
            original_phrase=original,
            context_keywords=[
                "array",
                "index",
            ],
        )

        assert found is not None
        assert found.id == stored.id
        assert found.corrected_phrase == "while loop"

        repository.mark_applied(
            stored.id
        )

        session.flush()
        session.refresh(stored)

        assert stored.times_applied == 1

        repository.set_status(
            stored.id,
            "disabled",
        )

        assert (
            repository.find_approved(
                original_phrase=original,
                context_keywords=[
                    "array",
                    "index",
                ],
            )
            is None
        )

        repository.delete(stored.id)

    print(
        "POSTGRESQL TECHNICAL CORRECTION "
        "INTEGRATION TEST PASSED"
    )


if __name__ == "__main__":
    main()