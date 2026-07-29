from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import case, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models.technical_correction import (
    TechnicalCorrection,
)
from app.schemas.technical_normalisation import MemoryStatus
from app.services.technical_vocabulary import (
    normalise_lookup_key,
)


class PostgreSQLTechnicalCorrectionRepository:
    """
    PostgreSQL implementation used by SelectiveTechnicalNormaliser.

    Repository methods flush but do not commit. Transaction ownership stays
    with session_scope() or the calling application layer.
    """

    def __init__(
        self,
        session: Session,
        *,
        min_context_similarity: float = 0.20,
    ) -> None:
        if not 0 <= min_context_similarity <= 1:
            raise ValueError(
                "min_context_similarity must be between 0 and 1."
            )

        self.session = session
        self.min_context_similarity = min_context_similarity

    def find_approved(
        self,
        *,
        original_phrase: str,
        context_keywords: Sequence[str],
    ) -> TechnicalCorrection | None:
        original_key = normalise_lookup_key(
            original_phrase
        )

        statement = (
            select(TechnicalCorrection)
            .where(
                TechnicalCorrection.original_key
                == original_key,
                TechnicalCorrection.status
                == "approved",
            )
        )

        records = list(
            self.session.scalars(statement)
        )

        if not records:
            return None

        query_context = _normalise_context_keywords(
            context_keywords
        )

        ranked: list[
            tuple[float, float, int, TechnicalCorrection]
        ] = []

        for record in records:
            record_context = set(
                record.context_keywords or []
            )

            similarity = _jaccard_similarity(
                query_context,
                record_context,
            )

            # Empty stored context means the correction was deliberately
            # approved as context-independent.
            if (
                record_context
                and similarity
                < self.min_context_similarity
            ):
                continue

            ranked.append(
                (
                    similarity,
                    record.confidence,
                    record.times_applied,
                    record,
                )
            )

        if not ranked:
            return None

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            ),
            reverse=True,
        )

        return ranked[0][3]

    def store_or_update(
        self,
        *,
        original_phrase: str,
        corrected_phrase: str,
        context_keywords: Sequence[str],
        confidence: float,
        status: MemoryStatus,
        source_model: str | None,
    ) -> TechnicalCorrection:
        if not 0 <= confidence <= 1:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        context = sorted(
            _normalise_context_keywords(
                context_keywords
            )
        )
        context_signature = "|".join(context)

        values = {
            "original_phrase": original_phrase,
            "original_key": normalise_lookup_key(
                original_phrase
            ),
            "corrected_phrase": corrected_phrase,
            "corrected_key": normalise_lookup_key(
                corrected_phrase
            ),
            "context_keywords": context,
            "context_signature": context_signature,
            "confidence": confidence,
            "status": status,
            "source_model": source_model,
            "times_seen": 1,
            "times_applied": 0,
        }

        insert_statement = insert(
            TechnicalCorrection
        ).values(**values)

        excluded = insert_statement.excluded

        # Status should never be silently downgraded:
        # rejected/disabled stay blocked; approved stays approved.
        next_status = case(
            (
                TechnicalCorrection.status.in_(
                    ["rejected", "disabled"]
                ),
                TechnicalCorrection.status,
            ),
            (
                TechnicalCorrection.status
                == "approved",
                "approved",
            ),
            (
                excluded.status == "approved",
                "approved",
            ),
            else_="candidate",
        )

        upsert_statement = (
            insert_statement
            .on_conflict_do_update(
                constraint=(
                    "uq_technical_correction_phrase_context"
                ),
                set_={
                    "confidence": func.greatest(
                        TechnicalCorrection.confidence,
                        excluded.confidence,
                    ),
                    "status": next_status,
                    "source_model": func.coalesce(
                        excluded.source_model,
                        TechnicalCorrection.source_model,
                    ),
                    "times_seen": (
                        TechnicalCorrection.times_seen
                        + 1
                    ),
                    "updated_at": func.now(),
                },
            )
            .returning(TechnicalCorrection.id)
        )

        record_id = self.session.execute(
            upsert_statement
        ).scalar_one()

        self.session.flush()

        record = self.session.get(
            TechnicalCorrection,
            record_id,
        )

        if record is None:
            raise RuntimeError(
                "Correction upsert succeeded but record "
                "could not be reloaded."
            )

        return record

    def mark_applied(self, record_id: int) -> None:
        statement = (
            update(TechnicalCorrection)
            .where(
                TechnicalCorrection.id
                == record_id
            )
            .values(
                times_applied=(
                    TechnicalCorrection.times_applied
                    + 1
                ),
                updated_at=func.now(),
            )
        )

        result = self.session.execute(statement)

        if result.rowcount == 0:
            raise KeyError(
                f"No technical correction with id {record_id}."
            )

        self.session.flush()

    def set_status(
        self,
        record_id: int,
        status: MemoryStatus,
    ) -> None:
        statement = (
            update(TechnicalCorrection)
            .where(
                TechnicalCorrection.id
                == record_id
            )
            .values(
                status=status,
                updated_at=func.now(),
            )
        )

        result = self.session.execute(statement)

        if result.rowcount == 0:
            raise KeyError(
                f"No technical correction with id {record_id}."
            )

        self.session.flush()

    def list_records(
        self,
        *,
        status: MemoryStatus | None = None,
        limit: int = 100,
    ) -> list[TechnicalCorrection]:
        statement = select(
            TechnicalCorrection
        )

        if status is not None:
            statement = statement.where(
                TechnicalCorrection.status
                == status
            )

        statement = (
            statement
            .order_by(
                TechnicalCorrection.updated_at.desc(),
                TechnicalCorrection.id.desc(),
            )
            .limit(limit)
        )

        return list(
            self.session.scalars(statement)
        )

    def delete(self, record_id: int) -> None:
        result = self.session.execute(
            delete(TechnicalCorrection)
            .where(
                TechnicalCorrection.id
                == record_id
            )
        )

        if result.rowcount == 0:
            raise KeyError(
                f"No technical correction with id {record_id}."
            )

        self.session.flush()


def _normalise_context_keywords(
    values: Sequence[str],
) -> set[str]:
    return {
        normalise_lookup_key(value)
        for value in values
        if value and normalise_lookup_key(value)
    }


def _jaccard_similarity(
    left: set[str],
    right: set[str],
) -> float:
    if not left and not right:
        return 1.0

    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)