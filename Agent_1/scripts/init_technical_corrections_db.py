from __future__ import annotations

from app.db.models.technical_correction import (
    TechnicalCorrection,
)
from app.db.session import get_engine


def main() -> None:
    engine = get_engine()

    TechnicalCorrection.__table__.create(
        bind=engine,
        checkfirst=True,
    )

    print(
        "technical_corrections table is ready."
    )


if __name__ == "__main__":
    main()