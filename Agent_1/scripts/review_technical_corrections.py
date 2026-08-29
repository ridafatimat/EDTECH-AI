from __future__ import annotations

import argparse

from app.db.repositories.technical_correction_repository import (
    PostgreSQLTechnicalCorrectionRepository,
)
from app.db.session import session_scope


VALID_STATUSES = (
    "candidate",
    "approved",
    "rejected",
    "disabled",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Review PostgreSQL technical correction memory."
        )
    )

    parser.add_argument(
        "action",
        choices=[
            "list",
            "approve",
            "reject",
            "disable",
        ],
    )

    parser.add_argument(
        "record_id",
        nargs="?",
        type=int,
    )

    parser.add_argument(
        "--status",
        choices=VALID_STATUSES,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    with session_scope() as session:
        repository = (
            PostgreSQLTechnicalCorrectionRepository(
                session
            )
        )

        if args.action == "list":
            records = repository.list_records(
                status=args.status,
                limit=args.limit,
            )

            if not records:
                print(
                    "No technical correction records found."
                )
                return

            for record in records:
                print("-" * 90)
                print(f"ID: {record.id}")
                print(
                    f"{record.original_phrase!r} "
                    f"-> {record.corrected_phrase!r}"
                )
                print(f"Status: {record.status}")
                print(
                    f"Confidence: {record.confidence}"
                )
                print(
                    "Context: "
                    f"{record.context_keywords}"
                )
                print(
                    "Seen/applied: "
                    f"{record.times_seen}/"
                    f"{record.times_applied}"
                )

            return

        if args.record_id is None:
            parser.error(
                "record_id is required for "
                "approve/reject/disable."
            )

        status_by_action = {
            "approve": "approved",
            "reject": "rejected",
            "disable": "disabled",
        }

        repository.set_status(
            args.record_id,
            status_by_action[args.action],
        )

    print(
        f"Record {args.record_id} updated to "
        f"{status_by_action[args.action]}."
    )


if __name__ == "__main__":
    main()