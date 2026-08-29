from __future__ import annotations

"""Read-only verification for the catalogue-rules PostgreSQL migration."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from app.db.session import get_engine
from scripts.migrate_catalogue_rules_to_postgres import (
    _catalogue_ids,
    _concept_id_overlap,
    _detect_target_table,
    _table_columns,
    _verify_migration,
    MIGRATION_COLUMNS,
)


def main() -> None:
    engine = get_engine()
    expected_count = len(_catalogue_ids())

    with engine.connect() as connection:
        schema, table, id_column = _detect_target_table(connection)
        columns = _table_columns(connection, schema, table)
        missing_columns = [
            name for name in MIGRATION_COLUMNS if name not in columns
        ]

        matched, _ = _concept_id_overlap(
            connection,
            schema=schema,
            table=table,
            id_column=id_column,
        )

        if missing_columns:
            raise RuntimeError(
                "Catalogue-rule migration is incomplete. Missing columns: "
                + ", ".join(missing_columns)
            )

        if matched != expected_count:
            raise RuntimeError(
                f"Only {matched}/{expected_count} catalogue concept IDs "
                "exist in the detected syllabus table."
            )

        _verify_migration(
            connection,
            schema=schema,
            table=table,
            id_column=id_column,
        )

        print(
            json.dumps(
                {
                    "status": "verified",
                    "table": f"{schema}.{table}",
                    "concept_id_column": id_column,
                    "concepts_verified": expected_count,
                    "fields_verified": list(MIGRATION_COLUMNS),
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()