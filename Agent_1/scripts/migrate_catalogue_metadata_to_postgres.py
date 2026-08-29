from __future__ import annotations

"""
One-time Agent 1 migration:
complete public.syllabus_concepts with the FULL static catalogue schema.

Run from Agent_1 root:
    python scripts/migrate_catalogue_metadata_to_postgres.py

Safe behaviour:
- Uses existing public.syllabus_concepts if present.
- Adds only missing columns with ALTER TABLE ... ADD COLUMN IF NOT EXISTS.
- Upserts the same 92 concept_ids (no duplicate rows).
- Populates all catalogue metadata/rule fields required by SyllabusStore.
- Verifies all 92 rows field-by-field before commit.
- Does not modify assessment/topic/memory tables.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import get_engine, load_environment  # noqa: E402
from app.services.cs_concept_catalog import CS_CONCEPTS  # noqa: E402


TABLE_NAME = "public.syllabus_concepts"
EXPECTED_CONCEPT_COUNT = 92
DEFAULT_SPECIFICATION_CODE = "8525"
DEFAULT_SPECIFICATION_VERSION = "AQA-8525-v1.2-2022-11-29"


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.syllabus_concepts (
    concept_id TEXT PRIMARY KEY,
    specification_code VARCHAR(20) NOT NULL,
    specification_version VARCHAR(120) NOT NULL,

    official_reference VARCHAR(20) NOT NULL,
    chapter_reference VARCHAR(20),
    chapter_title TEXT,
    official_title TEXT,

    label TEXT NOT NULL,
    domain TEXT,
    description TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,

    paper VARCHAR(20) NOT NULL,
    source_pages JSONB NOT NULL DEFAULT '[]'::jsonb,
    parent_concept_id TEXT NULL,

    excluded_phrases JSONB NOT NULL DEFAULT '[]'::jsonb,
    ambiguous_aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    supporting_context_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
    conflicting_context_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
    minimum_context_hits INTEGER NOT NULL DEFAULT 1,
    match_patterns JSONB NOT NULL DEFAULT '[]'::jsonb,

    embedding_text TEXT,

    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT syllabus_concepts_minimum_context_hits_chk
        CHECK (minimum_context_hits >= 0)
);
"""

# Important: CREATE TABLE IF NOT EXISTS does NOT add columns to a table that
# already exists, so the migration must explicitly upgrade the existing table.
ALTER_EXISTING_TABLE_SQL = """
ALTER TABLE public.syllabus_concepts
    ADD COLUMN IF NOT EXISTS chapter_reference VARCHAR(20),
    ADD COLUMN IF NOT EXISTS chapter_title TEXT,
    ADD COLUMN IF NOT EXISTS official_title TEXT,
    ADD COLUMN IF NOT EXISTS domain TEXT,
    ADD COLUMN IF NOT EXISTS embedding_text TEXT;
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_syllabus_concepts_official_reference
    ON public.syllabus_concepts (official_reference);

CREATE INDEX IF NOT EXISTS ix_syllabus_concepts_specification
    ON public.syllabus_concepts (specification_code, specification_version);

CREATE INDEX IF NOT EXISTS ix_syllabus_concepts_parent
    ON public.syllabus_concepts (parent_concept_id);
"""


UPSERT_SQL = text(
    """
    INSERT INTO public.syllabus_concepts (
        concept_id,
        specification_code,
        specification_version,
        official_reference,
        chapter_reference,
        chapter_title,
        official_title,
        label,
        domain,
        description,
        aliases,
        paper,
        source_pages,
        parent_concept_id,
        excluded_phrases,
        ambiguous_aliases,
        supporting_context_terms,
        conflicting_context_terms,
        minimum_context_hits,
        match_patterns,
        embedding_text,
        is_active
    ) VALUES (
        :concept_id,
        :specification_code,
        :specification_version,
        :official_reference,
        :chapter_reference,
        :chapter_title,
        :official_title,
        :label,
        :domain,
        :description,
        CAST(:aliases AS JSONB),
        :paper,
        CAST(:source_pages AS JSONB),
        :parent_concept_id,
        CAST(:excluded_phrases AS JSONB),
        CAST(:ambiguous_aliases AS JSONB),
        CAST(:supporting_context_terms AS JSONB),
        CAST(:conflicting_context_terms AS JSONB),
        :minimum_context_hits,
        CAST(:match_patterns AS JSONB),
        :embedding_text,
        TRUE
    )
    ON CONFLICT (concept_id) DO UPDATE SET
        specification_code = EXCLUDED.specification_code,
        specification_version = EXCLUDED.specification_version,
        official_reference = EXCLUDED.official_reference,
        chapter_reference = EXCLUDED.chapter_reference,
        chapter_title = EXCLUDED.chapter_title,
        official_title = EXCLUDED.official_title,
        label = EXCLUDED.label,
        domain = EXCLUDED.domain,
        description = EXCLUDED.description,
        aliases = EXCLUDED.aliases,
        paper = EXCLUDED.paper,
        source_pages = EXCLUDED.source_pages,
        parent_concept_id = EXCLUDED.parent_concept_id,
        excluded_phrases = EXCLUDED.excluded_phrases,
        ambiguous_aliases = EXCLUDED.ambiguous_aliases,
        supporting_context_terms = EXCLUDED.supporting_context_terms,
        conflicting_context_terms = EXCLUDED.conflicting_context_terms,
        minimum_context_hits = EXCLUDED.minimum_context_hits,
        match_patterns = EXCLUDED.match_patterns,
        embedding_text = EXCLUDED.embedding_text,
        is_active = TRUE,
        updated_at = NOW();
    """
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _normalise_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _patterns(concept: Any) -> list[dict[str, Any]]:
    return [
        {
            "label": pattern.label,
            "regex": pattern.regex,
            "weight": float(pattern.weight),
        }
        for pattern in concept.match_patterns
    ]


def _payload(concept: Any, specification_version: str) -> dict[str, Any]:
    return {
        "concept_id": concept.concept_id,
        "specification_code": DEFAULT_SPECIFICATION_CODE,
        "specification_version": specification_version,
        "official_reference": concept.official_reference,
        "chapter_reference": concept.chapter_reference,
        "chapter_title": concept.chapter_title,
        "official_title": concept.official_title,
        "label": concept.label,
        "domain": concept.domain,
        "description": concept.description,
        "aliases": _json(list(concept.aliases)),
        "paper": concept.paper,
        "source_pages": _json(list(concept.source_pages)),
        "parent_concept_id": concept.parent_concept_id,
        "excluded_phrases": _json(list(concept.excluded_phrases)),
        "ambiguous_aliases": _json(list(concept.ambiguous_aliases)),
        "supporting_context_terms": _json(list(concept.supporting_context_terms)),
        "conflicting_context_terms": _json(list(concept.conflicting_context_terms)),
        "minimum_context_hits": int(concept.minimum_context_hits),
        "match_patterns": _json(_patterns(concept)),
        "embedding_text": concept.embedding_text,
    }


def _run_multi_statement(connection: Any, sql: str) -> None:
    for statement in [part.strip() for part in sql.split(";") if part.strip()]:
        connection.execute(text(statement))


def _verify_columns(connection: Any) -> list[str]:
    required = {
        "concept_id",
        "specification_code",
        "specification_version",
        "official_reference",
        "chapter_reference",
        "chapter_title",
        "official_title",
        "label",
        "domain",
        "description",
        "aliases",
        "paper",
        "source_pages",
        "parent_concept_id",
        "excluded_phrases",
        "ambiguous_aliases",
        "supporting_context_terms",
        "conflicting_context_terms",
        "minimum_context_hits",
        "match_patterns",
        "embedding_text",
        "is_active",
    }

    existing = set(
        connection.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'syllabus_concepts'
                """
            )
        ).scalars().all()
    )

    missing = sorted(required - existing)
    if missing:
        raise RuntimeError(
            "Schema verification failed; columns still missing: "
            + ", ".join(missing)
        )

    return sorted(required)


def _verify_rows(
    connection: Any,
    expected: dict[str, dict[str, Any]],
    specification_version: str,
) -> None:
    rows = connection.execute(
        text(
            """
            SELECT
                concept_id,
                specification_code,
                specification_version,
                official_reference,
                chapter_reference,
                chapter_title,
                official_title,
                label,
                domain,
                description,
                aliases,
                paper,
                source_pages,
                parent_concept_id,
                excluded_phrases,
                ambiguous_aliases,
                supporting_context_terms,
                conflicting_context_terms,
                minimum_context_hits,
                match_patterns,
                embedding_text,
                is_active
            FROM public.syllabus_concepts
            WHERE specification_code = :code
              AND specification_version = :version
              AND is_active = TRUE
            ORDER BY concept_id
            """
        ),
        {
            "code": DEFAULT_SPECIFICATION_CODE,
            "version": specification_version,
        },
    ).mappings().all()

    if len(rows) != EXPECTED_CONCEPT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CONCEPT_COUNT} active rows, found {len(rows)}."
        )

    actual_by_id = {row["concept_id"]: row for row in rows}
    if set(actual_by_id) != set(expected):
        raise RuntimeError(
            "Concept ID mismatch. "
            f"missing={sorted(set(expected) - set(actual_by_id))}, "
            f"extra={sorted(set(actual_by_id) - set(expected))}"
        )

    json_fields = {
        "aliases",
        "source_pages",
        "excluded_phrases",
        "ambiguous_aliases",
        "supporting_context_terms",
        "conflicting_context_terms",
        "match_patterns",
    }

    compare_fields = [
        "specification_code",
        "specification_version",
        "official_reference",
        "chapter_reference",
        "chapter_title",
        "official_title",
        "label",
        "domain",
        "description",
        "aliases",
        "paper",
        "source_pages",
        "parent_concept_id",
        "excluded_phrases",
        "ambiguous_aliases",
        "supporting_context_terms",
        "conflicting_context_terms",
        "minimum_context_hits",
        "match_patterns",
        "embedding_text",
    ]

    mismatches: list[str] = []

    for concept_id, expected_row in expected.items():
        actual = actual_by_id[concept_id]

        for field in compare_fields:
            expected_value = expected_row[field]
            actual_value = actual[field]

            if field in json_fields:
                expected_value = _normalise_json(expected_value)
                actual_value = _normalise_json(actual_value)

            if actual_value != expected_value:
                mismatches.append(
                    f"{concept_id}.{field}: "
                    f"expected={expected_value!r}, actual={actual_value!r}"
                )
                if len(mismatches) >= 10:
                    break

        if len(mismatches) >= 10:
            break

    if mismatches:
        raise RuntimeError(
            "Row verification failed. First mismatches:\n- "
            + "\n- ".join(mismatches)
        )


def migrate() -> dict[str, Any]:
    load_environment()

    specification_version = os.getenv(
        "AQA_SPEC_VERSION",
        DEFAULT_SPECIFICATION_VERSION,
    ).strip()

    if len(CS_CONCEPTS) != EXPECTED_CONCEPT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CONCEPT_COUNT} CS_CONCEPTS, "
            f"found {len(CS_CONCEPTS)}."
        )

    ids = [concept.concept_id for concept in CS_CONCEPTS]
    if len(set(ids)) != EXPECTED_CONCEPT_COUNT:
        raise RuntimeError("Duplicate concept_id values found in CS_CONCEPTS.")

    expected = {
        concept.concept_id: _payload(concept, specification_version)
        for concept in CS_CONCEPTS
    }

    engine = get_engine()

    with engine.begin() as connection:
        _run_multi_statement(connection, CREATE_TABLE_SQL)
        _run_multi_statement(connection, ALTER_EXISTING_TABLE_SQL)
        _run_multi_statement(connection, INDEX_SQL)

        verified_columns = _verify_columns(connection)

        for payload in expected.values():
            connection.execute(UPSERT_SQL, payload)

        _verify_rows(connection, expected, specification_version)

        # Only after all 92 values have been populated and verified.
        connection.execute(
            text(
                """
                ALTER TABLE public.syllabus_concepts
                    ALTER COLUMN chapter_reference SET NOT NULL,
                    ALTER COLUMN chapter_title SET NOT NULL,
                    ALTER COLUMN official_title SET NOT NULL,
                    ALTER COLUMN domain SET NOT NULL,
                    ALTER COLUMN embedding_text SET NOT NULL;
                """
            )
        )

        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        row_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM public.syllabus_concepts
                WHERE specification_code = :code
                  AND specification_version = :version
                  AND is_active = TRUE
                """
            ),
            {
                "code": DEFAULT_SPECIFICATION_CODE,
                "version": specification_version,
            },
        ).scalar_one()

    return {
        "status": "verified",
        "database": database_name,
        "table": TABLE_NAME,
        "active_rows": int(row_count),
        "concepts_verified": EXPECTED_CONCEPT_COUNT,
        "required_columns_verified": verified_columns,
        "newly_required_fields": [
            "chapter_reference",
            "chapter_title",
            "official_title",
            "domain",
            "embedding_text",
        ],
        "other_tables_modified": False,
    }


def main() -> None:
    print(json.dumps(migrate(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()