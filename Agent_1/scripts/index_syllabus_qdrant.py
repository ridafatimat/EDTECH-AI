from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.services.syllabus_store import (  # noqa: E402
    get_syllabus_store,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed the AQA GCSE Computer Science syllabus "
            "catalogue with MiniLM and upload it to Qdrant."
        )
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Delete and rebuild the existing Qdrant collection."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Embedding and upload batch size. Default: 32."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be at least 1."
        )

    store = get_syllabus_store()
    config = store.config

    postgres_count = store.count_concepts()
    if postgres_count < 1:
        raise RuntimeError(
            "PostgreSQL returned zero active syllabus concepts."
        )

    print("=" * 72)
    print("AQA GCSE COMPUTER SCIENCE — QDRANT INDEXING")
    print("=" * 72)
    print(
        f"Qdrant URL: {config.qdrant_url}"
    )
    print(
        f"Collection: {config.qdrant_collection}"
    )
    print(
        f"Embedding model: {config.embedding_model}"
    )
    print(
        f"PostgreSQL concepts: {postgres_count}"
    )
    print(
        f"Recreate collection: {args.recreate}"
    )
    print()

    indexed_count = store.sync_qdrant(
        recreate=args.recreate,
        batch_size=args.batch_size,
    )

    stored_count = store.count_qdrant_points()
    verification = store.verify_qdrant_sync()

    print()
    print("Indexing complete.")
    print(
        f"Indexed this run: {indexed_count}"
    )
    print(
        f"Stored in collection: {stored_count}"
    )

    if verification.get("status") != "verified":
        raise RuntimeError(
            "Qdrant contents do not match the PostgreSQL syllabus concepts. "
            f"Missing: {verification.get('missing_in_qdrant', [])}; "
            f"Extra: {verification.get('extra_in_qdrant', [])}."
        )

    print(
        "PostgreSQL syllabus concepts and Qdrant are in sync."
    )


if __name__ == "__main__":
    main()