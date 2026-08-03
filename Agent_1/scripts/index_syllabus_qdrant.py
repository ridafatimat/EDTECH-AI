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


from app.services.cs_concept_catalog import (  # noqa: E402
    CS_CONCEPTS,
    validate_catalog,
)
from app.services.qdrant_syllabus_store import (  # noqa: E402
    QdrantSyllabusConfig,
    QdrantSyllabusStore,
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

    validate_catalog()

    config = (
        QdrantSyllabusConfig.from_environment()
    )

    store = QdrantSyllabusStore(
        config=config
    )

    print("=" * 72)
    print("AQA GCSE COMPUTER SCIENCE — QDRANT INDEXING")
    print("=" * 72)
    print(
        f"Qdrant URL: {config.url}"
    )
    print(
        f"Collection: {config.collection_name}"
    )
    print(
        f"Embedding model: {config.embedding_model}"
    )
    print(
        f"Catalogue concepts: {len(CS_CONCEPTS)}"
    )
    print(
        f"Recreate collection: {args.recreate}"
    )
    print()

    indexed_count = store.index_catalogue(
        recreate=args.recreate,
        batch_size=args.batch_size,
    )

    stored_count = store.count_points()

    print()
    print("Indexing complete.")
    print(
        f"Indexed this run: {indexed_count}"
    )
    print(
        f"Stored in collection: {stored_count}"
    )

    if stored_count != len(CS_CONCEPTS):
        raise RuntimeError(
            "Qdrant point count does not match "
            "the syllabus catalogue."
        )

    print(
        "Catalogue and Qdrant counts match."
    )


if __name__ == "__main__":
    main()