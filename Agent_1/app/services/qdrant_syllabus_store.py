from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models

from app.services.cs_concept_catalog import (
    CS_CONCEPTS,
    CSConcept,
)
from app.services.embedding_service import (
    TOPIC_EMBEDDING_MODEL,
    embed_texts,
    get_embedding_dimension,
)


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = (
    "aqa_gcse_computer_science_8525"
)

# Fixed namespace makes concept IDs deterministic across repeated indexing.
QDRANT_POINT_NAMESPACE = uuid.UUID(
    "27a6c1e1-01bb-4fdc-88b8-e89d87c85425"
)


@dataclass(frozen=True)
class QdrantSyllabusConfig:
    """
    Configuration for the local Qdrant syllabus collection.
    """

    url: str = DEFAULT_QDRANT_URL
    collection_name: str = DEFAULT_COLLECTION_NAME
    api_key: str | None = None
    timeout_seconds: float = 30.0
    top_k_per_unit: int = 20
    score_threshold: float | None = None
    embedding_model: str = TOPIC_EMBEDDING_MODEL

    @classmethod
    def from_environment(
        cls,
    ) -> "QdrantSyllabusConfig":
        api_key = (
            os.getenv("QDRANT_API_KEY", "").strip()
            or None
        )

        threshold_text = os.getenv(
            "QDRANT_SCORE_THRESHOLD",
            "",
        ).strip()

        score_threshold = (
            float(threshold_text)
            if threshold_text
            else None
        )

        return cls(
            url=os.getenv(
                "QDRANT_URL",
                DEFAULT_QDRANT_URL,
            ).strip(),
            collection_name=os.getenv(
                "QDRANT_COLLECTION",
                DEFAULT_COLLECTION_NAME,
            ).strip(),
            api_key=api_key,
            timeout_seconds=float(
                os.getenv(
                    "QDRANT_TIMEOUT_SECONDS",
                    "30",
                )
            ),
            top_k_per_unit=int(
                os.getenv(
                    "QDRANT_TOP_K_PER_UNIT",
                    "20",
                )
            ),
            score_threshold=score_threshold,
            embedding_model=os.getenv(
                "TOPIC_EMBEDDING_MODEL",
                TOPIC_EMBEDDING_MODEL,
            ).strip(),
        )

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError(
                "Qdrant URL cannot be empty."
            )

        if not self.collection_name:
            raise ValueError(
                "Qdrant collection name cannot be empty."
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "Qdrant timeout must be positive."
            )

        if self.top_k_per_unit < 1:
            raise ValueError(
                "Qdrant top_k_per_unit must be at least 1."
            )

        if (
            self.score_threshold is not None
            and not -1.0
            <= self.score_threshold
            <= 1.0
        ):
            raise ValueError(
                "Qdrant score threshold must be "
                "between -1 and 1."
            )


@dataclass(frozen=True)
class SemanticConceptMatch:
    """
    One syllabus concept returned by Qdrant.
    """

    concept_id: str
    score: float
    payload: dict[str, Any]


class QdrantSyllabusStore:
    """
    Store and retrieve official syllabus concept embeddings in Qdrant.

    Qdrant performs semantic nearest-neighbour search. Existing Module 3
    Python code remains responsible for:

    - lexical alias matching
    - contextual disambiguation
    - evidence quality
    - confidence
    - salience
    - primary/supporting classification
    """

    def __init__(
        self,
        config: QdrantSyllabusConfig | None = None,
    ) -> None:
        self.config = (
            config
            or QdrantSyllabusConfig.from_environment()
        )

        self.client = QdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
        )

    @staticmethod
    def point_id_for_concept(
        concept_id: str,
    ) -> str:
        """
        Generate a stable UUID point ID for a concept.
        """

        return str(
            uuid.uuid5(
                QDRANT_POINT_NAMESPACE,
                concept_id,
            )
        )

    def collection_exists(self) -> bool:
        return self.client.collection_exists(
            collection_name=(
                self.config.collection_name
            )
        )

    def ensure_collection(
        self,
        *,
        recreate: bool = False,
    ) -> None:
        """
        Create the syllabus collection if required.

        Recreate should be used when:
        - the embedding model changes;
        - vector size changes;
        - the catalogue needs a clean rebuild.
        """

        exists = self.collection_exists()

        if recreate and exists:
            self.client.delete_collection(
                collection_name=(
                    self.config.collection_name
                )
            )
            exists = False

        if exists:
            self._validate_existing_collection()
            return

        vector_size = get_embedding_dimension(
            self.config.embedding_model
        )

        self.client.create_collection(
            collection_name=(
                self.config.collection_name
            ),
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _validate_existing_collection(
        self,
    ) -> None:
        """
        Check that the existing collection uses the expected vector size.
        """

        collection = self.client.get_collection(
            collection_name=(
                self.config.collection_name
            )
        )

        vectors_config = (
            collection.config.params.vectors
        )

        if not isinstance(
            vectors_config,
            models.VectorParams,
        ):
            raise RuntimeError(
                "The Qdrant collection does not use a single "
                "unnamed dense vector."
            )

        expected_size = get_embedding_dimension(
            self.config.embedding_model
        )

        if vectors_config.size != expected_size:
            raise RuntimeError(
                "Qdrant vector dimension does not match the "
                "configured embedding model. "
                f"Collection size: {vectors_config.size}; "
                f"expected size: {expected_size}. "
                "Re-index with --recreate."
            )

    def index_catalogue(
        self,
        *,
        recreate: bool = False,
        batch_size: int = 32,
    ) -> int:
        """
        Embed and upload the complete official syllabus catalogue.
        """

        self.ensure_collection(
            recreate=recreate
        )

        concepts = list(CS_CONCEPTS)

        concept_texts = [
            concept.embedding_text
            for concept in concepts
        ]

        embeddings = embed_texts(
            concept_texts,
            model_name=self.config.embedding_model,
            batch_size=batch_size,
        )

        if len(embeddings) != len(concepts):
            raise RuntimeError(
                "Number of catalogue embeddings does not "
                "match number of concepts."
            )

        points = [
            models.PointStruct(
                id=self.point_id_for_concept(
                    concept.concept_id
                ),
                vector=embedding.tolist(),
                payload=self._concept_payload(
                    concept
                ),
            )
            for concept, embedding in zip(
                concepts,
                embeddings,
                strict=True,
            )
        ]

        self.client.upload_points(
            collection_name=(
                self.config.collection_name
            ),
            points=points,
            batch_size=batch_size,
            parallel=1,
            max_retries=3,
            wait=True,
        )

        return len(points)

    def search_by_vectors(
        self,
        query_vectors: np.ndarray,
        *,
        top_k: int | None = None,
    ) -> list[list[SemanticConceptMatch]]:
        """
        Search Qdrant once for each semantic-unit vector.
        """

        if query_vectors.ndim != 2:
            raise ValueError(
                "query_vectors must be a 2D NumPy array."
            )

        if query_vectors.size == 0:
            return []

        if not self.collection_exists():
            raise RuntimeError(
                "Qdrant syllabus collection does not exist. "
                "Run scripts/index_syllabus_qdrant.py first."
            )

        result_limit = (
            top_k
            or self.config.top_k_per_unit
        )

        all_matches: list[
            list[SemanticConceptMatch]
        ] = []

        for vector in query_vectors:
            response = self.client.query_points(
                collection_name=(
                    self.config.collection_name
                ),
                query=vector.tolist(),
                limit=result_limit,
                score_threshold=(
                    self.config.score_threshold
                ),
                with_payload=True,
                with_vectors=False,
            )

            matches: list[
                SemanticConceptMatch
            ] = []

            for point in response.points:
                payload = dict(
                    point.payload or {}
                )

                concept_id = str(
                    payload.get(
                        "concept_id",
                        "",
                    )
                ).strip()

                if not concept_id:
                    continue

                matches.append(
                    SemanticConceptMatch(
                        concept_id=concept_id,
                        score=float(point.score),
                        payload=payload,
                    )
                )

            all_matches.append(matches)

        return all_matches

    def retrieve_concept_vectors(
        self,
        concept_ids: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Retrieve vectors for a limited set of lexical candidates.

        This is not used to download the complete collection. It is used only
        when an exact lexical candidate was not present in Qdrant's semantic
        top-k results, allowing the existing confidence calculation to retain
        its semantic score.
        """

        unique_ids = list(
            dict.fromkeys(
                concept_id
                for concept_id in concept_ids
                if concept_id
            )
        )

        if not unique_ids:
            return {}

        point_ids = [
            self.point_id_for_concept(
                concept_id
            )
            for concept_id in unique_ids
        ]

        points = self.client.retrieve(
            collection_name=(
                self.config.collection_name
            ),
            ids=point_ids,
            with_payload=True,
            with_vectors=True,
        )

        vectors: dict[str, np.ndarray] = {}

        for point in points:
            payload = dict(
                point.payload or {}
            )

            concept_id = str(
                payload.get(
                    "concept_id",
                    "",
                )
            ).strip()

            if not concept_id:
                continue

            raw_vector = point.vector

            if isinstance(raw_vector, dict):
                raise RuntimeError(
                    "Expected a single unnamed dense vector."
                )

            if raw_vector is None:
                continue

            vectors[concept_id] = np.asarray(
                raw_vector,
                dtype=np.float32,
            )

        return vectors

    def count_points(self) -> int:
        result = self.client.count(
            collection_name=(
                self.config.collection_name
            ),
            exact=True,
        )

        return int(result.count)

    @staticmethod
    def _concept_payload(
        concept: CSConcept,
    ) -> dict[str, Any]:
        """
        Build metadata stored beside each syllabus embedding.
        """

        return {
            "concept_id": concept.concept_id,
            "board": "AQA",
            "qualification": "GCSE",
            "subject": "Computer Science",
            "specification_code": "8525",
            "official_reference": (
                concept.official_reference
            ),
            "chapter_reference": (
                concept.chapter_reference
            ),
            "chapter_title": (
                concept.chapter_title
            ),
            "official_title": (
                concept.official_title
            ),
            "label": concept.label,
            "domain": concept.domain,
            "description": concept.description,
            "aliases": list(concept.aliases),
            "paper": concept.paper,
            "source_pages": list(
                concept.source_pages
            ),
            "parent_concept_id": (
                concept.parent_concept_id
            ),
            "embedding_model": (
                TOPIC_EMBEDDING_MODEL
            ),
            "embedding_text": (
                concept.embedding_text
            ),
        }