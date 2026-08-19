from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Literal

import numpy as np
from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, text

from app.db.session import get_engine, load_environment
from app.services.embedding_service import (
    TOPIC_EMBEDDING_MODEL,
    embed_texts,
    get_embedding_dimension,
)


Paper = Literal["Paper 1", "Paper 2"]
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "aqa_gcse_computer_science_8525"
DEFAULT_SPECIFICATION_CODE = "8525"
DEFAULT_SPECIFICATION_VERSION = "AQA-8525-v1.2-2022-11-29"

# Keep the namespace already used by the old qdrant_syllabus_store.py so
# existing concept point IDs stay deterministic across the refactor.
QDRANT_POINT_NAMESPACE = uuid.UUID(
    "27a6c1e1-01bb-4fdc-88b8-e89d87c85425"
)


@dataclass(frozen=True)
class FlexiblePattern:
    """One flexible lexical pattern stored in PostgreSQL as JSONB."""

    label: str
    regex: str
    weight: float = 0.82


@dataclass(frozen=True)
class SyllabusConcept:
    """
    Runtime representation of one row from public.syllabus_concepts.

    Its public field names intentionally mirror the old CSConcept shape so
    Agent 1 can be migrated with minimal downstream logic changes.
    """

    concept_id: str
    official_reference: str
    chapter_reference: str
    chapter_title: str
    official_title: str
    label: str
    domain: str
    description: str
    aliases: tuple[str, ...]
    paper: Paper
    source_pages: tuple[int, ...]
    parent_concept_id: str | None = None
    excluded_phrases: tuple[str, ...] = ()
    ambiguous_aliases: tuple[str, ...] = ()
    supporting_context_terms: tuple[str, ...] = ()
    conflicting_context_terms: tuple[str, ...] = ()
    minimum_context_hits: int = 1
    match_patterns: tuple[FlexiblePattern, ...] = ()
    embedding_text: str = ""
    specification_code: str = DEFAULT_SPECIFICATION_CODE
    specification_version: str = DEFAULT_SPECIFICATION_VERSION
    is_active: bool = True


@dataclass(frozen=True)
class SemanticConceptMatch:
    """One nearest-neighbour result returned by Qdrant."""

    concept_id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class SyllabusStoreConfig:
    """Configuration for PostgreSQL-backed syllabus data and Qdrant search."""

    specification_code: str = DEFAULT_SPECIFICATION_CODE
    specification_version: str = DEFAULT_SPECIFICATION_VERSION
    qdrant_url: str = DEFAULT_QDRANT_URL
    qdrant_collection: str = DEFAULT_COLLECTION_NAME
    qdrant_api_key: str | None = None
    qdrant_timeout_seconds: float = 30.0
    qdrant_top_k: int = 20
    qdrant_score_threshold: float | None = None
    embedding_model: str = TOPIC_EMBEDDING_MODEL

    @classmethod
    def from_environment(cls) -> "SyllabusStoreConfig":
        load_environment()

        threshold_text = os.getenv("QDRANT_SCORE_THRESHOLD", "").strip()
        threshold = float(threshold_text) if threshold_text else None

        return cls(
            specification_code=os.getenv(
                "AQA_SPECIFICATION_CODE", DEFAULT_SPECIFICATION_CODE
            ).strip(),
            specification_version=os.getenv(
                "AQA_SPEC_VERSION", DEFAULT_SPECIFICATION_VERSION
            ).strip(),
            qdrant_url=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).strip(),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME
            ).strip(),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip() or None,
            qdrant_timeout_seconds=float(
                os.getenv("QDRANT_TIMEOUT_SECONDS", "30")
            ),
            qdrant_top_k=int(os.getenv("QDRANT_TOP_K_PER_UNIT", "20")),
            qdrant_score_threshold=threshold,
            embedding_model=os.getenv(
                "TOPIC_EMBEDDING_MODEL", TOPIC_EMBEDDING_MODEL
            ).strip(),
        )

    def __post_init__(self) -> None:
        if not self.specification_code:
            raise ValueError("specification_code cannot be empty")
        if not self.specification_version:
            raise ValueError("specification_version cannot be empty")
        if not self.qdrant_url:
            raise ValueError("qdrant_url cannot be empty")
        if not self.qdrant_collection:
            raise ValueError("qdrant_collection cannot be empty")
        if self.qdrant_timeout_seconds <= 0:
            raise ValueError("qdrant_timeout_seconds must be positive")
        if self.qdrant_top_k < 1:
            raise ValueError("qdrant_top_k must be at least 1")
        if (
            self.qdrant_score_threshold is not None
            and not -1.0 <= self.qdrant_score_threshold <= 1.0
        ):
            raise ValueError("qdrant_score_threshold must be between -1 and 1")


class SyllabusStore:
    """
    Single gateway for Agent 1 syllabus storage and lookup.

    PostgreSQL is the authoritative structured store.
    Qdrant is the semantic vector index rebuilt from PostgreSQL.

    This class intentionally does NOT decide whether a topic is primary,
    supporting, ambiguous, accepted or rejected. Existing Module 3 / HITL
    logic continues to make those decisions using metadata returned here.
    """

    TABLE_NAME = "public.syllabus_concepts"

    def __init__(
        self,
        config: SyllabusStoreConfig | None = None,
        *,
        engine: Engine | None = None,
        qdrant_client: QdrantClient | None = None,
    ) -> None:
        self.config = config or SyllabusStoreConfig.from_environment()
        self.engine = engine or get_engine()
        self._qdrant_client = qdrant_client

    # ------------------------------------------------------------------
    # PostgreSQL deserialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json_value(value: Any, default: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            return json.loads(stripped)
        return value

    @classmethod
    def _row_to_concept(cls, row: Any) -> SyllabusConcept:
        data = dict(row._mapping if hasattr(row, "_mapping") else row)

        aliases = cls._json_value(data.get("aliases"), [])
        source_pages = cls._json_value(data.get("source_pages"), [])
        excluded = cls._json_value(data.get("excluded_phrases"), [])
        ambiguous = cls._json_value(data.get("ambiguous_aliases"), [])
        supporting = cls._json_value(data.get("supporting_context_terms"), [])
        conflicting = cls._json_value(data.get("conflicting_context_terms"), [])
        raw_patterns = cls._json_value(data.get("match_patterns"), [])

        patterns = tuple(
            FlexiblePattern(
                label=str(item["label"]),
                regex=str(item["regex"]),
                weight=float(item.get("weight", 0.82)),
            )
            for item in raw_patterns
        )

        return SyllabusConcept(
            concept_id=str(data["concept_id"]),
            official_reference=str(data["official_reference"]),
            chapter_reference=str(data["chapter_reference"]),
            chapter_title=str(data["chapter_title"]),
            official_title=str(data["official_title"]),
            label=str(data["label"]),
            domain=str(data["domain"]),
            description=str(data["description"]),
            aliases=tuple(str(v) for v in aliases),
            paper=str(data["paper"]),
            source_pages=tuple(int(v) for v in source_pages),
            parent_concept_id=(
                str(data["parent_concept_id"])
                if data.get("parent_concept_id")
                else None
            ),
            excluded_phrases=tuple(str(v) for v in excluded),
            ambiguous_aliases=tuple(str(v) for v in ambiguous),
            supporting_context_terms=tuple(str(v) for v in supporting),
            conflicting_context_terms=tuple(str(v) for v in conflicting),
            minimum_context_hits=int(data.get("minimum_context_hits") or 1),
            match_patterns=patterns,
            embedding_text=str(data.get("embedding_text") or ""),
            specification_code=str(
                data.get("specification_code") or DEFAULT_SPECIFICATION_CODE
            ),
            specification_version=str(
                data.get("specification_version")
                or DEFAULT_SPECIFICATION_VERSION
            ),
            is_active=bool(data.get("is_active", True)),
        )

    @staticmethod
    def _base_select() -> str:
        return """
            SELECT
                concept_id,
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
                specification_code,
                specification_version,
                is_active
            FROM public.syllabus_concepts
        """

    def _spec_params(self) -> dict[str, str]:
        return {
            "specification_code": self.config.specification_code,
            "specification_version": self.config.specification_version,
        }

    # ------------------------------------------------------------------
    # PostgreSQL reads: authoritative structured lookup
    # ------------------------------------------------------------------

    def get_concept(self, concept_id: str) -> SyllabusConcept | None:
        concept_id = str(concept_id).strip()
        if not concept_id:
            return None

        query = text(
            self._base_select()
            + """
            WHERE concept_id = :concept_id
              AND specification_code = :specification_code
              AND specification_version = :specification_version
              AND is_active = TRUE
            LIMIT 1
            """
        )

        params = {**self._spec_params(), "concept_id": concept_id}
        with self.engine.connect() as connection:
            row = connection.execute(query, params).first()
        return self._row_to_concept(row) if row else None

    def get_all_concepts(self) -> list[SyllabusConcept]:
        query = text(
            self._base_select()
            + """
            WHERE specification_code = :specification_code
              AND specification_version = :specification_version
              AND is_active = TRUE
            ORDER BY official_reference, concept_id
            """
        )

        with self.engine.connect() as connection:
            rows = connection.execute(query, self._spec_params()).all()
        return [self._row_to_concept(row) for row in rows]

    def get_concepts_by_reference(self, reference: str) -> list[SyllabusConcept]:
        reference = str(reference).strip()
        if not reference:
            return []

        query = text(
            self._base_select()
            + """
            WHERE official_reference = :reference
              AND specification_code = :specification_code
              AND specification_version = :specification_version
              AND is_active = TRUE
            ORDER BY concept_id
            """
        )

        params = {**self._spec_params(), "reference": reference}
        with self.engine.connect() as connection:
            rows = connection.execute(query, params).all()
        return [self._row_to_concept(row) for row in rows]

    def get_concepts_by_paper(self, paper: Paper) -> list[SyllabusConcept]:
        if paper not in ("Paper 1", "Paper 2"):
            raise ValueError("paper must be 'Paper 1' or 'Paper 2'")

        query = text(
            self._base_select()
            + """
            WHERE paper = :paper
              AND specification_code = :specification_code
              AND specification_version = :specification_version
              AND is_active = TRUE
            ORDER BY official_reference, concept_id
            """
        )

        params = {**self._spec_params(), "paper": paper}
        with self.engine.connect() as connection:
            rows = connection.execute(query, params).all()
        return [self._row_to_concept(row) for row in rows]

    def get_technical_terms(self) -> tuple[str, ...]:
        """
        Build reusable vocabulary for preprocessing from PostgreSQL.

        We include concise naming fields and aliases, but not descriptions or
        contextual conflict terms because those are not vocabulary synonyms.
        """

        terms: set[str] = set()
        for concept in self.get_all_concepts():
            for value in (concept.label, concept.official_title, *concept.aliases):
                cleaned = str(value).strip()
                if cleaned:
                    terms.add(cleaned)
        return tuple(sorted(terms, key=str.casefold))

    def count_concepts(self) -> int:
        query = text(
            """
            SELECT COUNT(*)
            FROM public.syllabus_concepts
            WHERE specification_code = :specification_code
              AND specification_version = :specification_version
              AND is_active = TRUE
            """
        )
        with self.engine.connect() as connection:
            return int(connection.execute(query, self._spec_params()).scalar_one())

    # ------------------------------------------------------------------
    # PostgreSQL writes: authoritative storage
    # ------------------------------------------------------------------

    @staticmethod
    def _patterns_to_json(patterns: Iterable[FlexiblePattern]) -> str:
        return json.dumps(
            [
                {"label": p.label, "regex": p.regex, "weight": p.weight}
                for p in patterns
            ]
        )

    def upsert_concept(self, concept: SyllabusConcept) -> None:
        """
        Insert/update one concept in PostgreSQL.

        PostgreSQL is written first. Qdrant is intentionally not changed in
        this transaction; call sync_concept_to_qdrant() after a successful DB
        update when the semantic representation also needs refreshing.
        """

        query = text(
            """
            INSERT INTO public.syllabus_concepts (
                concept_id,
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
                specification_code,
                specification_version,
                is_active
            ) VALUES (
                :concept_id,
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
                :specification_code,
                :specification_version,
                :is_active
            )
            ON CONFLICT (concept_id) DO UPDATE SET
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
                specification_code = EXCLUDED.specification_code,
                specification_version = EXCLUDED.specification_version,
                is_active = EXCLUDED.is_active,
                updated_at = NOW()
            """
        )

        params = {
            "concept_id": concept.concept_id,
            "official_reference": concept.official_reference,
            "chapter_reference": concept.chapter_reference,
            "chapter_title": concept.chapter_title,
            "official_title": concept.official_title,
            "label": concept.label,
            "domain": concept.domain,
            "description": concept.description,
            "aliases": json.dumps(list(concept.aliases)),
            "paper": concept.paper,
            "source_pages": json.dumps(list(concept.source_pages)),
            "parent_concept_id": concept.parent_concept_id,
            "excluded_phrases": json.dumps(list(concept.excluded_phrases)),
            "ambiguous_aliases": json.dumps(list(concept.ambiguous_aliases)),
            "supporting_context_terms": json.dumps(
                list(concept.supporting_context_terms)
            ),
            "conflicting_context_terms": json.dumps(
                list(concept.conflicting_context_terms)
            ),
            "minimum_context_hits": concept.minimum_context_hits,
            "match_patterns": self._patterns_to_json(concept.match_patterns),
            "embedding_text": concept.embedding_text,
            "specification_code": concept.specification_code,
            "specification_version": concept.specification_version,
            "is_active": concept.is_active,
        }

        with self.engine.begin() as connection:
            connection.execute(query, params)

    # ------------------------------------------------------------------
    # Qdrant: semantic index derived from PostgreSQL
    # ------------------------------------------------------------------

    @property
    def qdrant(self) -> QdrantClient:
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.qdrant_timeout_seconds,
            )
        return self._qdrant_client

    @staticmethod
    def point_id_for_concept(concept_id: str) -> str:
        return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, concept_id))

    def qdrant_collection_exists(self) -> bool:
        return self.qdrant.collection_exists(
            collection_name=self.config.qdrant_collection
        )

    def ensure_qdrant_collection(self, *, recreate: bool = False) -> None:
        exists = self.qdrant_collection_exists()

        if recreate and exists:
            self.qdrant.delete_collection(
                collection_name=self.config.qdrant_collection
            )
            exists = False

        if exists:
            collection = self.qdrant.get_collection(
                collection_name=self.config.qdrant_collection
            )
            vectors_config = collection.config.params.vectors
            if not isinstance(vectors_config, models.VectorParams):
                raise RuntimeError(
                    "Expected one unnamed dense vector in syllabus collection."
                )
            expected = get_embedding_dimension(self.config.embedding_model)
            if vectors_config.size != expected:
                raise RuntimeError(
                    "Qdrant vector size mismatch: "
                    f"collection={vectors_config.size}, expected={expected}. "
                    "Rebuild the collection after confirming the embedding model."
                )
            return

        self.qdrant.create_collection(
            collection_name=self.config.qdrant_collection,
            vectors_config=models.VectorParams(
                size=get_embedding_dimension(self.config.embedding_model),
                distance=models.Distance.COSINE,
            ),
        )

    def _concept_payload(self, concept: SyllabusConcept) -> dict[str, Any]:
        """
        Qdrant keeps enough payload for result identification/display.

        PostgreSQL remains authoritative for rules and full structured data.
        """

        return {
            "concept_id": concept.concept_id,
            "board": "AQA",
            "qualification": "GCSE",
            "subject": "Computer Science",
            "specification_code": concept.specification_code,
            "specification_version": concept.specification_version,
            "official_reference": concept.official_reference,
            "chapter_reference": concept.chapter_reference,
            "chapter_title": concept.chapter_title,
            "official_title": concept.official_title,
            "label": concept.label,
            "paper": concept.paper,
            "embedding_model": self.config.embedding_model,
            "embedding_text": concept.embedding_text,
        }

    def sync_concept_to_qdrant(self, concept_id: str) -> None:
        concept = self.get_concept(concept_id)
        if concept is None:
            raise KeyError(f"Unknown or inactive syllabus concept: {concept_id}")

        self.ensure_qdrant_collection(recreate=False)
        vector = embed_texts(
            [concept.embedding_text], model_name=self.config.embedding_model
        )
        if vector.shape[0] != 1:
            raise RuntimeError("Expected exactly one concept embedding")

        self.qdrant.upsert(
            collection_name=self.config.qdrant_collection,
            points=[
                models.PointStruct(
                    id=self.point_id_for_concept(concept.concept_id),
                    vector=vector[0].tolist(),
                    payload=self._concept_payload(concept),
                )
            ],
            wait=True,
        )

    def sync_qdrant(
        self,
        *,
        recreate: bool = False,
        batch_size: int = 32,
    ) -> int:
        """Rebuild/upsert Qdrant directly from PostgreSQL concepts."""

        concepts = self.get_all_concepts()
        if not concepts:
            raise RuntimeError("PostgreSQL returned zero active syllabus concepts")

        self.ensure_qdrant_collection(recreate=recreate)

        embeddings = embed_texts(
            [concept.embedding_text for concept in concepts],
            model_name=self.config.embedding_model,
            batch_size=batch_size,
        )
        if len(embeddings) != len(concepts):
            raise RuntimeError(
                "Number of embeddings does not match PostgreSQL concept count"
            )

        points = [
            models.PointStruct(
                id=self.point_id_for_concept(concept.concept_id),
                vector=embedding.tolist(),
                payload=self._concept_payload(concept),
            )
            for concept, embedding in zip(concepts, embeddings, strict=True)
        ]

        self.qdrant.upload_points(
            collection_name=self.config.qdrant_collection,
            points=points,
            batch_size=batch_size,
            parallel=1,
            max_retries=3,
            wait=True,
        )
        return len(points)

    def semantic_search(
        self,
        text_value: str,
        *,
        top_k: int | None = None,
    ) -> list[SemanticConceptMatch]:
        cleaned = str(text_value).strip()
        if not cleaned:
            return []

        vector = embed_texts(
            [cleaned], model_name=self.config.embedding_model
        )
        if vector.shape[0] != 1:
            return []
        return self.search_by_vectors(vector, top_k=top_k)[0]

    def search_by_vectors(
        self,
        query_vectors: np.ndarray,
        *,
        top_k: int | None = None,
    ) -> list[list[SemanticConceptMatch]]:
        if query_vectors.ndim != 2:
            raise ValueError("query_vectors must be a 2D NumPy array")
        if query_vectors.size == 0:
            return []
        if not self.qdrant_collection_exists():
            raise RuntimeError(
                "Qdrant syllabus collection does not exist. "
                "Run sync_qdrant() first."
            )

        result_limit = top_k or self.config.qdrant_top_k
        all_matches: list[list[SemanticConceptMatch]] = []

        for vector in query_vectors:
            response = self.qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=vector.tolist(),
                limit=result_limit,
                score_threshold=self.config.qdrant_score_threshold,
                with_payload=True,
                with_vectors=False,
            )

            matches: list[SemanticConceptMatch] = []
            for point in response.points:
                payload = dict(point.payload or {})
                concept_id = str(payload.get("concept_id", "")).strip()
                if concept_id:
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
        concept_ids: Iterable[str],
    ) -> dict[str, np.ndarray]:
        unique_ids = list(
            dict.fromkeys(str(value).strip() for value in concept_ids if str(value).strip())
        )
        if not unique_ids:
            return {}

        points = self.qdrant.retrieve(
            collection_name=self.config.qdrant_collection,
            ids=[self.point_id_for_concept(value) for value in unique_ids],
            with_payload=True,
            with_vectors=True,
        )

        vectors: dict[str, np.ndarray] = {}
        for point in points:
            payload = dict(point.payload or {})
            concept_id = str(payload.get("concept_id", "")).strip()
            raw_vector = point.vector
            if not concept_id or raw_vector is None:
                continue
            if isinstance(raw_vector, dict):
                raise RuntimeError("Expected a single unnamed dense vector")
            vectors[concept_id] = np.asarray(raw_vector, dtype=np.float32)
        return vectors

    def count_qdrant_points(self) -> int:
        if not self.qdrant_collection_exists():
            return 0
        result = self.qdrant.count(
            collection_name=self.config.qdrant_collection,
            exact=True,
        )
        return int(result.count)

    def _qdrant_concept_ids(self) -> set[str]:
        if not self.qdrant_collection_exists():
            return set()

        concept_ids: set[str] = set()
        offset = None
        while True:
            points, next_offset = self.qdrant.scroll(
                collection_name=self.config.qdrant_collection,
                limit=256,
                offset=offset,
                with_payload=["concept_id"],
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                concept_id = str(payload.get("concept_id", "")).strip()
                if concept_id:
                    concept_ids.add(concept_id)
            if next_offset is None:
                break
            offset = next_offset
        return concept_ids

    def verify_qdrant_sync(self) -> dict[str, Any]:
        postgres_ids = {concept.concept_id for concept in self.get_all_concepts()}
        qdrant_ids = self._qdrant_concept_ids()

        missing = sorted(postgres_ids - qdrant_ids)
        extra = sorted(qdrant_ids - postgres_ids)

        return {
            "status": "verified" if not missing and not extra else "mismatch",
            "postgres_count": len(postgres_ids),
            "qdrant_count": len(qdrant_ids),
            "missing_in_qdrant": missing,
            "extra_in_qdrant": extra,
        }


@lru_cache(maxsize=1)
def get_syllabus_store() -> SyllabusStore:
    """Return one lazily-created store instance per Python process."""

    return SyllabusStore()


def clear_syllabus_store_cache() -> None:
    """Useful in tests after environment/configuration changes."""

    get_syllabus_store.cache_clear()
