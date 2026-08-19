from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from app.services import embedding_service as existing_embedding_service


class Agent1EditMemoryEmbeddingAdapter:
    """
    Thin read-only adapter around Agent 1's EXISTING embedding service.

    Safety properties:
    - does not load a new embedding implementation;
    - does not modify existing embedding_service.py;
    - does not change Module 3, Qdrant, Groq, PostgreSQL, or Streamlit;
    - uses the currently configured topic embedding model when available;
    - otherwise falls back to the existing embedding service default.

    The contextual matcher only requires:
        embed_texts(texts) -> sequence of vectors
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
    ) -> None:
        configured_model = (
            str(model_name).strip()
            if model_name is not None
            else ""
        )

        if not configured_model:
            configured_model = os.getenv(
                "DETECTED_TOPIC_EDIT_EMBEDDING_MODEL",
                "",
            ).strip()

        if not configured_model:
            configured_model = os.getenv(
                "TOPIC_EMBEDDING_MODEL",
                "",
            ).strip()

        if not configured_model:
            configured_model = str(
                getattr(
                    existing_embedding_service,
                    "DEFAULT_EMBEDDING_MODEL",
                    "",
                )
            ).strip()

        if not configured_model:
            raise RuntimeError(
                "No embedding model could be resolved. Set "
                "DETECTED_TOPIC_EDIT_EMBEDDING_MODEL or "
                "TOPIC_EMBEDDING_MODEL, or ensure the existing "
                "embedding service defines DEFAULT_EMBEDDING_MODEL."
            )

        self.model_name = configured_model
        self.batch_size = int(batch_size)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        cleaned = [
            str(text).strip()
            for text in texts
        ]

        if any(not text for text in cleaned):
            raise ValueError(
                "Contextual edit-memory embeddings require non-empty texts."
            )

        if not cleaned:
            return []

        vectors = existing_embedding_service.embed_texts(
            cleaned,
            model_name=self.model_name,
            batch_size=self.batch_size,
        )

        array = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError(
                "Existing embedding service returned an unexpected shape: "
                f"{array.shape!r}"
            )

        if array.shape[0] != len(cleaned):
            raise ValueError(
                "Existing embedding service returned the wrong number "
                "of embedding vectors."
            )

        return array.tolist()
