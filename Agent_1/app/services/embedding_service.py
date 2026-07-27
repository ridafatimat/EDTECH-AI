from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


# Main/high-accuracy embedding model selected for curriculum retrieval/mapping.
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Lightweight model used specifically for Module 2 semantic chunking.
CHUNKING_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    """
    Load and cache embedding models.

    Multiple models may be cached in the same process:
    - MiniLM for fast transcript chunking
    - Qwen3-Embedding-0.6B for curriculum retrieval/mapping
    """

    device = os.getenv("EMBEDDING_DEVICE")

    kwargs: dict[str, str] = {}

    if device:
        kwargs["device"] = device

    return SentenceTransformer(
        model_name,
        **kwargs,
    )


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate normalized embeddings for multiple pieces of text.

    Normalized embeddings allow cosine similarity to be calculated
    efficiently using a dot product.
    """

    cleaned_texts = [
        str(text).strip()
        for text in texts
        if str(text).strip()
    ]

    if not cleaned_texts:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    model = get_embedding_model(
        model_name
    )

    embeddings = model.encode(
        cleaned_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )