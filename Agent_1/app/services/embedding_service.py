from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


# Module 2 and Module 3 both use MiniLM.
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNKING_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
TOPIC_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    """
    Load and cache a SentenceTransformer model.

    The same MiniLM model is currently used for:
    - Module 2 semantic chunking
    - Module 3 syllabus topic retrieval
    - Qdrant syllabus indexing

    The model is cached so it is loaded only once per Python process.
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
    Generate normalized embeddings for multiple text values.

    Normalized vectors allow cosine similarity to be represented by
    a dot product. Qdrant also stores these vectors using cosine distance.
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

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
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


def get_embedding_dimension(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> int:
    """
    Return the output vector dimension of an embedding model.

    MiniLM-L6-v2 currently produces 384-dimensional embeddings.
    The value is obtained from the loaded model rather than hardcoded.
    """

    model = get_embedding_model(
        model_name
    )

    dimension = (
        model.get_sentence_embedding_dimension()
    )

    if dimension is None or dimension < 1:
        raise RuntimeError(
            "Could not determine embedding dimension for "
            f"{model_name}."
        )

    return int(dimension)