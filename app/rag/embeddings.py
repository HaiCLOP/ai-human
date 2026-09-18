"""Local ONNX-based embedding engine and vector serialization utilities."""

from __future__ import annotations

import struct
from typing import Sequence

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("rag.embeddings")

VECTOR_DIM: int = 384  # BAAI/bge-small-en-v1.5 dimensionality


def vector_to_bytes(vector: Sequence[float]) -> bytes:
    """Pack float sequence into binary IEEE 754 32-bit floats."""
    return struct.pack(f"{len(vector)}f", *vector)


def bytes_to_vector(blob: bytes, dim: int = VECTOR_DIM) -> np.ndarray:
    """Unpack binary blob into normalized NumPy float32 array."""
    floats = struct.unpack(f"{dim}f", blob)
    vec = np.array(floats, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


class EmbeddingEngine:
    """Generates local embeddings using fastembed ONNX Runtime."""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info("rag.loading_embedding_model", model=self.model_name)
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
                logger.info("rag.embedding_model_loaded", model=self.model_name)
            except Exception as e:
                logger.error("rag.embedding_model_load_failed", error=str(e))
                raise
        return self._model

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Generate normalized float32 embeddings for a batch of texts."""
        if not texts:
            return []
        model = self._get_model()
        # fastembed returns generator of numpy arrays
        embeddings_gen = model.embed(texts)
        result: list[np.ndarray] = []
        for emb in embeddings_gen:
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            result.append(vec)
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """Generate normalized float32 embedding for a single query."""
        results = self.embed_texts([query])
        return results[0]
