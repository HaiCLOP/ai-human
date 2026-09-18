"""In-memory vectorized NumPy retrieval over local SQLite chunk embeddings."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingEngine, bytes_to_vector
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import DocumentChunkRecord, RAGRepository

logger = get_logger("rag.retriever")


class LocalRAGRetriever:
    """Performs sub-millisecond vectorized cosine similarity search over local chunks."""

    def __init__(
        self,
        db: DatabaseManager | None = None,
        embedding_engine: EmbeddingEngine | None = None,
    ):
        self.db = db or get_db_manager()
        self.rag_repo = RAGRepository(self.db)
        self.embedding_engine = embedding_engine or EmbeddingEngine()

        self._chunks: list[DocumentChunkRecord] = []
        self._matrix: np.ndarray | None = None  # Shape (N, 384)
        self._is_loaded: bool = False

    def reload_index(self) -> None:
        """Load all chunks and vectors from SQLite into in-memory matrix."""
        chunks = self.rag_repo.get_all_chunks()
        if not chunks:
            self._chunks = []
            self._matrix = None
            self._is_loaded = True
            return

        vectors: list[np.ndarray] = []
        valid_chunks: list[DocumentChunkRecord] = []

        for c in chunks:
            try:
                vec = bytes_to_vector(c.embedding)
                vectors.append(vec)
                valid_chunks.append(c)
            except Exception as e:
                logger.warning("rag.failed_to_parse_chunk_vector", chunk_id=c.chunk_id, error=str(e))

        if vectors:
            self._matrix = np.vstack(vectors)  # (N, 384)
            self._chunks = valid_chunks
        else:
            self._matrix = None
            self._chunks = []

        self._is_loaded = True
        logger.info("rag.index_loaded_in_memory", total_chunks=len(self._chunks))

    def retrieve(
        self,
        query: str,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        """Query index and return top-K relevant chunks exceeding similarity threshold."""
        if not self._is_loaded or self._matrix is None:
            self.reload_index()

        if self._matrix is None or len(self._chunks) == 0:
            return []

        settings = get_settings()
        sim_threshold = threshold if threshold is not None else settings.RAG_SIMILARITY_THRESHOLD
        k = top_k if top_k is not None else settings.RAG_TOP_K

        start_time = time.perf_counter()
        query_vec = self.embedding_engine.embed_query(query)

        # Vectorized cosine similarity (pre-normalized dot product)
        scores = np.dot(self._matrix, query_vec)  # (N,)

        # Filter indices by threshold
        qualifying_indices = np.where(scores >= sim_threshold)[0]
        if len(qualifying_indices) == 0:
            logger.info("rag.no_chunks_above_threshold", max_score=float(np.max(scores)) if len(scores) > 0 else 0.0)
            return []

        # Sort qualifying indices by score descending
        sorted_indices = qualifying_indices[np.argsort(-scores[qualifying_indices])]
        top_indices = sorted_indices[:k]

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "rag.retrieval_complete",
            matched_count=len(top_indices),
            duration_ms=round(duration_ms, 2),
            best_score=round(float(scores[top_indices[0]]), 4),
        )

        return [self._chunks[idx].content for idx in top_indices]
