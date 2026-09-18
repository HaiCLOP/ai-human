"""Historical conversation window chunking and local semantic RAG indexing."""

from __future__ import annotations

import json
from typing import Any

from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingEngine, vector_to_bytes
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import RAGRepository

logger = get_logger("learning.rag_indexer")


class HistoricalRAGIndexer:
    """Indexes historical conversation windows and memory candidates for semantic retrieval."""

    def __init__(
        self,
        db: DatabaseManager | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        window_size: int = 5,
        window_stride: int = 3,
    ):
        self.db = db or get_db_manager()
        self.rag_repo = RAGRepository(self.db)
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.window_size = window_size
        self.window_stride = window_stride

    def index_conversations(
        self,
        conversations: list[dict[str, Any]],
        all_messages: list[dict[str, Any]],
    ) -> int:
        """Create conversation window chunks, compute embeddings locally, and persist to SQLite."""
        # Clear existing historical conversation chunks
        doc_id = "doc_historical_conversations"
        self.rag_repo.insert_document(
            document_id=doc_id,
            title="Historical Conversations Archive",
            category="historical_conversations",
            source_path="data/instagram_history",
        )
        self.rag_repo.clear_document_chunks(doc_id)

        # Group messages by conversation
        by_conv: dict[str, list[dict[str, Any]]] = {}
        for m in all_messages:
            cid = m.get("conversation_id", "")
            if cid not in by_conv:
                by_conv[cid] = []
            by_conv[cid].append(m)

        chunks_to_embed: list[str] = []
        chunk_metadatas: list[dict[str, Any]] = []

        for cid, msgs in by_conv.items():
            # Filter to text messages with content
            text_msgs = [m for m in msgs if m.get("text", "").strip()]
            if not text_msgs:
                continue

            # Prioritize recent messages if the thread is very large (focus on recent context)
            if len(text_msgs) > 1500:
                text_msgs = text_msgs[-1500:]

            conv_chunks: list[tuple[str, dict[str, Any]]] = []
            for i in range(0, len(text_msgs), self.window_stride):
                window = text_msgs[i : i + self.window_size]
                if len(window) < 2:
                    continue

                total_chars = sum(len(wm.get("text", "").strip()) for wm in window)
                if total_chars < 30:  # Skip empty or trivial exchanges
                    continue

                lines: list[str] = []
                for wm in window:
                    sender = wm.get("sender_name", "Unknown")
                    text = wm.get("text", "").strip()
                    lines.append(f"{sender}: {text}")

                chunk_text = "\n".join(lines)
                meta = {
                    "conversation_id": cid,
                    "contact_id": window[0].get("contact_id", ""),
                    "start_ts": window[0].get("timestamp_ms", 0),
                    "end_ts": window[-1].get("timestamp_ms", 0),
                }
                conv_chunks.append((chunk_text, meta))

            # Limit to most relevant and recent 150 chunks per conversation
            if len(conv_chunks) > 150:
                conv_chunks = conv_chunks[-150:]

            for text, meta in conv_chunks:
                chunks_to_embed.append(text)
                chunk_metadatas.append(meta)

        if not chunks_to_embed:
            return 0

        logger.info("learning.rag_embedding_started", total_chunks=len(chunks_to_embed))
        embeddings = self.embedding_engine.embed_texts(chunks_to_embed)

        for idx, (text, emb, meta) in enumerate(zip(chunks_to_embed, embeddings, chunk_metadatas)):
            chunk_id = f"{doc_id}_w{idx}"
            emb_bytes = vector_to_bytes(emb.tolist())
            self.rag_repo.insert_chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                chunk_index=idx,
                content=text,
                embedding=emb_bytes,
                metadata=meta,
            )

        logger.info("learning.rag_indexing_completed", indexed_count=len(chunks_to_embed))
        return len(chunks_to_embed)
