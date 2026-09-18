"""Local knowledge indexing pipeline: documents -> chunking -> embedding -> SQLite."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.core.logging import get_logger
from app.rag.chunker import MarkdownChunker
from app.rag.embeddings import EmbeddingEngine, vector_to_bytes
from app.storage.database import DatabaseManager, get_db_manager
from app.storage.repositories import RAGRepository

logger = get_logger("rag.indexer")


class KnowledgeIndexer:
    """Indexes markdown lore and documentation into SQLite vector storage."""

    def __init__(
        self,
        db: DatabaseManager | None = None,
        embedding_engine: EmbeddingEngine | None = None,
    ):
        self.db = db or get_db_manager()
        self.rag_repo = RAGRepository(self.db)
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.chunker = MarkdownChunker()

    def index_file(self, file_path: Path | str, category: str = "lore") -> int:
        """Parse, chunk, embed, and store a single document."""
        p = Path(file_path)
        if not p.exists():
            logger.warning("rag.file_not_found", path=str(p))
            return 0

        text = p.read_text(encoding="utf-8")
        title = p.stem.replace("_", " ").title()
        doc_id = f"doc_{p.stem}"

        self.rag_repo.insert_document(
            document_id=doc_id,
            title=title,
            category=category,
            source_path=str(p),
        )

        # Clear older chunks for this document if re-indexing
        self.rag_repo.clear_document_chunks(doc_id)

        chunks = self.chunker.chunk_document(text)
        if not chunks:
            return 0

        chunk_texts = [c.content for c in chunks]
        embeddings = self.embedding_engine.embed_texts(chunk_texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = f"{doc_id}_c{chunk.chunk_index}"
            emb_bytes = vector_to_bytes(emb.tolist())
            metadata = {"heading": chunk.heading, "char_count": chunk.char_count}
            self.rag_repo.insert_chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=emb_bytes,
                metadata=metadata,
            )

        logger.info("rag.indexed_file", file=p.name, chunk_count=len(chunks))
        return len(chunks)

    def index_directory(self, dir_path: Path | str, category: str = "lore") -> int:
        """Scan directory and index all .md files."""
        folder = Path(dir_path)
        if not folder.exists() or not folder.is_dir():
            return 0

        total_chunks = 0
        for md_file in folder.glob("*.md"):
            total_chunks += self.index_file(md_file, category=category)
        return total_chunks
