"""Unit and integration tests for Local RAG subsystem."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from app.rag.chunker import MarkdownChunker
from app.rag.embeddings import EmbeddingEngine, bytes_to_vector, vector_to_bytes
from app.rag.indexer import KnowledgeIndexer
from app.rag.retriever import LocalRAGRetriever
from app.storage.database import DatabaseManager


def test_markdown_chunker_header_splitting():
    sample_doc = """# Main Heading

First paragraph of text here.

## Subsection A

Details about Subsection A and some more interesting facts.

## Subsection B

Details about Subsection B.
"""
    chunker = MarkdownChunker(target_chunk_size=100, overlap=20)
    chunks = chunker.chunk_document(sample_doc)
    assert len(chunks) >= 2
    headings = [c.heading for c in chunks]
    assert "Subsection A" in headings
    assert "Subsection B" in headings


def test_vector_serialization_roundtrip():
    original_vec = np.random.randn(384).astype(np.float32)
    original_norm = original_vec / np.linalg.norm(original_vec)

    blob = vector_to_bytes(original_norm.tolist())
    assert len(blob) == 384 * 4  # 1536 bytes

    restored_norm = bytes_to_vector(blob, dim=384)
    assert np.allclose(original_norm, restored_norm, atol=1e-5)


def test_rag_end_to_end_indexing_and_retrieval():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_rag.db"
        db = DatabaseManager(db_path)
        db.initialize_schema()

        lore_file = Path("knowledge/lore.md")
        assert lore_file.exists()

        engine = EmbeddingEngine()
        indexer = KnowledgeIndexer(db=db, embedding_engine=engine)
        chunk_count = indexer.index_file(lore_file)
        assert chunk_count > 0

        retriever = LocalRAGRetriever(db=db, embedding_engine=engine)

        # Query specifically about printers
        results = retriever.retrieve("Why do you hate printers and toner?", threshold=0.4, top_k=2)
        assert len(results) > 0
        assert any("printer" in r.lower() for r in results)
