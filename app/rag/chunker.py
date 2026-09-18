"""Header-aware Markdown text chunking with overlap preservation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class TextChunk:
    chunk_index: int
    content: str
    heading: str
    char_count: int


class MarkdownChunker:
    """Chunks markdown documents while preserving semantic headings and sliding overlap."""

    def __init__(self, target_chunk_size: int = 400, overlap: int = 60):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_document(self, markdown_text: str) -> list[TextChunk]:
        """Split markdown text into clean semantic chunks."""
        sections = re.split(r"(^#{1,3}\s+.*$)", markdown_text, flags=re.MULTILINE)
        current_heading = "General"
        chunks: list[TextChunk] = []
        chunk_idx = 0

        i = 0
        while i < len(sections):
            part = sections[i].strip()
            if not part:
                i += 1
                continue

            if part.startswith("#"):
                current_heading = part.lstrip("#").strip()
                i += 1
                continue

            # This is a content block under current_heading
            paragraphs = [p.strip() for p in part.split("\n\n") if p.strip()]
            buffer = ""

            for p in paragraphs:
                if len(buffer) + len(p) + 2 <= self.target_chunk_size:
                    buffer = f"{buffer}\n\n{p}".strip()
                else:
                    if buffer:
                        chunks.append(
                            TextChunk(
                                chunk_index=chunk_idx,
                                content=f"[{current_heading}] {buffer}",
                                heading=current_heading,
                                char_count=len(buffer),
                            )
                        )
                        chunk_idx += 1
                        # Retain overlap from end of buffer
                        overlap_tail = buffer[-self.overlap :] if len(buffer) > self.overlap else ""
                        buffer = f"{overlap_tail}\n\n{p}".strip()
                    else:
                        # Single paragraph exceeds target size, split by sentences
                        sentences = re.split(r"(?<=[.!?])\s+", p)
                        for s in sentences:
                            if len(buffer) + len(s) + 1 <= self.target_chunk_size:
                                buffer = f"{buffer} {s}".strip()
                            else:
                                if buffer:
                                    chunks.append(
                                        TextChunk(
                                            chunk_index=chunk_idx,
                                            content=f"[{current_heading}] {buffer}",
                                            heading=current_heading,
                                            char_count=len(buffer),
                                        )
                                    )
                                    chunk_idx += 1
                                buffer = s

            if buffer:
                chunks.append(
                    TextChunk(
                        chunk_index=chunk_idx,
                        content=f"[{current_heading}] {buffer}",
                        heading=current_heading,
                        char_count=len(buffer),
                    )
                )
                chunk_idx += 1

            i += 1

        return chunks
