"""Legal-text chunker with source provenance tracking.

Splits long legal passages into overlapping chunks while preserving
document_id, source_path, and source_member for full traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .data_loader import LegalDocument


CHUNKER_VERSION = "nhan-chunker-v1"


@dataclass(frozen=True, slots=True)
class LegalChunk:
    """One chunk of legal text with traceable provenance."""

    chunk_id: str
    document_id: str
    source_path: str
    source_member: str | None
    raw_text: str
    start_offset: int
    end_offset: int
    content_hash: str

    @property
    def retrieval_text(self) -> str:
        """Text used for retrieval indexing."""
        return self.raw_text


def _split_by_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, preserving non-empty ones."""
    parts = text.split("\n")
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    doc: LegalDocument,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 100,
) -> list[LegalChunk]:
    """Chunk a legal document's passage into overlapping text chunks.

    Strategy:
    1. If passage <= max_chars, return as single chunk.
    2. Otherwise, split by paragraphs and greedily merge into chunks
       that respect max_chars, with overlap from previous chunk.
    """
    passage = doc.passage.strip()
    if not passage:
        return []

    chunks: list[LegalChunk] = []

    if len(passage) <= max_chars:
        content_hash = sha256(passage.encode("utf-8")).hexdigest()[:16]
        chunks.append(
            LegalChunk(
                chunk_id=f"{doc.id}_0",
                document_id=doc.id,
                source_path=doc.source_path,
                source_member=doc.source_member,
                raw_text=passage,
                start_offset=0,
                end_offset=len(passage),
                content_hash=content_hash,
            )
        )
        return chunks

    # Sliding window chunking
    start = 0
    chunk_idx = 0
    while start < len(passage):
        end = min(start + max_chars, len(passage))

        # Try to break at a paragraph/sentence boundary
        if end < len(passage):
            # Look for last newline within the window
            last_break = passage.rfind("\n", start, end)
            if last_break > start + min_chars:
                end = last_break + 1
            else:
                # Look for last period followed by space
                last_period = passage.rfind(". ", start, end)
                if last_period > start + min_chars:
                    end = last_period + 2

        chunk_text = passage[start:end].strip()
        if len(chunk_text) >= min_chars or chunk_idx == 0:
            content_hash = sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                LegalChunk(
                    chunk_id=f"{doc.id}_{chunk_idx}",
                    document_id=doc.id,
                    source_path=doc.source_path,
                    source_member=doc.source_member,
                    raw_text=chunk_text,
                    start_offset=start,
                    end_offset=end,
                    content_hash=content_hash,
                )
            )
            chunk_idx += 1

        # Advance with overlap
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = end  # Prevent infinite loop
        start = next_start

    return chunks


def chunk_corpus(
    documents: list[LegalDocument],
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 100,
) -> list[LegalChunk]:
    """Chunk all documents in the corpus."""
    all_chunks: list[LegalChunk] = []
    for doc in documents:
        doc_chunks = chunk_document(
            doc,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
        )
        all_chunks.extend(doc_chunks)
    return all_chunks
