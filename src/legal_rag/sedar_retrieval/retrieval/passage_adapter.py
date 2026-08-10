"""Adapt canonical passages to existing BM25 LegalChunk views."""

from __future__ import annotations

from hashlib import sha256

from legal_rag.schemas import LegalChunk

from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage


def passage_to_legal_chunk(passage: CanonicalPassage) -> LegalChunk:
    """Map a canonical passage onto the BM25-compatible LegalChunk contract."""

    section_parts: list[str] = []
    if passage.article_number:
        section_parts.append(f"Điều {passage.article_number}")
    if passage.clause_number:
        section_parts.append(f"Khoản {passage.clause_number}")
    if passage.point_label:
        section_parts.append(f"Điểm {passage.point_label}")
    section_label = " / ".join(section_parts) if section_parts else passage.retrieval_level

    return LegalChunk(
        chunk_id=passage.passage_id,
        document_id=passage.document_id,
        source_path=passage.source.source_path,
        source_member=passage.source.source_member,
        raw_text=passage.raw_text,
        retrieval_text=passage.retrieval_text,
        content_hash=sha256(passage.raw_text.encode("utf-8")).hexdigest(),
        chunker_version=f"sedar-retrieval-v3-{passage.retrieval_level}",
        section_label=section_label,
        start_offset=None,
        end_offset=None,
    )


def load_passages_jsonl(path: str) -> tuple[CanonicalPassage, ...]:
    import json
    from pathlib import Path

    rows: list[CanonicalPassage] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(CanonicalPassage.model_validate(json.loads(line)))
    return tuple(rows)
