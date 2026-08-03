from copy import deepcopy

import pytest

from legal_rag.evidence import (
    DeduplicationError,
    deduplicate_retrieved_chunks,
    normalize_chunk_text,
)
from legal_rag.schemas import LegalChunk, RetrievalHit


def _chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc-1",
    section_label: str | None = "Điều 1",
) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_path="selected-contexts.zip",
        source_member=f"context_{document_id}.json",
        section_label=section_label,
        raw_text=text,
        retrieval_text=text,
        content_hash=f"hash-{chunk_id}",
        chunker_version="chunker-v1",
    )


def _hit(chunk_id: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_path="selected-contexts.zip",
        rank=rank,
        bm25_score=float(100 - rank),
    )


def test_exact_normalized_duplicates_keep_the_higher_rank_and_preserve_text() -> None:
    first = _chunk("chunk-1", "Điều 1:\nNội dung pháp lý.")
    duplicate = _chunk("chunk-2", "  đIỀU 1: Nội dung pháp lý.  ")
    original = {chunk.chunk_id: chunk for chunk in (first, duplicate)}
    before = deepcopy(original)

    result = deduplicate_retrieved_chunks(
        [_hit("chunk-2", 2), _hit("chunk-1", 1)], original
    )

    assert result.kept_hits == (_hit("chunk-1", 1),)
    assert result.dropped_ids == ("chunk-2",)
    assert result.drop_reasons == {"chunk-2": "exact_normalized_duplicate"}
    assert original == before
    assert original["chunk-2"].retrieval_text == "  đIỀU 1: Nội dung pháp lý.  "
    assert normalize_chunk_text("A\n B") == normalize_chunk_text(" a  b ")


def test_same_source_article_high_overlap_drops_only_the_lower_rank_window() -> None:
    first = _chunk(
        "chunk-1",
        "alpha beta gamma delta epsilon zeta eta theta",
    )
    overlapping = _chunk(
        "chunk-2",
        "gamma delta epsilon zeta eta theta iota",
    )
    unrelated_article = _chunk(
        "chunk-3",
        "gamma delta epsilon zeta eta theta iota",
        section_label="Điều 2",
    )
    chunks = {
        chunk.chunk_id: chunk for chunk in (first, overlapping, unrelated_article)
    }

    result = deduplicate_retrieved_chunks(
        [_hit("chunk-3", 3), _hit("chunk-2", 2), _hit("chunk-1", 1)], chunks
    )

    assert tuple(hit.chunk_id for hit in result.kept_hits) == ("chunk-1", "chunk-3")
    assert result.dropped[0].chunk_id == "chunk-2"
    assert result.dropped[0].kept_chunk_id == "chunk-1"
    assert result.dropped[0].reason == "same_source_article_high_overlap"
    assert result.dropped[0].overlap_ratio == pytest.approx(6 / 7)


def test_high_overlap_is_conservative_across_documents_and_below_threshold() -> None:
    base = _chunk("chunk-1", "alpha beta gamma delta epsilon zeta")
    different_document = _chunk(
        "chunk-2",
        "alpha beta gamma delta epsilon zeta other",
        document_id="doc-2",
    )
    low_overlap = _chunk("chunk-3", "alpha beta other words entirely different")
    chunks = {
        chunk.chunk_id: chunk for chunk in (base, different_document, low_overlap)
    }

    result = deduplicate_retrieved_chunks(
        [_hit("chunk-3", 3), _hit("chunk-2", 2), _hit("chunk-1", 1)], chunks
    )

    assert tuple(hit.chunk_id for hit in result.kept_hits) == (
        "chunk-1",
        "chunk-2",
        "chunk-3",
    )
    assert result.dropped == ()


def test_deduplication_is_deterministic_and_missing_provenance_fails() -> None:
    chunks = {
        "chunk-1": _chunk("chunk-1", "alpha beta gamma"),
        "chunk-2": _chunk("chunk-2", "alpha beta gamma"),
    }
    hits = [_hit("chunk-2", 2), _hit("chunk-1", 1)]

    first = deduplicate_retrieved_chunks(hits, chunks).as_dict()
    second = deduplicate_retrieved_chunks(list(reversed(hits)), chunks).as_dict()

    assert first == second
    with pytest.raises(DeduplicationError, match="no corresponding legal chunk"):
        deduplicate_retrieved_chunks([_hit("missing", 1)], chunks)


def test_overlap_threshold_must_be_valid() -> None:
    chunk = _chunk("chunk-1", "alpha beta")
    with pytest.raises(ValueError, match="overlap_threshold"):
        deduplicate_retrieved_chunks(
            [_hit("chunk-1", 1)], {"chunk-1": chunk}, overlap_threshold=0
        )
