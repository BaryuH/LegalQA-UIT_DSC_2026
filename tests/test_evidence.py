from copy import deepcopy

import pytest

from legal_rag.evidence import (
    DeduplicationError,
    deduplicate_retrieved_chunks,
    normalize_chunk_text,
    rerank_hits_with_citations,
    score_citation_match,
)
from legal_rag.schemas import LegalChunk, LegalDocument, RetrievalHit


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


def test_score_citation_match_differentiates_statutes() -> None:
    gold_chunk = _chunk(
        "chunk-gold",
        "Trách nhiệm tổ chức đấu thầu theo Điều 37 Nghị định 153/2020/NĐ-CP...",
        section_label="Điều 37",
        document_id="doc-153",
    )
    wrong_art_chunk = _chunk(
        "chunk-wrong-art",
        "Quy định chung theo Điều 14 Nghị định 153/2020/NĐ-CP...",
        section_label="Điều 14",
        document_id="doc-153",
    )
    unrelated_chunk = _chunk(
        "chunk-unrelated",
        "Quy định an toàn lao động theo Điều 5...",
        section_label="Điều 5",
        document_id="doc-99",
    )
    doc_153 = LegalDocument(
        id="doc-153",
        name="Nghị định 153/2020/NĐ-CP",
        passage="...",
        source_path="selected-contexts.zip",
        content_hash="h153",
    )

    query = "Trách nhiệm của tổ chức theo Điều 37 Nghị định 153/2020/NĐ-CP?"
    score_gold = score_citation_match(query, gold_chunk, doc_153)
    score_wrong_art = score_citation_match(query, wrong_art_chunk, doc_153)
    score_unrelated = score_citation_match(query, unrelated_chunk, None)

    assert score_gold > score_wrong_art > score_unrelated
    assert score_unrelated == 0.0
    # Gold matches both article and document number
    assert score_gold >= 6.0


def test_rerank_hits_with_citations_promotes_exact_article_and_preserves_tie() -> None:
    c_match = _chunk(
        "c-match",
        "Điều 76 Bộ luật Lao động 2019...",
        section_label="Điều 76",
        document_id="doc-bllđ",
    )
    c_high_bm25 = _chunk(
        "c-high-bm25",
        "Nội dung chung về thỏa ước tập thể nhưng không nói rõ điều nào...",
        section_label="Điều 1",
        document_id="doc-other",
    )
    chunks = {c.chunk_id: c for c in (c_match, c_high_bm25)}
    doc = LegalDocument(
        id="doc-bllđ",
        name="Bộ luật Lao động 2019",
        passage="...",
        source_path="selected-contexts.zip",
        content_hash="hbllđ",
    )
    documents = {"doc-bllđ": doc}

    h_high_bm25 = RetrievalHit(
        chunk_id="c-high-bm25",
        document_id="doc-other",
        source_path="selected-contexts.zip",
        rank=1,
        bm25_score=15.0,
        rerank_score=2.0,
    )
    h_match = RetrievalHit(
        chunk_id="c-match",
        document_id="doc-bllđ",
        source_path="selected-contexts.zip",
        rank=2,
        bm25_score=10.0,
        rerank_score=2.5,
    )

    query = "Ký kết thỏa ước theo Điều 76 Bộ luật Lao động 2019?"
    hits = [h_high_bm25, h_match]  # Initially h_high_bm25 is rank 1

    reordered = rerank_hits_with_citations(
        hits, chunks, query, documents=documents, citation_weight=1.0
    )

    # c-match should now be promoted to rank 1
    assert reordered[0].chunk_id == "c-match"
    assert reordered[0].rank == 1
    assert reordered[0].rerank_score > reordered[1].rerank_score
    assert reordered[1].chunk_id == "c-high-bm25"
    assert reordered[1].rank == 2

    # Query without citations preserves exact original rank order
    query_no_cit = "Thỏa ước lao động là gì?"
    unchanged = rerank_hits_with_citations(
        hits, chunks, query_no_cit, documents=documents
    )
    assert tuple(h.chunk_id for h in unchanged) == ("c-high-bm25", "c-match")
    assert tuple(h.rank for h in unchanged) == (1, 2)
