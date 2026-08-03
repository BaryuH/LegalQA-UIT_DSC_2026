from copy import deepcopy

import pytest

from legal_rag.evidence import EvidencePackingError, pack_evidence
from legal_rag.schemas import LegalChunk, LegalDocument, RetrievalHit


def _chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc-1",
    section_label: str = "Điều 1",
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


def _hit(chunk_id: str, rank: int, *, document_id: str = "doc-1") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=document_id,
        source_path="selected-contexts.zip",
        rank=rank,
        bm25_score=float(100 - rank),
        rerank_score=float(10 - rank),
    )


def test_header_is_part_of_budget_and_scores_are_not_rendered() -> None:
    chunk = _chunk("chunk-1", "Điều 1. Phạm vi áp dụng\nNội dung pháp lý.")
    document = LegalDocument(
        id="doc-1",
        name="Luật mẫu",
        passage=chunk.raw_text,
        source_path="selected-contexts.zip",
        content_hash="doc-hash",
        link="https://example.test/law",
    )

    result = pack_evidence(
        [_hit("chunk-1", 1)],
        {"chunk-1": chunk},
        documents={"doc-1": document},
        max_total_chars=500,
        max_chunks_per_document=2,
    )

    assert len(result.rendered_text) <= 500
    assert result.rendered_text.startswith("[TRÍCH ĐOẠN 1]\n")
    assert "Văn bản: Luật mẫu" in result.rendered_text
    assert "Mã tài liệu: doc-1" in result.rendered_text
    assert "Điều/Khoản: Điều 1" in result.rendered_text
    assert "Nguồn: https://example.test/law" in result.rendered_text
    assert "Nội dung:\nĐiều 1. Phạm vi áp dụng" in result.rendered_text
    assert "bm25" not in result.rendered_text.casefold()
    assert "rerank" not in result.rendered_text.casefold()
    assert result.metadata["scores_rendered"] is False


def test_per_document_cap_and_rank_order_drop_lower_ranked_chunks() -> None:
    chunks = {
        "a-high": _chunk("a-high", "Điều A\nA", document_id="doc-a"),
        "a-low": _chunk("a-low", "Điều A2\nA2", document_id="doc-a"),
        "b": _chunk("b", "Điều B\nB", document_id="doc-b"),
    }
    result = pack_evidence(
        [
            _hit("a-low", 3, document_id="doc-a"),
            _hit("b", 2, document_id="doc-b"),
            _hit("a-high", 1, document_id="doc-a"),
        ],
        chunks,
        max_total_chars=2000,
        max_chunks_per_document=1,
    )

    assert result.included_ids == ("a-high", "b")
    assert result.dropped_ids == ("a-low",)
    assert result.dropped_reasons == {"a-low": "max_chunks_per_document"}


def test_oversized_chunk_preserves_heading_and_marks_tail_truncation() -> None:
    source_text = "Điều 37. Phạm vi điều chỉnh\n" + ("Nội dung rất dài. " * 30)
    chunk = _chunk("chunk-1", source_text)
    before = deepcopy(chunk)
    # Leave enough room for the complete header, heading, marker, and a short tail.
    generous = pack_evidence(
        [_hit("chunk-1", 1)],
        {"chunk-1": chunk},
        max_total_chars=5000,
        max_chunks_per_document=1,
    )
    full_header_length = len(generous.rendered_text) - len(source_text)
    budget = full_header_length + len("Điều 37. Phạm vi điều chỉnh") + 35

    result = pack_evidence(
        [_hit("chunk-1", 1)],
        {"chunk-1": chunk},
        max_total_chars=budget,
        max_chunks_per_document=1,
    )

    assert result.included_ids == ("chunk-1",)
    assert result.truncated_ids == ("chunk-1",)
    assert result.dropped_ids == ()
    assert "Điều 37. Phạm vi điều chỉnh" in result.rendered_text
    assert result.rendered_text.endswith("… [ĐÃ CẮT PHẦN ĐUÔI]")
    assert len(result.rendered_text) <= budget
    assert chunk == before


def test_single_line_oversized_chunk_uses_section_label_as_heading() -> None:
    chunk = _chunk(
        "chunk-1", "Điều 37 " + ("Nội dung dài. " * 40), section_label="Điều 37"
    )
    result = pack_evidence(
        [_hit("chunk-1", 1)],
        {"chunk-1": chunk},
        max_total_chars=220,
        max_chunks_per_document=1,
    )

    assert result.truncated_ids == ("chunk-1",)
    assert "Nội dung dài." in result.rendered_text
    assert result.rendered_text.endswith("… [ĐÃ CẮT PHẦN ĐUÔI]")


def test_budget_drop_and_serialization_are_deterministic() -> None:
    chunks = {
        "first": _chunk("first", "Điều 1\n" + ("alpha " * 40)),
        "second": _chunk("second", "Điều 2\nshort", section_label="Điều 2"),
    }
    hits = [_hit("second", 2), _hit("first", 1)]
    kwargs = dict(max_total_chars=250, max_chunks_per_document=2)
    first = pack_evidence(hits, chunks, **kwargs)
    second = pack_evidence(list(reversed(hits)), chunks, **kwargs)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.included_ids == ("first",)
    assert first.truncated_ids == ("first",)
    assert first.dropped_ids == ("second",)
    assert first.dropped_reasons == {"second": "max_total_chars"}


def test_no_chunk_fits_when_heading_cannot_fit() -> None:
    chunk = _chunk("chunk-1", "Điều 1. Một tiêu đề dài\nNội dung")
    with pytest.raises(EvidencePackingError, match="No evidence chunk fits"):
        pack_evidence(
            [_hit("chunk-1", 1)],
            {"chunk-1": chunk},
            max_total_chars=10,
            max_chunks_per_document=1,
        )
