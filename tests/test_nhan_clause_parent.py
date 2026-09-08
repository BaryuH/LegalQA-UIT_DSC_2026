"""Acceptance and unit tests for Clause-Level Chunking & Small-to-Big Retrieval in src_nhan."""

from __future__ import annotations

from hashlib import sha256

import pytest

from src_nhan.chunker import (
    LegalChunk,
    ParentChunk,
    chunk_corpus,
    chunk_document,
)
from src_nhan.data_loader import LegalDocument
from src_nhan.evidence import pack_evidence
from src_nhan.reranker import RerankHit


def _make_doc(passage: str, doc_id: str = "doc_tax", name: str = "Luật Quản lý thuế") -> LegalDocument:
    return LegalDocument(
        id=doc_id,
        name=name,
        passage=passage,
        source_path="data/test_contexts.zip",
        source_member=f"context_{doc_id}.json",
        content_hash=sha256(passage.encode("utf-8")).hexdigest()[:16],
    )


def test_clause_level_chunking_with_points() -> None:
    passage = (
        "Lời mở đầu văn bản pháp luật.\n"
        "Điều 37. Quyền của người nộp thuế\n"
        "Người nộp thuế có các quyền sau đây:\n"
        "1. Được hướng dẫn thực hiện nộp thuế, cung cấp thông tin.\n"
        "a) Điểm a quy định về biểu mẫu;\n"
        "b) Điểm b quy định về thời hạn.\n"
        "2. Được nhận văn bản giải thích của cơ quan thuế.\n"
        "3. Có quyền khiếu nại, khởi kiện quyết định hành chính thuế.\n"
        "Điều 38. Trách nhiệm của người nộp thuế\n"
        "1. Khai thuế chính xác, trung thực, đầy đủ.\n"
        "2. Nộp tiền thuế đầy đủ, đúng thời hạn."
    )
    doc = _make_doc(passage)
    res = chunk_document(doc, max_chars=1200)

    # 1. Preamble parent + 2 article parents = 3 parents
    assert len(res.parents) == 3
    parents_by_id = {p.parent_id: p for p in res.parents}

    assert "doc_tax_preamble" in parents_by_id
    assert "doc_tax_dieu_37" in parents_by_id
    assert "doc_tax_dieu_38" in parents_by_id

    # Verify Điều 37 ParentChunk
    p37 = parents_by_id["doc_tax_dieu_37"]
    assert p37.article_number == "37"
    assert "Điều 37: Quyền của người nộp thuế" in p37.header
    assert "Người nộp thuế có các quyền sau đây:" in p37.raw_text
    assert "3. Có quyền khiếu nại" in p37.raw_text
    assert doc.passage[p37.start_offset : p37.end_offset] == p37.raw_text

    # 2. Check child chunks
    chunks_by_id = {c.chunk_id: c for c in res.chunks}

    # Điều 37 should have 3 child chunks (Khoản 1, Khoản 2, Khoản 3)
    k1 = chunks_by_id["doc_tax_dieu_37_k1"]
    k2 = chunks_by_id["doc_tax_dieu_37_k2"]
    k3 = chunks_by_id["doc_tax_dieu_37_k3"]

    assert k1.parent_id == "doc_tax_dieu_37"
    assert k2.parent_id == "doc_tax_dieu_37"
    assert k3.parent_id == "doc_tax_dieu_37"

    assert k1.section_label == "Điều 37 / Khoản 1"
    assert k2.section_label == "Điều 37 / Khoản 2"
    assert k3.section_label == "Điều 37 / Khoản 3"

    # Khoản 1 must contain Điểm a and Điểm b
    assert "a) Điểm a quy định về biểu mẫu;" in k1.raw_text
    assert "b) Điểm b quy định về thời hạn." in k1.raw_text

    # Verify exact traceability
    for chunk in res.chunks:
        assert doc.passage[chunk.start_offset : chunk.end_offset] == chunk.raw_text
        assert chunk.document_id == doc.id
        assert chunk.source_path == doc.source_path
        assert chunk.source_member == doc.source_member
        assert chunk.content_hash == sha256(chunk.raw_text.encode("utf-8")).hexdigest()[:16]

    # Verify retrieval_text enrichment
    assert "[Luật Quản lý thuế]" in k1.retrieval_text
    assert "[Điều 37: Quyền của người nộp thuế]" in k1.retrieval_text
    assert "Người nộp thuế có các quyền sau đây:" in k1.retrieval_text
    assert "1. Được hướng dẫn thực hiện nộp thuế" in k1.retrieval_text


def test_article_without_numbered_clauses() -> None:
    passage = (
        "Điều 5. Hiệu lực thi hành\n"
        "Thông tư này có hiệu lực thi hành kể từ ngày 15 tháng 3 năm 2024.\n"
        "Các quy định trước đây trái với Thông tư này đều bị bãi bỏ."
    )
    doc = _make_doc(passage, doc_id="doc_eff")
    res = chunk_document(doc, max_chars=1200)

    assert len(res.parents) == 1
    assert res.parents[0].parent_id == "doc_eff_dieu_5"
    assert res.parents[0].header == "Điều 5: Hiệu lực thi hành"

    assert len(res.chunks) == 1
    assert res.chunks[0].parent_id == "doc_eff_dieu_5"
    assert res.chunks[0].section_label == "Điều 5"
    assert doc.passage[res.chunks[0].start_offset : res.chunks[0].end_offset] == res.chunks[0].raw_text


def test_document_without_articles_fallback() -> None:
    passage = (
        "Cộng hòa xã hội chủ nghĩa Việt Nam\n"
        "Độc lập - Tự do - Hạnh phúc\n\n"
        "Thông báo về việc tổ chức tập huấn kiến thức pháp luật năm 2024.\n"
        "Kính gửi toàn thể cán bộ công nhân viên cơ quan."
    )
    doc = _make_doc(passage, doc_id="doc_notice", name="Thông báo số 01")
    res = chunk_document(doc, max_chars=500)

    assert len(res.parents) == 1
    assert res.parents[0].parent_id == "doc_notice_doc"
    assert res.parents[0].header == "Thông báo số 01"

    assert len(res.chunks) >= 1
    assert all(c.parent_id == "doc_notice_doc" for c in res.chunks)


def test_long_clause_splits_by_points() -> None:
    passage = (
        "Điều 10. Hồ sơ khai thuế\n"
        "1. Hồ sơ khai thuế bao gồm các tài liệu quy định cụ thể dưới đây:\n"
        "a) Tờ khai thuế theo mẫu do Bộ Tài chính ban hành kèm theo bảng kê chi tiết toàn bộ hóa đơn chứng từ liên quan.\n"
        "b) Báo cáo tài chính năm đã được kiểm toán đối với doanh nghiệp có vốn đầu tư trực tiếp nước ngoài.\n"
        "c) Bản sao có chứng thực giấy chứng nhận đăng ký kinh doanh hoặc quyết định thành lập của cơ quan có thẩm quyền.\n"
        "d) Các tài liệu khác chứng minh điều kiện được hưởng ưu đãi thuế theo quy định của pháp luật."
    )
    doc = _make_doc(passage, doc_id="doc_long")
    # Setting max_chars=180 forces the long clause to split by points
    res = chunk_document(doc, max_chars=180, overlap_chars=20, min_chars=30)

    assert len(res.parents) == 1
    assert res.parents[0].parent_id == "doc_long_dieu_10"

    # Chunks should be split by points
    point_labels = [c.section_label for c in res.chunks]
    assert any("Điểm a" in label for label in point_labels)
    assert any("Điểm b" in label for label in point_labels)
    assert any("Điểm c" in label for label in point_labels)
    assert any("Điểm d" in label for label in point_labels)
    assert all(c.parent_id == "doc_long_dieu_10" for c in res.chunks)


def test_small_to_big_evidence_packing_deduplication() -> None:
    passage = (
        "Điều 37. Quyền của người nộp thuế\n"
        "1. Được hướng dẫn về thuế.\n"
        "2. Được giữ bí mật thông tin.\n"
        "3. Được bồi thường thiệt hại.\n"
        "Điều 38. Trách nhiệm của người nộp thuế\n"
        "1. Khai thuế chính xác.\n"
        "2. Nộp thuế đúng hạn."
    )
    doc = _make_doc(passage, doc_id="doc_tax")
    res = chunk_document(doc)
    chunks_by_id = {c.chunk_id: c for c in res.chunks}
    parents_by_id = {p.parent_id: p for p in res.parents}

    # Hit 1 is Khoản 2 of Điều 37 (rank 1)
    # Hit 2 is Khoản 1 of Điều 37 (rank 2) -> same parent as Hit 1
    # Hit 3 is Khoản 1 of Điều 38 (rank 3) -> different parent
    hits = [
        RerankHit(chunk_id="doc_tax_dieu_37_k2", document_id="doc_tax", rerank_score=0.95, original_rrf_score=0.03, rank=1),
        RerankHit(chunk_id="doc_tax_dieu_37_k1", document_id="doc_tax", rerank_score=0.90, original_rrf_score=0.02, rank=2),
        RerankHit(chunk_id="doc_tax_dieu_38_k1", document_id="doc_tax", rerank_score=0.85, original_rrf_score=0.01, rank=3),
    ]

    packed = pack_evidence(hits, chunks_by_id, parents_by_id=parents_by_id, max_total_chars=4000)

    # Both child chunks from Điều 37 are in included_ids
    assert "doc_tax_dieu_37_k2" in packed.included_ids
    assert "doc_tax_dieu_37_k1" in packed.included_ids
    assert "doc_tax_dieu_38_k1" in packed.included_ids

    # But Điều 37 is included ONCE in included_parent_ids (deduplication!)
    assert packed.included_parent_ids == ("doc_tax_dieu_37", "doc_tax_dieu_38")

    # Rendered text has full Điều 37 and full Điều 38, appearing exactly once each
    assert packed.rendered_text.count("[Văn bản: doc_tax - Điều 37: Quyền của người nộp thuế]") == 1
    assert packed.rendered_text.count("[Văn bản: doc_tax - Điều 38: Trách nhiệm của người nộp thuế]") == 1
    assert "3. Được bồi thường thiệt hại." in packed.rendered_text  # Even though Hit 3 wasn't retrieved, Parent brings it!


def test_small_to_big_budget_truncation() -> None:
    passage = "Điều 1. Quy định chung\n" + ("Nội dung điều khoản dài. " * 50)
    doc = _make_doc(passage, doc_id="doc_budget")
    res = chunk_document(doc, max_chars=200)
    chunks_by_id = {c.chunk_id: c for c in res.chunks}
    parents_by_id = {p.parent_id: p for p in res.parents}

    first_chunk_id = res.chunks[0].chunk_id
    hits = [
        RerankHit(chunk_id=first_chunk_id, document_id="doc_budget", rerank_score=0.99, original_rrf_score=0.05, rank=1),
    ]

    # Limit budget to 250 characters
    packed = pack_evidence(hits, chunks_by_id, parents_by_id=parents_by_id, max_total_chars=250)

    assert packed.total_chars <= 250
    assert len(packed.truncated_ids) == 1
    assert packed.truncated_ids[0] == "doc_budget_dieu_1"


def test_chunk_corpus_result_iteration() -> None:
    doc1 = _make_doc("Điều 1. Mục tiêu\nNội dung 1.", doc_id="d1")
    doc2 = _make_doc("Điều 2. Nguyên tắc\nNội dung 2.", doc_id="d2")

    result = chunk_corpus([doc1, doc2])

    # Test object properties
    assert len(result.chunks) == 2
    assert len(result.parents) == 2
    assert "d1_dieu_1" in result.parents_by_id
    assert "d2_dieu_2" in result.parents_by_id

    # Test tuple unpacking: chunks, parents = chunk_corpus(...)
    c_list, p_list = result
    assert c_list == result.chunks
    assert p_list == result.parents
