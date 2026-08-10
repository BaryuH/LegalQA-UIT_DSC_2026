"""Unit tests for canonical legal parser (TASK 03)."""

from __future__ import annotations

from legal_rag.schemas import LegalDocument
from legal_rag.sedar_retrieval.corpus.hierarchy import (
    audit_canonical_nodes,
    build_retrieval_text,
    nodes_to_passages,
)
from legal_rag.sedar_retrieval.corpus.parse_legal import parse_legal_document


def _doc(passage: str, doc_id: str = "doc1") -> LegalDocument:
    return LegalDocument(
        id=doc_id,
        name="Bộ luật Lao động 2019",
        passage=passage,
        source_path="selected-contexts.zip",
        source_member="context_doc1.json",
        content_hash="hash",
    )


def test_standard_article_with_title() -> None:
    passage = "Điều 35. Quyền đơn phương chấm dứt hợp đồng lao động\nNội dung điều."
    nodes = parse_legal_document(_doc(passage))
    articles = [n for n in nodes if n.level == "article"]
    assert len(articles) == 1
    assert articles[0].article_number == "35"
    assert articles[0].article_title == "Quyền đơn phương chấm dứt hợp đồng lao động"
    assert "Nội dung điều." in articles[0].raw_text


def test_clauses_and_vietnamese_point_d() -> None:
    passage = (
        "Điều 15. Thử việc\n"
        "1. Nội dung khoản 1.\n"
        "a) Điểm a.\n"
        "đ) Điểm đ.\n"
        "2. Nội dung khoản 2.\n"
    )
    nodes = parse_legal_document(_doc(passage))
    clauses = [n for n in nodes if n.level == "clause"]
    points = [n for n in nodes if n.level == "point"]
    assert {c.clause_number for c in clauses} == {"1", "2"}
    assert any(p.point_label and p.point_label.casefold() == "đ" for p in points)


def test_missing_chapter_still_parses_article() -> None:
    passage = "Điều 1. Phạm vi\nÁp dụng trên toàn quốc."
    nodes = parse_legal_document(_doc(passage))
    assert any(n.level == "article" and n.article_number == "1" for n in nodes)
    audit = audit_canonical_nodes(nodes)
    assert audit.orphan_node_count == 0
    assert audit.unique_id_rate == 1.0


def test_chapter_section_hierarchy() -> None:
    passage = (
        "Chương III - Hợp đồng lao động\n"
        "Mục 1 - Giao kết\n"
        "Điều 13. Định nghĩa\n"
        "1. Khoản một.\n"
    )
    nodes = parse_legal_document(_doc(passage))
    assert any(n.level == "chapter" for n in nodes)
    assert any(n.level == "section" for n in nodes)
    article = next(n for n in nodes if n.level == "article")
    assert article.chapter_id is not None
    assert article.section_id is not None


def test_malformed_numbering_is_retained_not_dropped() -> None:
    passage = "Phần mở đầu không có điều.\nNội dung tự do."
    nodes = parse_legal_document(_doc(passage))
    retained = [n for n in nodes if n.parse_status == "retained"]
    assert retained
    assert all("Phần mở đầu" in n.raw_text or "Nội dung" in n.raw_text for n in retained)


def test_retrieval_and_reader_text_separation() -> None:
    passage = (
        "Chương I - Quy định chung\n"
        "Điều 1. Phạm vi điều chỉnh\n"
        "1. Nội dung khoản 1.\n"
    )
    nodes = parse_legal_document(_doc(passage))
    passages = nodes_to_passages(nodes, levels=("article", "clause"))
    assert passages
    for item in passages:
        assert item.raw_text
        assert item.reader_text
        assert "[DOCUMENT]" in item.retrieval_text
        assert "[DOCUMENT]" not in item.reader_text or item.reader_text == item.raw_text
        built = build_retrieval_text(
            next(n for n in nodes if n.node_id == item.passage_id)
        )
        assert built == item.retrieval_text
