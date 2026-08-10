"""Citation parser starter tests (TASK 14)."""

from legal_rag.sedar_retrieval.query import parse_citations


def test_parse_article_clause_point() -> None:
    spans = parse_citations("Theo điểm a khoản 2 Điều 15 của Bộ luật Lao động")
    assert spans
    assert spans[0].article == "15"
    assert spans[0].clause == "2"
    assert spans[0].point and spans[0].point.casefold() == "a"


def test_parse_document_number() -> None:
    spans = parse_citations("Theo Nghị định 13/2023/NĐ-CP")
    assert any(s.document_number for s in spans)
