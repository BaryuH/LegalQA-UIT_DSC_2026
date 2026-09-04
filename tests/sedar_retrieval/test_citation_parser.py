"""Q1 regression table for the Vietnamese legal citation parser.

Every row here is a shape the pre-Q1 parser missed, or a false positive the
Q1 patch must not introduce. A miss costs three things at once: a silver label,
an LTR training label, and the citation feature block at inference.
"""

from __future__ import annotations

import pytest

from legal_rag.sedar_retrieval.query.citation_parser import parse_citations


def _kinds(text: str) -> list[str]:
    return [mention.kind for mention in parse_citations(text)]


# ── Original TASK 14 starter tests, kept verbatim ─────────────────────────
# These predate Q1 and must keep passing. test_parse_article_clause_point
# asserts on spans[0], so it also pins the ordering contract: mentions are
# sorted by (start, end, kind) and the contained clause_only/point_only
# mentions of "điểm a khoản 2 Điều 15" must not appear ahead of it.


def test_parse_article_clause_point() -> None:
    spans = parse_citations("Theo điểm a khoản 2 Điều 15 của Bộ luật Lao động")
    assert spans
    assert spans[0].article == "15"
    assert spans[0].clause == "2"
    assert spans[0].point and spans[0].point.casefold() == "a"


def test_parse_document_number() -> None:
    spans = parse_citations("Theo Nghị định 13/2023/NĐ-CP")
    assert any(s.document_number for s in spans)


def test_canonical_point_clause_article_still_parses() -> None:
    mentions = parse_citations("Theo điểm a khoản 2 Điều 15 Nghị định 13/2023/NĐ-CP")
    assert any(
        m.article == "15" and m.clause == "2" and m.point == "a" for m in mentions
    )
    assert any(
        m.document_number and "13/2023" in m.document_number for m in mentions
    )


@pytest.mark.parametrize(
    ("text", "expected_number"),
    [
        ("Mức xử phạt theo 153/2020/NĐ-CP là bao nhiêu?", "153/2020/NĐ-CP"),
        ("Nghị quyết 42/2017/QH14 quy định gì?", "42/2017/QH14"),
        (
            "Thông tư liên tịch 01/2016/TTLT-BYT-BTC áp dụng khi nào?",
            "01/2016/TTLT-BYT-BTC",
        ),
        ("Quyết định 1234/2020/QĐ-TTg còn hiệu lực không?", "1234/2020/QĐ-TTg"),
    ],
)
def test_document_numbers(text: str, expected_number: str) -> None:
    """Bare numbers, extra document types, digit and mixed-case issuers."""

    assert any(m.document_number == expected_number for m in parse_citations(text))


@pytest.mark.parametrize(
    "text",
    [
        "Theo Pháp lệnh 09/2014/UBTVQH13 thì sao?",
        "Chỉ thị 16/CT-TTg có nội dung gì?",
    ],
)
def test_additional_document_types(text: str) -> None:
    assert "document_number" in _kinds(text)


@pytest.mark.parametrize(
    ("text", "year"),
    [
        ("Luật Doanh nghiệp 2020 quy định về vốn điều lệ thế nào?", "2020"),
        ("Bộ luật Dân sự 2015 có bao nhiêu điều?", "2015"),
    ],
)
def test_document_name_and_year(text: str, year: str) -> None:
    """A name with no number at all: 'Luật Doanh nghiệp 2020'.

    'bộ luật' must win over 'luật' in the type alternation.
    """

    mentions = [m for m in parse_citations(text) if m.kind == "document_name"]
    assert mentions
    assert any(m.year == year for m in mentions)


def test_standalone_clause_and_point() -> None:
    assert any(
        m.kind == "clause_only" and m.clause == "2"
        for m in parse_citations("Khoản 2 quy định gì?")
    )
    assert any(
        m.kind == "point_only" and m.point == "b"
        for m in parse_citations("Điểm b được hiểu như thế nào?")
    )


def test_article_number_with_letter_suffix() -> None:
    assert any(m.article == "76a" for m in parse_citations("Điều 76a quy định gì?"))


# ── False positives the patch must not introduce ──────────────────────────


def test_law_this_plus_year_is_not_a_document_name() -> None:
    """'Luật này có hiệu lực từ năm 2015' is not a document name.

    This is why the name group requires an uppercase initial: Vietnamese law
    names capitalise the first word after the type.
    """

    assert "document_name" not in _kinds(
        "Luật này có hiệu lực từ năm 2015 và thay thế văn bản cũ"
    )


def test_date_is_not_a_document_number() -> None:
    assert "document_number" not in _kinds(
        "Hợp đồng ký ngày 16/09/2022 thì áp dụng quy định nào?"
    )


def test_no_citation_query_returns_nothing() -> None:
    assert parse_citations("Xin hỏi thủ tục đăng ký ra sao?") == ()


# ── Containment: overlapping matches must not inflate the mention count ───


def test_clause_inside_article_form_is_dropped() -> None:
    """'khoản 2 Điều 15' is one article mention, not article + clause_only.

    Without containment resolution, analyzer.classify_complexity over-counts
    citations and misclassifies simple queries as multi_hop.
    """

    kinds = _kinds("Theo khoản 2 Điều 15 thì sao?")
    assert kinds.count("article") == 1
    assert "clause_only" not in kinds


def test_bare_number_inside_typed_mention_is_dropped() -> None:
    kinds = _kinds(
        "Nghị định 13/2023/NĐ-CP và Nghị định 15/2020/NĐ-CP khác nhau ra sao?"
    )
    assert kinds.count("document_number") == 2


def test_multi_hop_shape() -> None:
    kinds = _kinds("Theo Điều 5 và Điều 12 của Luật Đất đai 2024")
    assert kinds.count("article") == 2
    assert "document_name" in kinds
