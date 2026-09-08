"""Acceptance tests for the SEDAR corpus v4 builder."""

from __future__ import annotations

from legal_rag.schemas import LegalDocument
from legal_rag.sedar_retrieval.corpus_v4 import (
    BuildOptions,
    audit_corpus,
    build_document,
    normalize_source_text,
    split_instruments,
)
from legal_rag.sedar_retrieval.corpus_v4.citations import (
    CitationIndex,
    extract_citations,
    fold,
)

SOFT = "\r\n\n"


def _document(passage: str, *, doc_id: str = "d1", name: str = "d1", link: str | None = None) -> LegalDocument:
    return LegalDocument(
        id=doc_id,
        name=name,
        passage=passage,
        source_path="data/selected-contexts.zip",
        content_hash="0" * 64,
        link=link,
        source_member=f"selected-contexts/context_{doc_id}.json",
    )


def test_soft_wrap_is_rejoined_and_hard_break_is_kept() -> None:
    raw = f"CHÍNH{SOFT}PHỦ\n\nSố: 99/2003/NĐ-CP\n\nĐiều 1. Phạm vi{SOFT}điều chỉnh\n\nQuy chế này quy định."
    text = normalize_source_text(raw)
    assert "Điều 1. Phạm vi điều chỉnh" in text
    assert "Quy chế này quy định." in text.split("\n")[-1]


def test_annexed_instrument_gets_its_own_numbering_scope() -> None:
    passage = (
        "CHÍNH PHỦ\n\nSố: 99/2003/NĐ-CP\n\nHà Nội, ngày 28 tháng 8 năm 2003\n\n"
        "NGHỊ ĐỊNH\n\nBan hành Quy chế Khu công nghệ cao\n\n"
        "Điều 1. Ban hành kèm theo Nghị định này Quy chế Khu công nghệ cao.\n\n"
        "Điều 2. Nghị định này có hiệu lực sau 15 ngày.\n\n"
        "QUY CHẾ\n\n(Ban hành kèm theo Nghị định số 99/2003/NĐ-CP)\n\n"
        "Chương 1\n\nNHỮNG QUY ĐỊNH CHUNG\n\n"
        "Điều 1. Phạm vi điều chỉnh\n\nQuy chế này quy định về tổ chức.\n\n"
        "Điều 2. Giải thích từ ngữ\n\nCông nghệ cao là công nghệ tiên tiến.\n"
    )
    record, units = build_document(_document(passage, link="https://x/Nghi-dinh-99-2003-ND-CP-abc-51305.aspx"))
    assert record.instrument_count >= 2
    article_ids = [u.unit_id for u in units if u.level == "article"]
    assert len(article_ids) == len(set(article_ids))
    scopes = {u.instrument_index for u in units if u.level == "article"}
    assert len(scopes) >= 2
    annex = [u for u in units if u.instrument_kind == "annex" and u.level == "article"]
    assert annex and annex[0].chapter_number is not None


def test_no_micro_unit_is_indexed_and_children_merge_to_parent() -> None:
    clauses = "\n\n".join(f"{i}. Khoản ngắn." for i in range(1, 30))
    passage = (
        "BỘ Y TẾ\n\nSố: 01/2020/TT-BYT\n\nTHÔNG TƯ\n\nQuy định thử nghiệm\n\n"
        f"Điều 1. Quy định chi tiết\n\n{clauses}\n"
    )
    options = BuildOptions(long_article_chars=200, min_child_chars=120, max_child_chars=600)
    _, units = build_document(_document(passage), options)
    indexed = [u for u in units if u.indexed]
    assert indexed, "at least one unit must be indexed"
    assert all(u.char_count >= 120 for u in indexed if u.level == "article_part")
    parts = [u for u in units if u.level == "article_part"]
    assert parts, "a long article must produce merged children"
    assert all(u.parent_unit_id for u in parts)
    # Exactly one granularity reaches the index for a long article.
    article = next(u for u in units if u.level == "article")
    assert article.indexed is False


def test_article_less_document_still_yields_indexed_blocks() -> None:
    body = "\n\n".join(
        [
            "BỘ GIÁO DỤC VÀ ĐÀO TẠO",
            "Số: 4039/BGDĐT-GDTX",
            "I. NHIỆM VỤ CHUNG",
            "Tiếp tục triển khai nhiệm vụ năm học. " * 20,
            "II. NHIỆM VỤ CỤ THỂ",
            "Củng cố mạng lưới trung tâm học tập cộng đồng. " * 20,
        ]
    )
    record, units = build_document(_document(body, doc_id="d2", name="d2"))
    assert record.parse_profile == "block"
    blocks = [u for u in units if u.level == "block"]
    assert blocks
    assert all(u.indexed for u in blocks)


def test_retrieval_text_carries_the_hierarchy_header() -> None:
    passage = (
        "CHÍNH PHỦ\n\nSố: 10/2021/NĐ-CP\n\nNGHỊ ĐỊNH\n\nQuy định về xây dựng\n\n"
        "Chương II\n\nQUẢN LÝ CHI PHÍ\n\nĐiều 5. Nguyên tắc quản lý\n\n"
        "1. Chi phí phải được quản lý chặt chẽ.\n"
    )
    _, units = build_document(_document(passage, link="https://x/Nghi-dinh-10-2021-ND-CP-xay-dung-1.aspx"))
    article = next(u for u in units if u.level == "article")
    assert article.retrieval_text.startswith(article.breadcrumb)
    assert "Nghị định 10/2021/NĐ-CP" in article.breadcrumb
    assert "Chương II" in article.breadcrumb
    assert "Điều 5" in article.breadcrumb
    # The reader text stays free of index-only decoration.
    assert not article.reader_text.startswith("Nghị định")


def test_enforcement_articles_are_tagged_and_not_packable() -> None:
    passage = (
        "CHÍNH PHỦ\n\nSố: 11/2021/NĐ-CP\n\nNGHỊ ĐỊNH\n\nQuy định mẫu\n\n"
        "Điều 9. Trách nhiệm thi hành\n\n"
        "Các Bộ trưởng chịu trách nhiệm thi hành Nghị định này.\n"
    )
    _, units = build_document(_document(passage))
    article = next(u for u in units if u.level == "article")
    assert article.role == "enforcement"
    assert article.packable is False
    assert article.indexed is True


def test_cross_reference_is_not_parsed_as_an_article_heading() -> None:
    passage = (
        "QUỐC HỘI\n\nSố: 91/2015/QH13\n\nBỘ LUẬT\n\nDÂN SỰ\n\n"
        "Điều 325. Thế chấp quyền sử dụng đất\n\n"
        "1. Trường hợp thế chấp quyền sử dụng đất.\n\n"
        "Điều 325. và Điều 326 của Bộ luật này thì hợp đồng vẫn còn hiệu lực.\n"
    )
    _, units = build_document(_document(passage))
    numbers = [u.article_number for u in units if u.level == "article"]
    assert numbers.count("325") == 1


def test_audit_reports_zero_duplicate_ids_and_full_document_coverage() -> None:
    passage = (
        "CHÍNH PHỦ\n\nSố: 12/2021/NĐ-CP\n\nNGHỊ ĐỊNH\n\nQuy định thử\n\n"
        "Điều 1. Phạm vi\n\nNghị định này quy định về thử nghiệm.\n\n"
        "Điều 1. Phạm vi\n\nBản sửa đổi được trích dẫn lại.\n"
    )
    record, units = build_document(_document(passage))
    report = audit_corpus([record], list(units))
    assert report.duplicate_unit_id_count == 0
    assert report.documents_without_units == 0
    assert report.duplicate_article_number_count == 1
    assert any("duplicate_article_number" in u.warnings for u in units)


def test_split_instruments_ignores_inline_mentions() -> None:
    lines = [
        "CHÍNH PHỦ",
        "Số: 13/2021/NĐ-CP",
        "Điều 1. Phạm vi",
        "Nội dung này thực hiện theo quy định tại",
        "Quy chế ban hành kèm theo Quyết định số 1/2020/QĐ-TTg",
    ]
    instruments = split_instruments(lines)
    assert len(instruments) == 1


def test_citation_index_resolves_folded_titles_and_flags_self_reference() -> None:
    index = CitationIndex()
    index.add_document(
        "d9",
        ["Bộ luật Lao động 2019", "Bộ luật 45/2019/QH14"],
        link="https://x/Bo-luat-lao-dong-2019-333670.aspx",
    )
    index.add_article_unit("d9", "76", "d9::i0::art76")
    citations = extract_citations(
        "Căn cứ Điều 76 Bộ luật Lao động 2019 quy định về ký kết. "
        "Theo Điều 5 Nghị định này thì áp dụng."
    )
    assert len(citations) == 2
    resolved = index.resolve(citations[0])
    assert resolved == ("d9::i0::art76",)
    assert citations[1].self_reference is True
    assert index.resolve(citations[1]) == ()
    assert fold("Bộ luật Lao động 2019") == "bo luat lao dong 2019"
