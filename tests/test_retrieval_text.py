"""Acceptance tests for retrieval-only text views."""

from legal_rag.text import normalize_retrieval_text, tokenize_legal_text


def test_normalization_is_nfc_and_collapses_only_the_derived_view() -> None:
    raw = "  Ta\u0300i   lie\u0302\u0323u\r\npha\u0301p   ly\u0301.  "

    derived = normalize_retrieval_text(raw)

    assert derived == "Tài liệu pháp lý."
    assert raw == "  Ta\u0300i   lie\u0302\u0323u\r\npha\u0301p   ly\u0301.  "


def test_vietnamese_diacritics_and_d_are_not_folded() -> None:
    tokens = tokenize_legal_text("Đất đai và dat đai.")

    assert "Đất" in tokens
    assert "đai" in tokens
    assert "dat" in tokens
    assert "đai" != "dai"


def test_legal_codes_dates_money_and_duration_tokens_are_retained() -> None:
    tokens = tokenize_legal_text(
        "Theo 153/2020/NĐ-CP và 65/2022/NĐ-CP, ngày 16/09/2022, "
        "phạt 10.000.000 đồng trong 12 tháng."
    )

    assert "153/2020/NĐ-CP" in tokens
    assert "65/2022/NĐ-CP" in tokens
    assert "16/09/2022" in tokens
    assert "10.000.000" in tokens
    assert "12" in tokens
    assert "tháng" in tokens


def test_numbers_slashes_hyphens_and_punctuation_are_not_deleted() -> None:
    tokens = tokenize_legal_text("Điều 37 khoản 2-3, tỷ lệ 1/2.")

    assert "37" in tokens
    assert "2" in tokens
    assert "3" in tokens
    assert "1/2" in tokens
    assert "-" in tokens
    assert "," in tokens
    assert "." in tokens
