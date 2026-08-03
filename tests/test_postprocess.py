from dataclasses import fields

import pytest

from legal_rag.generation import (
    AnswerPostprocessResult,
    PostprocessResult,
    postprocess_answer,
)


def test_postprocess_stores_raw_and_removes_outer_wrappers() -> None:
    raw = (
        "\r\n  ```text\r\n"
        "FINAL_ANSWER:\r\n"
        "- Khoản thứ nhất.\r\n\r\n"
        "Đoạn giải thích thứ hai.\r\n"
        "```\r\n  "
    )

    result = postprocess_answer(raw)

    assert isinstance(result, AnswerPostprocessResult)
    assert PostprocessResult is AnswerPostprocessResult
    assert result.raw_answer == raw
    assert result.cleaned_answer == ("- Khoản thứ nhất.\n\nĐoạn giải thích thứ hai.")
    assert {field.name for field in fields(result)} == {
        "raw_answer",
        "cleaned_answer",
    }


def test_postprocess_preserves_bullets_paragraphs_numbers_dates_and_old_law() -> None:
    raw = (
        "CÂU TRẢ LỜI:\r\n"
        "1. Quy định hiện hành tại Điều 37.\r\n"
        "\r\n"
        "2. Quy định trước đây áp dụng đến ngày 01/01/2020.\r\n"
        "\r\n"
        "Căn cứ Nghị định 123/2020/NĐ-CP."
    )

    result = postprocess_answer(raw)

    assert result.cleaned_answer == (
        "1. Quy định hiện hành tại Điều 37.\n\n"
        "2. Quy định trước đây áp dụng đến ngày 01/01/2020.\n\n"
        "Căn cứ Nghị định 123/2020/NĐ-CP."
    )


@pytest.mark.parametrize(
    "raw",
    [
        "FINAL_ANSWER:\nNội dung.",
        "CÂU TRẢ LỜI:\nNội dung.",
    ],
)
def test_only_exact_prefixes_are_removed(raw: str) -> None:
    assert postprocess_answer(raw).cleaned_answer == "Nội dung."


def test_non_exact_prefixes_and_non_outer_fences_are_preserved() -> None:
    lower_case = postprocess_answer("final_answer:\nNội dung.")
    spaced = postprocess_answer("FINAL_ANSWER :\nNội dung.")
    inner_fence = postprocess_answer("Mở đầu\n```\nNội dung\n```\nKết thúc")
    unmatched = postprocess_answer("```\nNội dung")

    assert lower_case.cleaned_answer == "final_answer:\nNội dung."
    assert spaced.cleaned_answer == "FINAL_ANSWER :\nNội dung."
    assert inner_fence.cleaned_answer == "Mở đầu\n```\nNội dung\n```\nKết thúc"
    assert unmatched.cleaned_answer == "```\nNội dung"


def test_blank_or_non_string_raw_answer_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-blank"):
        postprocess_answer(" \r\n \t")
    with pytest.raises(TypeError, match="must be a string"):
        postprocess_answer(None)  # type: ignore[arg-type]


def test_code_fence_without_language_tag_is_removed() -> None:
    result = postprocess_answer("```\n- Một ý.\n- Ý tiếp theo.\n```")

    assert result.cleaned_answer == "- Một ý.\n- Ý tiếp theo."
