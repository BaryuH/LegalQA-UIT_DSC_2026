"""Tests for post-generation repetition trimming."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "sedar_retrieval"
    / "trim_degenerate_repetition.py"
)
_spec = importlib.util.spec_from_file_location("_trim_module", _MODULE_PATH)
assert _spec and _spec.loader
_trim_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trim_module)
trim = _trim_module.trim


def test_a_run_of_repeats_is_cut_at_the_third_occurrence() -> None:
    answer = "\n".join(
        ["Căn cứ theo Điều 1 Thông tư 15/2020/TT-BGDĐT quy định như sau:"]
        + [
            f"{i}. Bài thi được làm thành 02 bản dùng để chấm thi và in kết quả."
            for i in range(1, 6)
        ]
    )
    trimmed = trim(answer)
    assert trimmed != answer
    assert trimmed.count("02 bản dùng để chấm thi") == 2


def test_quoting_then_restating_once_is_legitimate_and_kept() -> None:
    """Legal answers quote a list then restate it. That is not a defect."""

    answer = (
        "Căn cứ Điều 11 Nghị định 113/2015/NĐ-CP quy định như sau:\n"
        "1. Mức 0,1 áp dụng với nhà giáo dạy thực hành có một yếu tố độc hại.\n"
        "2. Mức 0,2 áp dụng với nhà giáo dạy thực hành có hai yếu tố độc hại.\n"
        "Như vậy, mức hưởng phụ cấp được quy định như sau:\n"
        "- Mức 0,1 áp dụng với nhà giáo dạy thực hành có một yếu tố độc hại.\n"
        "- Mức 0,2 áp dụng với nhà giáo dạy thực hành có hai yếu tố độc hại."
    )
    assert trim(answer) == answer


def test_list_markers_do_not_hide_a_repeat() -> None:
    """A renumbered repeat is still a repeat."""

    line = "Bài thi được làm thành 02 bản dùng để chấm thi và in kết quả thi."
    answer = "\n".join(f"{i}. {line}" for i in range(1, 5))
    assert trim(answer).count(line) == 2


def test_short_connectives_are_not_treated_as_repetition() -> None:
    answer = "\n".join(
        [
            "Căn cứ Điều 5 Luật Cư trú 2020 quy định như sau:",
            "Như vậy:",
            "- Điều kiện thứ nhất là có chỗ ở hợp pháp tại nơi đăng ký thường trú.",
            "Như vậy:",
            "- Điều kiện thứ hai là được chủ hộ đồng ý bằng văn bản theo quy định.",
            "Như vậy:",
        ]
    )
    assert trim(answer) == answer


def test_an_answer_without_repetition_is_returned_unchanged() -> None:
    answer = "Căn cứ Điều 28 Luật Cư trú 2020, hồ sơ đăng ký tạm trú gồm hai loại."
    assert trim(answer) == answer


def test_trimming_rolls_back_to_a_sentence_boundary() -> None:
    line = "Bài thi được làm thành 02 bản dùng để chấm thi và in kết quả thi."
    answer = "\n".join([f"{i}. {line}" for i in range(1, 5)] + ["Còn dang dở"])
    trimmed = trim(answer)
    assert not trimmed.endswith("Còn dang dở")
    assert trimmed.rstrip().endswith(".")
