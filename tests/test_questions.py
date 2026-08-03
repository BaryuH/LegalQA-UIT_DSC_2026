"""Contract tests for the observed competition question-map loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from legal_rag.questions import QuestionLoadError, inference_view, load_questions

REPO_ROOT = Path(__file__).resolve().parents[1]
WARMUP_PATH = REPO_ROOT / "data" / "warmup.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_observed_dict_and_nested_schema_with_explicit_split() -> None:
    records = load_questions(WARMUP_PATH, split="warmup")

    assert len(records) == 500
    assert records[0].id == min(record.id for record in records)
    assert all(record.split == "warmup" for record in records)
    assert all(record.answer is not None for record in records)
    assert list(records) == sorted(records, key=lambda record: record.id)


def test_id_is_string_and_raw_unicode_question_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "questions.json"
    question = "  Người lao động được nghỉ\nbao nhiêu ngày?  "
    source.write_text(
        json.dumps(
            {"42": {"question": question, "answer": "Đáp án."}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    record = load_questions(source, split="fixture")[0]

    assert record.id == "42"
    assert record.question == question
    assert record.answer == "Đáp án."


def test_missing_gold_is_allowed_and_inference_view_has_no_gold(tmp_path: Path) -> None:
    source = tmp_path / "inference.json"
    source.write_text(
        json.dumps({"7": {"question": "Câu hỏi không có đáp án."}}, ensure_ascii=False),
        encoding="utf-8",
    )

    record = load_questions(source, split="public-official")[0]
    safe = inference_view((record,))[0]

    assert record.answer is None
    assert safe.model_dump(mode="json") == {
        "id": "7",
        "question": "Câu hỏi không có đáp án.",
        "split": "public-official",
    }
    assert "answer" not in safe.model_dump_json()


def test_blank_question_fails_with_path_and_record_key(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text(
        json.dumps(
            {"bad-1": {"question": "   ", "answer": "Đáp án."}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(QuestionLoadError) as error:
        load_questions(source, split="fixture")

    message = str(error.value)
    assert str(source) in message
    assert "record key 'bad-1'" in message
    assert "Text must not be blank" in message


def test_duplicate_json_id_fails_with_path_and_key(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.json"
    source.write_text(
        '{"1": {"question": "Một?", "answer": "Một."}, '
        '"1": {"question": "Trùng?", "answer": "Trùng."}}',
        encoding="utf-8",
    )

    with pytest.raises(QuestionLoadError, match="duplicate") as error:
        load_questions(source, split="fixture")

    assert str(source) in str(error.value)
    assert "record key '1'" in str(error.value)


def test_unknown_fields_fail_closed_with_path_and_record_key(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps(
            {"1": {"question": "Câu hỏi?", "answer": "Đáp án.", "extra": 1}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(QuestionLoadError, match="unexpected field") as unknown_error:
        load_questions(unknown, split="fixture")
    assert str(unknown) in str(unknown_error.value)
    assert "record key '1'" in str(unknown_error.value)


def test_source_hash_is_unchanged_before_and_after_loading() -> None:
    before = _sha256(WARMUP_PATH)

    records = load_questions(WARMUP_PATH, split="warmup")

    after = _sha256(WARMUP_PATH)
    assert records
    assert before == after
