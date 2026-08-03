from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from pydantic import ValidationError

from legal_rag.schemas import SubmissionPayload
from legal_rag.submission import (
    SubmissionError,
    build_submission_payload,
    create_submission_zip,
    load_submission_question_ids,
    validate_submission_json,
    validate_submission_zip,
    write_submission_json,
)


def _questions(path: Path) -> tuple[str, ...]:
    path.write_text(
        json.dumps(
            {
                "007": {"question": "Câu hỏi bảy"},
                "2": {"question": "Câu hỏi hai"},
                "1": {"question": "Câu hỏi một"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return load_submission_question_ids(path)


def _write_submission(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in members:
            archive.writestr(name, data)


def test_typed_submission_payload_forbids_metadata_and_coercion() -> None:
    payload = SubmissionPayload.model_validate({"1": {"answer": ""}})
    assert payload.root["1"].answer == ""
    with pytest.raises(ValidationError):
        SubmissionPayload.model_validate({"1": {"answer": 1}})
    with pytest.raises(ValidationError):
        SubmissionPayload.model_validate({"1": {"answer": "ok", "model": "x"}})


def test_builder_preserves_dataset_order_and_string_id_spelling() -> None:
    expected = ("007", "2", "1")
    payload = build_submission_payload(
        (
            {"id": 1, "answer": "Một", "method": "direct"},
            {"id": "007", "answer": "Bảy", "status": "success"},
            {"id": "2", "answer": "Hai", "raw_answer": "Hai"},
        ),
        expected,
    )

    assert list(payload) == ["007", "2", "1"]
    assert payload["007"] == {"answer": "Bảy"}
    assert payload["1"] == {"answer": "Một"}


def test_builder_allows_empty_answer_but_never_coerces_answer() -> None:
    payload = build_submission_payload(({"id": "1", "answer": ""},), ("1",))
    assert payload == {"1": {"answer": ""}}
    with pytest.raises(SubmissionError) as error:
        build_submission_payload(({"id": "1", "answer": 1},), ("1",))
    assert error.value.code == "SUBMISSION_ANSWER_NOT_STRING"


@pytest.mark.parametrize(
    ("predictions", "expected_code"),
    [
        (({"id": "1", "answer": "one"},), "SUBMISSION_MISSING_QUESTION_ID"),
        (
            ({"id": "1", "answer": "one"}, {"id": "1", "answer": "again"}),
            "SUBMISSION_DUPLICATE_QUESTION_ID",
        ),
        (({"id": True, "answer": "bad"},), "SUBMISSION_INVALID_QUESTION_ID"),
        (({"id": "1"},), "SUBMISSION_MISSING_ANSWER"),
        (({"id": "1", "answer": [], "metadata": {}},), "SUBMISSION_EXTRA_FIELDS"),
    ],
)
def test_builder_fails_closed_with_typed_errors(
    predictions: tuple[object, ...], expected_code: str
) -> None:
    with pytest.raises(SubmissionError) as error:
        build_submission_payload(predictions, ("1", "2"))
    assert error.value.code == expected_code


def test_builder_rejects_extra_prediction_id() -> None:
    with pytest.raises(SubmissionError) as error:
        build_submission_payload(
            ({"id": "1", "answer": "one"}, {"id": "3", "answer": "three"}),
            ("1",),
        )
    assert error.value.code == "SUBMISSION_EXTRA_QUESTION_ID"


def test_question_id_loader_preserves_input_order_and_does_not_need_answers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "questions.json"
    expected = _questions(path)
    assert expected == ("007", "2", "1")

    list_path = tmp_path / "records.json"
    list_path.write_text(
        json.dumps(
            [{"id": 9, "question": "q9"}, {"id": "9", "question": "q9 again"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(SubmissionError) as error:
        load_submission_question_ids(list_path)
    assert error.value.code == "SUBMISSION_DUPLICATE_QUESTION_ID"


def test_json_writer_is_utf8_non_ascii_and_parse_back_validated(tmp_path: Path) -> None:
    expected = ("007", "1")
    path = tmp_path / "submission.json"
    report = write_submission_json(
        {"007": {"answer": "Đáp án tiếng Việt"}, "1": {"answer": ""}},
        path,
        expected,
    )
    raw = path.read_bytes()
    assert report.valid
    assert "Đáp án tiếng Việt".encode() in raw
    assert b"\\u0110" not in raw
    assert json.loads(raw.decode("utf-8")) == {
        "007": {"answer": "Đáp án tiếng Việt"},
        "1": {"answer": ""},
    }
    assert report.warnings == ("EMPTY_ANSWER:1",)


def test_json_validator_rejects_top_level_value_types_and_extra_fields(
    tmp_path: Path,
) -> None:
    cases = (
        ([], "SUBMISSION_TOP_LEVEL_NOT_OBJECT"),
        ({"1": "answer"}, "SUBMISSION_VALUE_NOT_OBJECT"),
        ({"1": {}}, "SUBMISSION_MISSING_ANSWER"),
        ({"1": {"answer": None}}, "SUBMISSION_ANSWER_NOT_STRING"),
        ({"1": {"answer": 3}}, "SUBMISSION_ANSWER_NOT_STRING"),
        ({"1": {"answer": True}}, "SUBMISSION_ANSWER_NOT_STRING"),
        ({"1": {"answer": []}}, "SUBMISSION_ANSWER_NOT_STRING"),
        ({"1": {"answer": "ok", "method": "direct"}}, "SUBMISSION_EXTRA_FIELDS"),
    )
    for index, (payload, code) in enumerate(cases):
        path = tmp_path / f"case-{index}" / "submission.json"
        path.parent.mkdir()
        _write_submission(path, payload)
        report = validate_submission_json(path, ("1",))
        assert not report.valid
        assert code in report.error_codes


def test_json_validator_detects_duplicate_keys_invalid_encoding_and_filename(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate" / "submission.json"
    duplicate.parent.mkdir()
    duplicate.write_bytes(b'{"1":{"answer":"one"},"1":{"answer":"two"}}')
    report = validate_submission_json(duplicate, ("1",))
    assert "SUBMISSION_DUPLICATE_QUESTION_ID" in report.error_codes

    invalid_utf8 = tmp_path / "invalid-utf8" / "submission.json"
    invalid_utf8.parent.mkdir()
    invalid_utf8.write_bytes(b"\xff")
    report = validate_submission_json(invalid_utf8, ("1",))
    assert "SUBMISSION_INVALID_UTF8" in report.error_codes

    wrong_name = tmp_path / "wrong.json"
    wrong_name.write_text('{"1":{"answer":"one"}}', encoding="utf-8")
    report = validate_submission_json(wrong_name, ("1",))
    assert "SUBMISSION_WRONG_FILENAME" in report.error_codes
    assert not report.valid


def test_empty_answer_rejection_is_explicit_policy(tmp_path: Path) -> None:
    path = tmp_path / "submission.json"
    _write_submission(path, {"1": {"answer": ""}})
    report = validate_submission_json(path, ("1",), reject_empty_answers=True)
    assert not report.valid
    assert "SUBMISSION_EMPTY_ANSWER" in report.error_codes


def test_json_writer_escapes_newlines_and_quotes_without_rewriting(
    tmp_path: Path,
) -> None:
    answer = 'Dòng 1\n"Dòng 2"'
    first = tmp_path / "first" / "submission.json"
    second = tmp_path / "second" / "submission.json"
    write_submission_json({"1": {"answer": answer}}, first, ("1",))
    write_submission_json({"1": {"answer": answer}}, second, ("1",))
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8"))["1"]["answer"] == answer


def test_zip_packaging_has_exact_single_root_member_and_is_validated(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "submission.json"
    _write_submission(json_path, {"1": {"answer": "Đáp án"}})
    zip_path = tmp_path / "submission.zip"
    report = create_submission_zip(json_path, zip_path, ("1",))
    assert report.valid
    assert validate_submission_zip(zip_path, ("1",)).valid
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["submission.json"]
        assert json.loads(archive.read("submission.json")) == {
            "1": {"answer": "Đáp án"}
        }


@pytest.mark.parametrize(
    "members",
    [
        [],
        [("nested/submission.json", b'{"1":{"answer":"one"}}')],
        [
            ("submission.json", b'{"1":{"answer":"one"}}'),
            ("__MACOSX/._submission.json", b"metadata"),
        ],
        [("submission.json", b"not-json")],
    ],
)
def test_zip_validator_rejects_wrong_layout_extra_member_and_bad_inner_json(
    tmp_path: Path, members: list[tuple[str, bytes]]
) -> None:
    path = tmp_path / "submission.zip"
    _write_zip(path, members)
    report = validate_submission_zip(path, ("1",))
    assert not report.valid
    assert report.error_codes


def test_zip_validator_rejects_wrong_filename_and_corruption(tmp_path: Path) -> None:
    wrong_name = tmp_path / "answers.zip"
    _write_zip(wrong_name, [("submission.json", b'{"1":{"answer":"one"}}')])
    report = validate_submission_zip(wrong_name, ("1",))
    assert "SUBMISSION_WRONG_FILENAME" in report.error_codes

    corrupt = tmp_path / "submission.zip"
    corrupt.write_bytes(b"not-a-zip")
    report = validate_submission_zip(corrupt, ("1",))
    assert "SUBMISSION_ZIP_INVALID" in report.error_codes


def test_failed_packaging_does_not_leave_old_or_partial_final_zip(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "submission.json"
    _write_submission(json_path, {"1": {"answer": "one"}})
    output = tmp_path / "submission.zip"
    output.write_bytes(b"existing")
    with pytest.raises(SubmissionError):
        create_submission_zip(json_path, output, ("1",))
    assert output.read_bytes() == b"existing"
