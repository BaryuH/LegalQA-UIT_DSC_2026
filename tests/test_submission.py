from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.cli import main
from legal_rag.submission import (
    SubmissionError,
    SubmissionSpec,
    build_submission,
    validate_submission,
)


def _list_spec(*, ordering: str = "target") -> SubmissionSpec:
    return SubmissionSpec(
        container="list",
        fields=("case_id", "response"),
        id_field="case_id",
        answer_field="response",
        id_type="string",
        ordering=ordering,  # type: ignore[arg-type]
    )


def test_build_submission_uses_exact_fields_and_target_order() -> None:
    payload = build_submission(
        [
            {"id": "1", "answer": "Answer one.", "method": "direct"},
            {"id": "2", "answer": "Answer two.", "method": "direct"},
        ],
        ["2", "1"],
        _list_spec(),
    )

    assert payload == [
        {"case_id": "2", "response": "Answer two."},
        {"case_id": "1", "response": "Answer one."},
    ]
    assert all("method" not in record for record in payload)
    assert all("question" not in record for record in payload)


def test_build_submission_can_use_deterministic_canonical_id_order() -> None:
    payload = build_submission(
        [{"id": "2", "answer": "Two."}, {"id": "10", "answer": "Ten."}],
        ["2", "10"],
        _list_spec(ordering="id"),
    )

    assert [record["case_id"] for record in payload] == ["10", "2"]


def test_build_submission_enforces_exact_coverage_and_unique_predictions() -> None:
    with pytest.raises(SubmissionError, match="exactly cover"):
        build_submission([{"id": "1", "answer": "One."}], ["1", "2"], _list_spec())

    with pytest.raises(SubmissionError, match="Duplicate prediction ID"):
        build_submission(
            [{"id": "1", "answer": "One."}, {"id": "1", "answer": "Other."}],
            ["1"],
            _list_spec(),
        )

    with pytest.raises(SubmissionError, match="non-blank string"):
        build_submission([{"id": "1", "answer": "  "}], ["1"], _list_spec())


def test_dict_submission_formats_are_explicit_and_validated() -> None:
    answer_map_spec = SubmissionSpec(
        container="dict",
        fields=("id", "answer"),
        id_field="id",
        answer_field="answer",
        id_type="string",
        ordering="target",
        dict_mode="answer_map",
    )
    answer_map = build_submission(
        [{"id": "2", "answer": "Two."}, {"id": "1", "answer": "One."}],
        ["1", "2"],
        answer_map_spec,
    )
    assert answer_map == {"1": "One.", "2": "Two."}
    assert (
        validate_submission(
            answer_map,
            target_ids=["1", "2"],
            spec=answer_map_spec,
        ).record_count
        == 2
    )

    record_map_spec = SubmissionSpec(
        container="dict",
        fields=("case_id", "response"),
        id_field="case_id",
        answer_field="response",
        id_type="string",
        ordering="target",
        dict_mode="record_map",
    )
    assert build_submission(
        [{"id": "1", "answer": "One."}],
        ["1"],
        record_map_spec,
    ) == {"1": {"response": "One."}}


def test_submission_schema_rejects_unapproved_question_and_metadata_fields() -> None:
    with pytest.raises(SubmissionError, match="unsupported field"):
        SubmissionSpec(
            container="list",
            fields=("id", "answer", "question"),
            id_field="id",
            answer_field="answer",
            id_type="string",
            ordering="target",
        )

    with pytest.raises(SubmissionError, match="internal field"):
        SubmissionSpec(
            container="list",
            fields=("id", "metadata"),
            id_field="id",
            answer_field="metadata",
            id_type="string",
            ordering="target",
        )


def test_cli_create_and_validate_submission(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {"id": "1", "answer": "Một.", "method": "direct"},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"id": "2", "answer": "Hai.", "method": "direct"},
                    ensure_ascii=False,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {"2": {"question": "q2"}, "1": {"question": "q1"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "submission.zip"

    assert (
        main(
            [
                "create-submission",
                "--predictions",
                str(predictions_path),
                "--questions",
                str(questions_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    create_output = json.loads(capsys.readouterr().out)
    assert create_output["written_answers"] == 2
    assert create_output["zip_members"] == ["submission.json"]

    from zipfile import ZipFile

    with ZipFile(output_path) as archive:
        assert archive.namelist() == ["submission.json"]
        written = json.loads(archive.read("submission.json").decode("utf-8"))
    assert written == {
        "2": {"answer": "Hai."},
        "1": {"answer": "Một."},
    }

    assert (
        main(
            [
                "validate-submission",
                "--submission",
                str(output_path),
                "--questions",
                str(questions_path),
            ]
        )
        == 0
    )
    validate_output = json.loads(capsys.readouterr().out)
    assert validate_output["written_answers"] == 2
