"""Golden and failure-mode tests for the local evaluation contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from legal_rag.evaluation import (
    AlignmentError,
    DuplicateIDError,
    InputRecord,
    align_records,
    evaluate_records,
    normalize_text,
)
from legal_rag.evaluation.io import load_records

REPO_ROOT = Path(__file__).resolve().parents[1]


def _evaluate(reference: str, prediction: str):
    return evaluate_records(
        [InputRecord(id="case-1", answer=reference)],
        [InputRecord(id="case-1", answer=prediction)],
    )


def test_exact_match_has_perfect_local_scores() -> None:
    report = _evaluate("Căn cứ Điều 37.", "Căn cứ Điều 37.")

    assert report.artifact["metrics"]["meteor"] == 1.0
    assert report.artifact["metrics"]["rouge_l"] == 1.0
    assert report.cases[0].status == "scored"
    assert report.cases[0].meteor is not None
    assert report.cases[0].meteor.matches == 5


def test_whitespace_and_newline_normalize_to_the_same_tokens() -> None:
    report = _evaluate(
        "Người sử dụng lao động phải trả lương.",
        "  Người  sử dụng\r\nlao động phải trả lương.  ",
    )

    case = report.cases[0]
    assert case.normalized_prediction.text == "Người  sử dụng\nlao động phải trả lương."
    assert case.normalized_reference.tokens == case.normalized_prediction.tokens
    assert report.artifact["metrics"]["meteor"] == 1.0
    assert report.artifact["metrics"]["rouge_l"] == 1.0


def test_partial_phrase_overlap_is_between_zero_and_one() -> None:
    report = _evaluate(
        "Người lao động có quyền nghỉ phép năm.",
        "Người lao động có quyền.",
    )

    assert 0.0 < report.artifact["metrics"]["meteor"] < 1.0
    assert 0.0 < report.artifact["metrics"]["rouge_l"] < 1.0


def test_macro_aggregate_uses_scored_cases_and_records_versions() -> None:
    report = evaluate_records(
        [
            InputRecord(id="1", answer="Một câu."),
            InputRecord(id="2", answer="Hai câu."),
        ],
        [
            InputRecord(id="1", answer="Một câu."),
            InputRecord(id="2", answer="Hai."),
        ],
    )

    assert report.artifact["counts"]["scored"] == 2
    assert report.artifact["aggregation"]["method"] == "macro_over_scored_cases"
    assert report.artifact["metrics"]["meteor"] == pytest.approx(
        sum(case.meteor.score for case in report.cases if case.meteor) / 2
    )
    assert report.artifact["evaluator_version"] == "local-v1"
    assert report.artifact["normalization"]["unicode_form"] == "NFC"


def test_empty_prediction_is_structured_and_not_scored() -> None:
    report = _evaluate("Một câu trả lời hợp lệ.", "   \n")

    assert report.cases[0].status == "empty_prediction"
    assert report.cases[0].error_code == "EMPTY_PREDICTION"
    assert report.artifact["counts"]["empty_prediction"] == 1
    assert report.artifact["counts"]["scored"] == 0
    assert report.artifact["metrics"]["meteor"] is None
    assert report.artifact["metrics"]["rouge_l"] is None


def test_missing_extra_and_duplicate_ids_fail_closed() -> None:
    references = [
        InputRecord(id="1", answer="Một."),
        InputRecord(id="2", answer="Hai."),
    ]
    with pytest.raises(AlignmentError, match="missing prediction ID"):
        align_records(references, [InputRecord(id="1", answer="Một.")])
    with pytest.raises(AlignmentError, match="extra prediction ID"):
        align_records(
            [InputRecord(id="1", answer="Một.")],
            [
                InputRecord(id="1", answer="Một."),
                InputRecord(id="2", answer="Hai."),
            ],
        )
    with pytest.raises(DuplicateIDError, match="duplicate ID"):
        align_records(
            [InputRecord(id="1", answer="Một.")],
            [
                InputRecord(id="1", answer="Một."),
                InputRecord(id="1", answer="Một lần nữa."),
            ],
        )


def test_vietnamese_decomposed_diacritics_are_normalized() -> None:
    reference = "Tài liệu pháp lý."
    prediction = "Ta\u0300i lie\u0302\u0323u pha\u0301p ly\u0301."
    normalized_reference = normalize_text(reference)
    normalized_prediction = normalize_text(prediction)

    assert normalized_reference.text == normalized_prediction.text
    assert normalized_reference.tokens == normalized_prediction.tokens
    assert _evaluate(reference, prediction).artifact["metrics"]["meteor"] == 1.0


def test_repeated_tokens_are_matched_one_to_one_deterministically() -> None:
    report = _evaluate("điều điều khoản", "điều khoản điều")
    repeat = _evaluate("điều điều khoản", "điều khoản điều")

    assert report.cases[0].meteor is not None
    assert report.cases[0].meteor.matches == 3
    assert report.cases[0].meteor.chunks == 3
    assert report.artifact == repeat.artifact


def test_punctuation_and_legal_code_tokens_are_explicit() -> None:
    normalized = normalize_text("Theo Điều 153/2020/NĐ-CP, khoản 2.")

    assert normalized.tokens == (
        "Theo",
        "Điều",
        "153/2020/NĐ-CP",
        ",",
        "khoản",
        "2",
        ".",
    )
    punctuation_difference = _evaluate("Điều 37.", "Điều 37")
    assert punctuation_difference.cases[0].normalized_reference.tokens[-1] == "."
    assert punctuation_difference.artifact["metrics"]["rouge_l"] < 1.0


def test_long_multibullet_answer_is_scored_without_truncation() -> None:
    answer = (
        "1. Người lao động có quyền nghỉ hằng năm.\n"
        "2. Người sử dụng lao động phải bảo đảm điều kiện làm việc an toàn.\n"
        "3. Tranh chấp được giải quyết theo trình tự pháp luật hiện hành."
    )
    report = _evaluate(answer, answer)

    assert report.cases[0].normalized_reference.text.count("\n") == 2
    assert report.artifact["metrics"]["meteor"] == 1.0
    assert report.artifact["metrics"]["rouge_l"] == 1.0


def test_jsonl_duplicate_ids_are_detected_by_alignment(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        '{"id": "1", "answer": "Một."}\n{"id": "1", "answer": "Trùng."}\n',
        encoding="utf-8",
    )
    predictions = load_records(predictions_path, "Prediction")

    with pytest.raises(DuplicateIDError, match="duplicate ID"):
        align_records([InputRecord(id="1", answer="Một.")], predictions)


def test_cli_writes_deterministic_metric_artifact(tmp_path: Path) -> None:
    references_path = tmp_path / "references.json"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "metrics.json"
    references_path.write_text(
        json.dumps(
            {"1": {"question": "q", "answer": "Căn cứ Điều 37."}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    predictions_path.write_text(
        json.dumps({"id": "1", "answer": "Căn cứ Điều 37."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "scripts/evaluate_predictions.py",
        "--references",
        str(references_path),
        "--predictions",
        str(predictions_path),
        "--output",
        str(output_path),
        "--run-id",
        "golden-cli",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["run_id"] == "golden-cli"
    assert artifact["metrics"]["meteor"] == 1.0
    assert artifact["per_case"][0]["id"] == "1"
    assert "normalized_reference" not in artifact["per_case"][0]


def test_cli_help_works() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_predictions.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--references" in completed.stdout
