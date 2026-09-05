"""Acceptance tests for the deterministic error taxonomy classifier."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from legal_rag.evaluation import (
    ClassifierSignals,
    ClassifierThresholds,
    classify_case,
    classify_cases,
    generate_error_report,
)
from scripts.generate_error_report import main as generate_error_report_cli

ARTICLE_12 = "doc-a::art::12"
ARTICLE_13 = "doc-a::art::13"
ARTICLE_14 = "doc-a::art::14"


def _signals(**overrides: object) -> ClassifierSignals:
    values: dict[str, object] = {
        "identifier": "case-a",
        "prediction": "Theo Điều 12, người sử dụng phải thực hiện nghĩa vụ.",
        "reference": "Theo Điều 12, người sử dụng phải thực hiện nghĩa vụ.",
        "meteor": 0.5,
        "rouge_l": 0.5,
        "metric_status": "scored",
        "retrieval_status": "success",
        "raw_hit_ids": ("hit-a",),
        "packed_chunk_ids": ("pack-a",),
        "packed_dropped_ids": (),
        "packed_truncated_ids": (),
        "gold_article_keys": frozenset({ARTICLE_12}),
        "gold_document_ids": frozenset({"doc-a"}),
        "packed_article_keys": frozenset({ARTICLE_12}),
        "packed_document_ids": frozenset({"doc-a"}),
        "hit_article_keys": frozenset({ARTICLE_12}),
        "prediction_tokens": 20,
        "max_new_tokens": 100,
        "dropped_article_keys": frozenset(),
        "truncated_article_keys": frozenset(),
        "reference_tokens": 20,
    }
    values.update(overrides)
    return ClassifierSignals(**values)  # type: ignore[arg-type]


def test_rule_1_generation_failure() -> None:
    result = classify_case(_signals(prediction=None))
    assert (result.error_type, result.reason_code) == (
        "GENERATION_FAILURE",
        "metric_status_not_ok",
    )


def test_rule_2_empty_answer() -> None:
    result = classify_case(_signals(prediction=" \n\t"))
    assert (result.error_type, result.reason_code) == (
        "EMPTY_ANSWER",
        "blank_prediction",
    )


def test_rule_3_context_load_error() -> None:
    result = classify_case(_signals(retrieval_status="no_packed_evidence"))
    assert (result.error_type, result.reason_code) == (
        "CONTEXT_LOAD_ERROR",
        "retrieval_status_no_packed_evidence",
    )


def test_rule_4_no_silver_label() -> None:
    result = classify_case(_signals(gold_article_keys=frozenset()))
    assert (result.error_type, result.reason_code) == (
        "OTHER",
        "no_silver_label",
    )


def test_rule_5_retrieval_miss() -> None:
    result = classify_case(
        _signals(
            gold_article_keys=frozenset({ARTICLE_14}),
            gold_document_ids=frozenset({"doc-a"}),
            packed_article_keys=frozenset({ARTICLE_12}),
            hit_article_keys=frozenset({ARTICLE_12}),
        )
    )
    assert (result.error_type, result.reason_code) == (
        "RETRIEVAL_MISS",
        "gold_article_absent_from_hits",
    )


def test_rule_6_evidence_truncation_when_gold_was_dropped_by_budget() -> None:
    result = classify_case(
        _signals(
            packed_article_keys=frozenset(),
            packed_document_ids=frozenset(),
            packed_dropped_ids=("dropped-a",),
            dropped_article_keys=frozenset({ARTICLE_12}),
        )
    )
    assert (result.error_type, result.reason_code, result.confidence) == (
        "EVIDENCE_TRUNCATION",
        "gold_dropped_by_budget",
        "rule_certain",
    )


def test_rule_6_also_fires_for_a_truncated_gold_passage() -> None:
    result = classify_case(
        _signals(
            packed_article_keys=frozenset(),
            packed_document_ids=frozenset(),
            packed_truncated_ids=("truncated-a",),
            truncated_article_keys=frozenset({ARTICLE_12}),
        )
    )
    assert result.error_type == "EVIDENCE_TRUNCATION"


def test_budget_drop_without_projection_is_not_guessed() -> None:
    """A dropped pack with no article projection must not be called either way."""

    result = classify_case(
        _signals(
            packed_article_keys=frozenset(),
            packed_document_ids=frozenset(),
            gold_document_ids=frozenset({"doc-z"}),
            packed_dropped_ids=("dropped-a",),
        )
    )
    assert (result.error_type, result.reason_code) == (
        "OTHER",
        "budget_projection_unavailable",
    )


def test_certain_document_rule_still_wins_over_budget_ambiguity() -> None:
    result = classify_case(
        _signals(
            packed_article_keys=frozenset({ARTICLE_13}),
            packed_document_ids=frozenset({"doc-a"}),
            packed_dropped_ids=("dropped-a",),
        )
    )
    assert result.error_type == "RIGHT_DOCUMENT_WRONG_CHUNK"


def test_rule_7_reranking_regression() -> None:
    result = classify_case(
        _signals(
            packed_article_keys=frozenset(),
            packed_document_ids=frozenset(),
        )
    )
    assert (result.error_type, result.reason_code) == (
        "RERANKING_REGRESSION",
        "gold_outranked_in_pack",
    )


def test_rule_8_right_document_wrong_article() -> None:
    result = classify_case(
        _signals(
            packed_article_keys=frozenset({ARTICLE_13}),
            packed_document_ids=frozenset({"doc-a"}),
            packed_dropped_ids=("dropped-a",),
        )
    )
    assert (result.error_type, result.reason_code) == (
        "RIGHT_DOCUMENT_WRONG_CHUNK",
        "right_doc_wrong_article",
    )


def test_rule_9_wrong_article_citation() -> None:
    result = classify_case(
        _signals(
            prediction="Căn cứ Điều 13, người sử dụng phải thực hiện nghĩa vụ.",
            reference="Căn cứ Điều 12, người sử dụng phải thực hiện nghĩa vụ.",
        )
    )
    assert (result.error_type, result.reason_code) == (
        "WRONG_ARTICLE_CITATION",
        "citation_set_disjoint",
    )


def test_rule_10_generation_cap() -> None:
    result = classify_case(_signals(prediction_tokens=97))
    assert (result.error_type, result.reason_code) == (
        "UNDER_SPECIFIED",
        "hit_generation_cap",
    )


def test_rule_11_over_verbose_needs_both_length_and_precision_loss() -> None:
    result = classify_case(
        _signals(
            prediction_tokens=60,
            reference_tokens=20,
            max_new_tokens=None,
            meteor=0.50,
            rouge_l=0.30,
        )
    )
    assert (result.error_type, result.reason_code, result.confidence) == (
        "OVER_VERBOSE",
        "long_and_precision_loss",
        "rule_heuristic",
    )


def test_rule_11_does_not_fire_on_length_alone() -> None:
    """A long answer that kept its precision is not OVER_VERBOSE."""

    result = classify_case(
        _signals(
            prediction_tokens=60,
            reference_tokens=20,
            max_new_tokens=None,
            meteor=0.50,
            rouge_l=0.49,
        )
    )
    assert result.error_type == "OTHER"


def test_rule_12_under_specified_by_length_ratio() -> None:
    result = classify_case(
        _signals(prediction_tokens=8, reference_tokens=40, max_new_tokens=None)
    )
    assert (result.error_type, result.reason_code, result.confidence) == (
        "UNDER_SPECIFIED",
        "too_short_vs_reference",
        "rule_heuristic",
    )


def test_generation_cap_outranks_the_length_ratio_rules() -> None:
    result = classify_case(
        _signals(prediction_tokens=97, reference_tokens=20, max_new_tokens=100)
    )
    assert result.reason_code == "hit_generation_cap"


def test_length_rules_fail_closed_without_a_token_count() -> None:
    """A missing token count must never fall back to a word count."""

    for overrides in (
        {"prediction_tokens": None, "max_new_tokens": None},
        {"reference_tokens": None, "max_new_tokens": None},
    ):
        result = classify_case(_signals(**overrides))
        assert (result.error_type, result.reason_code) == (
            "OTHER",
            "token_count_unavailable",
        )


def test_zero_reference_token_count_does_not_divide_by_zero() -> None:
    result = classify_case(
        _signals(prediction_tokens=20, reference_tokens=0, max_new_tokens=None)
    )
    assert (result.error_type, result.reason_code) == (
        "OTHER",
        "reference_token_count_invalid",
    )


def test_rule_13_unclassified() -> None:
    # Gold is packed, citations agree, the cap was not reached and the
    # length ratio is unremarkable: nothing in the cascade applies.
    result = classify_case(_signals())
    assert (result.error_type, result.reason_code) == (
        "OTHER",
        "unclassified",
    )


def test_rule_priority_retrieval_miss_precedes_citation_mismatch() -> None:
    result = classify_case(
        _signals(
            gold_article_keys=frozenset({ARTICLE_14}),
            packed_article_keys=frozenset({ARTICLE_12}),
            hit_article_keys=frozenset({ARTICLE_12}),
            prediction="Căn cứ Điều 13.",
            reference="Căn cứ Điều 12.",
        )
    )
    assert (result.error_type, result.reason_code) == (
        "RETRIEVAL_MISS",
        "gold_article_absent_from_hits",
    )


def test_classify_cases_is_deterministic_for_input_order() -> None:
    first = _signals(identifier="case-a")
    second = _signals(
        identifier="case-b",
        prediction="Căn cứ Điều 13.",
        reference="Căn cứ Điều 12.",
    )
    assert classify_cases([first, second]) == classify_cases([second, first])


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _report_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "predictions": tmp_path / "predictions.jsonl",
        "references": tmp_path / "references.json",
        "metrics": tmp_path / "metrics.json",
        "retrieval": tmp_path / "retrieval.jsonl",
        "labels": tmp_path / "labels.jsonl",
        "passages": tmp_path / "passages.jsonl",
    }
    _write_jsonl(
        paths["predictions"],
        [{"id": "case-a", "answer": "Căn cứ Điều 12."}],
    )
    _write_json(paths["references"], {"case-a": "Căn cứ Điều 12."})
    _write_json(
        paths["metrics"],
        {
            "run_id": "taxonomy-test",
            "method": "fixture",
            "split": "warmup",
            "reference_role": "evaluation_reference_only",
            "per_case": [
                {
                    "id": "case-a",
                    "status": "scored",
                    "meteor": 0.5,
                    "rouge_l": 0.5,
                }
            ],
        },
    )
    _write_jsonl(
        paths["retrieval"],
        [
            {
                "id": "case-a",
                "status": "success",
                "raw_hit_ids": ["p12"],
                "packed_chunk_ids": ["p12"],
            }
        ],
    )
    _write_jsonl(
        paths["labels"],
        [
            {
                "query_id": "case-a",
                "relevant_ids": ["p12"],
                "provenance": "silver",
            }
        ],
    )
    passage = {
        "passage_id": "p12",
        "document_id": "doc-a",
        "article_number": "12",
        "retrieval_level": "article",
        "raw_text": "Điều 12.",
        "reader_text": "Điều 12.",
        "retrieval_text": "Điều 12.",
        "source": {
            "source_path": "selected-contexts.zip",
            "document_id": "doc-a",
            "content_hash": "hash-a",
        },
    }
    _write_jsonl(paths["passages"], [passage])
    return paths


def test_manual_label_overrides_auto_label_and_summary_is_final_type(
    tmp_path: Path,
) -> None:
    paths = _report_fixture(tmp_path)
    report = generate_error_report(
        paths["predictions"],
        paths["references"],
        paths["metrics"],
        retrieval_path=paths["retrieval"],
        auto_classify=True,
        manual_error_types={"case-a": "FORMAT_ERROR"},
        gold_article_keys={"case-a": frozenset({ARTICLE_12})},
        gold_document_ids={"case-a": frozenset({"doc-a"})},
        passage_article_keys={"p12": ARTICLE_12},
    )
    assert report.cases[0].error_type == "FORMAT_ERROR"
    assert report.cases[0].reason_code == ""
    assert report.classification_summary == {"FORMAT_ERROR": 1}


def test_default_report_schema_has_no_classification_columns(
    tmp_path: Path,
) -> None:
    paths = _report_fixture(tmp_path)
    report = generate_error_report(
        paths["predictions"],
        paths["references"],
        paths["metrics"],
        retrieval_path=paths["retrieval"],
    )
    assert "reason_code" not in report.to_markdown()
    assert "reason_code" not in report.to_csv()


def test_cli_auto_classify_is_fail_closed_without_labels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _report_fixture(tmp_path)
    exit_code = generate_error_report_cli(
        [
            "--predictions",
            str(paths["predictions"]),
            "--references",
            str(paths["references"]),
            "--metrics",
            str(paths["metrics"]),
            "--auto-classify",
            "--passages",
            str(paths["passages"]),
            "--markdown",
            str(tmp_path / "report.md"),
            "--csv",
            str(tmp_path / "report.csv"),
        ]
    )
    assert exit_code == 2
    assert "--labels" in capsys.readouterr().err


def test_cli_summary_contains_codes_but_not_reference_text(tmp_path: Path) -> None:
    paths = _report_fixture(tmp_path)
    reference = "SECRET_REFERENCE_TEXT"
    _write_json(paths["references"], {"case-a": reference})
    markdown = tmp_path / "report.md"
    csv_path = tmp_path / "report.csv"
    summary = tmp_path / "summary.json"
    exit_code = generate_error_report_cli(
        [
            "--predictions",
            str(paths["predictions"]),
            "--references",
            str(paths["references"]),
            "--metrics",
            str(paths["metrics"]),
            "--retrieval",
            str(paths["retrieval"]),
            "--labels",
            str(paths["labels"]),
            "--passages",
            str(paths["passages"]),
            "--auto-classify",
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--summary-out",
            str(summary),
        ]
    )
    assert exit_code == 0
    summary_text = summary.read_text(encoding="utf-8")
    assert reference not in summary_text
    assert "WRONG_ARTICLE_CITATION" in summary_text
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["reason_code"] == "citation_set_disjoint"


def test_classifier_inputs_reject_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        ClassifierThresholds(cap_tolerance=-1)


def test_cli_summary_does_not_require_gold_inference_artifacts(
    tmp_path: Path,
) -> None:
    paths = _report_fixture(tmp_path)
    summary = tmp_path / "summary.json"
    generate_error_report_cli(
        [
            "--predictions",
            str(paths["predictions"]),
            "--references",
            str(paths["references"]),
            "--metrics",
            str(paths["metrics"]),
            "--retrieval",
            str(paths["retrieval"]),
            "--labels",
            str(paths["labels"]),
            "--passages",
            str(paths["passages"]),
            "--auto-classify",
            "--summary-out",
            str(summary),
            "--markdown",
            str(tmp_path / "report.md"),
            "--csv",
            str(tmp_path / "report.csv"),
        ]
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert set(payload) <= {
        "DATA_SCHEMA_ERROR",
        "CONTEXT_LOAD_ERROR",
        "CHUNK_BOUNDARY_ERROR",
        "RETRIEVAL_MISS",
        "RIGHT_DOCUMENT_WRONG_CHUNK",
        "WRONG_DOCUMENT_VERSION",
        "RERANKING_REGRESSION",
        "EVIDENCE_TRUNCATION",
        "MISSING_REQUIRED_ITEM",
        "UNSUPPORTED_ADDITION",
        "WRONG_ARTICLE_CITATION",
        "TEMPORAL_CONFUSION",
        "OVER_VERBOSE",
        "UNDER_SPECIFIED",
        "FORMAT_ERROR",
        "EMPTY_ANSWER",
        "GENERATION_FAILURE",
        "REFERENCE_STYLE_VARIATION",
        "OTHER",
    }
