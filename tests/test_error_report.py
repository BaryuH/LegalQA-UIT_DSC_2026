"""Acceptance tests for evaluation-only I2 error reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from legal_rag.evaluation import (
    ErrorReportError,
    generate_error_report,
    write_error_report,
)
from scripts.generate_error_report import main as generate_error_report_cli


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    predictions = tmp_path / "predictions.jsonl"
    references = tmp_path / "references.json"
    metrics = tmp_path / "metrics.json"
    retrieval = tmp_path / "retrieval.jsonl"
    _write_jsonl(
        predictions,
        [
            {"id": "case-a", "answer": "A generated answer", "method": "bm25_rag"},
            {"id": "case-b", "answer": "B generated answer", "method": "bm25_rag"},
        ],
    )
    _write_json(
        references,
        {"case-a": "A reference answer", "case-b": "B reference answer"},
    )
    _write_json(
        metrics,
        {
            "schema_version": "a3.metrics.v1",
            "run_id": "i2-fixture-run",
            "method": "bm25_rag",
            "split": "warmup",
            "reference_role": "evaluation_reference_only",
            "per_case": [
                {"id": "case-a", "status": "scored", "meteor": 0.1, "rouge_l": 0.2},
                {"id": "case-b", "status": "scored", "meteor": 0.8, "rouge_l": 0.9},
            ],
        },
    )
    _write_jsonl(
        retrieval,
        [
            {
                "id": "case-a",
                "status": "success",
                "packed_evidence": {
                    "included_ids": ["chunk-a"],
                    "dropped_ids": [],
                    "truncated_ids": [],
                    "included_hits": [
                        {
                            "chunk_id": "chunk-a",
                            "document_id": "document-a",
                            "source_path": "contexts.zip",
                            "source_member": "context_a.json",
                            "section_label": "Article 1",
                            "start_offset": None,
                            "end_offset": None,
                            "rank": 1,
                            "bm25_score": 2.0,
                            "rerank_score": None,
                        }
                    ],
                    "rendered_text": "[TRICH DOAN 1]\nArticle 1 legal evidence.",
                    "dropped_reasons": {},
                    "metadata": {"included_count": 1},
                },
            },
            {"id": "case-b", "status": "error"},
        ],
    )
    return predictions, references, metrics, retrieval


def test_error_report_joins_by_id_sorts_worst_and_exports_both_formats(
    tmp_path: Path,
) -> None:
    predictions, references, metrics, retrieval = _fixture(tmp_path)
    report = generate_error_report(
        predictions,
        references,
        metrics,
        retrieval_path=retrieval,
        manual_error_types={
            "case-a": "RETRIEVAL_MISS",
            "case-b": "REFERENCE_STYLE_VARIATION",
        },
    )

    assert [case.id for case in report.cases] == ["case-a", "case-b"]
    assert report.cases[0].meteor == 0.1
    assert report.cases[0].error_type == "RETRIEVAL_MISS"
    assert "Article 1 legal evidence." in report.cases[0].evidence_text
    assert report.cases[1].retrieval_status == "error"

    markdown_path = tmp_path / "report.md"
    csv_path = tmp_path / "report.csv"
    write_error_report(report, markdown_path, csv_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "A reference answer" in markdown
    assert "RETRIEVAL_MISS" in markdown
    assert "Article 1 legal evidence." in markdown

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["case-a", "case-b"]
    assert rows[0]["meteor"] == "0.1"
    assert rows[0]["error_type"] == "RETRIEVAL_MISS"


def test_error_report_requires_evaluation_reference_role(tmp_path: Path) -> None:
    predictions, references, metrics, retrieval = _fixture(tmp_path)
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload["reference_role"] = "none"
    _write_json(metrics, payload)

    with pytest.raises(ErrorReportError, match="evaluation_reference_only"):
        generate_error_report(
            predictions,
            references,
            metrics,
            retrieval_path=retrieval,
        )


def test_private_report_is_disabled_before_reference_read(tmp_path: Path) -> None:
    predictions, references, metrics, retrieval = _fixture(tmp_path)
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload["split"] = "private"
    _write_json(metrics, payload)
    references.unlink()

    with pytest.raises(ErrorReportError, match="Private-answer"):
        generate_error_report(
            predictions,
            references,
            metrics,
            retrieval_path=retrieval,
        )


def test_missing_prediction_is_joined_without_silent_case_drop(tmp_path: Path) -> None:
    predictions, references, metrics, retrieval = _fixture(tmp_path)
    _write_jsonl(predictions, [{"id": "case-a", "answer": "A generated answer"}])

    report = generate_error_report(
        predictions,
        references,
        metrics,
        retrieval_path=retrieval,
    )

    missing = next(case for case in report.cases if case.id == "case-b")
    assert missing.prediction is None
    assert missing.reference == "B reference answer"


def test_error_report_cli_writes_markdown_and_csv(tmp_path: Path) -> None:
    predictions, references, metrics, retrieval = _fixture(tmp_path)
    manual_types = tmp_path / "manual-types.json"
    _write_json(manual_types, {"case-a": "RETRIEVAL_MISS"})
    markdown = tmp_path / "cli-report.md"
    csv_path = tmp_path / "cli-report.csv"

    exit_code = generate_error_report_cli(
        [
            "--predictions",
            str(predictions),
            "--references",
            str(references),
            "--metrics",
            str(metrics),
            "--retrieval",
            str(retrieval),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--error-types",
            str(manual_types),
        ]
    )

    assert exit_code == 0
    assert markdown.is_file()
    assert csv_path.is_file()
