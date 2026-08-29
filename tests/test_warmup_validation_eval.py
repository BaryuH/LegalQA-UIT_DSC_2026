"""VAL-01 clean warmup local evaluation acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.evaluation import LOCAL_SCORER_ID, EvaluationOptions
from legal_rag.evaluation.models import InputRecord
from legal_rag.finetuned_reader.warmup_eval import (
    VAL01_POLICY_ID,
    WarmupEvalError,
    build_eval_summary,
    evaluate_clean_warmup,
    filter_by_included_ids,
    load_clean_inference_questions,
    load_clean_reference_records,
    load_clean_warmup_manifest,
    select_included_ids,
    write_eval_only_references,
    write_warmup_eval_artifacts,
)
from legal_rag.finetuned_reader.warmup_validation import (
    POLICY_ID,
    build_clean_warmup_validation,
    write_clean_warmup_validation_artifacts,
)
from legal_rag.schemas import InferenceQuestion


def _write_questions(path: Path, records: dict[str, dict[str, str]]) -> Path:
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def clean_fixture(tmp_path: Path) -> dict[str, Path]:
    data = tmp_path / "data"
    data.mkdir()
    warmup = _write_questions(
        data / "warmup.json",
        {
            "w_keep": {"question": "Câu giữ lại?", "answer": "Gold keep"},
            "w_overlap": {
                "question": "Câu trùng public?",
                "answer": "Gold overlap",
            },
            "w_other": {"question": "Câu khác?", "answer": "Gold other"},
        },
    )
    public = _write_questions(
        data / "public-official.json",
        {
            "w_overlap": {
                "question": "Câu trùng public?",
                "answer": "PUBLIC GOLD MUST NOT ENTER INFERENCE",
            }
        },
    )
    result = build_clean_warmup_validation(warmup_path=warmup, public_path=public)
    manifest_dir = tmp_path / "validation"
    paths = write_clean_warmup_validation_artifacts(result, manifest_dir)
    return {
        "warmup": warmup,
        "public": public,
        "manifest": paths["manifest"],
    }


def test_load_manifest_and_filter_inference_is_question_only(
    clean_fixture: dict[str, Path],
) -> None:
    manifest = load_clean_warmup_manifest(clean_fixture["manifest"])
    assert manifest.policy_id == POLICY_ID
    assert set(manifest.included_ids) == {"w_keep", "w_other"}
    cases = load_clean_inference_questions(
        clean_fixture["warmup"],
        split="warmup",
        included_ids=manifest.included_ids,
    )
    assert [case.id for case in cases] == ["w_keep", "w_other"]
    assert all(isinstance(case, InferenceQuestion) for case in cases)
    assert "answer" not in InferenceQuestion.model_fields


def test_missing_id_fail_closed(clean_fixture: dict[str, Path]) -> None:
    with pytest.raises(WarmupEvalError, match="missing"):
        load_clean_inference_questions(
            clean_fixture["warmup"],
            split="warmup",
            included_ids=("w_keep", "missing_id"),
        )


def test_eval_join_opens_gold_only_for_clean_ids(
    clean_fixture: dict[str, Path],
    tmp_path: Path,
) -> None:
    manifest = load_clean_warmup_manifest(clean_fixture["manifest"])
    selected = select_included_ids(manifest.included_ids)
    references = load_clean_reference_records(
        clean_fixture["warmup"],
        split="warmup",
        included_ids=selected,
    )
    assert {record.id for record in references} == {"w_keep", "w_other"}
    assert "w_overlap" not in {record.id for record in references}

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(
                {"id": record.id, "answer": f"Pred {record.id}"},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for record in references
        ),
        encoding="utf-8",
    )
    report = evaluate_clean_warmup(
        references=references,
        predictions_path=predictions_path,
        options=EvaluationOptions(
            run_id="val01-fixture",
            method="hybrid_rag",
            split="warmup",
            scorer=LOCAL_SCORER_ID,
        ),
    )
    assert report.artifact["counts"]["evaluated"] == 2
    assert "per_case" in report.artifact
    assert {case["id"] for case in report.artifact["per_case"]} == {
        "w_keep",
        "w_other",
    }


def test_public_answer_never_in_eval_artifacts(
    clean_fixture: dict[str, Path],
    tmp_path: Path,
) -> None:
    manifest = load_clean_warmup_manifest(clean_fixture["manifest"])
    references = load_clean_reference_records(
        clean_fixture["warmup"],
        split="warmup",
        included_ids=manifest.included_ids,
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "w_keep", "answer": "A"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"id": "w_other", "answer": "B"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    report = evaluate_clean_warmup(
        references=references,
        predictions_path=predictions_path,
        options=EvaluationOptions(
            run_id="x",
            method="hybrid_rag",
            split="warmup",
            scorer=LOCAL_SCORER_ID,
        ),
    )
    summary = build_eval_summary(
        manifest=manifest,
        selected_ids=manifest.included_ids,
        method="hybrid_rag",
        run_id="x",
        predictions_path=predictions_path,
        metrics_path=tmp_path / "metrics.json",
        report=report,
        inference_used_answers=False,
        references_opened_before_predictions=False,
    )
    paths = write_warmup_eval_artifacts(
        output_dir=tmp_path / "eval",
        summary=summary,
        report=report,
        references=references,
        overwrite=True,
    )
    blob = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            paths["metrics"],
            paths["summary"],
            paths["references_eval_only"],
        )
    )
    assert "PUBLIC GOLD MUST NOT ENTER INFERENCE" not in blob
    assert summary["policy_id"] == VAL01_POLICY_ID
    assert summary["inference_used_answers"] is False


def test_summary_rejects_gold_in_inference_flag(
    clean_fixture: dict[str, Path],
) -> None:
    manifest = load_clean_warmup_manifest(clean_fixture["manifest"])
    report_stub_metrics = {
        "metrics": {"meteor": 0.0, "rouge_l": 0.0},
        "counts": {"evaluated": 0},
    }

    class _Report:
        artifact = report_stub_metrics

    with pytest.raises(WarmupEvalError, match="must not enter"):
        build_eval_summary(
            manifest=manifest,
            selected_ids=manifest.included_ids,
            method="hybrid_rag",
            run_id="x",
            predictions_path=Path("predictions.jsonl"),
            metrics_path=Path("metrics.json"),
            report=_Report(),  # type: ignore[arg-type]
            inference_used_answers=True,
            references_opened_before_predictions=False,
        )


def test_filter_ordering_deterministic() -> None:
    items = (
        InputRecord(id="b", answer="2"),
        InputRecord(id="a", answer="1"),
        InputRecord(id="c", answer="3"),
    )
    filtered = filter_by_included_ids(
        items,
        ("a", "c"),
        id_of=lambda record: record.id,
        role="reference",
    )
    assert [item.id for item in filtered] == ["a", "c"]


def test_limit_prefix(clean_fixture: dict[str, Path]) -> None:
    manifest = load_clean_warmup_manifest(clean_fixture["manifest"])
    selected = select_included_ids(manifest.included_ids, limit=1)
    assert len(selected) == 1
    assert selected[0] == sorted(manifest.included_ids)[0]


def test_write_eval_only_references_shape(tmp_path: Path) -> None:
    path = tmp_path / "refs.json"
    write_eval_only_references(
        [InputRecord(id="1", answer="Gold")],
        path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"1": {"answer": "Gold", "question": ""}}
