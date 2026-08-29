"""Promotion-boundary tests for the dense ensemble."""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag.sedar_retrieval.eval.ensemble_promotion import (
    evaluate_ensemble_promotion,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _make_bundle(
    root: Path,
    name: str,
    *,
    meteor: float,
    rouge_l: float,
    reader_frozen: bool = True,
    evidence_top_k: int = 4,
) -> tuple[Path, Path]:
    run_dir = root / f"{name}-run"
    eval_dir = root / f"{name}-eval"
    _write_json(
        run_dir / "config.json",
        {
            "split": "warmup",
            "id_source": "clean_manifest",
            "selected_ids_count": 460,
            "reader_frozen": reader_frozen,
            "evidence": {
                "evidence_top_k": evidence_top_k,
                "max_total_chars": 4000,
                "max_chunks_per_document": 2,
            },
        },
    )
    _write_json(
        run_dir / "checkpoint_reference.json",
        {
            "adapter_hash": "reader-adapter-hash",
            "checkpoint_manifest_hash": "reader-checkpoint-hash",
        },
    )
    _write_json(
        eval_dir / "metrics.json",
        {
            "metric_contract_version": "A2-local-v1",
            "evaluator_kind": "local",
            "evaluator_name": "btc_source_scorer_v1",
            "evaluator_version": "1",
            "split": "warmup",
            "split_policy": "warmup_evaluation",
            "split_role": "validation",
            "data_manifest_hash": "warmup-data-hash",
            "reference_role": "evaluation_reference_only",
            "scorer": {"name": "btc_source_scorer_v1", "version": "1"},
            "metrics": {"meteor": meteor, "rouge_l": rouge_l},
            "counts": {
                "target": 460,
                "evaluated": 460,
                "missing": 0,
                "duplicate": 0,
                "extra": 0,
                "errors": 0,
            },
            "provenance": {
                "adapter_hash": "reader-adapter-hash",
                "checkpoint_manifest_hash": "reader-checkpoint-hash",
                "included_ids_hash": "clean-warmup-ids-hash",
            },
        },
    )
    _write_json(
        eval_dir / "summary.json",
        {
            "split": "warmup",
            "selected_ids_count": 460,
            "inference_used_answers": False,
            "references_opened_before_predictions": False,
            "gold_boundary": "approved_evaluation",
        },
    )
    return run_dir, eval_dir


def test_ensemble_promotion_requires_matching_boundaries(tmp_path: Path) -> None:
    baseline_run, baseline_eval = _make_bundle(
        tmp_path,
        "baseline",
        meteor=0.500,
        rouge_l=0.470,
    )
    candidate_run, candidate_eval = _make_bundle(
        tmp_path,
        "candidate",
        meteor=0.506,
        rouge_l=0.471,
    )

    report = evaluate_ensemble_promotion(
        baseline_run_dir=baseline_run,
        candidate_run_dir=candidate_run,
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
    )

    assert report.status == "PASS"
    assert report.recommendation == "PROMOTE"
    assert all(check.status == "PASS" for check in report.checks)

    _write_json(
        candidate_run / "config.json",
        {
            "split": "warmup",
            "id_source": "clean_manifest",
            "selected_ids_count": 460,
            "reader_frozen": False,
            "evidence": {
                "evidence_top_k": 5,
                "max_total_chars": 4000,
                "max_chunks_per_document": 2,
            },
        },
    )
    rejected = evaluate_ensemble_promotion(
        baseline_run_dir=baseline_run,
        candidate_run_dir=candidate_run,
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
    )

    assert rejected.status == "FAIL"
    assert rejected.recommendation == "FIX"
    failed_names = {check.name for check in rejected.checks if check.status == "FAIL"}
    assert {"same_frozen_reader", "same_evidence_budget"} <= failed_names


def test_ensemble_promotion_rejects_insufficient_gain(tmp_path: Path) -> None:
    baseline_run, baseline_eval = _make_bundle(
        tmp_path,
        "baseline",
        meteor=0.500,
        rouge_l=0.470,
    )
    candidate_run, candidate_eval = _make_bundle(
        tmp_path,
        "candidate",
        meteor=0.501,
        rouge_l=0.471,
    )

    report = evaluate_ensemble_promotion(
        baseline_run_dir=baseline_run,
        candidate_run_dir=candidate_run,
        baseline_eval_dir=baseline_eval,
        candidate_eval_dir=candidate_eval,
    )

    assert report.status == "FAIL"
    assert (
        next(
            check for check in report.checks if check.name == "minimum_metric_gain"
        ).status
        == "FAIL"
    )
