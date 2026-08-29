"""Acceptance tests for scorer selection and evaluation provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.evaluation import (
    SOURCE_SCORER_ID,
    InputRecord,
    evaluate_records,
)
from legal_rag.evaluation.source_scorer import (
    SourceScorerDependencyError,
    compute_source_meteor,
    compute_source_rouge_l,
    source_scorer_metadata,
)
from legal_rag.finetuned_reader.warmup_eval import (
    WarmupEvalError,
    load_prediction_run_provenance,
)


def test_source_adapter_matches_declared_library_calls() -> None:
    pytest.importorskip("nltk")
    pytest.importorskip("rouge_score")

    from nltk.translate.meteor_score import meteor_score
    from rouge_score.rouge_scorer import RougeScorer

    reference = "Căn cứ Điều 37. Người lao động."
    prediction = "Căn cứ Điều 37."
    expected_meteor = meteor_score([reference.split()], prediction.split())
    expected_rouge = (
        RougeScorer(["rougeL"], use_stemmer=False)
        .score(
            reference,
            prediction,
        )["rougeL"]
        .fmeasure
    )

    assert compute_source_meteor(reference, prediction).score == pytest.approx(
        expected_meteor
    )
    assert compute_source_rouge_l(reference, prediction).score == pytest.approx(
        expected_rouge
    )


def test_default_evaluator_records_source_scorer_metadata() -> None:
    try:
        metadata = source_scorer_metadata()
        report = evaluate_records(
            [InputRecord(id="1", answer="Một câu.")],
            [InputRecord(id="1", answer="Một câu.")],
        )
    except SourceScorerDependencyError as exc:
        pytest.skip(str(exc))

    assert report.artifact["scorer"]["scorer_id"] == SOURCE_SCORER_ID
    assert report.artifact["metric_contract_version"] == "A2-btc-source-scorer-v1"
    assert (
        report.artifact["scorer"]["meteor"]["version"] == metadata["meteor"]["version"]
    )
    assert report.artifact["normalization"]["tokenizer_name"] == (
        "str.split + rouge_score.DefaultTokenizer"
    )
    assert report.artifact["provenance"]["checkpoint_manifest_hash"] == "UNRESOLVED"
    assert report.artifact["provenance"]["prediction_source_run_dir"] == "UNRESOLVED"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_prediction_run_provenance_requires_matching_config_hash(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "sedar-run"
    run_dir.mkdir()
    _write_json(
        run_dir / "run_summary.json",
        {
            "run_id": "sedar-infer-v2",
            "method": "finetuned_reader",
            "config_hash": "config-hash",
            "index_fingerprint": "index-hash",
        },
    )
    _write_json(run_dir / "config.json", {"config_hash": "different-config"})
    _write_json(
        run_dir / "checkpoint_reference.json",
        {
            "checkpoint_manifest_hash": "checkpoint-hash",
            "adapter_hash": "adapter-hash",
            "base_model": "Qwen",
            "base_revision": "revision",
            "retrieval_config_hash": "retrieval-hash",
            "index_fingerprint": "index-hash",
        },
    )
    (run_dir / "predictions.jsonl").write_text(
        '{"id":"1","answer":"prediction"}\n',
        encoding="utf-8",
    )

    with pytest.raises(WarmupEvalError, match="config hash"):
        load_prediction_run_provenance(
            run_dir,
            method="sedar_sft",
            method_version="sedar-sft-v2",
        )


def test_prediction_run_provenance_captures_checkpoint_and_prediction_hash(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "sedar-run"
    run_dir.mkdir()
    _write_json(
        run_dir / "run_summary.json",
        {
            "run_id": "sedar-infer-v2",
            "method": "finetuned_reader",
            "config_hash": "config-hash",
            "index_fingerprint": "index-hash",
        },
    )
    _write_json(run_dir / "config.json", {"config_hash": "config-hash"})
    _write_json(
        run_dir / "checkpoint_reference.json",
        {
            "checkpoint_manifest_hash": "checkpoint-hash",
            "adapter_hash": "adapter-hash",
            "base_model": "Qwen",
            "base_revision": "revision",
            "retrieval_config_hash": "retrieval-hash",
            "index_fingerprint": "index-hash",
        },
    )
    (run_dir / "predictions.jsonl").write_text(
        '{"id":"1","answer":"prediction"}\n',
        encoding="utf-8",
    )

    provenance = load_prediction_run_provenance(
        run_dir,
        method="sedar_sft",
        method_version="sedar-sft-v2",
    )

    assert provenance.source_run_id == "sedar-infer-v2"
    assert provenance.pipeline_method == "finetuned_reader"
    assert provenance.checkpoint_manifest_hash == "checkpoint-hash"
    assert provenance.adapter_hash == "adapter-hash"
    assert provenance.source_run_dir == run_dir.resolve()
    assert len(provenance.source_prediction_artifact_sha256) == 64
