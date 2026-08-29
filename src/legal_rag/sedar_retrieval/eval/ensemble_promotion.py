"""Promotion-boundary checks for frozen-reader ensemble comparisons."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PromotionStatus = Literal["PASS", "FAIL"]


class EnsemblePromotionError(ValueError):
    """Raised when promotion artifacts cannot be checked safely."""


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    """One explicit promotion-boundary check."""

    name: str
    status: PromotionStatus
    value: object
    expected: object
    detail: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "expected": self.expected,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class EnsemblePromotionReport:
    """Machine-readable promotion decision without answer text."""

    status: PromotionStatus
    recommendation: Literal["PROMOTE", "FIX"]
    min_gain: float
    checks: tuple[PromotionCheck, ...]
    metrics: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "sedar-retrieval-ensemble-promotion-v1",
            "status": self.status,
            "recommendation": self.recommendation,
            "min_gain": self.min_gain,
            "checks": [check.as_dict() for check in self.checks],
            "metrics": self.metrics,
        }


_MISSING = "<missing>"


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EnsemblePromotionError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise EnsemblePromotionError(f"JSON artifact must be an object: {path}")
    return payload


def _value(payload: Mapping[str, object], key: str) -> object:
    return payload[key] if key in payload else _MISSING


def _check(
    name: str,
    *,
    passed: bool,
    value: object,
    expected: object,
    detail: str | None = None,
) -> PromotionCheck:
    return PromotionCheck(
        name=name,
        status="PASS" if passed else "FAIL",
        value=value,
        expected=expected,
        detail=detail,
    )


def _numeric_metric(payload: Mapping[str, object], name: str) -> float | None:
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        return None
    raw_value = raw_metrics.get(name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None
    value = float(raw_value)
    return value if math.isfinite(value) else None


def _scope_values(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    provenance = evaluation.get("provenance")
    provenance_map = provenance if isinstance(provenance, Mapping) else {}
    return {
        "split": _value(evaluation, "split"),
        "split_policy": _value(evaluation, "split_policy"),
        "split_role": _value(evaluation, "split_role"),
        "data_manifest_hash": _value(evaluation, "data_manifest_hash"),
        "included_ids_hash": _value(provenance_map, "included_ids_hash"),
    }


def _complete_evaluation(payload: Mapping[str, object]) -> bool:
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        return False
    target = counts.get("target")
    evaluated = counts.get("evaluated")
    if (
        isinstance(target, bool)
        or not isinstance(target, int)
        or isinstance(evaluated, bool)
        or not isinstance(evaluated, int)
        or target <= 0
        or evaluated != target
    ):
        return False
    return all(
        counts.get(key) == 0 for key in ("missing", "duplicate", "extra", "errors")
    )


def _same_evaluator(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> bool:
    keys = (
        "metric_contract_version",
        "evaluator_kind",
        "evaluator_name",
        "evaluator_version",
        "split",
        "split_policy",
        "split_role",
        "reference_role",
        "scorer",
    )
    return all(
        baseline.get(key, _MISSING) != _MISSING
        and baseline.get(key) == candidate.get(key)
        for key in keys
    )


def evaluate_ensemble_promotion(
    *,
    baseline_run_dir: str | Path,
    candidate_run_dir: str | Path,
    baseline_eval_dir: str | Path,
    candidate_eval_dir: str | Path,
    min_gain: float = 0.003,
) -> EnsemblePromotionReport:
    """Check whether an ensemble run is comparable and eligible to promote."""

    if not math.isfinite(min_gain) or min_gain < 0.0:
        raise EnsemblePromotionError("min_gain must be finite and non-negative")

    baseline_run = Path(baseline_run_dir)
    candidate_run = Path(candidate_run_dir)
    baseline_eval = Path(baseline_eval_dir)
    candidate_eval = Path(candidate_eval_dir)
    baseline_config = _read_json_object(baseline_run / "config.json")
    candidate_config = _read_json_object(candidate_run / "config.json")
    baseline_checkpoint = _read_json_object(baseline_run / "checkpoint_reference.json")
    candidate_checkpoint = _read_json_object(
        candidate_run / "checkpoint_reference.json"
    )
    baseline_metrics = _read_json_object(baseline_eval / "metrics.json")
    candidate_metrics = _read_json_object(candidate_eval / "metrics.json")
    baseline_summary = _read_json_object(baseline_eval / "summary.json")
    candidate_summary = _read_json_object(candidate_eval / "summary.json")

    baseline_scope = _scope_values(baseline_metrics)
    candidate_scope = _scope_values(candidate_metrics)
    source_scope = {
        "baseline": {
            "split": _value(baseline_config, "split"),
            "id_source": _value(baseline_config, "id_source"),
            "selected_ids_count": _value(baseline_config, "selected_ids_count"),
        },
        "candidate": {
            "split": _value(candidate_config, "split"),
            "id_source": _value(candidate_config, "id_source"),
            "selected_ids_count": _value(candidate_config, "selected_ids_count"),
        },
    }
    source_scope_complete = all(
        value != _MISSING for scope in source_scope.values() for value in scope.values()
    )
    scope_passed = (
        baseline_scope == candidate_scope
        and _MISSING not in baseline_scope.values()
        and source_scope["baseline"] == source_scope["candidate"]
        and source_scope["baseline"]["split"] == baseline_scope["split"]
        and source_scope_complete
    )

    baseline_reader_reference = {
        "adapter_hash": _value(baseline_checkpoint, "adapter_hash"),
        "checkpoint_manifest_hash": _value(
            baseline_checkpoint, "checkpoint_manifest_hash"
        ),
        "reader_frozen": _value(baseline_config, "reader_frozen"),
    }
    candidate_reader_reference = {
        "adapter_hash": _value(candidate_checkpoint, "adapter_hash"),
        "checkpoint_manifest_hash": _value(
            candidate_checkpoint, "checkpoint_manifest_hash"
        ),
        "reader_frozen": _value(candidate_config, "reader_frozen"),
    }
    baseline_provenance = baseline_metrics.get("provenance")
    baseline_provenance = (
        baseline_provenance if isinstance(baseline_provenance, Mapping) else {}
    )
    candidate_provenance = candidate_metrics.get("provenance")
    candidate_provenance = (
        candidate_provenance if isinstance(candidate_provenance, Mapping) else {}
    )
    baseline_reader = {
        **baseline_reader_reference,
        "eval_adapter_hash": _value(baseline_provenance, "adapter_hash"),
        "eval_checkpoint_manifest_hash": _value(
            baseline_provenance, "checkpoint_manifest_hash"
        ),
    }
    candidate_reader = {
        **candidate_reader_reference,
        "eval_adapter_hash": _value(candidate_provenance, "adapter_hash"),
        "eval_checkpoint_manifest_hash": _value(
            candidate_provenance, "checkpoint_manifest_hash"
        ),
    }
    reader_passed = (
        baseline_reader == candidate_reader
        and baseline_reader_reference["reader_frozen"] is True
        and candidate_reader_reference["reader_frozen"] is True
        and baseline_reader_reference["adapter_hash"]
        == baseline_reader["eval_adapter_hash"]
        and candidate_reader_reference["adapter_hash"]
        == candidate_reader["eval_adapter_hash"]
        and baseline_reader_reference["checkpoint_manifest_hash"]
        == baseline_reader["eval_checkpoint_manifest_hash"]
        and _MISSING not in baseline_reader.values()
        and candidate_reader_reference["checkpoint_manifest_hash"]
        == candidate_reader["eval_checkpoint_manifest_hash"]
        and _MISSING not in candidate_reader.values()
    )

    baseline_evidence = baseline_config.get("evidence")
    candidate_evidence = candidate_config.get("evidence")
    evidence_passed = (
        isinstance(baseline_evidence, Mapping)
        and isinstance(candidate_evidence, Mapping)
        and bool(baseline_evidence)
        and baseline_evidence == candidate_evidence
    )

    baseline_flags = {
        "inference_used_answers": _value(baseline_summary, "inference_used_answers"),
        "references_opened_before_predictions": _value(
            baseline_summary, "references_opened_before_predictions"
        ),
        "gold_boundary": _value(baseline_summary, "gold_boundary"),
    }
    candidate_flags = {
        "inference_used_answers": _value(candidate_summary, "inference_used_answers"),
        "references_opened_before_predictions": _value(
            candidate_summary, "references_opened_before_predictions"
        ),
        "gold_boundary": _value(candidate_summary, "gold_boundary"),
    }
    leakage_passed = (
        baseline_flags == candidate_flags
        and baseline_flags["inference_used_answers"] is False
        and baseline_flags["references_opened_before_predictions"] is False
        and baseline_flags["gold_boundary"] == "approved_evaluation"
    )

    baseline_meteor = _numeric_metric(baseline_metrics, "meteor")
    candidate_meteor = _numeric_metric(candidate_metrics, "meteor")
    baseline_rouge = _numeric_metric(baseline_metrics, "rouge_l")
    candidate_rouge = _numeric_metric(candidate_metrics, "rouge_l")
    meteor_delta = (
        candidate_meteor - baseline_meteor
        if baseline_meteor is not None and candidate_meteor is not None
        else None
    )
    rouge_delta = (
        candidate_rouge - baseline_rouge
        if baseline_rouge is not None and candidate_rouge is not None
        else None
    )
    gain_passed = (
        meteor_delta is not None
        and rouge_delta is not None
        and (meteor_delta >= min_gain or rouge_delta >= min_gain)
    )

    checks = (
        _check(
            "same_evaluation_scope",
            passed=scope_passed,
            value={"baseline": baseline_scope, "candidate": candidate_scope},
            expected="same split, split policy, data manifest, and included IDs",
        ),
        _check(
            "same_frozen_reader",
            passed=reader_passed,
            value={"baseline": baseline_reader, "candidate": candidate_reader},
            expected="same adapter/checkpoint hashes and reader_frozen=true",
        ),
        _check(
            "same_evidence_budget",
            passed=evidence_passed,
            value={"baseline": baseline_evidence, "candidate": candidate_evidence},
            expected="identical non-empty evidence configuration",
        ),
        _check(
            "complete_evaluation",
            passed=_complete_evaluation(baseline_metrics)
            and _complete_evaluation(candidate_metrics),
            value={
                "baseline": baseline_metrics.get("counts", _MISSING),
                "candidate": candidate_metrics.get("counts", _MISSING),
            },
            expected="target=evaluated and missing/duplicate/extra/errors=0",
        ),
        _check(
            "no_gold_leakage",
            passed=leakage_passed,
            value={"baseline": baseline_flags, "candidate": candidate_flags},
            expected="question-only inference before approved evaluation boundary",
        ),
        _check(
            "same_evaluator",
            passed=_same_evaluator(baseline_metrics, candidate_metrics),
            value={
                "baseline": {
                    key: baseline_metrics.get(key, _MISSING)
                    for key in (
                        "metric_contract_version",
                        "evaluator_kind",
                        "evaluator_name",
                        "evaluator_version",
                        "split",
                        "split_policy",
                        "split_role",
                        "reference_role",
                    )
                },
                "candidate": {
                    key: candidate_metrics.get(key, _MISSING)
                    for key in (
                        "metric_contract_version",
                        "evaluator_kind",
                        "evaluator_name",
                        "evaluator_version",
                        "split",
                        "split_policy",
                        "split_role",
                        "reference_role",
                    )
                },
            },
            expected="identical evaluation contract and scorer metadata",
        ),
        _check(
            "minimum_metric_gain",
            passed=gain_passed,
            value={
                "baseline_meteor": baseline_meteor,
                "candidate_meteor": candidate_meteor,
                "meteor_delta": meteor_delta,
                "baseline_rouge_l": baseline_rouge,
                "candidate_rouge_l": candidate_rouge,
                "rouge_l_delta": rouge_delta,
            },
            expected=f"METEOR or ROUGE-L delta >= {min_gain}",
        ),
    )
    passed = all(check.status == "PASS" for check in checks)
    return EnsemblePromotionReport(
        status="PASS" if passed else "FAIL",
        recommendation="PROMOTE" if passed else "FIX",
        min_gain=min_gain,
        checks=checks,
        metrics={
            "baseline": {"meteor": baseline_meteor, "rouge_l": baseline_rouge},
            "candidate": {"meteor": candidate_meteor, "rouge_l": candidate_rouge},
            "delta": {"meteor": meteor_delta, "rouge_l": rouge_delta},
        },
    )


__all__ = [
    "EnsemblePromotionError",
    "EnsemblePromotionReport",
    "PromotionCheck",
    "evaluate_ensemble_promotion",
]
