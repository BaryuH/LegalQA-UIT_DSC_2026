"""Evaluation orchestration and deterministic metric artifact creation."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alignment import align_records
from .meteor import MeteorResult, compute_meteor
from .models import AlignedRecord, InputRecord
from .normalization import NormalizationConfig, NormalizedText, normalize_text
from .rouge_l import RougeLResult, compute_rouge_l

EVALUATOR_NAME = "legal_rag.local_exact_token_metrics"
EVALUATOR_VERSION = "local-v1"
METRIC_CONTRACT_VERSION = "A2-local-v1"


@dataclass(frozen=True)
class EvaluationOptions:
    """Metadata and local aggregation choices for one evaluation run."""

    run_id: str = "local-evaluation"
    method: str = "local"
    split: str = "unknown"
    data_manifest_hash: str = "UNRESOLVED"
    prediction_artifact: str = "UNRESOLVED"
    command: str = "UNRESOLVED"
    normalization: NormalizationConfig = NormalizationConfig()


@dataclass(frozen=True)
class CaseEvaluation:
    """Per-case result, including intermediate values for unit diagnostics."""

    id: str
    status: str
    normalized_reference: NormalizedText
    normalized_prediction: NormalizedText
    meteor: MeteorResult | None
    rouge_l: RougeLResult | None
    error_code: str | None = None

    def as_artifact_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "meteor": self.meteor.score if self.meteor else None,
            "rouge_l": self.rouge_l.score if self.rouge_l else None,
            "reference_token_count": len(self.normalized_reference.tokens),
            "prediction_token_count": len(self.normalized_prediction.tokens),
            "meteor_details": self.meteor.as_dict() if self.meteor else None,
            "rouge_l_details": self.rouge_l.as_dict() if self.rouge_l else None,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Metric artifact plus in-memory cases used by tests and diagnostics."""

    artifact: dict[str, Any]
    cases: tuple[CaseEvaluation, ...]


def _score_case(record: AlignedRecord, config: NormalizationConfig) -> CaseEvaluation:
    reference = normalize_text(record.reference, config)
    prediction = normalize_text(record.prediction, config)
    if not prediction.text:
        return CaseEvaluation(
            id=record.id,
            status="empty_prediction",
            normalized_reference=reference,
            normalized_prediction=prediction,
            meteor=None,
            rouge_l=None,
            error_code="EMPTY_PREDICTION",
        )
    if not reference.text:
        return CaseEvaluation(
            id=record.id,
            status="empty_reference",
            normalized_reference=reference,
            normalized_prediction=prediction,
            meteor=None,
            rouge_l=None,
            error_code="EMPTY_REFERENCE",
        )

    return CaseEvaluation(
        id=record.id,
        status="scored",
        normalized_reference=reference,
        normalized_prediction=prediction,
        meteor=compute_meteor(reference.tokens, prediction.tokens),
        rouge_l=compute_rouge_l(reference.tokens, prediction.tokens),
    )


def _macro(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_records(
    references: list[InputRecord] | tuple[InputRecord, ...],
    predictions: list[InputRecord] | tuple[InputRecord, ...],
    options: EvaluationOptions | None = None,
) -> EvaluationReport:
    """Strictly align and locally score a complete prediction set."""

    selected_options = options or EvaluationOptions()
    aligned = align_records(references, predictions)
    cases = tuple(
        _score_case(record, selected_options.normalization) for record in aligned
    )
    scored_cases = [case for case in cases if case.status == "scored"]
    meteor_scores = [case.meteor.score for case in scored_cases if case.meteor]
    rouge_scores = [case.rouge_l.score for case in scored_cases if case.rouge_l]
    empty_prediction_count = sum(case.status == "empty_prediction" for case in cases)
    empty_reference_count = sum(case.status == "empty_reference" for case in cases)
    errors_count = sum(case.error_code is not None for case in cases)

    artifact = {
        "schema_version": "a3.metrics.v1",
        "run_id": selected_options.run_id,
        "method": selected_options.method,
        "split": selected_options.split,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "evaluator_kind": "local",
        "evaluator_name": EVALUATOR_NAME,
        "evaluator_version": EVALUATOR_VERSION,
        "python_version": platform.python_version(),
        "data_manifest_hash": selected_options.data_manifest_hash,
        "prediction_artifact": selected_options.prediction_artifact,
        "reference_role": "evaluation_reference_only",
        "command": selected_options.command,
        "normalization": selected_options.normalization.as_dict(),
        "aggregation": {
            "method": "macro_over_scored_cases",
            "score_denominator": "scored_cases",
            "official_equivalence": "UNRESOLVED",
        },
        "error_policy": {
            "alignment": "fail_fast",
            "empty_prediction": "continue_with_structured_status",
            "empty_reference": "continue_with_structured_status",
        },
        "counts": {
            "target": len(references),
            "evaluated": len(cases),
            "scored": len(scored_cases),
            "empty_prediction": empty_prediction_count,
            "empty_reference": empty_reference_count,
            "missing": 0,
            "duplicate": 0,
            "extra": 0,
            "errors": errors_count,
        },
        "metrics": {
            "meteor": _macro(meteor_scores),
            "rouge_l": _macro(rouge_scores),
            "primary_metric": "meteor",
            "secondary_metric": "rouge_l",
            "higher_is_better": True,
            "aggregation_method": "macro_over_scored_cases",
        },
        "per_case": [case.as_artifact_dict() for case in cases],
    }
    return EvaluationReport(artifact=artifact, cases=cases)


def write_report(
    report: EvaluationReport, output_path: str | Path, overwrite: bool = False
) -> None:
    """Write a UTF-8 deterministic artifact without overwriting by default."""

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Metric artifact already exists: {path}; pass --overwrite explicitly"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        report.artifact,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(serialized + "\n", encoding="utf-8", newline="\n")
