"""Evaluation orchestration and deterministic metric artifact creation."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..splits import ReferenceAccess, validate_reference_access
from .alignment import align_records
from .meteor import compute_meteor
from .models import AlignedRecord, InputRecord
from .normalization import NormalizationConfig, NormalizedText, normalize_text
from .rouge_l import compute_rouge_l
from .source_scorer import (
    SOURCE_METRIC_CONTRACT_VERSION,
    SOURCE_SCORER_ID,
    SOURCE_SCORER_NAME,
    SOURCE_SCORER_VERSION,
    SourceScorerDependencyError,
    compute_source_meteor,
    compute_source_rouge_l,
    source_normalization_metadata,
    source_scorer_metadata,
)

LOCAL_SCORER_ID = "local_exact_token_metrics"
EVALUATOR_NAME = "legal_rag.local_exact_token_metrics"
EVALUATOR_VERSION = "local-v1"
METRIC_CONTRACT_VERSION = "A2-local-v1"


class MetricResult(Protocol):
    """Minimal interface shared by local and source-scorer metric results."""

    @property
    def score(self) -> float:
        """Return the aggregate metric score."""

    def as_dict(self) -> Mapping[str, object]:
        """Serialize bounded metric diagnostics."""


@dataclass(frozen=True)
class EvaluationOptions:
    """Metadata and local aggregation choices for one evaluation run."""

    run_id: str = "local-evaluation"
    method: str = "local"
    split: str = "unknown"
    reference_access: ReferenceAccess = "approved_evaluation"
    data_manifest_hash: str = "UNRESOLVED"
    prediction_artifact: str = "UNRESOLVED"
    command: str = "UNRESOLVED"
    scorer: str = SOURCE_SCORER_ID
    scorer_source_path: str = "UNRESOLVED"
    scorer_source_sha256: str = "UNRESOLVED"
    method_version: str = "UNRESOLVED"
    pipeline_method: str = "UNRESOLVED"
    prediction_run_id: str = "UNRESOLVED"
    prediction_source_run_dir: str = "UNRESOLVED"
    prediction_artifact_sha256: str = "UNRESOLVED"
    source_prediction_artifact_sha256: str = "UNRESOLVED"
    checkpoint_manifest_hash: str = "UNRESOLVED"
    adapter_hash: str = "UNRESOLVED"
    base_model: str = "UNRESOLVED"
    base_revision: str = "UNRESOLVED"
    inference_config_hash: str = "UNRESOLVED"
    retrieval_config_hash: str = "UNRESOLVED"
    index_fingerprint: str = "UNRESOLVED"
    validation_manifest_sha256: str = "UNRESOLVED"
    included_ids_hash: str = "UNRESOLVED"
    normalization: NormalizationConfig = NormalizationConfig()


@dataclass(frozen=True)
class CaseEvaluation:
    """Per-case result, including intermediate values for unit diagnostics."""

    id: str
    status: str
    normalized_reference: NormalizedText
    normalized_prediction: NormalizedText
    meteor: MetricResult | None
    rouge_l: MetricResult | None
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


def _score_case_local(
    record: AlignedRecord, config: NormalizationConfig
) -> CaseEvaluation:
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


def _score_case_source(record: AlignedRecord) -> CaseEvaluation:
    """Score using the declared BTC-source library/tokenization boundary."""

    reference = NormalizedText(
        text=record.reference,
        tokens=tuple(record.reference.split()),
    )
    prediction = NormalizedText(
        text=record.prediction,
        tokens=tuple(record.prediction.split()),
    )
    if not record.prediction.strip():
        return CaseEvaluation(
            id=record.id,
            status="empty_prediction",
            normalized_reference=reference,
            normalized_prediction=prediction,
            meteor=None,
            rouge_l=None,
            error_code="EMPTY_PREDICTION",
        )
    if not record.reference.strip():
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
        meteor=compute_source_meteor(record.reference, record.prediction),
        rouge_l=compute_source_rouge_l(record.reference, record.prediction),
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
    split_usage = None
    if selected_options.split != "unknown":
        split_usage = validate_reference_access(
            selected_options.split, selected_options.reference_access
        )
    aligned = align_records(references, predictions)
    if selected_options.scorer == SOURCE_SCORER_ID:
        try:
            scorer_metadata = source_scorer_metadata()
        except SourceScorerDependencyError:
            raise
        cases = tuple(_score_case_source(record) for record in aligned)
        evaluator_name = SOURCE_SCORER_NAME
        evaluator_version = SOURCE_SCORER_VERSION
        metric_contract_version = SOURCE_METRIC_CONTRACT_VERSION
        normalization = source_normalization_metadata()
    elif selected_options.scorer == LOCAL_SCORER_ID:
        scorer_metadata = {
            "scorer_id": LOCAL_SCORER_ID,
            "evaluator_name": "legal_rag.local_exact_token_metrics",
            "evaluator_version": "local-v1",
            "metric_contract_version": "A2-local-v1",
            "official_equivalence": "UNVERIFIED",
        }
        cases = tuple(
            _score_case_local(record, selected_options.normalization)
            for record in aligned
        )
        evaluator_name = str(scorer_metadata["evaluator_name"])
        evaluator_version = str(scorer_metadata["evaluator_version"])
        metric_contract_version = str(scorer_metadata["metric_contract_version"])
        normalization = selected_options.normalization.as_dict()
    else:
        raise ValueError(f"Unknown evaluation scorer: {selected_options.scorer}")
    scored_cases = [case for case in cases if case.status == "scored"]
    meteor_scores = [case.meteor.score for case in scored_cases if case.meteor]
    rouge_scores = [case.rouge_l.score for case in scored_cases if case.rouge_l]
    empty_prediction_count = sum(case.status == "empty_prediction" for case in cases)
    empty_reference_count = sum(case.status == "empty_reference" for case in cases)
    errors_count = sum(case.error_code is not None for case in cases)

    artifact = {
        "schema_version": "a3.metrics.v2",
        "run_id": selected_options.run_id,
        "method": selected_options.method,
        "method_version": selected_options.method_version,
        "split": selected_options.split,
        "split_policy": split_usage.policy if split_usage else "UNRESOLVED",
        "split_role": split_usage.purpose if split_usage else "UNRESOLVED",
        "metric_contract_version": metric_contract_version,
        "evaluator_kind": "local",
        "evaluator_name": evaluator_name,
        "evaluator_version": evaluator_version,
        "python_version": platform.python_version(),
        "data_manifest_hash": selected_options.data_manifest_hash,
        "prediction_artifact": selected_options.prediction_artifact,
        "reference_role": "evaluation_reference_only",
        "reference_access": selected_options.reference_access,
        "command": selected_options.command,
        "scorer": scorer_metadata,
        "normalization": normalization,
        "provenance": {
            "pipeline_method": selected_options.pipeline_method,
            "prediction_run_id": selected_options.prediction_run_id,
            "prediction_source_run_dir": selected_options.prediction_source_run_dir,
            "prediction_artifact_sha256": selected_options.prediction_artifact_sha256,
            "source_prediction_artifact_sha256": (
                selected_options.source_prediction_artifact_sha256
            ),
            "checkpoint_manifest_hash": selected_options.checkpoint_manifest_hash,
            "adapter_hash": selected_options.adapter_hash,
            "base_model": selected_options.base_model,
            "base_revision": selected_options.base_revision,
            "inference_config_hash": selected_options.inference_config_hash,
            "retrieval_config_hash": selected_options.retrieval_config_hash,
            "index_fingerprint": selected_options.index_fingerprint,
            "validation_manifest_sha256": selected_options.validation_manifest_sha256,
            "included_ids_hash": selected_options.included_ids_hash,
            "scorer_source_path": selected_options.scorer_source_path,
            "scorer_source_sha256": selected_options.scorer_source_sha256,
        },
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
