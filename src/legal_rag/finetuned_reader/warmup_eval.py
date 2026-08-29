"""VAL-01: local clean-warmup evaluation (question-only infer → eval gold join)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from legal_rag.evaluation import (
    EvaluationOptions,
    EvaluationReport,
    evaluate_records,
    write_report,
)
from legal_rag.evaluation.error_report import (
    ErrorReport,
    generate_error_report,
    write_error_report,
)
from legal_rag.evaluation.io import load_records
from legal_rag.evaluation.models import InputRecord
from legal_rag.finetuned_reader.warmup_validation import (
    POLICY_ID as CLEAN_WARMUP_POLICY_ID,
)
from legal_rag.finetuned_reader.warmup_validation import (
    write_json_atomic,
)
from legal_rag.questions import load_inference_questions
from legal_rag.schemas import InferenceQuestion
from legal_rag.splits import validate_reference_access

VAL01_POLICY_ID = "sedar-warmup-local-eval-v1"
SCHEMA_VERSION = 2

T = TypeVar("T")


class WarmupEvalError(ValueError):
    """Raised when clean-warmup local evaluation cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class CleanWarmupManifestView:
    """IDs-only view of the VAL-00 clean warmup validation manifest."""

    path: Path
    policy_id: str
    included_ids: tuple[str, ...]
    included_ids_hash: str
    excluded_ids_hash: str
    source_warmup_sha256: str
    source_public_sha256: str
    included_count: int
    excluded_count: int
    source_warmup_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "policy_id": self.policy_id,
            "included_ids_hash": self.included_ids_hash,
            "excluded_ids_hash": self.excluded_ids_hash,
            "source_warmup_sha256": self.source_warmup_sha256,
            "source_public_sha256": self.source_public_sha256,
            "included_count": self.included_count,
            "excluded_count": self.excluded_count,
            "source_warmup_count": self.source_warmup_count,
        }


@dataclass(frozen=True, slots=True)
class PredictionRunProvenance:
    """Immutable provenance loaded from the inference run that produced predictions."""

    source_run_dir: Path
    source_run_id: str
    pipeline_method: str
    config_hash: str
    checkpoint_manifest_hash: str
    adapter_hash: str
    base_model: str
    base_revision: str
    retrieval_config_hash: str
    index_fingerprint: str
    source_prediction_artifact_sha256: str

    def as_dict(self) -> dict[str, str]:
        """Return content-free prediction provenance for evaluation artifacts."""

        return {
            "source_run_dir": self.source_run_dir.as_posix(),
            "source_run_id": self.source_run_id,
            "pipeline_method": self.pipeline_method,
            "config_hash": self.config_hash,
            "checkpoint_manifest_hash": self.checkpoint_manifest_hash,
            "adapter_hash": self.adapter_hash,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "retrieval_config_hash": self.retrieval_config_hash,
            "index_fingerprint": self.index_fingerprint,
            "source_prediction_artifact_sha256": (
                self.source_prediction_artifact_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class WarmupEvalResult:
    """Outputs from one VAL-01 evaluation boundary pass."""

    manifest: CleanWarmupManifestView
    selected_ids: tuple[str, ...]
    report: EvaluationReport
    summary: dict[str, Any]
    error_report: ErrorReport | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WarmupEvalError(f"Unable to load inference provenance: {path}") from exc
    if not isinstance(payload, Mapping):
        raise WarmupEvalError(f"Inference provenance must be a JSON object: {path}")
    return payload


def load_prediction_run_provenance(
    run_dir: Path,
    *,
    method: str,
    method_version: str,
) -> PredictionRunProvenance:
    """Load and cross-check provenance for a prediction-producing run.

    A filtered prediction file is not sufficient to identify a checkpoint.  The
    caller must point to the immutable source run directory containing
    ``run_summary.json``, ``config.json``, ``checkpoint_reference.json`` and
    ``predictions.jsonl``.
    """

    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.json"
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoint_reference.json"
    predictions_path = run_dir / "predictions.jsonl"
    for path in (summary_path, config_path, predictions_path):
        if not path.is_file():
            raise WarmupEvalError(f"Prediction run provenance file is missing: {path}")
    if method in {"sedar_sft", "finetuned_reader"} and method_version == "UNRESOLVED":
        raise WarmupEvalError(
            "SEDAR/FTR evaluation requires explicit --method-version; "
            "do not infer the model version from a folder name"
        )

    summary = _read_json_object(summary_path)
    config = _read_json_object(config_path)
    checkpoint = _read_json_object(checkpoint_path) if checkpoint_path.is_file() else {}
    required_summary = ("run_id", "method", "config_hash", "index_fingerprint")
    missing_summary = [key for key in required_summary if key not in summary]
    if missing_summary:
        raise WarmupEvalError(
            "Prediction run summary missing provenance fields: "
            + ", ".join(missing_summary)
        )
    config_hash = str(summary["config_hash"])
    if str(config.get("config_hash", "")) != config_hash:
        raise WarmupEvalError(
            "Prediction run config hash does not match run_summary.json"
        )

    if method in {"sedar_sft", "finetuned_reader"}:
        required_checkpoint = (
            "checkpoint_manifest_hash",
            "adapter_hash",
            "base_model",
            "base_revision",
            "retrieval_config_hash",
            "index_fingerprint",
        )
        missing_checkpoint = [
            key for key in required_checkpoint if key not in checkpoint
        ]
        if missing_checkpoint:
            raise WarmupEvalError(
                "FTR prediction run missing checkpoint provenance fields: "
                + ", ".join(missing_checkpoint)
            )
        if str(checkpoint["index_fingerprint"]) != str(summary["index_fingerprint"]):
            raise WarmupEvalError(
                "Checkpoint index fingerprint does not match run_summary.json"
            )
        checkpoint_values = {key: str(checkpoint[key]) for key in required_checkpoint}
    else:
        checkpoint_values = {
            "checkpoint_manifest_hash": "NOT_APPLICABLE",
            "adapter_hash": "NOT_APPLICABLE",
            "base_model": "NOT_APPLICABLE",
            "base_revision": "NOT_APPLICABLE",
            "retrieval_config_hash": "NOT_APPLICABLE",
            "index_fingerprint": str(summary["index_fingerprint"]),
        }

    return PredictionRunProvenance(
        source_run_dir=run_dir.resolve(),
        source_run_id=str(summary["run_id"]),
        pipeline_method=str(summary["method"]),
        config_hash=config_hash,
        checkpoint_manifest_hash=checkpoint_values["checkpoint_manifest_hash"],
        adapter_hash=checkpoint_values["adapter_hash"],
        base_model=checkpoint_values["base_model"],
        base_revision=checkpoint_values["base_revision"],
        retrieval_config_hash=checkpoint_values["retrieval_config_hash"],
        index_fingerprint=checkpoint_values["index_fingerprint"],
        source_prediction_artifact_sha256=_sha256_file(predictions_path),
    )


def load_clean_warmup_manifest(path: Path) -> CleanWarmupManifestView:
    """Load the VAL-00 IDs-only manifest; refuse payloads that carry answers."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WarmupEvalError(f"Unable to load clean warmup manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise WarmupEvalError(f"Clean warmup manifest must be a JSON object: {path}")
    serialized = json.dumps(payload, ensure_ascii=False)
    if '"answer"' in serialized or '"reference"' in serialized:
        raise WarmupEvalError(
            f"Clean warmup manifest must be IDs-only (no answer/reference): {path}"
        )
    policy_id = str(payload.get("policy_id", ""))
    if policy_id != CLEAN_WARMUP_POLICY_ID:
        raise WarmupEvalError(
            f"Unexpected clean warmup policy_id {policy_id!r}; "
            f"expected {CLEAN_WARMUP_POLICY_ID!r}"
        )
    included_raw = payload.get("included_ids")
    if not isinstance(included_raw, list) or not all(
        isinstance(item, str) for item in included_raw
    ):
        raise WarmupEvalError(
            "Clean warmup manifest.included_ids must be a string list"
        )
    included_ids = tuple(included_raw)
    if list(included_ids) != sorted(included_ids):
        raise WarmupEvalError(
            "Clean warmup included_ids must be deterministically sorted"
        )
    required = (
        "included_ids_hash",
        "excluded_ids_hash",
        "source_warmup_sha256",
        "source_public_sha256",
        "included_count",
        "excluded_count",
        "source_warmup_count",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise WarmupEvalError(
            "Clean warmup manifest missing fields: " + ", ".join(missing)
        )
    if int(payload["included_count"]) != len(included_ids):
        raise WarmupEvalError("included_count does not match included_ids length")
    return CleanWarmupManifestView(
        path=path,
        policy_id=policy_id,
        included_ids=included_ids,
        included_ids_hash=str(payload["included_ids_hash"]),
        excluded_ids_hash=str(payload["excluded_ids_hash"]),
        source_warmup_sha256=str(payload["source_warmup_sha256"]),
        source_public_sha256=str(payload["source_public_sha256"]),
        included_count=int(payload["included_count"]),
        excluded_count=int(payload["excluded_count"]),
        source_warmup_count=int(payload["source_warmup_count"]),
    )


def select_included_ids(
    included_ids: Sequence[str],
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Optionally take the first N sorted included IDs for smoke runs."""

    selected = tuple(included_ids)
    if limit is None:
        return selected
    if limit < 1:
        raise WarmupEvalError("--limit must be >= 1 when provided")
    return selected[:limit]


def filter_by_included_ids(
    items: Sequence[T],
    included_ids: Sequence[str],
    *,
    id_of: Callable[[T], str],
    role: str,
) -> tuple[T, ...]:
    """Return items for exactly ``included_ids`` in manifest order; fail closed."""

    by_id: dict[str, T] = {}
    for item in items:
        item_id = str(id_of(item))
        if item_id in by_id:
            raise WarmupEvalError(f"Duplicate {role} ID: {item_id}")
        by_id[item_id] = item
    missing = [item_id for item_id in included_ids if item_id not in by_id]
    if missing:
        preview = ", ".join(missing[:10])
        raise WarmupEvalError(
            f"{role} missing {len(missing)} clean-warmup ID(s); examples: {preview}"
        )
    return tuple(by_id[item_id] for item_id in included_ids)


def load_clean_inference_questions(
    question_path: Path,
    *,
    split: str,
    included_ids: Sequence[str],
) -> tuple[InferenceQuestion, ...]:
    """Load question-only cases and keep exactly the clean validation IDs."""

    cases = load_inference_questions(question_path, split=split)  # type: ignore[arg-type]
    for case in cases:
        if type(case) is not InferenceQuestion:
            raise WarmupEvalError("Inference questions must be InferenceQuestion")
    return filter_by_included_ids(
        cases,
        included_ids,
        id_of=lambda case: case.id,
        role="inference question",
    )


def load_clean_reference_records(
    references_path: Path,
    *,
    split: str,
    included_ids: Sequence[str],
) -> list[InputRecord]:
    """Open gold only after the approved evaluation gate, then subset to clean IDs."""

    validate_reference_access(split, "approved_evaluation")
    records = load_records(references_path, "Reference")
    filtered = filter_by_included_ids(
        records,
        included_ids,
        id_of=lambda record: record.id,
        role="reference",
    )
    return list(filtered)


def write_eval_only_references(
    records: Sequence[InputRecord],
    path: Path,
) -> None:
    """Write an evaluation-boundary-only map for scoring/error-report joiners.

    Uses the competition question-map shape expected by ``load_records``. Question
    text is intentionally blank so this artifact cannot seed retrieval/prompts;
    only ``answer`` is used inside the approved evaluation boundary.
    """

    shaped = {
        record.id: {"question": "", "answer": record.answer} for record in records
    }
    write_json_atomic(path, shaped)


def evaluate_clean_warmup(
    *,
    references: Sequence[InputRecord],
    predictions_path: Path,
    options: EvaluationOptions,
) -> EvaluationReport:
    """Score predictions against clean-warmup references already opened for eval."""

    predictions = load_records(predictions_path, "Prediction")
    prediction_ids = {record.id for record in predictions}
    reference_ids = {record.id for record in references}
    if prediction_ids != reference_ids:
        missing = sorted(reference_ids - prediction_ids)
        extra = sorted(prediction_ids - reference_ids)
        raise WarmupEvalError(
            "Prediction/reference ID sets differ after clean-warmup filter; "
            f"missing={len(missing)} extra={len(extra)}"
        )
    ordered_predictions = filter_by_included_ids(
        predictions,
        [record.id for record in references],
        id_of=lambda record: record.id,
        role="prediction",
    )
    return evaluate_records(list(references), list(ordered_predictions), options)


def build_eval_summary(
    *,
    manifest: CleanWarmupManifestView,
    selected_ids: Sequence[str],
    method: str,
    run_id: str,
    predictions_path: Path,
    metrics_path: Path,
    report: EvaluationReport,
    inference_used_answers: bool,
    references_opened_before_predictions: bool,
) -> dict[str, Any]:
    """Content-free provenance for one VAL-01 run."""

    if inference_used_answers:
        raise WarmupEvalError("Gold answers must not enter the inference path")
    if references_opened_before_predictions:
        raise WarmupEvalError(
            "References must be opened only after predictions exist "
            "(evaluation boundary)"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": VAL01_POLICY_ID,
        "clean_warmup_policy_id": manifest.policy_id,
        "clean_warmup_manifest": manifest.as_dict(),
        "selected_ids_count": len(selected_ids),
        "selected_ids_hash_scope": (
            "full_manifest"
            if len(selected_ids) == manifest.included_count
            else "limit_prefix"
        ),
        "method": method,
        "method_version": report.artifact.get("method_version", "UNRESOLVED"),
        "run_id": run_id,
        "split": "warmup",
        "predictions_path": predictions_path.as_posix(),
        "metrics_path": metrics_path.as_posix(),
        "evaluator": {
            "kind": report.artifact.get("evaluator_kind"),
            "name": report.artifact.get("evaluator_name"),
            "version": report.artifact.get("evaluator_version"),
            "metric_contract_version": report.artifact.get("metric_contract_version"),
            "scorer": report.artifact.get("scorer"),
        },
        "provenance": report.artifact.get("provenance"),
        "metrics": report.artifact.get("metrics"),
        "counts": report.artifact.get("counts"),
        "inference_used_answers": False,
        "references_opened_before_predictions": False,
        "gold_boundary": "approved_evaluation",
    }


def write_warmup_eval_artifacts(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    report: EvaluationReport,
    references: Sequence[InputRecord] | None = None,
    error_report: ErrorReport | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write metrics/summary and optional eval-only references + error report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "metrics": output_dir / "metrics.json",
        "summary": output_dir / "summary.json",
    }
    write_report(report, paths["metrics"], overwrite=overwrite)
    if paths["summary"].exists() and not overwrite:
        raise FileExistsError(paths["summary"])
    write_json_atomic(paths["summary"], dict(summary))
    if references is not None:
        paths["references_eval_only"] = output_dir / "references_eval_only.json"
        if paths["references_eval_only"].exists() and not overwrite:
            raise FileExistsError(paths["references_eval_only"])
        # Allow rebuilds when overwrite=True by removing the prior eval-only file.
        if paths["references_eval_only"].exists() and overwrite:
            paths["references_eval_only"].unlink()
        write_eval_only_references(references, paths["references_eval_only"])
    if error_report is not None:
        paths["error_report_md"] = output_dir / "error_report.md"
        paths["error_report_csv"] = output_dir / "error_report.csv"
        write_error_report(
            error_report,
            markdown_path=paths["error_report_md"],
            csv_path=paths["error_report_csv"],
            overwrite=overwrite,
        )
    return paths


def build_error_report_for_clean_eval(
    *,
    predictions_path: Path,
    references_eval_only_path: Path,
    metrics_path: Path,
    retrieval_path: Path | None = None,
    split: str = "warmup",
) -> ErrorReport:
    """Join per-case diagnostics inside the evaluation boundary only."""

    return generate_error_report(
        predictions_path,
        references_eval_only_path,
        metrics_path,
        retrieval_path=retrieval_path,
        split=split,
    )


__all__ = [
    "CLEAN_WARMUP_POLICY_ID",
    "CleanWarmupManifestView",
    "SCHEMA_VERSION",
    "VAL01_POLICY_ID",
    "WarmupEvalError",
    "WarmupEvalResult",
    "build_error_report_for_clean_eval",
    "build_eval_summary",
    "evaluate_clean_warmup",
    "filter_by_included_ids",
    "load_clean_inference_questions",
    "load_clean_reference_records",
    "load_clean_warmup_manifest",
    "select_included_ids",
    "write_eval_only_references",
    "write_warmup_eval_artifacts",
]
