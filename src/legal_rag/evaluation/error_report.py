"""Evaluation-only retrieval and generation error report generation."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..schemas import PackedEvidence
from ..splits import SplitAccessError, validate_reference_access
from .alignment import canonical_id
from .io import load_records

ERROR_TYPES = frozenset(
    {
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
)
_EVALUATION_REFERENCE_ROLE = "evaluation_reference_only"


class ErrorReportError(ValueError):
    """Raised when an evaluation-only error report cannot be built safely."""


@dataclass(frozen=True, slots=True)
class PredictionReportRow:
    """Prediction fields allowed into an evaluation report join."""

    id: str
    answer: str | None
    method: str | None
    status: str | None


@dataclass(frozen=True, slots=True)
class EvidencePreview:
    """Bounded preview of one packed evidence block."""

    rank: int
    chunk_id: str
    document_id: str
    preview: str

    def as_text(self) -> str:
        return f"{self.rank}:{self.chunk_id} ({self.document_id}) {self.preview}"


@dataclass(frozen=True, slots=True)
class ErrorCase:
    """One joined prediction/reference/retrieval/metric case."""

    id: str
    prediction: str | None
    reference: str
    meteor: float | None
    rouge_l: float | None
    metric_status: str
    retrieval_status: str
    evidence_previews: tuple[EvidencePreview, ...]
    error_type: str

    @property
    def evidence_text(self) -> str:
        return "\n".join(preview.as_text() for preview in self.evidence_previews)


@dataclass(frozen=True, slots=True)
class ErrorReport:
    """Deterministically ordered evaluation-only error report."""

    run_id: str
    method: str
    split: str
    reference_role: str
    cases: tuple[ErrorCase, ...]

    def to_markdown(self) -> str:
        """Render a worst-first Markdown report containing joined text/evidence."""

        lines = [
            "# Retrieval/Generation Error Report",
            "",
            f"- run_id: `{self.run_id}`",
            f"- method: `{self.method}`",
            f"- split: `{self.split}`",
            f"- reference_role: `{self.reference_role}`",
            f"- cases: `{len(self.cases)}`",
            "",
            (
                "| rank | id | METEOR | ROUGE-L | error_type | retrieval | "
                "evidence previews | prediction | reference |"
            ),
            "|---:|---|---:|---:|---|---|---|---|---|",
        ]
        for rank, case in enumerate(self.cases, start=1):
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(rank),
                        _markdown_cell(case.id),
                        _metric_cell(case.meteor),
                        _metric_cell(case.rouge_l),
                        _markdown_cell(case.error_type),
                        _markdown_cell(case.retrieval_status),
                        _markdown_cell(case.evidence_text),
                        _markdown_cell(case.prediction or "<missing>"),
                        _markdown_cell(case.reference),
                    )
                )
                + " |"
            )
        return "\n".join(lines) + "\n"

    def to_csv(self) -> str:
        """Render the same joined cases as UTF-8 CSV text."""

        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            (
                "rank",
                "id",
                "meteor",
                "rouge_l",
                "error_type",
                "retrieval_status",
                "evidence_previews",
                "prediction",
                "reference",
            )
        )
        for rank, case in enumerate(self.cases, start=1):
            writer.writerow(
                (
                    rank,
                    case.id,
                    "" if case.meteor is None else case.meteor,
                    "" if case.rouge_l is None else case.rouge_l,
                    case.error_type,
                    case.retrieval_status,
                    case.evidence_text,
                    "" if case.prediction is None else case.prediction,
                    case.reference,
                )
            )
        return output.getvalue()


def _markdown_cell(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", "")
        .replace("\n", "<br>")
    )


def _metric_cell(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _read_json(path: str | Path) -> Mapping[str, object]:
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ErrorReportError(f"Unable to read JSON artifact: {input_path}") from exc
    if not isinstance(payload, Mapping):
        raise ErrorReportError(f"JSON artifact must be an object: {input_path}")
    return payload


def _read_jsonl(path: str | Path) -> tuple[Mapping[str, object], ...]:
    input_path = Path(path)
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ErrorReportError(f"Unable to read JSONL artifact: {input_path}") from exc
    records: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ErrorReportError(
                f"Invalid JSONL at {input_path}:{line_number}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ErrorReportError(
                f"JSONL row must be an object at {input_path}:{line_number}"
            )
        records.append(payload)
    return tuple(records)


def _row_id(row: Mapping[str, object], location: str) -> str:
    if "id" not in row:
        raise ErrorReportError(f"{location} is missing id")
    try:
        return canonical_id(row["id"])
    except ValueError as exc:
        raise ErrorReportError(f"{location} has an invalid id") from exc


def _unique_rows(
    rows: tuple[Mapping[str, object], ...], role: str
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for index, row in enumerate(rows):
        identifier = _row_id(row, f"{role}[{index}]")
        if identifier in result:
            raise ErrorReportError(f"Duplicate {role} id: {identifier}")
        result[identifier] = row
    return result


def _prediction_rows(path: str | Path) -> dict[str, PredictionReportRow]:
    result: dict[str, PredictionReportRow] = {}
    for index, row in enumerate(_read_jsonl(path)):
        identifier = _row_id(row, f"prediction[{index}]")
        if identifier in result:
            raise ErrorReportError(f"Duplicate prediction id: {identifier}")
        answer = row.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise ErrorReportError(f"prediction[{index}].answer must be a string")
        result[identifier] = PredictionReportRow(
            id=identifier,
            answer=answer,
            method=_optional_text(row.get("method")),
            status=_optional_text(row.get("status")),
        )
    return result


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ErrorReportError("Expected a string or null")
    return value


def _metric_value(value: object, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ErrorReportError(f"Metric {name} must be numeric or null")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ErrorReportError(f"Metric {name} must be between zero and one")
    return numeric


def _metrics_rows(
    path: str | Path,
) -> tuple[Mapping[str, object], dict[str, Mapping[str, object]]]:
    payload = _read_json(path)
    if payload.get("reference_role") != _EVALUATION_REFERENCE_ROLE:
        raise ErrorReportError(
            "Error reports require an evaluation metric artifact with "
            "reference_role='evaluation_reference_only'"
        )
    rows = payload.get("per_case")
    if not isinstance(rows, list):
        raise ErrorReportError("Evaluation artifact is missing per_case metrics")
    return payload, _unique_rows(tuple(rows), "metric")


def _retrieval_rows(
    path: str | Path | None,
    *,
    top_evidence_k: int,
    preview_chars: int,
) -> dict[str, tuple[str, tuple[EvidencePreview, ...]]]:
    if path is None:
        return {}
    if top_evidence_k <= 0 or preview_chars <= 0:
        raise ValueError("Evidence limits must be greater than zero")
    result: dict[str, tuple[str, tuple[EvidencePreview, ...]]] = {}
    for index, row in enumerate(_read_jsonl(path)):
        identifier = _row_id(row, f"retrieval[{index}]")
        if identifier in result:
            raise ErrorReportError(f"Duplicate retrieval id: {identifier}")
        packed_raw = row.get("packed_evidence")
        previews: list[EvidencePreview] = []
        status = str(row.get("status", "unknown"))
        if packed_raw is not None:
            try:
                packed = PackedEvidence.model_validate(packed_raw)
            except (TypeError, ValueError) as exc:
                raise ErrorReportError(
                    f"Invalid packed evidence for retrieval id: {identifier}"
                ) from exc
            blocks = packed.rendered_text.split("\n\n")
            for position, hit in enumerate(packed.included_hits[:top_evidence_k]):
                block = blocks[position] if position < len(blocks) else ""
                previews.append(
                    EvidencePreview(
                        rank=hit.rank,
                        chunk_id=hit.chunk_id,
                        document_id=hit.document_id,
                        preview=_bounded_preview(block, preview_chars),
                    )
                )
            if status == "success" and not previews:
                status = "miss"
        elif status == "success":
            status = "no_packed_evidence"
        result[identifier] = (status, tuple(previews))
    return result


def _bounded_preview(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def _sort_worst_first(case: ErrorCase) -> tuple[int, float, float, str]:
    return (
        0 if case.meteor is None else 1,
        -1.0 if case.meteor is None else case.meteor,
        -1.0 if case.rouge_l is None else case.rouge_l,
        case.id,
    )


def generate_error_report(
    predictions_path: str | Path,
    references_path: str | Path,
    metrics_path: str | Path,
    *,
    retrieval_path: str | Path | None = None,
    split: str | None = None,
    manual_error_types: Mapping[str, str] | None = None,
    default_error_type: str = "OTHER",
    top_evidence_k: int = 3,
    preview_chars: int = 320,
    allow_private: bool = False,
) -> ErrorReport:
    """Join evaluation-only references with predictions, metrics, and retrieval."""

    metrics_payload, metric_rows = _metrics_rows(metrics_path)
    artifact_split = str(metrics_payload.get("split", "unknown"))
    selected_split = artifact_split if split is None else split
    if split is not None and artifact_split not in {"unknown", split}:
        raise ErrorReportError(
            f"Requested split {split!r} does not match metrics split {artifact_split!r}"
        )
    if selected_split == "private" and not allow_private:
        raise ErrorReportError(
            "Private-answer error reports are disabled; pass allow_private=True "
            "only with explicit evaluation authorization"
        )
    if selected_split not in {"private", "unknown"}:
        try:
            validate_reference_access(selected_split, "approved_evaluation")
        except SplitAccessError as exc:
            raise ErrorReportError(str(exc)) from exc
    if default_error_type not in ERROR_TYPES:
        raise ErrorReportError(f"Unknown default error_type: {default_error_type}")
    selected_manual = dict(manual_error_types or {})
    unknown_manual_ids = set(selected_manual) - set(metric_rows)
    if unknown_manual_ids:
        raise ErrorReportError(
            "Manual error_type contains unknown IDs: "
            + ", ".join(sorted(unknown_manual_ids))
        )
    for identifier, error_type in selected_manual.items():
        if error_type not in ERROR_TYPES:
            raise ErrorReportError(f"Unknown error_type for {identifier}: {error_type}")

    references = load_records(references_path, "Reference")
    reference_rows = {record.id: record.answer for record in references}
    if set(reference_rows) != set(metric_rows):
        raise ErrorReportError("Reference and metric IDs do not match exactly")
    predictions = _prediction_rows(predictions_path)
    extra_predictions = set(predictions) - set(reference_rows)
    if extra_predictions:
        raise ErrorReportError(
            "Prediction IDs not present in references: "
            + ", ".join(sorted(extra_predictions))
        )
    retrieval = _retrieval_rows(
        retrieval_path,
        top_evidence_k=top_evidence_k,
        preview_chars=preview_chars,
    )
    extra_retrieval = set(retrieval) - set(reference_rows)
    if extra_retrieval:
        raise ErrorReportError(
            "Retrieval IDs not present in references: "
            + ", ".join(sorted(extra_retrieval))
        )

    cases: list[ErrorCase] = []
    for identifier in sorted(reference_rows):
        metric = metric_rows[identifier]
        retrieval_status, previews = retrieval.get(identifier, ("not_available", ()))
        cases.append(
            ErrorCase(
                id=identifier,
                prediction=(
                    predictions[identifier].answer
                    if identifier in predictions
                    else None
                ),
                reference=reference_rows[identifier],
                meteor=_metric_value(metric.get("meteor"), "meteor"),
                rouge_l=_metric_value(metric.get("rouge_l"), "rouge_l"),
                metric_status=str(metric.get("status", "unknown")),
                retrieval_status=retrieval_status,
                evidence_previews=previews,
                error_type=selected_manual.get(identifier, default_error_type),
            )
        )
    return ErrorReport(
        run_id=str(metrics_payload.get("run_id", "unknown")),
        method=str(metrics_payload.get("method", "unknown")),
        split=selected_split,
        reference_role=_EVALUATION_REFERENCE_ROLE,
        cases=tuple(sorted(cases, key=_sort_worst_first)),
    )


def write_error_report(
    report: ErrorReport,
    markdown_path: str | Path,
    csv_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write Markdown and CSV reports without overwriting by default."""

    _write_text(Path(markdown_path), report.to_markdown(), overwrite=overwrite)
    _write_text(Path(csv_path), report.to_csv(), overwrite=overwrite)


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


__all__ = [
    "ERROR_TYPES",
    "ErrorCase",
    "ErrorReport",
    "ErrorReportError",
    "EvidencePreview",
    "generate_error_report",
    "write_error_report",
]
