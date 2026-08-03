"""Local, deterministic evaluation for Vietnamese legal QA answers."""

from .alignment import AlignmentError, DuplicateIDError, align_records
from .error_report import (
    ERROR_TYPES,
    ErrorCase,
    ErrorReport,
    ErrorReportError,
    EvidencePreview,
    generate_error_report,
    write_error_report,
)
from .evaluator import (
    EVALUATOR_NAME,
    EVALUATOR_VERSION,
    EvaluationOptions,
    EvaluationReport,
    evaluate_records,
    write_report,
)
from .models import InputRecord
from .normalization import NormalizationConfig, NormalizedText, normalize_text

__all__ = [
    "AlignmentError",
    "DuplicateIDError",
    "EVALUATOR_NAME",
    "EVALUATOR_VERSION",
    "EvaluationOptions",
    "EvaluationReport",
    "InputRecord",
    "NormalizationConfig",
    "NormalizedText",
    "align_records",
    "evaluate_records",
    "normalize_text",
    "write_report",
    "ERROR_TYPES",
    "ErrorCase",
    "ErrorReport",
    "ErrorReportError",
    "EvidencePreview",
    "generate_error_report",
    "write_error_report",
]
