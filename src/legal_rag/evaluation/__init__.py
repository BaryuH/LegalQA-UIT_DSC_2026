"""Local, deterministic evaluation for Vietnamese legal QA answers."""

from .alignment import AlignmentError, DuplicateIDError, align_records
from .error_report import (
    ERROR_TYPES,
    ErrorCase,
    ErrorReport,
    ErrorReportError,
    EvidencePreview,
    RetrievalTrace,
    generate_error_report,
    write_error_report,
)
from .evaluator import (
    EVALUATOR_NAME,
    EVALUATOR_VERSION,
    LOCAL_SCORER_ID,
    METRIC_CONTRACT_VERSION,
    EvaluationOptions,
    EvaluationReport,
    evaluate_records,
    write_report,
)
from .models import InputRecord
from .normalization import NormalizationConfig, NormalizedText, normalize_text
from .source_scorer import (
    SOURCE_METRIC_CONTRACT_VERSION,
    SOURCE_SCORER_ID,
    SOURCE_SCORER_NAME,
    SOURCE_SCORER_VERSION,
    SourceScorerDependencyError,
)

__all__ = [
    "AlignmentError",
    "DuplicateIDError",
    "EVALUATOR_NAME",
    "EVALUATOR_VERSION",
    "EvaluationOptions",
    "EvaluationReport",
    "InputRecord",
    "LOCAL_SCORER_ID",
    "METRIC_CONTRACT_VERSION",
    "NormalizationConfig",
    "NormalizedText",
    "SOURCE_METRIC_CONTRACT_VERSION",
    "SOURCE_SCORER_ID",
    "SOURCE_SCORER_NAME",
    "SOURCE_SCORER_VERSION",
    "SourceScorerDependencyError",
    "align_records",
    "evaluate_records",
    "normalize_text",
    "write_report",
    "ERROR_TYPES",
    "ErrorCase",
    "ErrorReport",
    "ErrorReportError",
    "EvidencePreview",
    "RetrievalTrace",
    "generate_error_report",
    "write_error_report",
]
