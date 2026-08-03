"""Auxiliary extractive-reader baselines."""

from .backends import (
    MockExtractiveReader,
    ReaderUnavailableError,
    TransformersExtractiveReader,
    create_local_reader,
    validate_checkpoint,
)
from .bm25 import ReaderBM25Hit, ReaderBM25Index
from .data import ReaderDataError, ReaderDataset, ReaderSplit, load_reader_dataset
from .metrics import ReaderMetrics, evaluate_reader_predictions
from .pipeline import (
    ReaderMethod,
    ReaderPipelineError,
    ReaderRunResult,
    run_reader,
    run_reader_from_config,
)
from .types import (
    ExtractiveReader,
    ReaderCandidate,
    ReaderCase,
    ReaderInferenceCase,
    ReaderPrediction,
    ReaderSpan,
)

__all__ = [
    "ExtractiveReader",
    "MockExtractiveReader",
    "ReaderBM25Hit",
    "ReaderBM25Index",
    "ReaderCandidate",
    "ReaderCase",
    "ReaderDataError",
    "ReaderDataset",
    "ReaderInferenceCase",
    "ReaderMethod",
    "ReaderMetrics",
    "ReaderPipelineError",
    "ReaderPrediction",
    "ReaderRunResult",
    "ReaderSpan",
    "ReaderSplit",
    "ReaderUnavailableError",
    "TransformersExtractiveReader",
    "create_local_reader",
    "evaluate_reader_predictions",
    "load_reader_dataset",
    "run_reader",
    "run_reader_from_config",
    "validate_checkpoint",
]
