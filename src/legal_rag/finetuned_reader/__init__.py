"""Generative finetuned_reader support (LegalQACompetition experimental path)."""

from importlib import import_module
from typing import Any

from .b2_freeze import (
    B2ControlIdentity,
    B2FreezeDriftError,
    B2FreezeFingerprint,
    B2FreezeIncompleteError,
    build_b2_freeze_fingerprint,
    control_identity_from_config,
    load_b2_freeze_fingerprint,
    require_complete_b2_freeze,
    validate_against_b2_freeze,
)
from .checkpoint import (
    CheckpointValidationError,
    ValidatedCheckpoint,
    hash_directory,
    validate_checkpoint,
)
from .collator import (
    TargetDoesNotFitError,
    TokenizedSFTExample,
    build_answer_only_labels,
    collate_tokenized,
    tokenize_sft_example,
)
from .contracts import EvidenceRecord, ExcludedExample, GenerationResult, SFTExample
from .dataset import (
    DatasetBuildError,
    DatasetBuildResult,
    FrozenB2EvidenceRetriever,
    FrozenRetrievalResult,
    build_sft_dataset,
    build_sft_dataset_from_config,
    write_dataset_artifacts,
)
from .inference import (
    FineTunedReaderGenerator,
    GenerativeReaderError,
    load_finetuned_reader_generator,
)
from .pipeline import (
    FineTunedReaderPipelineError,
    FineTunedReaderRunResult,
    run_finetuned_reader,
    run_finetuned_reader_from_config,
)
from .prompting import GenerativePromptBuilder, GenerativePromptError, RenderedPrompt
from .trainer import RealTrainingError, RealTrainingResult, run_real_sft
from .training import (
    TrainingGateError,
    TrainingStackReport,
    inspect_training_stack,
    require_training_stack,
    run_training_smoke,
)

_LAZY_AUDIT_EXPORTS = frozenset(
    {
        "AuditResult",
        "run_data_feasibility_audit",
        "training_overlap_remediation",
        "write_audit_artifacts",
    }
)


def __getattr__(name: str) -> Any:
    """Load audit exports lazily so ``python -m`` does not pre-import its module."""

    if name in _LAZY_AUDIT_EXPORTS:
        module = import_module(".data_feasibility_audit", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AuditResult",
    "B2ControlIdentity",
    "B2FreezeDriftError",
    "B2FreezeFingerprint",
    "B2FreezeIncompleteError",
    "CheckpointValidationError",
    "DatasetBuildError",
    "DatasetBuildResult",
    "EvidenceRecord",
    "ExcludedExample",
    "FineTunedReaderGenerator",
    "FineTunedReaderPipelineError",
    "FineTunedReaderRunResult",
    "FrozenB2EvidenceRetriever",
    "FrozenRetrievalResult",
    "GenerativePromptBuilder",
    "GenerativePromptError",
    "GenerativeReaderError",
    "GenerationResult",
    "RenderedPrompt",
    "RealTrainingError",
    "RealTrainingResult",
    "SFTExample",
    "TargetDoesNotFitError",
    "TokenizedSFTExample",
    "TrainingGateError",
    "TrainingStackReport",
    "ValidatedCheckpoint",
    "build_answer_only_labels",
    "build_b2_freeze_fingerprint",
    "build_sft_dataset",
    "build_sft_dataset_from_config",
    "collate_tokenized",
    "control_identity_from_config",
    "hash_directory",
    "inspect_training_stack",
    "load_b2_freeze_fingerprint",
    "load_finetuned_reader_generator",
    "require_complete_b2_freeze",
    "require_training_stack",
    "run_data_feasibility_audit",
    "run_finetuned_reader",
    "run_finetuned_reader_from_config",
    "run_training_smoke",
    "run_real_sft",
    "tokenize_sft_example",
    "training_overlap_remediation",
    "validate_checkpoint",
    "validate_against_b2_freeze",
    "write_dataset_artifacts",
    "write_audit_artifacts",
]
