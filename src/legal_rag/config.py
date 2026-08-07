"""Validated, reproducible configuration profiles for the Legal RAG pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .splits import (
    ReferenceAccess,
    SplitName,
    SplitPolicy,
    SplitUsage,
    get_split_usage,
    validate_reference_access,
)


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("Text must not be blank")
    return value


NonBlankText = Annotated[str, AfterValidator(_require_non_blank)]
GenerationProvider = Literal["mock", "openai", "anthropic", "ollama"]
RetrievalStrategy = Literal["none", "bm25", "bm25_rerank"]
RerankerProvider = Literal["none", "mock", "sentence_transformers"]
RerankerDevice = Literal["auto", "cpu", "cuda"]
ReaderMode = Literal["original_context", "train_context_bm25"]
ReaderDevice = Literal["auto", "cpu", "cuda"]
FineTunedDType = Literal["auto", "float32", "float16", "bfloat16"]
FineTunedAdapterType = Literal["lora", "qlora"]
FineTunedAdapterBias = Literal["none", "all", "lora_only"]
FineTunedOverlapPolicy = Literal["fail", "exclude_and_record"]
SubmissionFormat = Literal["object_by_question_id"]
SubmissionOrder = Literal["dataset"]

_SECRET_KEY_MARKERS = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "passwd",
    "secret",
)
_SECRET_EXACT_KEYS = frozenset({"token"})
_REDACTED = "<redacted>"


def _normalized_key(value: object) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _is_secret_key(value: object) -> bool:
    normalized = _normalized_key(value)
    return normalized in _SECRET_EXACT_KEYS or any(
        marker in normalized for marker in _SECRET_KEY_MARKERS
    )


def redact_secrets(value: object) -> object:
    """Return a serializable copy that replaces values beneath secret-bearing keys."""

    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if _is_secret_key(key) else redact_secrets(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact_secrets(item) for item in value]
    return value


def _reject_secret_keys(value: object, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _is_secret_key(key):
                raise ValueError(
                    f"Configuration must not contain secret key: {path}.{key}"
                )
            _reject_secret_keys(nested, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _reject_secret_keys(nested, f"{path}[{index}]")


def _validate_relative_path(value: object) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError("Configuration paths must be strings")
    raw = str(value)
    if not raw.strip():
        raise ValueError("Path must not be blank")
    path = Path(value)
    if path.is_absolute() or raw.startswith(("/", "\\")):
        raise ValueError("Configuration paths must be relative")
    return path


class ConfigSection(BaseModel):
    """Base for strict configuration sections."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProjectSection(ConfigSection):
    name: NonBlankText
    profile: NonBlankText
    python_version: NonBlankText


class DataSection(ConfigSection):
    data_dir: Path
    question_path: Path
    selected_contexts_path: Path
    split: SplitName
    split_policy: SplitPolicy

    @field_validator(
        "data_dir", "question_path", "selected_contexts_path", mode="before"
    )
    @classmethod
    def validate_relative_paths(cls, value: object) -> Path:
        return _validate_relative_path(value)


class ChunkingSection(ConfigSection):
    max_chars: int = Field(gt=0)
    overlap_chars: int = Field(ge=0)
    min_chars: int = Field(gt=0)
    version: NonBlankText

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.max_chars <= self.overlap_chars:
            raise ValueError("chunking.max_chars must be greater than overlap_chars")
        return self


class RetrievalSection(ConfigSection):
    strategy: RetrievalStrategy
    rough_top_n: int = Field(ge=1)
    k1: float = Field(gt=0, default=1.5)
    b: float = Field(ge=0.0, le=1.0, default=0.75)


class RerankerSection(ConfigSection):
    enabled: bool
    required: bool
    provider: RerankerProvider
    model: NonBlankText | None = None
    model_revision: NonBlankText | None = None
    device: RerankerDevice = "auto"
    batch_size: int = Field(gt=0, default=8)
    max_length: int = Field(gt=0, default=512)


class EvidenceSection(ConfigSection):
    evidence_top_k: int = Field(ge=1)
    max_total_chars: int = Field(gt=0)
    max_chunks_per_document: int = Field(ge=1)


class GenerationSection(ConfigSection):
    provider: GenerationProvider
    model: NonBlankText
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_chars: int = Field(gt=0)
    max_completion_length: int | None = Field(default=None, gt=0)
    retries: int = Field(ge=0)
    base_url: NonBlankText | None = None
    api_key_env: NonBlankText | None = None
    timeout_seconds: float = Field(gt=0, default=30.0)
    backoff_seconds: float = Field(ge=0, default=0.5)

    @model_validator(mode="after")
    def validate_provider_settings(self) -> Self:
        if self.provider in {"openai", "ollama"} and self.base_url is None:
            raise ValueError("Configured HTTP provider requires generation.base_url")
        if self.provider == "ollama" and self.api_key_env is not None:
            raise ValueError(
                "Ollama local provider must not use generation.api_key_env"
            )
        return self


class PromptsSection(ConfigSection):
    direct_version: NonBlankText
    rag_version: NonBlankText


class EvaluationSection(ConfigSection):
    enabled: bool
    reference_access: ReferenceAccess
    primary_metric: Literal["meteor"]
    secondary_metric: Literal["rouge_l"]

    @model_validator(mode="after")
    def validate_reference_access(self) -> Self:
        if self.enabled and self.reference_access != "approved_evaluation":
            raise ValueError("Enabled evaluation requires approved reference access")
        return self


class RuntimeSection(ConfigSection):
    cache_dir: Path
    outputs_dir: Path
    artifacts_dir: Path
    fail_fast: bool
    seed: int = Field(ge=0)

    @field_validator("cache_dir", "outputs_dir", "artifacts_dir", mode="before")
    @classmethod
    def validate_relative_paths(cls, value: object) -> Path:
        return _validate_relative_path(value)


class SubmissionSection(ConfigSection):
    enabled: bool
    format: SubmissionFormat
    answer_field: Literal["answer"] = "answer"
    output_filename: Literal["submission.zip"] = "submission.zip"
    inner_filename: Literal["submission.json"] = "submission.json"
    encoding: Literal["utf-8"] = "utf-8"
    ensure_ascii: bool = False
    forbid_extra_fields: bool = True
    deterministic_order: SubmissionOrder = "dataset"
    require_full_coverage: bool = True
    reject_extra_ids: bool = True
    reject_duplicate_ids: bool = True
    reject_non_string_answers: bool = True
    reject_empty_answers: bool = False

    @model_validator(mode="after")
    def validate_official_contract(self) -> Self:
        if self.ensure_ascii:
            raise ValueError("submission.ensure_ascii must be false")
        if not self.forbid_extra_fields:
            raise ValueError("submission.forbid_extra_fields must be true")
        if not self.reject_non_string_answers:
            raise ValueError("submission.reject_non_string_answers must be true")
        return self


class ReaderSection(ConfigSection):
    """Optional extractive-reader settings, isolated from the Legal-RAG pipeline."""

    enabled: bool
    mode: ReaderMode
    dataset_path: Path
    split_manifest_path: Path
    checkpoint_path: Path
    checkpoint_manifest_path: Path
    device: ReaderDevice = "auto"
    batch_size: int = Field(gt=0, default=8)
    max_seq_length: int = Field(gt=0, default=384)
    doc_stride: int = Field(ge=0, default=128)
    max_answer_length: int = Field(gt=0, default=50)
    retrieval_top_k: int = Field(ge=1, default=5)
    k1: float = Field(gt=0, default=1.5)
    b: float = Field(ge=0.0, le=1.0, default=0.75)
    local_files_only: Literal[True] = True

    @field_validator(
        "dataset_path",
        "split_manifest_path",
        "checkpoint_path",
        "checkpoint_manifest_path",
        mode="before",
    )
    @classmethod
    def validate_relative_paths(cls, value: object) -> Path:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def validate_reader_window(self) -> Self:
        if self.doc_stride >= self.max_seq_length:
            raise ValueError("reader.doc_stride must be smaller than max_seq_length")
        if self.max_answer_length > self.max_seq_length:
            raise ValueError("reader.max_answer_length must not exceed max_seq_length")
        return self


class FineTunedModelSection(ConfigSection):
    """Exact local Transformers identity for the generative reader."""

    base_model: NonBlankText
    revision: NonBlankText
    tokenizer: NonBlankText
    loader: Literal["auto", "causal_lm", "multimodal_lm"] = "auto"
    context_length: int = Field(gt=0)
    dtype: FineTunedDType = "auto"
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    local_files_only: Literal[True] = True
    license: NonBlankText | None = None


class FineTunedLoRASection(ConfigSection):
    """LoRA/QLoRA adapter settings; empty targets remain an explicit blocker."""

    enabled: Literal[True] = True
    adapter_type: FineTunedAdapterType = "lora"
    r: int = Field(gt=0)
    alpha: float = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: tuple[NonBlankText, ...] = ()
    bias: FineTunedAdapterBias = "none"


class FineTunedTrainingSection(ConfigSection):
    """Deterministic SFT hyperparameters with answer-only-loss defaults."""

    seed: int = Field(ge=0)
    max_seq_length: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    train_batch_size: int = Field(gt=0)
    eval_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    weight_decay: float = Field(ge=0.0)
    max_grad_norm: float = Field(gt=0)
    logging_steps: int = Field(gt=0)
    packing: Literal[False] = False
    gradient_checkpointing: bool = False


class FineTunedOutputSection(ConfigSection):
    """Repository-relative output roots for datasets and checkpoints."""

    dataset_root: Path
    checkpoint_root: Path

    @field_validator("dataset_root", "checkpoint_root", mode="before")
    @classmethod
    def validate_relative_paths(cls, value: object) -> Path:
        return _validate_relative_path(value)


class FineTunedReaderSection(ConfigSection):
    """Generative SFT reader contract, isolated from the extractive reader."""

    enabled: Literal[True] = True
    required: Literal[True] = True
    type: Literal["generative_sft_reader"] = "generative_sft_reader"
    dataset_version: NonBlankText
    train_prompt_path: Path
    inference_prompt_path: Path
    checkpoint_path: Path
    checkpoint_manifest_path: Path
    max_new_tokens: int = Field(gt=0)
    overlap_policy: FineTunedOverlapPolicy = "fail"
    overlap_remediation_id: NonBlankText | None = None
    stop_sequences: tuple[str, ...] = ()
    model: FineTunedModelSection
    lora: FineTunedLoRASection
    training: FineTunedTrainingSection
    output: FineTunedOutputSection

    @field_validator(
        "train_prompt_path",
        "inference_prompt_path",
        "checkpoint_path",
        "checkpoint_manifest_path",
        mode="before",
    )
    @classmethod
    def validate_relative_paths(cls, value: object) -> Path:
        return _validate_relative_path(value)

    @field_validator("stop_sequences")
    @classmethod
    def validate_stop_sequences(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("finetuned_reader.stop_sequences must not contain blanks")
        return value

    @model_validator(mode="after")
    def validate_sequence_budget(self) -> Self:
        if self.max_new_tokens >= self.training.max_seq_length:
            raise ValueError(
                "finetuned_reader.max_new_tokens must be smaller than "
                "training.max_seq_length"
            )
        if self.model.load_in_4bit and self.lora.adapter_type != "qlora":
            raise ValueError(
                "load_in_4bit requires finetuned_reader.lora.adapter_type=qlora"
            )
        if (
            self.overlap_policy == "exclude_and_record"
            and self.overlap_remediation_id is None
        ):
            raise ValueError(
                "exclude_and_record requires finetuned_reader.overlap_remediation_id"
            )
        if self.overlap_policy == "fail" and self.overlap_remediation_id is not None:
            raise ValueError(
                "overlap_remediation_id is only valid with overlap_policy="
                "exclude_and_record"
            )
        return self


class ProjectConfig(ConfigSection):
    """Complete profile with validation, redaction, path resolution, and identity."""

    project: ProjectSection
    data: DataSection
    chunking: ChunkingSection
    retrieval: RetrievalSection
    reranker: RerankerSection
    evidence: EvidenceSection
    generation: GenerationSection
    prompts: PromptsSection
    evaluation: EvaluationSection
    runtime: RuntimeSection
    submission: SubmissionSection
    reader: ReaderSection | None = None
    finetuned_reader: FineTunedReaderSection | None = None

    @model_validator(mode="after")
    def validate_profile_invariants(self) -> Self:
        if self.retrieval.rough_top_n < self.evidence.evidence_top_k:
            raise ValueError("retrieval.rough_top_n must be >= evidence.evidence_top_k")

        split_usage = get_split_usage(self.data.split)
        expected_policy = split_usage.policy
        if self.data.split_policy != expected_policy:
            raise ValueError(
                f"data.split_policy must be {expected_policy!r} for split "
                f"{self.data.split!r}"
            )
        if self.data.split == "private" and self.evaluation.reference_access != "none":
            raise ValueError("Private split must disallow evaluator reference access")
        try:
            validate_reference_access(self.data.split, self.evaluation.reference_access)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        uses_reranker = self.retrieval.strategy == "bm25_rerank"
        if uses_reranker != self.reranker.enabled:
            raise ValueError("reranker.enabled must match retrieval.strategy")
        if self.reranker.required and not self.reranker.enabled:
            raise ValueError("reranker.required requires reranker.enabled")
        if self.reranker.enabled and self.reranker.provider == "none":
            raise ValueError("An enabled reranker requires a non-none provider")
        if not self.reranker.enabled and self.reranker.provider != "none":
            raise ValueError("A disabled reranker must use provider 'none'")

        reader_profiles = {"finetuned-reader", "tuned-bm25-reader"}
        is_reader_profile = self.project.profile in reader_profiles
        if is_reader_profile != (self.reader is not None and self.reader.enabled):
            raise ValueError(
                "Reader profiles require reader.enabled=true and non-reader "
                "profiles must omit the reader section"
            )
        if self.reader is not None and not is_reader_profile:
            raise ValueError("Non-reader profiles must omit the reader section")
        if self.reader is not None:
            expected_mode: ReaderMode = (
                "original_context"
                if self.project.profile == "finetuned-reader"
                else "train_context_bm25"
            )
            if self.reader.mode != expected_mode:
                raise ValueError(
                    f"reader.mode must be {expected_mode!r} for profile "
                    f"{self.project.profile!r}"
                )
            if self.retrieval.strategy != "none" or self.reranker.enabled:
                raise ValueError(
                    "Reader profiles must not reuse the Legal-RAG retrieval stack"
                )

        is_generative_profile = self.project.profile == "finetuned_reader"
        if is_generative_profile != (
            self.finetuned_reader is not None and self.finetuned_reader.enabled
        ):
            raise ValueError(
                "Generative finetuned_reader requires "
                "finetuned_reader.enabled=true and non-generative profiles must "
                "omit that section"
            )
        if self.finetuned_reader is not None and not is_generative_profile:
            raise ValueError("Non-generative profiles must omit finetuned_reader")
        if is_generative_profile:
            if self.reader is not None:
                raise ValueError(
                    "Generative finetuned_reader must not use reader settings"
                )
            if self.retrieval.strategy != "bm25_rerank" or not self.reranker.enabled:
                raise ValueError(
                    "Generative finetuned_reader must use the frozen BM25+reranker "
                    "retrieval contract"
                )
        return self

    @property
    def project_name(self) -> str:
        """Compatibility accessor for the B1 CLI skeleton."""

        return self.project.name

    @property
    def python_version(self) -> str:
        """Compatibility accessor for the B1 CLI skeleton."""

        return self.project.python_version

    @property
    def data_dir(self) -> Path:
        """Compatibility accessor for the B1 CLI skeleton."""

        return self.data.data_dir

    @property
    def mode(self) -> str:
        """Compatibility accessor exposing the selected profile name."""

        return self.project.profile

    @property
    def split_usage(self) -> SplitUsage:
        """Return the immutable registry entry for the configured split."""

        return get_split_usage(self.data.split)

    def redacted_dict(self) -> dict[str, Any]:
        """Serialize the validated profile while redacting secret-shaped keys."""

        payload = self.model_dump(mode="json")
        # Preserve hashes of pre-FTR profiles when the optional generative
        # section is absent; adding a new optional section must not drift the
        # frozen B2 control identity.
        if payload.get("finetuned_reader") is None:
            payload.pop("finetuned_reader")
        redacted = redact_secrets(payload)
        if not isinstance(redacted, dict):  # pragma: no cover - structural guard
            raise TypeError("Configuration serialization must produce a mapping")
        return redacted

    def config_hash(self) -> str:
        """Return a stable SHA256 identity for the redacted, normalized profile."""

        serialized = json.dumps(
            self.redacted_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def resolved_dict(self, repo_root: str | Path) -> dict[str, Any]:
        """Resolve all declared relative paths beneath ``repo_root`` for execution."""

        root = Path(repo_root).resolve()
        resolved = self.redacted_dict()
        path_fields = {
            "data": ("data_dir", "question_path", "selected_contexts_path"),
            "runtime": ("cache_dir", "outputs_dir", "artifacts_dir"),
            "reader": (
                "dataset_path",
                "split_manifest_path",
                "checkpoint_path",
                "checkpoint_manifest_path",
            ),
            "finetuned_reader": (
                "train_prompt_path",
                "inference_prompt_path",
                "checkpoint_path",
                "checkpoint_manifest_path",
            ),
        }
        for section_name, field_names in path_fields.items():
            section = resolved.get(section_name)
            if section is None:
                continue
            if not isinstance(section, dict):  # pragma: no cover - structural guard
                raise TypeError(
                    f"Configuration section {section_name!r} must be a mapping"
                )
            for field_name in field_names:
                candidate = (root / str(section[field_name])).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError as exc:
                    raise ValueError(
                        f"Resolved path escapes repository root: {section[field_name]}"
                    ) from exc
                section[field_name] = candidate.as_posix()
        generative = resolved.get("finetuned_reader")
        if isinstance(generative, dict):
            output = generative.get("output")
            if isinstance(output, dict):
                for field_name in ("dataset_root", "checkpoint_root"):
                    candidate = (root / str(output[field_name])).resolve()
                    try:
                        candidate.relative_to(root)
                    except ValueError as exc:
                        raise ValueError(
                            "Resolved path escapes repository root: "
                            f"{output[field_name]}"
                        ) from exc
                    output[field_name] = candidate.as_posix()
        return resolved


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a YAML profile without creating or changing files."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    _reject_secret_keys(raw)
    return ProjectConfig.model_validate(raw)
