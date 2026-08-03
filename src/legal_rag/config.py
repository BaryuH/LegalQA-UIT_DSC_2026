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


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("Text must not be blank")
    return value


NonBlankText = Annotated[str, AfterValidator(_require_non_blank)]
GenerationProvider = Literal["mock", "openai", "anthropic"]
RetrievalStrategy = Literal["none", "bm25", "bm25_rerank"]
RerankerProvider = Literal["none", "mock", "sentence_transformers"]
RerankerDevice = Literal["auto", "cpu", "cuda"]
SplitName = Literal["train", "warmup", "public", "private"]
SplitPolicy = Literal[
    "train_development",
    "warmup_evaluation",
    "public_inference",
    "private_final_inference",
]
SubmissionFormat = Literal["object_by_question_id"]
SubmissionOrder = Literal["dataset"]

_SPLIT_POLICIES: dict[SplitName, SplitPolicy] = {
    "train": "train_development",
    "warmup": "warmup_evaluation",
    "public": "public_inference",
    "private": "private_final_inference",
}
_SECRET_KEY_MARKERS = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
)
_REDACTED = "<redacted>"


def _normalized_key(value: object) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _is_secret_key(value: object) -> bool:
    return any(marker in _normalized_key(value) for marker in _SECRET_KEY_MARKERS)


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
    retries: int = Field(ge=0)
    base_url: NonBlankText | None = None
    api_key_env: NonBlankText | None = None
    timeout_seconds: float = Field(gt=0, default=30.0)
    backoff_seconds: float = Field(ge=0, default=0.5)

    @model_validator(mode="after")
    def validate_provider_settings(self) -> Self:
        if self.provider == "openai" and self.base_url is None:
            raise ValueError("OpenAI-compatible provider requires generation.base_url")
        return self


class PromptsSection(ConfigSection):
    direct_version: NonBlankText
    rag_version: NonBlankText


class EvaluationSection(ConfigSection):
    enabled: bool
    reference_access: Literal["none", "approved_evaluation"]
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

    @model_validator(mode="after")
    def validate_profile_invariants(self) -> Self:
        if self.retrieval.rough_top_n < self.evidence.evidence_top_k:
            raise ValueError("retrieval.rough_top_n must be >= evidence.evidence_top_k")

        expected_policy = _SPLIT_POLICIES[self.data.split]
        if self.data.split_policy != expected_policy:
            raise ValueError(
                f"data.split_policy must be {expected_policy!r} for split "
                f"{self.data.split!r}"
            )
        if self.data.split == "private" and self.evaluation.reference_access != "none":
            raise ValueError("Private split must disallow evaluator reference access")

        uses_reranker = self.retrieval.strategy == "bm25_rerank"
        if uses_reranker != self.reranker.enabled:
            raise ValueError("reranker.enabled must match retrieval.strategy")
        if self.reranker.required and not self.reranker.enabled:
            raise ValueError("reranker.required requires reranker.enabled")
        if self.reranker.enabled and self.reranker.provider == "none":
            raise ValueError("An enabled reranker requires a non-none provider")
        if not self.reranker.enabled and self.reranker.provider != "none":
            raise ValueError("A disabled reranker must use provider 'none'")
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

    def redacted_dict(self) -> dict[str, Any]:
        """Serialize the validated profile while redacting secret-shaped keys."""

        redacted = redact_secrets(self.model_dump(mode="json"))
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
        }
        for section_name, field_names in path_fields.items():
            section = resolved[section_name]
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
