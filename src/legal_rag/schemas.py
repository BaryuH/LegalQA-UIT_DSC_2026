"""Typed, inference-safe domain records for the Vietnamese Legal RAG pipeline."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Annotated
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    StrictStr,
    model_validator,
)


def _require_non_blank_text(value: str) -> str:
    if not value.strip():
        raise ValueError("Text must not be blank")
    return value


def _normalize_identifier(value: object) -> str:
    """Represent identifiers as non-blank strings without rewriting their value."""

    if value is None or isinstance(value, bool):
        raise ValueError("Identifiers must not be null or boolean values")
    normalized = str(value)
    if not normalized.strip():
        raise ValueError("Identifier must not be blank")
    return normalized


NonBlankText = Annotated[str, AfterValidator(_require_non_blank_text)]
CanonicalID = Annotated[
    str,
    BeforeValidator(_normalize_identifier),
    AfterValidator(_require_non_blank_text),
]


class DomainModel(BaseModel):
    """Base model that rejects undeclared fields and non-finite float values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class InferenceQuestion(DomainModel):
    """Question-only view permitted to reach retrieval and generation."""

    id: CanonicalID
    question: NonBlankText
    split: NonBlankText | None = None


class LegalQuestion(DomainModel):
    """Question record that may contain gold only in approved non-inference scope."""

    id: CanonicalID
    question: NonBlankText
    answer: NonBlankText | None = None
    split: NonBlankText | None = None

    def inference_view(self) -> InferenceQuestion:
        """Return the question-only view, deliberately excluding any gold answer."""

        return InferenceQuestion(id=self.id, question=self.question, split=self.split)


class LegalDocument(DomainModel):
    """Selected legal context with immutable-source provenance."""

    id: CanonicalID
    name: NonBlankText
    passage: NonBlankText
    source_path: NonBlankText
    content_hash: NonBlankText
    link: NonBlankText | None = None
    source_member: NonBlankText | None = None


class LegalChunk(DomainModel):
    """Derived legal-text chunk with traceable document and source provenance."""

    chunk_id: CanonicalID
    document_id: CanonicalID
    source_path: NonBlankText
    raw_text: NonBlankText
    retrieval_text: NonBlankText
    content_hash: NonBlankText
    chunker_version: NonBlankText
    source_member: NonBlankText | None = None
    section_label: NonBlankText | None = None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        """Require offsets to be supplied together and in source order."""

        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("Chunk offsets must be provided together")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError("Chunk end_offset must be greater than start_offset")
        return self


class RetrievalHit(DomainModel):
    """One retrieved chunk, preserving provenance and separate retrieval scores."""

    chunk_id: CanonicalID
    document_id: CanonicalID
    source_path: NonBlankText
    source_member: NonBlankText | None = None
    section_label: NonBlankText | None = None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    rank: int = Field(gt=0)
    bm25_score: float
    rerank_score: float | None = None

    @model_validator(mode="after")
    def validate_retrieval_offsets(self) -> Self:
        """Require complete, ordered source ranges when retrieval has offsets."""

        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("Retrieval hit offsets must be provided together")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError(
                "Retrieval hit end_offset must be greater than start_offset"
            )
        return self


class PackedEvidence(DomainModel):
    """Bounded evidence with explicit inclusion, exclusion, and truncation records."""

    included_ids: tuple[CanonicalID, ...]
    dropped_ids: tuple[CanonicalID, ...] = ()
    truncated_ids: tuple[CanonicalID, ...] = ()
    included_hits: tuple[RetrievalHit, ...]
    rendered_text: NonBlankText
    dropped_reasons: dict[CanonicalID, NonBlankText] = Field(default_factory=dict)
    metadata: dict[str, str | int | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_included_hits(self) -> Self:
        """Keep explicit included IDs synchronized with their retrieval hits."""

        if not self.included_ids:
            raise ValueError("Packed evidence must include at least one chunk ID")
        if tuple(hit.chunk_id for hit in self.included_hits) != self.included_ids:
            raise ValueError("included_ids must match included_hits in order")
        if len(set(self.included_ids)) != len(self.included_ids):
            raise ValueError("included_ids must be unique")
        if set(self.included_ids) & set(self.dropped_ids):
            raise ValueError("A chunk cannot be both included and dropped")
        if not set(self.truncated_ids).issubset(self.included_ids):
            raise ValueError("truncated_ids must be present in included_ids")
        if set(self.dropped_ids) & set(self.truncated_ids):
            raise ValueError("A chunk cannot be both dropped and truncated")
        if not set(self.dropped_reasons).issubset(self.dropped_ids):
            raise ValueError("dropped_reasons keys must be present in dropped_ids")
        return self


class Prediction(DomainModel):
    """Inference output that intentionally has no gold/reference answer field."""

    id: CanonicalID
    answer: NonBlankText
    method: NonBlankText
    status: NonBlankText = "success"
    error_code: NonBlankText | None = None
    fallback_reason: NonBlankText | None = None


class SubmissionAnswer(DomainModel):
    """Typed official answer object; empty strings remain schema-valid."""

    answer: StrictStr


class SubmissionPayload(RootModel[dict[str, SubmissionAnswer]]):
    """Typed fixed official payload before JSON/ZIP boundary validation."""


class CaseMetric(DomainModel):
    """Evaluation-only metric values for one case, without answer text."""

    id: CanonicalID
    status: NonBlankText
    meteor: float | None = Field(default=None, ge=0.0, le=1.0)
    rouge_l: float | None = Field(default=None, ge=0.0, le=1.0)
    error_code: NonBlankText | None = None


class EvaluationSummary(DomainModel):
    """Aggregate local or official evaluation metadata and per-case scores."""

    run_id: CanonicalID
    split: NonBlankText
    evaluator_name: NonBlankText
    evaluator_version: NonBlankText
    metric_contract_version: NonBlankText
    primary_metric: NonBlankText = "meteor"
    secondary_metric: NonBlankText = "rouge_l"
    meteor: float | None = Field(default=None, ge=0.0, le=1.0)
    rouge_l: float | None = Field(default=None, ge=0.0, le=1.0)
    target_count: int = Field(ge=0)
    evaluated_count: int = Field(ge=0)
    scored_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    cases: tuple[CaseMetric, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Prevent aggregate counts from claiming impossible evaluation coverage."""

        if self.evaluated_count > self.target_count:
            raise ValueError("evaluated_count cannot exceed target_count")
        if self.scored_count > self.evaluated_count:
            raise ValueError("scored_count cannot exceed evaluated_count")
        if self.error_count > self.evaluated_count:
            raise ValueError("error_count cannot exceed evaluated_count")
        return self


_SECRET_KEY_MARKERS = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
)


def _validate_secret_free_mapping(value: object, path: str = "metadata") -> None:
    """Reject common secret-bearing mapping keys at every nesting level."""

    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            key_text = str(key)
            normalized_key = "".join(
                character for character in key_text.casefold() if character.isalnum()
            )
            if any(marker in normalized_key for marker in _SECRET_KEY_MARKERS):
                raise ValueError(
                    f"Run metadata must not contain secret key: {path}.{key_text}"
                )
            _validate_secret_free_mapping(nested_value, f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            _validate_secret_free_mapping(nested_value, f"{path}[{index}]")


class RunMetadata(DomainModel):
    """Reproducibility metadata for a run, with secret-bearing keys rejected."""

    run_id: CanonicalID
    method: NonBlankText
    split: NonBlankText
    config_fingerprint: NonBlankText
    data_manifest_hash: NonBlankText
    prompt_fingerprint: NonBlankText
    model_fingerprint: NonBlankText
    environment: dict[NonBlankText, NonBlankText] = Field(default_factory=dict)
    index_fingerprint: NonBlankText | None = None
    errors: tuple[NonBlankText, ...] = ()
    fallback_reason: NonBlankText | None = None

    @model_validator(mode="after")
    def validate_no_secrets(self) -> Self:
        """Ensure stored environment metadata does not use secret-bearing keys."""

        _validate_secret_free_mapping(self.environment, path="environment")
        return self


_INTERNAL_SUBMISSION_FIELDS = {
    "evidence",
    "gold",
    "gold_answer",
    "metadata",
    "method",
    "prediction",
    "reference",
    "reference_answer",
    "retrieval",
    "score",
    "scores",
    "status",
}


class SubmissionRecord(RootModel[dict[str, str]]):
    """Legacy generic submission-field boundary kept for internal compatibility.

    Official artifacts must use :mod:`legal_rag.submission`'s fixed ZIP contract.
    """

    @model_validator(mode="after")
    def validate_official_payload(self) -> Self:
        if not self.root:
            raise ValueError("Submission payload must contain official fields")
        for field_name, value in self.root.items():
            if not field_name.strip():
                raise ValueError("Submission field names must not be blank")
            if field_name.casefold() in _INTERNAL_SUBMISSION_FIELDS:
                raise ValueError(
                    f"Submission payload contains internal field: {field_name}"
                )
            _require_non_blank_text(value)
        return self

    @classmethod
    def from_official_fields(
        cls,
        fields: Mapping[str, object],
        *,
        required_fields: Collection[str],
    ) -> Self:
        """Validate a payload against a supplied, approved official schema."""

        expected = set(required_fields)
        provided = set(fields)
        if provided != expected:
            missing = sorted(expected - provided)
            extra = sorted(provided - expected)
            raise ValueError(
                f"Submission fields do not match approved schema; "
                f"missing={missing}, extra={extra}"
            )

        serialized: dict[str, str] = {}
        for field_name, value in fields.items():
            if field_name.casefold().endswith("id"):
                serialized[field_name] = _normalize_identifier(value)
            else:
                serialized[field_name] = _require_non_blank_text(str(value))
        return cls.model_validate(serialized)
