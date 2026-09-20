"""Provider-neutral, inference-safe LLM client contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypeAlias

MetadataValue: TypeAlias = str | int | float | bool | None

_SECRET_KEY_MARKERS = (
    "apikey",
    "authorization",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
)
_FORBIDDEN_METADATA_MARKERS = (
    "chainofthought",
    "reasoning",
    "thought",
)
_SECRET_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[_ -]?key|authorization|bearer|password|secret|token)\b"
    r"\s*[:=]\s*[^\s,;]+"
)
_BEARER_VALUE_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")


def _normalized_key(value: object) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _require_non_blank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value


def _canonical_case_id(value: object) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError("case_id must not be null or boolean")
    return _require_non_blank(str(value), "case_id")


def _redact_message(value: str) -> str:
    redacted = _SECRET_VALUE_PATTERN.sub(
        lambda match: f"{match.group(1)}=<redacted>", value
    )
    return _BEARER_VALUE_PATTERN.sub("Bearer <redacted>", redacted)


def _validate_metadata(
    metadata: Mapping[str, MetadataValue],
) -> dict[str, MetadataValue]:
    sanitized: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        key_text = _require_non_blank(key, "metadata key")
        normalized_key = _normalized_key(key_text)
        if any(marker in normalized_key for marker in _SECRET_KEY_MARKERS):
            raise ValueError(f"LLM metadata must not contain secret key: {key_text}")
        if any(marker in normalized_key for marker in _FORBIDDEN_METADATA_MARKERS):
            raise ValueError("LLM metadata must not contain chain-of-thought data")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise TypeError(f"LLM metadata value for {key_text!r} is not scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"LLM metadata value for {key_text!r} must be finite")
        sanitized[key_text] = value
    return sanitized


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Minimal successful provider response with operational metadata only."""

    text: str
    latency_ms: float
    retries: int
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_blank(self.text, "LLMResponse.text")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("LLMResponse.latency_ms must be finite and non-negative")
        if self.retries < 0:
            raise ValueError("LLMResponse.retries must be non-negative")
        normalized = _validate_metadata(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(normalized))

    @property
    def latency(self) -> float:
        """Compatibility alias; latency is measured in milliseconds."""

        return self.latency_ms

    def safe_metadata(self) -> dict[str, MetadataValue]:
        """Return the redaction-checked metadata for non-sensitive artifacts."""

        return dict(self.metadata)


@dataclass(frozen=True, slots=True)
class CaseError:
    """Structured failure for one inference case, without prompt or gold text."""

    case_id: str
    error_code: str
    message: str
    retries: int = 0
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _canonical_case_id(self.case_id))
        _require_non_blank(self.error_code, "CaseError.error_code")
        _require_non_blank(self.message, "CaseError.message")
        if self.retries < 0:
            raise ValueError("CaseError.retries must be non-negative")
        object.__setattr__(self, "message", _redact_message(self.message))

    def for_case(self, case_id: str, *, retries: int | None = None) -> CaseError:
        """Attach a fixture/provider error to the actual case being processed."""

        return CaseError(
            case_id=case_id,
            error_code=self.error_code,
            message=self.message,
            retries=self.retries if retries is None else retries,
            retryable=self.retryable,
        )

    def as_dict(self) -> dict[str, str | int | bool]:
        """Serialize only safe case-level diagnostics."""

        return {
            "case_id": self.case_id,
            "error_code": self.error_code,
            "message": self.message,
            "retries": self.retries,
            "retryable": self.retryable,
        }


class LLMClientError(RuntimeError):
    """Raised when one case cannot receive a valid LLM response."""

    def __init__(self, case_error: CaseError) -> None:
        self.case_error = case_error
        super().__init__(
            f"{case_error.error_code} for case {case_error.case_id!r}: "
            f"{case_error.message}"
        )


class LLMConfigurationError(ValueError):
    """Raised when no approved adapter exists for a configured provider."""


class LLMClient(Protocol):
    """Small protocol shared by offline and approved provider adapters."""

    provider: str
    model: str

    def generate(self, prompt: str, *, case_id: str) -> LLMResponse:
        """Generate one response or raise a structured case-level error."""
