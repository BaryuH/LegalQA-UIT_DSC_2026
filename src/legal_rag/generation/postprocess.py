"""Minimal, non-semantic cleanup for generated legal answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

_EXACT_PREFIXES: Final[tuple[str, ...]] = (
    "FINAL_ANSWER:",
    "CÂU TRẢ LỜI:",
)
_CODE_FENCE_OPEN: Final[re.Pattern[str]] = re.compile(r"^```(?:[A-Za-z0-9_-]+)?$")


@dataclass(frozen=True, slots=True)
class AnswerPostprocessResult:
    """Keep the provider output and its minimally cleaned answer together."""

    raw_answer: str
    cleaned_answer: str

    def __post_init__(self) -> None:
        if not isinstance(self.raw_answer, str) or not self.raw_answer.strip():
            raise ValueError("raw_answer must be a non-blank string")
        if not isinstance(self.cleaned_answer, str) or not self.cleaned_answer.strip():
            raise ValueError("cleaned_answer must be a non-blank string")


PostprocessResult = AnswerPostprocessResult


def postprocess_answer(raw_answer: str) -> AnswerPostprocessResult:
    """Apply only wrappers/whitespace cleanup and preserve legal content exactly."""

    if not isinstance(raw_answer, str):
        raise TypeError("raw_answer must be a string")
    if not raw_answer.strip():
        raise ValueError("raw_answer must be a non-blank string")

    cleaned = _normalize_line_endings(raw_answer).strip()
    for _ in range(2):
        before = cleaned
        cleaned = _remove_outer_code_fence(cleaned).strip()
        cleaned = _remove_exact_prefix(cleaned).strip()
        if cleaned == before:
            break
    if not cleaned:
        raise ValueError("cleaned_answer must be a non-blank string")
    return AnswerPostprocessResult(
        raw_answer=raw_answer,
        cleaned_answer=cleaned,
    )


def _normalize_line_endings(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _remove_outer_code_fence(value: str) -> str:
    lines = value.split("\n")
    if len(lines) < 3:
        return value
    if _CODE_FENCE_OPEN.fullmatch(lines[0].strip()) is None:
        return value
    if lines[-1].strip() != "```":
        return value
    return "\n".join(lines[1:-1])


def _remove_exact_prefix(value: str) -> str:
    for prefix in _EXACT_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix) :].lstrip()
    return value


__all__ = [
    "AnswerPostprocessResult",
    "PostprocessResult",
    "postprocess_answer",
]
