"""Verbatim-overlap helpers for grounding / answerability analysis.

Legal answers quote statute text rather than paraphrase, so n-gram (shingle)
overlap between an answer/gold and the evidence is a good proxy for "is this
content actually in the evidence". Kept dependency-free (same normalization as
``legal_rag.evaluation.answer_in_context``: NFC + lowercase + collapse space,
word tokens, 8-gram shingles).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

DEFAULT_SHINGLE_SIZE = 8
_WORD_RE = re.compile(r"[0-9\w]+", re.UNICODE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(normalize(text))


def shingles(text: str, size: int = DEFAULT_SHINGLE_SIZE) -> set[tuple[str, ...]]:
    toks = tokens(text)
    if len(toks) < size:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + size]) for i in range(len(toks) - size + 1)}


def shingle_coverage(part: str, whole: str, size: int = DEFAULT_SHINGLE_SIZE) -> float:
    """Fraction of ``part``'s shingles that also appear in ``whole`` (verbatim)."""

    part_shingles = shingles(part, size)
    if not part_shingles:
        return 0.0
    return len(part_shingles & shingles(whole, size)) / len(part_shingles)


def unigram_recall(part: str, whole: str) -> float:
    """Fraction of ``part``'s distinct word tokens present in ``whole``."""

    part_tokens = set(tokens(part))
    if not part_tokens:
        return 0.0
    return len(part_tokens & set(tokens(whole))) / len(part_tokens)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "DEFAULT_SHINGLE_SIZE",
    "mean",
    "normalize",
    "shingle_coverage",
    "shingles",
    "tokens",
    "unigram_recall",
]
