"""Vietnamese word segmentation wrapper using underthesea.

Provides deterministic, cache-friendly tokenization for BM25 indexing
and query processing of Vietnamese legal text.
"""

from __future__ import annotations

from functools import lru_cache

from underthesea import word_tokenize


def segment(text: str) -> list[str]:
    """Segment Vietnamese text into word tokens.

    Uses underthesea word_tokenize which handles compound Vietnamese words
    and legal terminology. Returns lowercased tokens for BM25 matching.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = word_tokenize(text, format="list")
    # Lowercase for BM25 matching; preserve Vietnamese diacritics
    return [t.lower().replace(" ", "_") for t in tokens if t.strip()]


@lru_cache(maxsize=8192)
def segment_cached(text: str) -> tuple[str, ...]:
    """Cached version of segment() returning a tuple for hashability."""
    return tuple(segment(text))
