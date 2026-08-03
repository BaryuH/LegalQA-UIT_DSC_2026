"""Retrieval-only normalization and conservative legal tokenization."""

from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "retrieval-normalization-v1"


def normalize_retrieval_text(text: str) -> str:
    """Return an NFC and whitespace-normalized retrieval/deduplication view.

    The returned value is derived from ``text``.  Callers must keep the source
    passage or chunk text separately and use that raw value when rendering
    evidence.
    """

    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip()


# Alternatives are ordered from the most structured legal forms to the most
# general forms.  This keeps codes and dates intact without dropping any
# number, slash, hyphen, or punctuation that the patterns do not consume.
_TOKEN_PATTERN = re.compile(
    r"""
    \d+(?:/\d+)+/[^\W\d_]+(?:-[^\W\d_]+)+  # e.g. 153/2020/NĐ-CP
    |\d{1,2}[-/]\d{1,2}[-/]\d{2,4}          # dates such as 16/09/2022
    |\d+(?:/\d+)+                           # other slash-number forms
    |\d+[^\W\d_]+                            # article numbers such as 37a
    |\d+(?:[.,]\d+)*                         # integers and grouped amounts
    |[^\W_]+(?:[-'][^\W_]+)*                 # Unicode words and hyphenated words
    |[^\w\s]|_                               # punctuation, including / and -
    """,
    flags=re.UNICODE | re.VERBOSE,
)


def tokenize_legal_text(text: str) -> tuple[str, ...]:
    """Tokenize a normalized retrieval view while retaining legal signals.

    Vietnamese diacritics, ``đ``/``d``, legal codes, dates, numeric amounts,
    duration numbers, and punctuation remain observable in the token stream.
    """

    normalized = normalize_retrieval_text(text)
    return tuple(_TOKEN_PATTERN.findall(normalized))
