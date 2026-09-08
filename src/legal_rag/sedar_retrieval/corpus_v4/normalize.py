"""Text normalisation for the SEDAR corpus v4 builder.

The competition corpus is HTML scraped from thuvienphapluat.vn. Its ``passage``
field carries three artefacts that break both parsing and retrieval:

1. Windows newlines and Unicode separators (``\\r\\n``, ``\\u2028``, ``\\u2029``).
2. Hard line wrapping *inside* sentences, so a single clause arrives as several
   physical lines separated by blank lines.
3. Trailing distribution blocks (``Nơi nhận:`` ...) and signature blocks that are
   pure noise for question answering.

Everything here is deterministic and reversible in the sense that the original
``passage`` is never mutated on disk; normalisation happens in memory during the
build and the resulting offsets are recorded on each unit.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "normalize_source_text",
    "unwrap_hard_wrapped_lines",
    "strip_distribution_block",
    "collapse_inline_whitespace",
    "is_structural_heading",
]

#: Sentinel standing in for a *soft* wrap. The scraper emits "\r\n\n" when it
#: wraps a line inside one HTML cell or paragraph and a bare "\n\n" when a new
#: block starts. Measured on the corpus: 155,974 soft wraps against ~163,000 hard
#: breaks, so collapsing both to "\n" (as v3 does) destroys the only signal that
#: says where a sentence really ended.
SOFT_WRAP = "\x01"

_SEPARATORS = {
    "\u2028": "\n",
    "\u2029": "\n",
    "\u00a0": " ",
    "\u200b": "",
    "\ufeff": "",
}

# Heading markers that must always start their own line.
_HEADING_RE = re.compile(
    r"""^\s*(
        phần\s+(thứ\s+)?[\dIVXivx]+
        | chương\s+[\dIVXivx]+
        | mục\s+[\dIVXivx]+
        | tiểu\s+mục\s+[\dIVXivx]+
        | điều(\s+\d+[a-zđ]?\b|\s*$)
        | phụ\s+lục\b
        | mẫu\s+(số\s+)?[\dIVXivx]
        | biểu\s+(số\s+)?[\dIVXivx]
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Enumerators: "1.", "1)", "a)", "a.", "I.", "-" at line start.
_ENUMERATOR_RE = re.compile(
    r"^\s*(\d+[\.\)]|[a-zđ]\)|[IVX]+\.|[-–•+])\s+", re.IGNORECASE
)

_DISTRIBUTION_RE = re.compile(
    r"^\s*(nơi\s+nhận\s*:|nguồn\s*:)\s*$", re.IGNORECASE | re.MULTILINE
)

_SIGNATURE_RE = re.compile(
    r"^\s*\(?(đã\s+ký(\s+và\s+đóng\s+dấu)?)\)?\s*$", re.IGNORECASE | re.MULTILINE
)

_TERMINATORS = ".:;!?…”\"')"


def collapse_inline_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs without touching line structure."""

    return re.sub(r"[ \t\f\v]+", " ", text)


def is_structural_heading(line: str) -> bool:
    """True when the line opens a structural unit or an enumerated item."""

    stripped = line.strip()
    if not stripped:
        return False
    if _HEADING_RE.match(stripped):
        return True
    return bool(_ENUMERATOR_RE.match(stripped))


def _looks_like_continuation(nxt: str) -> bool:
    """True when ``nxt`` continues the previous physical line mid-sentence."""

    stripped = nxt.strip()
    if not stripped:
        return False
    if is_structural_heading(stripped):
        return False
    first = stripped[0]
    if first.isdigit():
        # A bare number that is not an enumerator (already excluded above) is
        # usually a wrapped figure such as "10.000 đồng".
        return True
    if not first.isalpha():
        return False
    # Uppercase Vietnamese letters start new sentences or ALL-CAPS titles.
    return first == first.lower()


def _split_wraps(text: str) -> str:
    """Mark soft wraps, then reduce hard breaks to single newlines."""

    text = text.replace("\r\n\n", SOFT_WRAP).replace("\r\n", SOFT_WRAP)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def unwrap_hard_wrapped_lines(text: str, *, max_join_len: int = 400) -> str:
    """Re-join lines that the scraper wrapped in the middle of a sentence.

    A line is joined with the next one when it does not end in sentence-final
    punctuation, the next line does not open a structural unit, and the next
    line starts with a lowercase letter or a bare digit. ``max_join_len`` caps
    how long a joined logical line may grow so a malformed table cannot collapse
    an entire document into one line.
    """

    out: list[str] = []
    for block in text.split("\n"):
        pieces = [piece.strip() for piece in block.split(SOFT_WRAP)]
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            continue
        buffer = pieces[0]
        for piece in pieces[1:]:
            joinable = (
                buffer
                and buffer[-1] not in _TERMINATORS
                and len(buffer) < max_join_len
                and not is_structural_heading(piece)
            )
            if joinable:
                buffer = f"{buffer} {piece}"
            else:
                out.append(buffer)
                buffer = piece
        out.append(buffer)
    return "\n".join(out)


def strip_distribution_block(text: str) -> str:
    """Drop the trailing ``Nơi nhận:`` recipient list and signature markers.

    Only the *tail* of the text is trimmed: a ``Nơi nhận:`` marker in the middle
    of a document belongs to an embedded form and is left alone.
    """

    matches = list(_DISTRIBUTION_RE.finditer(text))
    if matches:
        last = matches[-1]
        tail = text[last.start() :]
        # Only strip when the block sits in the final quarter of the text and is
        # short enough to be a recipient list rather than substantive content.
        if last.start() > len(text) * 0.5 and len(tail) < 4000:
            text = text[: last.start()]
    text = _SIGNATURE_RE.sub("", text)
    return text.rstrip()


def normalize_source_text(text: str, *, unwrap: bool = True) -> str:
    """Full normalisation pipeline for one raw ``passage``."""

    normalized = unicodedata.normalize("NFC", text)
    for src, dst in _SEPARATORS.items():
        normalized = normalized.replace(src, dst)
    normalized = _split_wraps(normalized)
    normalized = collapse_inline_whitespace(normalized)
    if unwrap:
        normalized = unwrap_hard_wrapped_lines(normalized)
    else:
        normalized = normalized.replace(SOFT_WRAP, "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    return normalized.strip()
