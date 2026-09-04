"""Deterministic Vietnamese legal citation parser (TASK 14).

Q1 fix (2026-09-02). The original parser only recognised

* ``[điểm X] [khoản Y] điều Z`` — and only in that exact order, always
  requiring the word ``điều``; and
* a document number that was **preceded** by one of four document types
  (``nghị định``, ``thông tư``, ``quyết định``, ``luật``/``bộ luật``).

That left four real citation shapes unparsed, and every one of them costs three
things at once: a silver label (evaluation), a training label (LTR), and the
citation feature block at inference.

===========================================  ==========  ============
shape                                        before      now
===========================================  ==========  ============
``153/2020/NĐ-CP`` standing alone            missed      ``document_number``
``Nghị quyết``/``Pháp lệnh``/``Chỉ thị``/…   missed      ``document_number``
``Luật Doanh nghiệp 2020`` (name, no number) missed      ``document_name``
``khoản 2`` / ``điểm a`` standing alone      missed      ``clause_only`` / ``point_only``
===========================================  ==========  ============

Overlapping matches are resolved by containment: a mention wholly inside a
longer mention is dropped, so ``Nghị định 153/2020/NĐ-CP`` yields one mention
rather than a typed one plus a bare-number one, and ``khoản 2 Điều 15`` yields
the article mention rather than an extra clause-only mention. Without this,
``analyzer.classify_complexity`` would over-count citations and misclassify
simple queries as ``multi_hop``.

The parser stays purely lexical. It never resolves a mention to a document in
the corpus: that is ``eval/silver_labels.py``'s job, which is document-scoped
and fails closed. Nothing here may be used to guess that a bare ``khoản 2``
belongs to the article mentioned earlier in the sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ResolutionStatus = Literal["resolved", "ambiguous", "unresolved", "none"]
CitationKind = Literal[
    "article",
    "document_number",
    "document_name",
    "clause_only",
    "point_only",
]


@dataclass(frozen=True, slots=True)
class CitationMention:
    raw: str
    start: int
    end: int
    document_name: str | None = None
    document_number: str | None = None
    year: str | None = None
    article: str | None = None
    clause: str | None = None
    point: str | None = None
    resolution_status: ResolutionStatus = "unresolved"
    canonical_target_id: str | None = None
    kind: CitationKind = "article"


# ── Building blocks ────────────────────────────────────────────────────────

# Document types, longest-first inside a shared prefix so that alternation
# never stops at the shorter form: "bộ luật" before "luật", "thông tư liên
# tịch" before "thông tư".
_DOC_TYPE = (
    r"(?:hiến\s*pháp"
    r"|bộ\s*luật"
    r"|luật"
    r"|nghị\s*định"
    r"|nghị\s*quyết"
    r"|thông\s*tư\s*liên\s*tịch"
    r"|thông\s*tư"
    r"|quyết\s*định"
    r"|chỉ\s*thị"
    r"|pháp\s*lệnh"
    r"|công\s*văn"
    r")"
)

# Issuing-body suffix of a document number. Must accept mixed case and digits:
# NĐ-CP, TT-BTC, QĐ-TTg, QH13, QH14, TTLT-BYT-BTC.
_ISSUER = r"[A-ZĐ][A-Za-zĐđ0-9]*(?:-[A-ZĐ][A-Za-zĐđ0-9]*)*"

# A full document number: 153/2020/NĐ-CP, 91/2015/QH13.
_FULL_NUMBER = rf"\d{{1,4}}/\d{{4}}/{_ISSUER}"

# Any Unicode letter run (no digits, no underscore).
_WORD = r"[^\W\d_]+"

# Vietnamese uppercase initial. Vietnamese law names capitalise only the first
# word after the type ("Luật Doanh nghiệp", "Bộ luật Tố tụng dân sự"), so the
# first name word must start uppercase and the rest may be lowercase. This is
# what keeps "Luật này có hiệu lực từ năm 2015" from parsing as a name.
_UPPER = (
    r"[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂƠƯ"
    r"ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]"
)
_DOC_NAME = rf"{_UPPER}[^\W\d_]*(?:\s+{_WORD}){{0,3}}"


# ── Patterns ───────────────────────────────────────────────────────────────

# 1. điểm / khoản / điều, in the canonical order.
_POINT_CLAUSE_ARTICLE = re.compile(
    r"(?P<raw>"
    r"(?:điểm\s+(?P<point>[A-Za-zĐđ])\s+)?"
    r"(?:khoản\s+(?P<clause>\d+)\s+)?"
    r"điều\s+(?P<article>\d+[A-Za-z]?)"
    r")",
    flags=re.IGNORECASE | re.UNICODE,
)

# 2. Document type followed by its number: "Nghị định số 13/2023/NĐ-CP".
_DOC_NUMBER = re.compile(
    rf"(?P<raw>{_DOC_TYPE}"
    rf"\s*(?:số\s*)?(?P<number>\d+(?:/\d+)?(?:/{_ISSUER})?)(?:/(?P<year>\d{{4}}))?)",
    flags=re.IGNORECASE | re.UNICODE,
)

# 3. A document number standing alone: "153/2020/NĐ-CP". Requires the full
#    number/year/issuer shape, so a date such as 16/09/2022 cannot match.
_BARE_DOC_NUMBER = re.compile(
    rf"(?P<raw>(?P<number>{_FULL_NUMBER}))",
    flags=re.UNICODE,
)

# 4. Document type plus a name and a year: "Luật Doanh nghiệp 2020".
_DOC_NAME_YEAR = re.compile(
    rf"(?P<raw>(?i:{_DOC_TYPE})\s+(?P<name>{_DOC_NAME})\s+(?P<year>19\d{{2}}|20\d{{2}}))",
    flags=re.UNICODE,
)

# 5/6. A clause or a point standing alone. No lookahead is needed: when the
#      canonical "khoản 2 Điều 15" form matches, its span contains these and
#      containment resolution drops them.
_CLAUSE_ONLY = re.compile(
    r"(?P<raw>khoản\s+(?P<clause>\d+))",
    flags=re.IGNORECASE | re.UNICODE,
)
_POINT_ONLY = re.compile(
    r"(?P<raw>điểm\s+(?P<point>[A-Za-zĐđ]))\b",
    flags=re.IGNORECASE | re.UNICODE,
)

_YEAR = re.compile(r"\b(?P<year>19\d{2}|20\d{2})\b")


def _drop_contained(
    mentions: list[CitationMention],
) -> list[CitationMention]:
    """Drop every mention that lies wholly inside a strictly longer mention."""

    spans = [(item.start, item.end) for item in mentions]
    kept: list[CitationMention] = []
    for index, item in enumerate(mentions):
        length = item.end - item.start
        contained = any(
            other_index != index
            and other_start <= item.start
            and item.end <= other_end
            and (other_end - other_start) > length
            for other_index, (other_start, other_end) in enumerate(spans)
        )
        if not contained:
            kept.append(item)
    return kept


def parse_citations(query: str) -> tuple[CitationMention, ...]:
    """Extract explicit legal citations without LLM assistance."""

    mentions: list[CitationMention] = []

    for match in _POINT_CLAUSE_ARTICLE.finditer(query):
        mentions.append(
            CitationMention(
                raw=match.group("raw"),
                start=match.start(),
                end=match.end(),
                article=match.group("article"),
                clause=match.group("clause"),
                point=match.group("point"),
                resolution_status="unresolved",
                kind="article",
            )
        )
    for match in _DOC_NUMBER.finditer(query):
        mentions.append(
            CitationMention(
                raw=match.group("raw"),
                start=match.start(),
                end=match.end(),
                document_number=match.group("number"),
                year=match.group("year"),
                resolution_status="unresolved",
                kind="document_number",
            )
        )
    for match in _BARE_DOC_NUMBER.finditer(query):
        mentions.append(
            CitationMention(
                raw=match.group("raw"),
                start=match.start(),
                end=match.end(),
                document_number=match.group("number"),
                resolution_status="unresolved",
                kind="document_number",
            )
        )
    for match in _DOC_NAME_YEAR.finditer(query):
        mentions.append(
            CitationMention(
                raw=match.group("raw"),
                start=match.start(),
                end=match.end(),
                document_name=match.group("name"),
                year=match.group("year"),
                resolution_status="unresolved",
                kind="document_name",
            )
        )
    for match in _CLAUSE_ONLY.finditer(query):
        mentions.append(
            CitationMention(
                raw=match.group("raw"),
                start=match.start(),
                end=match.end(),
                clause=match.group("clause"),
                resolution_status="unresolved",
                kind="clause_only",
            )
        )
    for match in _POINT_ONLY.finditer(query):
        mentions.append(
            CitationMention(
                raw=match.group("raw"),
                start=match.start(),
                end=match.end(),
                point=match.group("point"),
                resolution_status="unresolved",
                kind="point_only",
            )
        )

    # Dedup identical spans first, then drop spans nested inside longer ones.
    unique: dict[tuple[int, int, str], CitationMention] = {}
    for item in mentions:
        unique.setdefault((item.start, item.end, item.raw.casefold()), item)
    deduped = _drop_contained(list(unique.values()))
    return tuple(sorted(deduped, key=lambda m: (m.start, m.end, m.kind)))


def extract_years(query: str) -> tuple[str, ...]:
    return tuple(match.group("year") for match in _YEAR.finditer(query))


__all__ = [
    "CitationKind",
    "CitationMention",
    "ResolutionStatus",
    "extract_years",
    "parse_citations",
]
