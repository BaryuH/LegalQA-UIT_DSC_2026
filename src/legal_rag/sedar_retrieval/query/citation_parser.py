"""Deterministic Vietnamese legal citation parser (TASK 14)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ResolutionStatus = Literal["resolved", "ambiguous", "unresolved", "none"]


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


_POINT_CLAUSE_ARTICLE = re.compile(
    r"(?P<raw>"
    r"(?:điểm\s+(?P<point>[A-Za-zĐđ])\s+)?"
    r"(?:khoản\s+(?P<clause>\d+)\s+)?"
    r"điều\s+(?P<article>\d+[A-Za-z]?)"
    r")",
    flags=re.IGNORECASE | re.UNICODE,
)
_DOC_NUMBER = re.compile(
    r"(?P<raw>(?:nghị\s*định|thông\s*tư|quyết\s*định|luật|bộ\s*luật)"
    r"\s*(?:số\s*)?(?P<number>\d+(?:/\d+)?(?:/[A-ZĐ\-]+)?)(?:/(?P<year>\d{4}))?)",
    flags=re.IGNORECASE | re.UNICODE,
)
_YEAR = re.compile(r"\b(?P<year>19\d{2}|20\d{2})\b")


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
            )
        )
    # Dedup overlapping identical spans.
    unique: dict[tuple[int, int, str], CitationMention] = {}
    for item in mentions:
        unique[(item.start, item.end, item.raw.casefold())] = item
    return tuple(sorted(unique.values(), key=lambda m: (m.start, m.end)))


def extract_years(query: str) -> tuple[str, ...]:
    return tuple(match.group("year") for match in _YEAR.finditer(query))
