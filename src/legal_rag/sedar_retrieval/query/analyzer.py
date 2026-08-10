"""Legal query analyzer scaffold (TASK 15) — deterministic layer only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from legal_rag.sedar_retrieval.query.citation_parser import (
    CitationMention,
    parse_citations,
)

Complexity = Literal["simple", "implicit", "multi_condition", "multi_hop"]


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    original_query: str
    citations: tuple[CitationMention, ...]
    complexity: Complexity
    actor: tuple[str, ...] = ()
    subject: tuple[str, ...] = ()
    action: tuple[str, ...] = ()
    condition: tuple[str, ...] = ()
    exception: tuple[str, ...] = ()
    legal_issue: tuple[str, ...] = ()


_MULTI_HINTS = ("và", "hoặc", "trừ trường hợp", "ngoại trừ", "đồng thời")
_HOP_HINTS = ("theo điều", "quy định tại", "theo khoản")


def classify_complexity(query: str, citations: tuple[CitationMention, ...]) -> Complexity:
    lowered = query.casefold()
    if any(hint in lowered for hint in _HOP_HINTS) and len(citations) >= 2:
        return "multi_hop"
    if any(hint in lowered for hint in ("trừ trường hợp", "ngoại trừ", "nếu")):
        return "multi_condition"
    if not citations and ("?" in query or len(query.split()) >= 12):
        return "implicit"
    return "simple"


def analyze_query_deterministic(query: str) -> QueryAnalysis:
    citations = parse_citations(query)
    return QueryAnalysis(
        original_query=query,
        citations=citations,
        complexity=classify_complexity(query, citations),
    )
