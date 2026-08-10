"""Deterministic legal reference graph builder (TASK 17)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from legal_rag.sedar_retrieval.corpus.schema import CanonicalNode

ResolutionStatus = Literal["resolved", "ambiguous", "unresolved"]


@dataclass(frozen=True, slots=True)
class ReferenceEdge:
    source_id: str
    target_id: str | None
    relation_type: str
    original_reference_text: str
    resolution_status: ResolutionStatus


_REF_RE = re.compile(
    r"(?P<raw>"
    r"(?:theo|quy định tại|trừ trường hợp quy định tại)\s+"
    r"(?:điểm\s+(?P<point>[A-Za-zĐđ])\s+)?"
    r"(?:khoản\s+(?P<clause>\d+)\s+)?"
    r"(?:điều\s+(?P<article>\d+[A-Za-z]?))"
    r")",
    flags=re.IGNORECASE | re.UNICODE,
)


def parse_reference_edges(
    nodes: tuple[CanonicalNode, ...],
) -> tuple[ReferenceEdge, ...]:
    """Parse same-document explicit references from authoritative raw text."""

    by_article: dict[tuple[str, str], str] = {}
    by_clause: dict[tuple[str, str, str], str] = {}
    for node in nodes:
        if node.level == "article" and node.article_number:
            by_article[(node.document_id, node.article_number)] = node.node_id
        if node.level == "clause" and node.article_number and node.clause_number:
            by_clause[
                (node.document_id, node.article_number, node.clause_number)
            ] = node.node_id

    edges: list[ReferenceEdge] = []
    for node in nodes:
        if node.level not in {"article", "clause", "point"}:
            continue
        for match in _REF_RE.finditer(node.raw_text):
            article = match.group("article")
            clause = match.group("clause")
            target: str | None = None
            status: ResolutionStatus = "unresolved"
            if clause and article:
                target = by_clause.get((node.document_id, article, clause))
                status = "resolved" if target else "unresolved"
            elif article:
                target = by_article.get((node.document_id, article))
                status = "resolved" if target else "unresolved"
            edges.append(
                ReferenceEdge(
                    source_id=node.node_id,
                    target_id=target,
                    relation_type="explicit_reference",
                    original_reference_text=match.group("raw"),
                    resolution_status=status,
                )
            )
    return tuple(edges)


def outgoing_references(
    edges: tuple[ReferenceEdge, ...], source_id: str
) -> tuple[ReferenceEdge, ...]:
    return tuple(edge for edge in edges if edge.source_id == source_id)
