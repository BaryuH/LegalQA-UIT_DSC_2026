"""Hierarchy audit and retrieval-unit builders for SEDAR Retrieval v3."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .schema import (
    CanonicalNode,
    CanonicalPassage,
    CorpusAuditReport,
    RetrievalLevel,
)


def audit_canonical_nodes(
    nodes: Sequence[CanonicalNode],
    *,
    source_texts: dict[str, str] | None = None,
) -> CorpusAuditReport:
    """Audit unique IDs, orphans, and optional text preservation."""

    ids = [node.node_id for node in nodes]
    unique = len(set(ids))
    unique_rate = 1.0 if not ids else unique / len(ids)
    id_set = set(ids)
    orphans = 0
    for node in nodes:
        if node.parent_id is None:
            continue
        if node.parent_id not in id_set:
            orphans += 1

    empty_units = sum(
        1
        for node in nodes
        if node.level in {"article", "clause"} and not node.raw_text.strip()
    )
    warning_count = sum(len(node.warnings) for node in nodes)
    unparsed_chars = sum(
        len(node.raw_text)
        for node in nodes
        if node.parse_status in {"retained", "unparsed"}
    )

    preservation = 1.0
    if source_texts:
        covered = 0
        total = 0
        by_doc: dict[str, list[CanonicalNode]] = {}
        for node in nodes:
            by_doc.setdefault(node.document_id, []).append(node)
        for doc_id, text in source_texts.items():
            total += len(text)
            # Prefer article+retained coverage approximation via concatenation of
            # non-overlapping top-level content nodes is complex; use exact
            # character membership of leaf raw texts where possible.
            joined = "".join(
                node.raw_text
                for node in by_doc.get(doc_id, [])
                if node.level in {"article", "retained", "document"}
                and node.node_id.endswith("::doc") is False
            )
            # Fallback: if article nodes cover full article spans, compare lengths.
            article_len = sum(
                len(node.raw_text)
                for node in by_doc.get(doc_id, [])
                if node.level == "article"
            )
            retained_len = sum(
                len(node.raw_text)
                for node in by_doc.get(doc_id, [])
                if node.parse_status == "retained"
            )
            covered += min(len(text), article_len + retained_len)
            _ = joined
        preservation = 1.0 if total == 0 else covered / total

    documents = {node.document_id for node in nodes}
    return CorpusAuditReport(
        document_count=len(documents),
        node_count=len(nodes),
        passage_count=0,
        unique_id_rate=unique_rate,
        orphan_node_count=orphans,
        duplicate_passage_id_count=max(0, len(ids) - unique),
        text_preservation_rate=preservation,
        unparsed_char_count=unparsed_chars,
        empty_article_or_clause_count=empty_units,
        warning_count=warning_count,
    )


def build_reader_text(node: CanonicalNode) -> str:
    """Authoritative reader evidence without synthetic retrieval wrappers."""

    if node.level == "article":
        heading = f"Điều {node.article_number}"
        if node.article_title:
            heading = f"{heading}. {node.article_title}"
        body = node.raw_text
        # Avoid duplicating heading if already present.
        if body.lstrip().lower().startswith("điều"):
            return body.strip()
        return f"{heading}\n\n{body.strip()}"
    if node.level == "clause":
        return node.raw_text.strip()
    return node.raw_text.strip()


def build_retrieval_text(node: CanonicalNode) -> str:
    """Hierarchy-prefixed retrieval representation (TASK 04 / R1)."""

    parts: list[str] = []
    if node.document_name:
        parts.append(f"[DOCUMENT] {node.document_name}")
    if node.chapter_title or node.chapter_id:
        title = node.chapter_title or node.chapter_id
        parts.append(f"[CHAPTER] {title}")
    if node.section_title or node.section_id:
        title = node.section_title or node.section_id
        parts.append(f"[SECTION] {title}")
    if node.article_number:
        article = f"Điều {node.article_number}"
        if node.article_title:
            article = f"{article}. {node.article_title}"
        parts.append(f"[ARTICLE] {article}")
    if node.clause_number:
        parts.append(f"[CLAUSE] Khoản {node.clause_number}")
    if node.point_label:
        parts.append(f"[POINT] Điểm {node.point_label}")
    parts.append("")
    parts.append(node.raw_text.strip())
    return "\n".join(parts)


def nodes_to_passages(
    nodes: Iterable[CanonicalNode],
    *,
    levels: Sequence[RetrievalLevel] = ("article", "clause"),
) -> tuple[CanonicalPassage, ...]:
    """Build retrieval passages for selected hierarchy levels."""

    allowed = set(levels)
    passages: list[CanonicalPassage] = []
    for node in nodes:
        if node.level not in allowed:
            continue
        if not node.raw_text.strip():
            continue
        reader = build_reader_text(node)
        retrieval = build_retrieval_text(node)
        passages.append(
            CanonicalPassage(
                passage_id=node.node_id,
                document_id=node.document_id,
                article_id=node.article_id,
                clause_id=node.clause_id,
                point_id=node.point_id,
                retrieval_level=node.level,
                document_name=node.document_name,
                article_number=node.article_number,
                article_title=node.article_title,
                clause_number=node.clause_number,
                point_label=node.point_label,
                status=node.status,
                raw_text=node.raw_text,
                reader_text=reader,
                retrieval_text=retrieval,
                source=node.source,
                parent_id=node.parent_id,
                parse_status=node.parse_status,
            )
        )
    return tuple(passages)
