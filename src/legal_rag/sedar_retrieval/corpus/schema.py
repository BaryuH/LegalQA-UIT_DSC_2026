"""Canonical Vietnamese legal hierarchy schema (SEDAR Retrieval v3)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from legal_rag.schemas import CanonicalID, DomainModel, NonBlankText

ParseStatus = Literal["ok", "partial", "unparsed", "retained"]
RetrievalLevel = Literal["document", "chapter", "section", "article", "clause", "point"]
DocumentStatus = Literal[
    "effective",
    "expired",
    "repealed",
    "amended",
    "unknown",
]


class SourceProvenance(DomainModel):
    """Immutable pointer back to the competition corpus member."""

    source_path: NonBlankText
    source_member: NonBlankText | None = None
    document_id: CanonicalID
    content_hash: NonBlankText
    link: NonBlankText | None = None


class CanonicalNode(DomainModel):
    """One hierarchical legal node with stable IDs and nullable metadata."""

    node_id: CanonicalID
    parent_id: CanonicalID | None = None
    level: RetrievalLevel
    document_id: CanonicalID
    document_type: str | None = None
    document_number: str | None = None
    document_name: str | None = None
    issuer: str | None = None
    promulgation_date: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    status: DocumentStatus = "unknown"

    chapter_id: CanonicalID | None = None
    chapter_title: str | None = None
    section_id: CanonicalID | None = None
    section_title: str | None = None
    article_id: CanonicalID | None = None
    article_number: str | None = None
    article_title: str | None = None
    clause_id: CanonicalID | None = None
    clause_number: str | None = None
    point_id: CanonicalID | None = None
    point_label: str | None = None

    raw_text: NonBlankText
    parse_status: ParseStatus = "ok"
    source: SourceProvenance
    warnings: tuple[str, ...] = ()


class CanonicalPassage(DomainModel):
    """Retrieval unit derived from a canonical node."""

    passage_id: CanonicalID
    document_id: CanonicalID
    article_id: CanonicalID | None = None
    clause_id: CanonicalID | None = None
    point_id: CanonicalID | None = None
    retrieval_level: RetrievalLevel
    document_name: str | None = None
    article_number: str | None = None
    article_title: str | None = None
    clause_number: str | None = None
    point_label: str | None = None
    status: DocumentStatus = "unknown"
    raw_text: NonBlankText
    reader_text: NonBlankText
    retrieval_text: NonBlankText
    summary: str | None = None
    references: tuple[CanonicalID, ...] = ()
    source: SourceProvenance
    parent_id: CanonicalID | None = None
    parse_status: ParseStatus = "ok"


class CorpusAuditReport(DomainModel):
    """Parser / hierarchy audit counters for Exit Gate checks."""

    document_count: int = Field(ge=0)
    node_count: int = Field(ge=0)
    passage_count: int = Field(ge=0)
    unique_id_rate: float = Field(ge=0.0, le=1.0)
    orphan_node_count: int = Field(ge=0)
    duplicate_passage_id_count: int = Field(ge=0)
    text_preservation_rate: float = Field(ge=0.0, le=1.0)
    unparsed_char_count: int = Field(ge=0)
    empty_article_or_clause_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
