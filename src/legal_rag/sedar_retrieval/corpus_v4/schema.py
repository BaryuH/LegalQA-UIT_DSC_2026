"""Typed schema for the SEDAR corpus v4 retrieval units.

v4 keeps the v3 hierarchy vocabulary but changes what a *retrieval unit* is.
v3 emitted one unit per ``Điều`` **and** one per ``Khoản`` into the same index,
which put 903,562 mixed-length units in competition with each other; 33.1% of
the clause units are under 120 characters. v4 emits exactly one indexed unit per
``Điều`` unless that article is long, in which case its children are merged
clause runs that always carry the parent heading and always resolve back to the
parent before ranking.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from legal_rag.schemas import CanonicalID, DomainModel, NonBlankText

UnitLevel = Literal["article", "article_part", "block", "preamble"]
UnitRole = Literal[
    "substantive",
    "promulgation",
    "effect",
    "enforcement",
    "annex_container",
    "form",
    "unknown",
]
InstrumentKind = Literal["main", "annex"]

__all__ = [
    "CorpusV4Audit",
    "DocumentRecord",
    "InstrumentKind",
    "RetrievalUnit",
    "UnitLevel",
    "UnitRole",
]


class DocumentRecord(DomainModel):
    """Normalised identity and provenance of one corpus document."""

    document_id: CanonicalID
    source_path: NonBlankText
    source_member: NonBlankText | None = None
    content_hash: NonBlankText
    link: NonBlankText | None = None
    raw_name: str | None = None
    doc_type: str | None = None
    doc_number: str | None = None
    title: str | None = None
    issuer: str | None = None
    issued_date: str | None = None
    year: str | None = None
    citation: NonBlankText
    citation_aliases: tuple[str, ...] = ()
    metadata_source: NonBlankText
    instrument_count: int = Field(default=1, ge=1)
    source_chars: int = Field(default=0, ge=0)
    covered_chars: int = Field(default=0, ge=0)
    unit_count: int = Field(default=0, ge=0)
    parse_profile: NonBlankText = "article"


class RetrievalUnit(DomainModel):
    """One indexable unit with its full ancestry rendered into text.

    ``retrieval_text`` is what goes into the BM25 and dense indexes: it carries a
    deterministic hierarchy header, which both supplies the missing context that
    a bare clause lacks and lengthens short units, countering the documented
    brevity bias of BM25 and dense encoders. ``reader_text`` is what the
    generator quotes and stays free of index-only decoration.
    """

    unit_id: CanonicalID
    document_id: CanonicalID
    parent_unit_id: CanonicalID | None = None
    level: UnitLevel
    role: UnitRole = "substantive"

    instrument_index: int = Field(default=0, ge=0)
    instrument_kind: InstrumentKind = "main"
    instrument_title: str | None = None

    part_label: str | None = None
    chapter_number: str | None = None
    chapter_title: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    subsection_title: str | None = None
    article_number: str | None = None
    article_title: str | None = None
    child_label: str | None = None
    child_index: int | None = Field(default=None, ge=0)
    child_count: int | None = Field(default=None, ge=0)

    citation: NonBlankText
    breadcrumb: NonBlankText
    reader_text: NonBlankText
    retrieval_text: NonBlankText

    char_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    indexed: bool = True
    packable: bool = True
    content_hash: NonBlankText
    source_path: NonBlankText
    source_member: NonBlankText | None = None
    link: str | None = None
    warnings: tuple[str, ...] = ()


class CorpusV4Audit(DomainModel):
    """Exit-gate counters for one v4 build."""

    document_count: int = Field(ge=0)
    instrument_count: int = Field(ge=0)
    unit_count: int = Field(ge=0)
    indexed_unit_count: int = Field(ge=0)
    article_unit_count: int = Field(ge=0)
    article_part_count: int = Field(ge=0)
    block_unit_count: int = Field(ge=0)
    preamble_unit_count: int = Field(ge=0)
    documents_without_units: int = Field(ge=0)
    duplicate_unit_id_count: int = Field(ge=0)
    duplicate_article_number_count: int = Field(ge=0)
    micro_unit_count: int = Field(ge=0)
    micro_unit_rate: float = Field(ge=0.0, le=1.0)
    text_coverage_rate: float = Field(ge=0.0, le=1.0)
    metadata_recovery_rate: float = Field(ge=0.0, le=1.0)
    median_unit_chars: int = Field(ge=0)
    p90_unit_chars: int = Field(ge=0)
    role_counts: dict[str, int] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
