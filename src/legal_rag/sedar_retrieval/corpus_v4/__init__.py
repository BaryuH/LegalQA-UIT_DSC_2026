"""SEDAR corpus v4: instrument-aware, article-primary retrieval units."""

from .audit import audit_corpus
from .citations import Citation, CitationIndex, extract_citations, fold
from .export import (
    V4_TO_V3_LEVEL,
    ExportCounters,
    article_id_for,
    build_article_id_map,
    export_units,
    unit_to_passage_row,
)
from .metadata import DocumentMeta, citation_aliases, extract_document_meta
from .normalize import normalize_source_text
from .schema import CorpusV4Audit, DocumentRecord, RetrievalUnit
from .segment import parse_articles, segment_blocks, split_instruments
from .units import BuildOptions, build_document

__all__ = [
    "V4_TO_V3_LEVEL",
    "BuildOptions",
    "Citation",
    "CitationIndex",
    "ExportCounters",
    "article_id_for",
    "build_article_id_map",
    "export_units",
    "extract_citations",
    "fold",
    "unit_to_passage_row",
    "CorpusV4Audit",
    "DocumentMeta",
    "DocumentRecord",
    "RetrievalUnit",
    "audit_corpus",
    "build_document",
    "citation_aliases",
    "extract_document_meta",
    "normalize_source_text",
    "parse_articles",
    "segment_blocks",
    "split_instruments",
]
