"""Corpus package for SEDAR Retrieval v3."""

from .hierarchy import (
    audit_canonical_nodes,
    build_reader_text,
    build_retrieval_text,
    nodes_to_passages,
)
from .parse_legal import parse_legal_document
from .reference_graph import outgoing_references, parse_reference_edges
from .schema import (
    CanonicalNode,
    CanonicalPassage,
    CorpusAuditReport,
    SourceProvenance,
)

__all__ = [
    "CanonicalNode",
    "CanonicalPassage",
    "CorpusAuditReport",
    "SourceProvenance",
    "audit_canonical_nodes",
    "build_reader_text",
    "build_retrieval_text",
    "nodes_to_passages",
    "outgoing_references",
    "parse_legal_document",
    "parse_reference_edges",
]
