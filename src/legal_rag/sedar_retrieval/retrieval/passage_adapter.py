"""Adapt canonical passages to existing BM25 LegalChunk views."""

from __future__ import annotations

from hashlib import sha256
from typing import Literal

from legal_rag.schemas import LegalChunk
from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage

BodySource = Literal["raw_text", "reader_text"]


def passage_to_legal_chunk(
    passage: CanonicalPassage,
    *,
    body_source: BodySource = "raw_text",
) -> LegalChunk:
    """Map a canonical passage onto the BM25-compatible LegalChunk contract.

    ``body_source`` selects which passage field becomes ``LegalChunk.raw_text``,
    i.e. what the reader actually sees once ``pack_evidence`` renders the block.

    ``raw_text``
        the historical default and the only behaviour before this flag existed.
    ``reader_text``
        the field ``hierarchy.build_reader_text()`` documents as *"Authoritative
        reader evidence"*: for an article-level passage it prepends
        ``Điều N. <tiêu đề>`` when the body does not already start with it.

    The mismatch this flag exposes: ``reader_text`` was built for the reader and
    then never reached it — the inference path always used ``raw_text``, and
    ``reader_text`` was read only by the synthetic-query and hard-negative
    training code.

    Note that ``build_reader_text`` currently returns bare ``raw_text`` for
    clause-level passages, so switching to ``reader_text`` only changes
    article-level blocks until that builder is extended (a corpus/view rebuild,
    tracked as Phase 7 in ``docs/sedar_retrieval/RETRIEVAL_FIX_PLAN.md``).

    ``content_hash`` always hashes ``passage.raw_text`` so that chunk identity
    stays tied to the source text and does not shift with this flag.
    """

    body = passage.reader_text if body_source == "reader_text" else passage.raw_text

    section_parts: list[str] = []
    if passage.article_number:
        section_parts.append(f"Điều {passage.article_number}")
    if passage.clause_number:
        section_parts.append(f"Khoản {passage.clause_number}")
    if passage.point_label:
        section_parts.append(f"Điểm {passage.point_label}")
    section_label = (
        " / ".join(section_parts) if section_parts else passage.retrieval_level
    )

    return LegalChunk(
        chunk_id=passage.passage_id,
        document_id=passage.document_id,
        source_path=passage.source.source_path,
        source_member=passage.source.source_member,
        raw_text=body,
        retrieval_text=passage.retrieval_text,
        content_hash=sha256(passage.raw_text.encode("utf-8")).hexdigest(),
        chunker_version=f"sedar-retrieval-v3-{passage.retrieval_level}",
        section_label=section_label,
        start_offset=None,
        end_offset=None,
    )


def load_passages_jsonl(path: str) -> tuple[CanonicalPassage, ...]:
    import json

    from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines

    rows: list[CanonicalPassage] = []
    for line in iter_jsonl_lines(path):
        rows.append(CanonicalPassage.model_validate(json.loads(line)))
    return tuple(rows)
