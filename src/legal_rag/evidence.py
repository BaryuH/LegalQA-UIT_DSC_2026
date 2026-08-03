"""Deterministic, provenance-preserving evidence selection utilities."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .schemas import LegalChunk, LegalDocument, PackedEvidence, RetrievalHit

DropReason = Literal[
    "exact_normalized_duplicate",
    "same_source_article_high_overlap",
]


@dataclass(frozen=True, slots=True)
class DeduplicationDrop:
    """A dropped hit and the higher-ranked hit that made it redundant."""

    chunk_id: str
    kept_chunk_id: str
    reason: DropReason
    overlap_ratio: float | None = None

    def as_dict(self) -> dict[str, str | float | None]:
        """Return content-free, JSON-compatible provenance for this decision."""

        return {
            "chunk_id": self.chunk_id,
            "kept_chunk_id": self.kept_chunk_id,
            "reason": self.reason,
            "overlap_ratio": self.overlap_ratio,
        }


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """Kept ranked hits plus explicit, deterministic drop decisions."""

    kept_hits: tuple[RetrievalHit, ...]
    dropped: tuple[DeduplicationDrop, ...]

    @property
    def dropped_ids(self) -> tuple[str, ...]:
        """Return dropped IDs in deterministic decision order."""

        return tuple(item.chunk_id for item in self.dropped)

    @property
    def drop_reasons(self) -> dict[str, str]:
        """Return a stable ID-to-reason mapping without source text."""

        return {item.chunk_id: item.reason for item in self.dropped}

    def as_dict(self) -> dict[str, object]:
        """Serialize the result without embedding chunk text."""

        return {
            "kept_ids": [hit.chunk_id for hit in self.kept_hits],
            "dropped_ids": list(self.dropped_ids),
            "dropped": [item.as_dict() for item in self.dropped],
        }


class DeduplicationError(ValueError):
    """Raised when hits cannot be resolved to provenance-preserving chunks."""


class EvidencePackingError(ValueError):
    """Raised when no evidence can satisfy the requested context budget."""


_WHITESPACE = re.compile(r"\s+")


def normalize_chunk_text(text: str) -> str:
    """Create an exact-dedup key while preserving legal token content."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(normalize_chunk_text(text).split())


def _longest_contiguous_overlap(first: tuple[str, ...], second: tuple[str, ...]) -> int:
    """Find the longest contiguous token window shared by two chunks."""

    if not first or not second:
        return 0
    previous = [0] * (len(second) + 1)
    longest = 0
    for first_token in first:
        current = [0]
        for second_index, second_token in enumerate(second, start=1):
            if first_token == second_token:
                value = previous[second_index - 1] + 1
                current.append(value)
                longest = max(longest, value)
            else:
                current.append(0)
        previous = current
    return longest


def _source_article_key(chunk: LegalChunk) -> tuple[str, str, str, str]:
    """Identify a conservative same-source/article comparison group."""

    return (
        chunk.document_id,
        chunk.source_path,
        chunk.source_member or "",
        normalize_chunk_text(chunk.section_label or ""),
    )


def _overlap_ratio(first: LegalChunk, second: LegalChunk) -> float:
    first_tokens = _tokens(first.retrieval_text)
    second_tokens = _tokens(second.retrieval_text)
    shorter_length = min(len(first_tokens), len(second_tokens))
    if shorter_length == 0:
        return 0.0
    return _longest_contiguous_overlap(first_tokens, second_tokens) / shorter_length


def deduplicate_retrieved_chunks(
    hits: Sequence[RetrievalHit],
    chunks: Mapping[str, LegalChunk],
    *,
    overlap_threshold: float = 0.8,
) -> DeduplicationResult:
    """Remove exact and conservative high-overlap duplicate chunks.

    Hits are considered in ascending ``rank`` and canonical ``chunk_id`` order, so
    lower (better) ranks always win. Exact normalized duplicates are removed globally.
    High-overlap removal requires the same document, archive member, and article label;
    when no article label exists, the document/source identity is the comparison group.
    Input chunks and hits are never mutated or text-merged.
    """

    if not 0.0 < overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be > 0 and <= 1")

    ordered_hits = sorted(hits, key=lambda hit: (hit.rank, hit.chunk_id))
    resolved: dict[str, LegalChunk] = {}
    for hit in ordered_hits:
        chunk = chunks.get(hit.chunk_id)
        if chunk is None:
            raise DeduplicationError(
                f"Retrieval hit {hit.chunk_id!r} has no corresponding legal chunk"
            )
        resolved[hit.chunk_id] = chunk

    kept_hits: list[RetrievalHit] = []
    kept_chunks: list[LegalChunk] = []
    exact_keys: dict[str, str] = {}
    dropped: list[DeduplicationDrop] = []

    for hit in ordered_hits:
        chunk = resolved[hit.chunk_id]
        exact_key = normalize_chunk_text(chunk.retrieval_text)
        duplicate_of = exact_keys.get(exact_key)
        if duplicate_of is not None:
            dropped.append(
                DeduplicationDrop(
                    chunk_id=hit.chunk_id,
                    kept_chunk_id=duplicate_of,
                    reason="exact_normalized_duplicate",
                )
            )
            continue

        overlap_duplicate: tuple[str, float] | None = None
        source_key = _source_article_key(chunk)
        for kept_hit, kept_chunk in zip(kept_hits, kept_chunks, strict=True):
            if _source_article_key(kept_chunk) != source_key:
                continue
            ratio = _overlap_ratio(chunk, kept_chunk)
            if ratio >= overlap_threshold:
                overlap_duplicate = (kept_hit.chunk_id, ratio)
                break

        if overlap_duplicate is not None:
            kept_chunk_id, ratio = overlap_duplicate
            dropped.append(
                DeduplicationDrop(
                    chunk_id=hit.chunk_id,
                    kept_chunk_id=kept_chunk_id,
                    reason="same_source_article_high_overlap",
                    overlap_ratio=ratio,
                )
            )
            continue

        exact_keys[exact_key] = hit.chunk_id
        kept_hits.append(hit)
        kept_chunks.append(chunk)

    return DeduplicationResult(kept_hits=tuple(kept_hits), dropped=tuple(dropped))


# Short alias for callers that already use retrieval-hit terminology.
deduplicate_hits = deduplicate_retrieved_chunks
deduplicate_chunks = deduplicate_retrieved_chunks


_TRUNCATION_MARKER = "\n… [ĐÃ CẮT PHẦN ĐUÔI]"


def _header_value(value: str) -> str:
    """Keep metadata on one stable line without changing the source chunk."""

    return value.replace("\r", " ").replace("\n", " ").strip()


def _document_metadata(
    chunk: LegalChunk, documents: Mapping[str, LegalDocument] | None
) -> tuple[str, str]:
    document = documents.get(chunk.document_id) if documents is not None else None
    if document is not None:
        name = document.name
        source = document.link or document.source_path
    else:
        name = chunk.source_member or Path(chunk.source_path).name
        source = chunk.source_path
    if chunk.source_member and document is None:
        source = f"{source}::{chunk.source_member}"
    return _header_value(name), _header_value(source)


def _header(
    position: int,
    chunk: LegalChunk,
    documents: Mapping[str, LegalDocument] | None,
) -> str:
    document_name, source = _document_metadata(chunk, documents)
    section = _header_value(chunk.section_label or "Không xác định")
    return "\n".join(
        (
            f"[TRÍCH ĐOẠN {position}]",
            f"Văn bản: {document_name}",
            f"Mã tài liệu: {_header_value(chunk.document_id)}",
            f"Điều/Khoản: {section}",
            f"Nguồn: {source}",
            "Nội dung:",
        )
    )


def _heading_and_tail(text: str, heading_hint: str | None = None) -> tuple[str, str]:
    stripped = text.lstrip("\r\n \t")
    lines = stripped.splitlines(keepends=True)
    if not lines:
        return "", ""
    heading = lines[0].rstrip("\r\n")
    tail = "".join(lines[1:])
    if not tail and heading_hint:
        hint = heading_hint.strip()
        if hint:
            normalized_heading = heading.casefold()
            normalized_hint = hint.casefold()
            if normalized_heading.startswith(normalized_hint):
                heading, tail = hint, heading[len(hint) :]
            elif heading != hint:
                heading, tail = hint, heading
    return heading, tail


def _truncate_block(
    header: str,
    raw_text: str,
    budget: int,
    heading_hint: str | None = None,
) -> str | None:
    """Render a budget-fitting block while retaining its first heading line."""

    prefix = f"{header}\n"
    available_body = budget - len(prefix) - len(_TRUNCATION_MARKER)
    heading, tail = _heading_and_tail(raw_text, heading_hint)
    if not heading or len(heading) > available_body:
        return None
    body = heading
    remaining = available_body - len(heading)
    if tail and remaining > 0:
        body += tail[:remaining]
    return f"{prefix}{body}{_TRUNCATION_MARKER}"


def pack_evidence(
    hits: Sequence[RetrievalHit],
    chunks: Mapping[str, LegalChunk],
    *,
    max_total_chars: int,
    max_chunks_per_document: int,
    documents: Mapping[str, LegalDocument] | None = None,
) -> PackedEvidence:
    """Pack ranked chunks into a stable, header-inclusive character budget.

    Candidates are processed in rank/chunk-ID order. The first chunks from each
    document win the per-document cap. A chunk that cannot fit whole is included as a
    heading-preserving tail truncation when possible; otherwise it is dropped. Scores
    are retained in ``included_hits`` but never rendered in the default format.
    """

    if max_total_chars <= 0:
        raise ValueError("max_total_chars must be greater than zero")
    if max_chunks_per_document <= 0:
        raise ValueError("max_chunks_per_document must be greater than zero")

    ordered_hits = sorted(hits, key=lambda hit: (hit.rank, hit.chunk_id))
    resolved: dict[str, LegalChunk] = {}
    for hit in ordered_hits:
        chunk = chunks.get(hit.chunk_id)
        if chunk is None:
            raise EvidencePackingError(
                f"Retrieval hit {hit.chunk_id!r} has no corresponding legal chunk"
            )
        resolved[hit.chunk_id] = chunk

    blocks: list[str] = []
    included_hits: list[RetrievalHit] = []
    included_ids: list[str] = []
    dropped_ids: list[str] = []
    truncated_ids: list[str] = []
    dropped_reasons: dict[str, str] = {}
    document_counts: dict[str, int] = {}

    for hit in ordered_hits:
        chunk = resolved[hit.chunk_id]
        document_count = document_counts.get(chunk.document_id, 0)
        if document_count >= max_chunks_per_document:
            dropped_ids.append(hit.chunk_id)
            dropped_reasons[hit.chunk_id] = "max_chunks_per_document"
            continue

        position = len(blocks) + 1
        header = _header(position, chunk, documents)
        full_block = f"{header}\n{chunk.raw_text}"
        separator_chars = 2 if blocks else 0
        remaining = (
            max_total_chars - sum(len(block) for block in blocks) - separator_chars
        )
        if remaining <= 0:
            dropped_ids.append(hit.chunk_id)
            dropped_reasons[hit.chunk_id] = "max_total_chars"
            continue

        if len(full_block) <= remaining:
            block = full_block
        else:
            truncated_block = _truncate_block(
                header,
                chunk.raw_text,
                remaining,
                chunk.section_label,
            )
            if truncated_block is None:
                dropped_ids.append(hit.chunk_id)
                dropped_reasons[hit.chunk_id] = "max_total_chars"
                continue
            block = truncated_block
            truncated_ids.append(hit.chunk_id)

        blocks.append(block)
        included_hits.append(hit)
        included_ids.append(hit.chunk_id)
        document_counts[chunk.document_id] = document_count + 1

    if not blocks:
        raise EvidencePackingError(
            "No evidence chunk fits the configured budget and heading requirements"
        )

    rendered_text = "\n\n".join(blocks)
    metadata: dict[str, str | int | bool] = {
        "max_total_chars": max_total_chars,
        "max_chunks_per_document": max_chunks_per_document,
        "candidate_count": len(ordered_hits),
        "included_count": len(included_hits),
        "dropped_count": len(dropped_ids),
        "truncated_count": len(truncated_ids),
        "scores_rendered": False,
    }
    return PackedEvidence(
        included_ids=tuple(included_ids),
        dropped_ids=tuple(dropped_ids),
        truncated_ids=tuple(truncated_ids),
        included_hits=tuple(included_hits),
        rendered_text=rendered_text,
        dropped_reasons=dropped_reasons,
        metadata=metadata,
    )


pack_retrieved_evidence = pack_evidence
