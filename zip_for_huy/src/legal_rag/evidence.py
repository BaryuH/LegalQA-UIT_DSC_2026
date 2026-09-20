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
    chunk: LegalChunk,
    documents: Mapping[str, LegalDocument] | None,
    document_names: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve the ``Văn bản`` / ``Nguồn`` header lines for one chunk.

    Name resolution order:

    1. a full ``LegalDocument`` (best: carries the link as well);
    2. ``document_names`` — a document_id -> display name map, for callers that
       hold the real document name but not a whole ``LegalDocument``;
    3. the zip member or file name.

    Step 2 exists because the SEDAR passage path has ``document_name`` on every
    ``CanonicalPassage`` yet had no way to hand it over, so the reader was shown
    ``Văn bản: <zip member>.txt`` instead of ``Nghị định 153/2020/NĐ-CP``. That
    matters beyond cosmetics: ``sedar_sft/verifier.py`` drops any legal
    identifier the answer cites that is absent from the evidence text, so a
    missing document name actively strips correct citations from answers.
    """

    document = documents.get(chunk.document_id) if documents is not None else None
    if document is not None:
        name = document.name
        source = document.link or document.source_path
    else:
        named = (
            document_names.get(chunk.document_id)
            if document_names is not None
            else None
        )
        name = named or chunk.source_member or Path(chunk.source_path).name
        source = chunk.source_path
    if chunk.source_member and document is None:
        source = f"{source}::{chunk.source_member}"
    return _header_value(name), _header_value(source)


def _header(
    position: int,
    chunk: LegalChunk,
    documents: Mapping[str, LegalDocument] | None,
    document_names: Mapping[str, str] | None = None,
) -> str:
    document_name, source = _document_metadata(chunk, documents, document_names)
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
    document_names: Mapping[str, str] | None = None,
    target_blocks: int | None = None,
) -> PackedEvidence:
    """Pack ranked chunks into a stable, header-inclusive character budget.

    Candidates are processed in rank/chunk-ID order. The first chunks from each
    document win the per-document cap. A chunk that cannot fit whole is included as a
    heading-preserving tail truncation when possible; otherwise it is dropped. Scores
    are retained in ``included_hits`` but never rendered in the default format.

    ``target_blocks`` separates *how many candidates the caller offers* from *how
    many blocks the pack should end up with*. Without it, a caller that wants four
    blocks must pass exactly four hits, and every hit dropped by
    ``max_chunks_per_document`` or by the character budget shrinks the pack with no
    backfill — three same-document hits in a top-4 silently yields a two-block pack.
    Pass a wider candidate window plus ``target_blocks=4`` and the constraints are
    applied first, then the pack stops once four blocks are in. Left ``None`` the
    behaviour is unchanged.
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
        if target_blocks is not None and len(blocks) >= target_blocks:
            dropped_ids.append(hit.chunk_id)
            dropped_reasons[hit.chunk_id] = "target_blocks_reached"
            continue

        document_count = document_counts.get(chunk.document_id, 0)
        if document_count >= max_chunks_per_document:
            dropped_ids.append(hit.chunk_id)
            dropped_reasons[hit.chunk_id] = "max_chunks_per_document"
            continue

        position = len(blocks) + 1
        header = _header(position, chunk, documents, document_names)
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
        "target_blocks": -1 if target_blocks is None else target_blocks,
        "target_blocks_met": (
            True if target_blocks is None else len(included_hits) >= target_blocks
        ),
        "document_names_supplied": document_names is not None,
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


def score_citation_match(
    query: str,
    chunk: LegalChunk,
    document: LegalDocument | None = None,
) -> float:
    """Calculate explicit legal citation match score between query and chunk.

    Extracts citation mentions from query using parse_citations (article, clause,
    document_number, document_name, year) and computes composite match score:
    - Document number match: +3.5 (in doc context) or +2.0 (in text)
    - Document name match: +2.5 (+1.0 if year also matches)
    - Article match: +3.0 (exact section label) or +2.5 (in heading) or +1.5 (in text)
    - Clause match: +1.0 (if article matched and clause in text)
    Returns 0.0 if query contains no citations.
    """
    from .sedar_retrieval.query.citation_parser import parse_citations

    citations = parse_citations(query)
    if not citations:
        return 0.0

    doc_name = document.name if document is not None else ""
    doc_context = f"{doc_name} {chunk.source_path}".casefold()
    full_text = f"{doc_context} {chunk.section_label or ''} {chunk.retrieval_text}".casefold()
    section = (chunk.section_label or "").casefold()

    total_score = 0.0
    matched_articles: set[str] = set()
    matched_doc_numbers: set[str] = set()
    matched_doc_names: set[str] = set()

    for c in citations:
        # 1. Document number match (highest precision)
        if c.document_number and c.document_number.casefold() not in matched_doc_numbers:
            num = c.document_number.casefold()
            if num in doc_context:
                total_score += 3.5
                matched_doc_numbers.add(num)
            elif num in full_text:
                total_score += 2.0
                matched_doc_numbers.add(num)

        # 2. Document name match
        if c.document_name and c.document_name.casefold() not in matched_doc_names:
            name = c.document_name.casefold()
            if name in doc_context:
                total_score += 2.5
                matched_doc_names.add(name)
                if c.year and c.year in doc_context:
                    total_score += 1.0
            elif name in full_text:
                total_score += 1.5
                matched_doc_names.add(name)
                if c.year and c.year in full_text:
                    total_score += 0.5

        # 3. Article match
        if c.article and c.article.casefold() not in matched_articles:
            art_pat = re.compile(rf"\bđiều\s+{re.escape(c.article.casefold())}\b")
            if section and art_pat.search(section):
                total_score += 3.0
                matched_articles.add(c.article.casefold())
            elif art_pat.search(full_text[:300]):
                total_score += 2.5
                matched_articles.add(c.article.casefold())
            elif art_pat.search(full_text):
                total_score += 1.5
                matched_articles.add(c.article.casefold())

            # 4. Clause match
            if c.clause:
                clause_pat = re.compile(rf"\bkhoản\s+{re.escape(c.clause.casefold())}\b")
                if clause_pat.search(full_text):
                    total_score += 1.0

    return total_score


def rerank_hits_with_citations(
    hits: Sequence[RetrievalHit],
    chunks: Mapping[str, LegalChunk],
    query: str,
    *,
    documents: Mapping[str, LegalDocument] | None = None,
    citation_weight: float = 1.0,
) -> tuple[RetrievalHit, ...]:
    """Reorder ranked hits by combining base retrieval score with citation match bonus.

    For queries containing explicit legal citations, relevant chunks matching the
    cited document or article receive a score bonus, prioritizing controlling statutes
    into the evidence pack. Queries without citations preserve exact original hit order.
    """
    from .sedar_retrieval.query.citation_parser import parse_citations

    citations = parse_citations(query)
    if not citations or citation_weight <= 0.0:
        return tuple(hits)

    scored_hits: list[tuple[float, RetrievalHit]] = []
    for hit in hits:
        chunk = chunks.get(hit.chunk_id)
        if chunk is None:
            base = hit.rerank_score if hit.rerank_score is not None else hit.bm25_score
            scored_hits.append((base, hit))
            continue
        doc = documents.get(chunk.document_id) if documents is not None else None
        cit_score = score_citation_match(query, chunk, doc)
        base = hit.rerank_score if hit.rerank_score is not None else hit.bm25_score
        combined = base + citation_weight * cit_score
        scored_hits.append((combined, hit))

    scored_hits.sort(key=lambda item: (item[0], -item[1].rank), reverse=True)
    return tuple(
        hit.model_copy(update={"rank": rank, "rerank_score": float(combined)})
        for rank, (combined, hit) in enumerate(scored_hits, start=1)
    )


def select_dynamic_evidence_hits(
    hits: Sequence[RetrievalHit],
    *,
    min_k: int = 2,
    max_k: int = 6,
    margin_top1: float = 3.0,
    margin_top2: float = 2.0,
) -> tuple[RetrievalHit, ...]:
    """Dynamically select variable-size evidence hits based on rerank score margins.

    Instead of a fixed top-k cutoff, keeps candidates up to max_k if they are
    competitive with Top 1 (within margin_top1) or Top 2 (within margin_top2)
    and have positive relevance logits. Guarantees at least min_k hits are kept.
    """
    if min_k <= 0:
        raise ValueError("min_k must be positive")
    if max_k < min_k:
        raise ValueError("max_k must be >= min_k")
    if len(hits) <= min_k:
        return tuple(hits[:max_k])

    scores: list[float] = [
        float(h.rerank_score) if h.rerank_score is not None else float(h.bm25_score)
        for h in hits
    ]
    s1 = scores[0]
    s2 = scores[1] if len(scores) > 1 else s1
    top2_is_competitive = (s1 - s2) <= margin_top1

    selected: list[RetrievalHit] = []
    for idx, hit in enumerate(hits):
        rank = idx + 1
        if rank > max_k:
            break
        if rank <= min_k:
            selected.append(hit)
            continue
        sk = scores[idx]
        diff_top1 = s1 - sk
        diff_top2 = s2 - sk

        is_relevant = sk >= 0.0
        close_to_top1 = diff_top1 <= margin_top1
        close_to_top2 = top2_is_competitive and (diff_top2 <= margin_top2)

        if is_relevant and (close_to_top1 or close_to_top2):
            selected.append(hit)
        else:
            break

    return tuple(selected)
