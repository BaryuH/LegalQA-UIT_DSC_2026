"""Evidence packing from reranked chunks for LLM context.

Supports Small-to-Big / Parent Document Retrieval:
Resolves top reranked child chunks to their Parent Documents (Điều),
deduplicating parents while preserving relevance ranking and character budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .chunker import LegalChunk, ParentChunk
from .reranker import RerankHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PackedEvidence:
    """Evidence text with inclusion/exclusion tracking."""

    rendered_text: str
    included_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    truncated_ids: tuple[str, ...]
    total_chars: int
    included_parent_ids: tuple[str, ...] = ()


def pack_evidence(
    reranked_hits: list[RerankHit],
    chunks_by_id: dict[str, LegalChunk],
    parents_by_id: dict[str, ParentChunk] | None = None,
    *,
    max_total_chars: int = 4000,
) -> PackedEvidence:
    """Pack reranked chunks into a character-budgeted evidence string.

    If parents_by_id is provided (Small-to-Big architecture):
      Resolves top child hits -> parent_id -> Parent Document (entire Điều).
      Deduplicates parent documents if multiple child hits map to the same Điều,
      preserving the rank order of the best-matching child chunk.
    Otherwise:
      Falls back to chunk-level packing.
    """
    included_ids: list[str] = []
    dropped_ids: list[str] = []
    truncated_ids: list[str] = []
    included_parent_ids: list[str] = []
    seen_parent_ids: set[str] = set()
    parts: list[str] = []
    current_chars = 0

    for hit in reranked_hits:
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            dropped_ids.append(hit.chunk_id)
            continue

        # Small-to-Big: resolve to Parent Document if parents_by_id is available
        if parents_by_id is not None:
            parent = parents_by_id.get(chunk.parent_id)
            if parent is not None:
                # Deduplicate: if this parent document is already included, record chunk and continue
                if parent.parent_id in seen_parent_ids:
                    included_ids.append(hit.chunk_id)
                    continue

                header = f"[Văn bản: {parent.document_id} - {parent.header}]"
                doc_text = f"{header}\n{parent.raw_text.strip()}"
                target_id = parent.parent_id
            else:
                header = f"[Văn bản: {chunk.document_id} - {chunk.section_label}]"
                doc_text = f"{header}\n{chunk.raw_text.strip()}"
                target_id = chunk.chunk_id
        else:
            header = f"[Văn bản: {chunk.document_id}]"
            doc_text = f"{header}\n{chunk.raw_text.strip()}"
            target_id = chunk.chunk_id

        sep_len = 2 if parts else 0
        if current_chars + sep_len + len(doc_text) <= max_total_chars:
            parts.append(doc_text)
            included_ids.append(hit.chunk_id)
            if parents_by_id is not None and target_id not in seen_parent_ids:
                seen_parent_ids.add(target_id)
                included_parent_ids.append(target_id)
            current_chars += sep_len + len(doc_text)
        else:
            # Try to fit a truncated version
            remaining = max_total_chars - current_chars - sep_len
            if remaining > len(header) + 100:  # At least 100 useful chars
                truncated_text = doc_text[:remaining].rstrip()
                parts.append(truncated_text)
                included_ids.append(hit.chunk_id)
                truncated_ids.append(target_id)
                if parents_by_id is not None and target_id not in seen_parent_ids:
                    seen_parent_ids.add(target_id)
                    included_parent_ids.append(target_id)
                current_chars += sep_len + len(truncated_text)
            else:
                dropped_ids.append(hit.chunk_id)
            break

    # Mark remaining hits as dropped
    processed = set(included_ids) | set(dropped_ids)
    for hit in reranked_hits:
        if hit.chunk_id not in processed:
            dropped_ids.append(hit.chunk_id)

    rendered = "\n\n".join(parts)

    if not rendered.strip():
        logger.warning("Evidence packing produced empty text.")

    logger.info(
        "Evidence packed: %d chunks included, %d parents included, %d dropped, %d truncated, %d chars",
        len(included_ids),
        len(included_parent_ids),
        len(dropped_ids),
        len(truncated_ids),
        len(rendered),
    )

    return PackedEvidence(
        rendered_text=rendered,
        included_ids=tuple(included_ids),
        dropped_ids=tuple(dropped_ids),
        truncated_ids=tuple(truncated_ids),
        total_chars=len(rendered),
        included_parent_ids=tuple(included_parent_ids),
    )
