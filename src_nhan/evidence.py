"""Evidence packing from reranked chunks for LLM context.

Renders the top reranked chunks into a formatted evidence string
with document provenance, respecting the max character budget.
Tracks included, dropped, and truncated chunks explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .chunker import LegalChunk
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


def pack_evidence(
    reranked_hits: list[RerankHit],
    chunks_by_id: dict[str, LegalChunk],
    *,
    max_total_chars: int = 4000,
) -> PackedEvidence:
    """Pack reranked chunks into a character-budgeted evidence string.

    Includes chunks in rerank order until the character budget is exceeded.
    Each chunk is prefixed with its document name for provenance.
    """
    included_ids: list[str] = []
    dropped_ids: list[str] = []
    truncated_ids: list[str] = []
    parts: list[str] = []
    current_chars = 0

    for hit in reranked_hits:
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            dropped_ids.append(hit.chunk_id)
            continue

        # Format: [Document: <doc_id>] <text>
        header = f"[Văn bản: {chunk.document_id}]"
        chunk_text = f"{header}\n{chunk.raw_text}"

        if current_chars + len(chunk_text) > max_total_chars:
            # Try to fit a truncated version
            remaining = max_total_chars - current_chars
            if remaining > len(header) + 100:  # At least 100 useful chars
                truncated_text = chunk_text[:remaining].rstrip()
                parts.append(truncated_text)
                included_ids.append(hit.chunk_id)
                truncated_ids.append(hit.chunk_id)
                current_chars += len(truncated_text)
            else:
                dropped_ids.append(hit.chunk_id)
            break

        parts.append(chunk_text)
        included_ids.append(hit.chunk_id)
        current_chars += len(chunk_text)

    # Also drop remaining hits that weren't processed
    processed = set(included_ids) | set(dropped_ids)
    for hit in reranked_hits:
        if hit.chunk_id not in processed:
            dropped_ids.append(hit.chunk_id)

    rendered = "\n\n".join(parts)

    if not rendered.strip():
        logger.warning("Evidence packing produced empty text.")

    logger.info(
        "Evidence packed: %d included, %d dropped, %d truncated, %d chars",
        len(included_ids),
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
    )
