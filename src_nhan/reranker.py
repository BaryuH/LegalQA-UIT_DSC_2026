"""ViRanker cross-encoder reranker for Vietnamese legal text.

Uses namdp-ptit/ViRanker (PhoBERT-based, ~0.135B) to rerank
candidate chunks by query-passage relevance. No silent fallback:
failures are reported explicitly per AGENTS.md rule #10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .chunker import LegalChunk
from .retriever_hybrid import HybridHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RerankHit:
    """One reranked result with cross-encoder score."""

    chunk_id: str
    document_id: str
    rerank_score: float
    original_rrf_score: float
    rank: int


class RerankerError(RuntimeError):
    """Raised when the reranker cannot complete — no silent fallback."""


class ViReranker:
    """Cross-encoder reranker using ViRanker model."""

    def __init__(
        self,
        *,
        model_name: str = "namdp-ptit/ViRanker",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model: %s", model_name)
        self._model = CrossEncoder(model_name, device=device)
        self._batch_size = batch_size
        self._model_name = model_name
        logger.info("Reranker loaded: %s", model_name)

    def rerank(
        self,
        query: str,
        hits: list[HybridHit],
        chunks_by_id: dict[str, LegalChunk],
        *,
        top_k: int = 3,
    ) -> list[RerankHit]:
        """Rerank hybrid hits using cross-encoder scores.

        Raises RerankerError on failure — no silent fallback.
        """
        if not query.strip():
            raise RerankerError("Query must not be blank")

        if not hits:
            return []

        # Build query-passage pairs
        pairs: list[tuple[str, str]] = []
        valid_hits: list[HybridHit] = []
        for hit in hits:
            chunk = chunks_by_id.get(hit.chunk_id)
            if chunk is None:
                logger.warning(
                    "Chunk %s not found in index, skipping", hit.chunk_id
                )
                continue
            pairs.append((query, chunk.retrieval_text))
            valid_hits.append(hit)

        if not pairs:
            raise RerankerError(
                f"No valid chunks found for reranking "
                f"(query={query[:50]!r}...)"
            )

        try:
            scores = self._model.predict(
                pairs, batch_size=self._batch_size, show_progress_bar=False
            )
        except Exception as exc:
            raise RerankerError(
                f"Reranker model failed: {exc}"
            ) from exc

        # Sort by rerank score descending, chunk_id tie-break
        scored_hits = list(zip(valid_hits, scores))
        scored_hits.sort(key=lambda x: (-float(x[1]), x[0].chunk_id))

        results: list[RerankHit] = []
        for rank, (hit, score) in enumerate(scored_hits[:top_k], start=1):
            results.append(
                RerankHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    rerank_score=float(score),
                    original_rrf_score=hit.rrf_score,
                    rank=rank,
                )
            )

        logger.info(
            "Reranked %d → %d hits (top_k=%d)",
            len(valid_hits),
            len(results),
            top_k,
        )
        return results
