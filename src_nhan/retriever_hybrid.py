"""Hybrid retriever combining BM25 and dense retrieval via Reciprocal Rank Fusion.

Merges results from BM25 (lexical) and vietlegal-e5 (semantic) retrievers
using RRF to produce a single ranked list that captures both exact-match
legal terms and semantic similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .retriever_bm25 import BM25Hit
from .retriever_dense import DenseHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HybridHit:
    """One hybrid retrieval result with fused score."""

    chunk_id: str
    document_id: str
    rrf_score: float
    bm25_score: float | None
    dense_score: float | None
    rank: int


def reciprocal_rank_fusion(
    bm25_hits: list[BM25Hit],
    dense_hits: list[DenseHit],
    *,
    k: int = 60,
    top_n: int = 30,
) -> list[HybridHit]:
    """Fuse BM25 and dense retrieval results using Reciprocal Rank Fusion.

    RRF score for document d = sum over systems S: 1 / (k + rank_S(d))
    where k is a constant (typically 60).
    """
    # Collect RRF contributions and original scores
    rrf_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}
    dense_scores: dict[str, float] = {}
    doc_ids: dict[str, str] = {}

    for hit in bm25_hits:
        rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (
            k + hit.rank
        )
        bm25_scores[hit.chunk_id] = hit.score
        doc_ids[hit.chunk_id] = hit.document_id

    for hit in dense_hits:
        rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + 1.0 / (
            k + hit.rank
        )
        dense_scores[hit.chunk_id] = hit.score
        doc_ids[hit.chunk_id] = hit.document_id

    # Sort by RRF score descending, then by chunk_id for deterministic tie-break
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: (-rrf_scores[cid], cid),
    )[:top_n]

    results: list[HybridHit] = []
    for rank, chunk_id in enumerate(sorted_ids, start=1):
        results.append(
            HybridHit(
                chunk_id=chunk_id,
                document_id=doc_ids[chunk_id],
                rrf_score=rrf_scores[chunk_id],
                bm25_score=bm25_scores.get(chunk_id),
                dense_score=dense_scores.get(chunk_id),
                rank=rank,
            )
        )

    logger.info(
        "Hybrid fusion: %d BM25 + %d dense → %d merged hits",
        len(bm25_hits),
        len(dense_hits),
        len(results),
    )
    return results
