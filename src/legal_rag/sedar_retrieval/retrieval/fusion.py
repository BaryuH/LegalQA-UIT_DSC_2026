"""Multi-retriever candidate fusion (TASK 08 / R3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrieverHit:
    passage_id: str
    score: float
    rank: int
    source: str


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    passage_id: str
    bm25_score: float | None
    bm25_rank: int | None
    dense_score: float | None
    dense_rank: int | None
    colbert_score: float | None
    colbert_rank: int | None
    rrf_score: float
    fused_rank: int


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[RetrieverHit]],
    *,
    rrf_k: int = 60,
    union_cap: int = 250,
) -> tuple[FusedCandidate, ...]:
    """Fuse ranked lists with RRF. Missing sources stay None (not zero)."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if union_cap <= 0:
        raise ValueError("union_cap must be positive")

    scores: dict[str, float] = {}
    bm25: dict[str, RetrieverHit] = {}
    dense: dict[str, RetrieverHit] = {}
    colbert: dict[str, RetrieverHit] = {}

    for source, hits in ranked_lists.items():
        seen: set[str] = set()
        for hit in hits:
            if hit.passage_id in seen:
                continue
            seen.add(hit.passage_id)
            scores[hit.passage_id] = scores.get(hit.passage_id, 0.0) + 1.0 / (
                rrf_k + hit.rank
            )
            if source == "bm25":
                bm25[hit.passage_id] = hit
            elif source == "dense":
                dense[hit.passage_id] = hit
            elif source == "colbert":
                colbert[hit.passage_id] = hit

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    capped = ordered[:union_cap]
    fused: list[FusedCandidate] = []
    for index, (passage_id, rrf_score) in enumerate(capped, start=1):
        b = bm25.get(passage_id)
        d = dense.get(passage_id)
        c = colbert.get(passage_id)
        fused.append(
            FusedCandidate(
                passage_id=passage_id,
                bm25_score=None if b is None else b.score,
                bm25_rank=None if b is None else b.rank,
                dense_score=None if d is None else d.score,
                dense_rank=None if d is None else d.rank,
                colbert_score=None if c is None else c.score,
                colbert_rank=None if c is None else c.rank,
                rrf_score=rrf_score,
                fused_rank=index,
            )
        )
    return tuple(fused)
