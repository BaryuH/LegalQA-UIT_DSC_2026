"""Multi-retriever candidate fusion (TASK 08 / R3)."""

from __future__ import annotations

import math
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
    rrf_score: float | None
    fused_rank: int
    legal_score: float | None = None
    legal_rank: int | None = None


def _validated_source_hits(
    ranked_lists: Mapping[str, Sequence[RetrieverHit]],
) -> dict[str, dict[str, RetrieverHit]]:
    source_hits: dict[str, dict[str, RetrieverHit]] = {}
    for source, hits in ranked_lists.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError("RRF source names must be non-blank strings")
        by_passage: dict[str, RetrieverHit] = {}
        for hit in hits:
            if hit.rank <= 0:
                raise ValueError("Retriever hit ranks must be positive")
            if not math.isfinite(float(hit.score)):
                raise ValueError("Retriever hit scores must be finite")
            by_passage.setdefault(hit.passage_id, hit)
        source_hits[source] = by_passage
    return source_hits


def _source_order(source_names: Sequence[str]) -> tuple[str, ...]:
    priority = {"bm25": 0, "dense": 1, "legal": 2, "colbert": 3}
    return tuple(
        sorted(source_names, key=lambda source: (priority.get(source, 4), source))
    )


def _fused_candidate(
    passage_id: str,
    *,
    bm25: Mapping[str, RetrieverHit],
    dense: Mapping[str, RetrieverHit],
    colbert: Mapping[str, RetrieverHit],
    legal: Mapping[str, RetrieverHit],
    rrf_score: float | None,
    fused_rank: int,
) -> FusedCandidate:
    bm25_hit = bm25.get(passage_id)
    dense_hit = dense.get(passage_id)
    colbert_hit = colbert.get(passage_id)
    legal_hit = legal.get(passage_id)
    return FusedCandidate(
        passage_id=passage_id,
        bm25_score=None if bm25_hit is None else bm25_hit.score,
        bm25_rank=None if bm25_hit is None else bm25_hit.rank,
        dense_score=None if dense_hit is None else dense_hit.score,
        dense_rank=None if dense_hit is None else dense_hit.rank,
        colbert_score=None if colbert_hit is None else colbert_hit.score,
        colbert_rank=None if colbert_hit is None else colbert_hit.rank,
        rrf_score=rrf_score,
        fused_rank=fused_rank,
        legal_score=None if legal_hit is None else legal_hit.score,
        legal_rank=None if legal_hit is None else legal_hit.rank,
    )


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[RetrieverHit]],
    *,
    rrf_k: int = 60,
    union_cap: int = 250,
    weights: Mapping[str, float] | None = None,
) -> tuple[FusedCandidate, ...]:
    """Fuse ranked lists with optional source weights.

    The default is the existing unweighted RRF behavior.  A weight of zero
    keeps the source in the candidate union but contributes no RRF score,
    which is useful for controlled ablations.  Missing source-specific
    metadata remains ``None`` rather than being converted to zero.
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if union_cap <= 0:
        raise ValueError("union_cap must be positive")
    validated_weights: dict[str, float] = {}
    for source, weight in (weights or {}).items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError("RRF source names must be non-blank strings")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"RRF weight must be numeric: {source!r}")
        numeric_weight = float(weight)
        if not math.isfinite(numeric_weight) or numeric_weight < 0.0:
            raise ValueError(f"RRF weight must be finite and non-negative: {source!r}")
        validated_weights[source] = numeric_weight

    source_hits = _validated_source_hits(ranked_lists)
    scores: dict[str, float] = {}
    for source, hits_by_passage in source_hits.items():
        weight = validated_weights.get(source, 1.0)
        for hit in hits_by_passage.values():
            scores[hit.passage_id] = scores.get(hit.passage_id, 0.0) + weight / (
                rrf_k + hit.rank
            )

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    capped = ordered[:union_cap]
    fused: list[FusedCandidate] = []
    bm25 = source_hits.get("bm25", {})
    dense = source_hits.get("dense", {})
    colbert = source_hits.get("colbert", {})
    legal = source_hits.get("legal", {})
    for index, (passage_id, rrf_score) in enumerate(capped, start=1):
        fused.append(
            _fused_candidate(
                passage_id,
                bm25=bm25,
                dense=dense,
                colbert=colbert,
                legal=legal,
                rrf_score=rrf_score,
                fused_rank=index,
            )
        )
    return tuple(fused)


def candidate_union(
    ranked_lists: Mapping[str, Sequence[RetrieverHit]],
    *,
    union_cap: int = 250,
) -> tuple[FusedCandidate, ...]:
    """Return a pure candidate union without an RRF score.

    Sources are traversed in the deterministic order BM25, dense, legal,
    colbert, then any custom names alphabetically.  A passage keeps its
    source-specific metadata for every source in which it appears.
    """

    if union_cap <= 0:
        raise ValueError("union_cap must be positive")
    source_hits = _validated_source_hits(ranked_lists)
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for source in _source_order(tuple(source_hits)):
        for passage_id in source_hits[source]:
            if passage_id in seen:
                continue
            seen.add(passage_id)
            ordered_ids.append(passage_id)
    capped = ordered_ids[:union_cap]
    bm25 = source_hits.get("bm25", {})
    dense = source_hits.get("dense", {})
    colbert = source_hits.get("colbert", {})
    legal = source_hits.get("legal", {})
    return tuple(
        _fused_candidate(
            passage_id,
            bm25=bm25,
            dense=dense,
            colbert=colbert,
            legal=legal,
            rrf_score=None,
            fused_rank=index,
        )
        for index, passage_id in enumerate(capped, start=1)
    )
