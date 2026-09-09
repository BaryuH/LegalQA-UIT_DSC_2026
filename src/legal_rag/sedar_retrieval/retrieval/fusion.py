"""Multi-retriever candidate fusion (TASK 08 / R3, TASK 23 convex fusion).

Two fusion families live here.

``reciprocal_rank_fusion`` merges ranks and discards score magnitude. That is
what makes it robust across retrievers with incomparable score scales, and it is
also its cost: a candidate ranked #1 by both legs is unbeatable under RRF no
matter how much better a #2 candidate's actual scores were.

``convex_score_fusion`` keeps magnitude: it normalises each leg's scores and
takes a weighted sum. The published evidence favours it. Bruch et al. (ACM TOIS
2023, arXiv:2210.11934) measure convex combination above RRF both in and out of
domain on MS MARCO and BEIR (nDCG@1000 0.454 vs 0.425), find RRF's ``k`` does not
transfer between collections while a single alpha does, and show the convex
parameter is sample-efficient to tune. OpenSearch's own benchmark of its own
default reports RRF 3.86% below score normalisation on nDCG@10 across six BEIR
sets. For Vietnamese specifically, "Which Works Best for Vietnamese?" (Findings
of EACL 2026) reports linear alpha-fusion above both standalone legs on all ten
datasets tested, with the optimum at alpha 0.6-0.8, and states that RRF
"generally underperformed linear interpolation"; the top-3 DRiLL@VLSP 2025 system
used a weighted sum with lambda 0.6 rather than RRF.

Neither function replaces the other. RRF stays the frozen default so existing
runs stay reproducible; convex fusion is the tunable alternative to ablate
against it.
"""

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
    #: Score produced by a non-RRF fusion function, kept in its own field so
    #: ``rrf_score`` never carries a value that is not an RRF score.
    fusion_score: float | None = None
    fusion_method: str | None = None


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
    fusion_score: float | None = None,
    fusion_method: str | None = None,
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
        fusion_score=fusion_score,
        fusion_method=fusion_method,
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


NORMALIZATIONS = ("minmax", "theoretical_minmax", "zscore", "none")
MISSING_POLICIES = ("theoretical_min", "observed_min", "zero", "skip")


def _normalize_scores(
    scores: Mapping[str, float],
    *,
    method: str,
    theoretical_min: float | None,
    theoretical_max: float | None,
) -> tuple[dict[str, float], float]:
    """Map one retriever's raw scores onto a comparable scale.

    Returns the normalised scores and the value a passage missing from this
    retriever should take under the ``theoretical_min`` policy. Bruch et al.
    normalise with the *theoretical* minimum where the retriever has one -
    cosine similarity on L2-normalised vectors is bounded below by -1 and BM25 by
    0 - because the observed minimum of a truncated top-k list is an artefact of
    the cut-off, not of the score distribution.
    """

    if method not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization: {method!r}")
    values = list(scores.values())
    if not values:
        return {}, 0.0
    if method == "none":
        return dict(scores), float(theoretical_min if theoretical_min is not None else 0.0)
    if method == "zscore":
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        deviation = math.sqrt(variance)
        if deviation <= 0.0:
            return {key: 0.0 for key in scores}, 0.0
        return {key: (value - mean) / deviation for key, value in scores.items()}, min(
            (value - mean) / deviation for value in values
        )

    observed_min = min(values)
    observed_max = max(values)
    lower = observed_min
    upper = observed_max
    if method == "theoretical_minmax":
        if theoretical_min is not None:
            lower = min(theoretical_min, observed_min)
        if theoretical_max is not None:
            upper = max(theoretical_max, observed_max)
    span = upper - lower
    if span <= 0.0:
        return {key: 1.0 for key in scores}, 0.0
    normalized = {key: (value - lower) / span for key, value in scores.items()}
    return normalized, 0.0


def convex_score_fusion(
    ranked_lists: Mapping[str, Sequence[RetrieverHit]],
    *,
    weights: Mapping[str, float] | None = None,
    normalization: str = "minmax",
    missing_score: str = "theoretical_min",
    theoretical_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    union_cap: int = 250,
) -> tuple[FusedCandidate, ...]:
    """Fuse ranked lists by a weighted sum of per-query normalised scores.

    ``weights`` are normalised to sum to one, so a two-source call with
    ``{"bm25": 0.3, "dense": 0.7}`` is the usual ``alpha`` parameterisation with
    ``alpha = 0.7`` on the dense leg. Weights are per source and need not sum to
    one on input.

    ``normalization`` is applied *per query and per source*, which is what makes
    the two legs comparable at all:

    ``minmax``
        ``(s - min) / (max - min)`` over the candidates this retriever returned.
    ``theoretical_minmax``
        the same, but the bounds are widened to the retriever's theoretical range
        where one is supplied via ``theoretical_bounds`` (for example ``(0.0,
        None)`` for BM25 or ``(-1.0, 1.0)`` for cosine). Preferred when the
        candidate lists are deep, because the observed minimum then reflects the
        top-k cut rather than the score distribution.
    ``zscore``
        standardise to zero mean and unit deviation.
    ``none``
        use raw scores. Only sensible when the legs already share a scale.

    ``missing_score`` decides what a passage that one retriever did not return
    contributes: its theoretical minimum (the default, and the choice in Bruch et
    al.), the observed minimum of that retriever's list, zero, or nothing at all
    (``skip``, i.e. average over the sources that did return it).

    ``skip`` is **pathological under ``minmax``** and measured as such: it lost
    0.06-0.10 article@4 at every alpha in the A1 sweep
    (``docs/sedar_retrieval/A1_CONVEX_FUSION_RESULTS.md``). Min-max maps each
    leg's rank-1 hit to exactly 1.0, so under ``skip`` a passage only one leg
    returned is scored on that leg alone, undiluted - the lexical leg's rank-1
    hit then scores 1.0 and tops the fused list however weak it is, with the
    dense leg's opinion discarded. ``theoretical_min`` charges an absent leg its
    minimum, which is what makes cross-leg agreement earn a top position. The
    two policies are numerically identical when only one leg has weight, which
    is how the A1 sweep identified this as the mechanism rather than a
    coincidence. Revisit ``skip`` only under ``theoretical_minmax``, where the
    observed maximum no longer pins each leg's top hit to 1.0.
    """

    if union_cap <= 0:
        raise ValueError("union_cap must be positive")
    if missing_score not in MISSING_POLICIES:
        raise ValueError(f"Unknown missing_score policy: {missing_score!r}")

    source_hits = _validated_source_hits(ranked_lists)
    raw_weights: dict[str, float] = {}
    for source in source_hits:
        weight = (weights or {}).get(source, 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"Fusion weight must be numeric: {source!r}")
        numeric = float(weight)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"Fusion weight must be finite and non-negative: {source!r}")
        raw_weights[source] = numeric
    total_weight = sum(raw_weights.values())
    if total_weight <= 0.0:
        raise ValueError("At least one fusion weight must be positive")
    normalized_weights = {
        source: weight / total_weight for source, weight in raw_weights.items()
    }

    bounds = dict(theoretical_bounds or {})
    per_source_scores: dict[str, dict[str, float]] = {}
    per_source_missing: dict[str, float] = {}
    for source, hits_by_passage in source_hits.items():
        lower, upper = bounds.get(source, (None, None))
        raw = {
            passage_id: float(hit.score)
            for passage_id, hit in hits_by_passage.items()
        }
        scaled, missing_value = _normalize_scores(
            raw,
            method=normalization,
            theoretical_min=lower,
            theoretical_max=upper,
        )
        per_source_scores[source] = scaled
        if missing_score == "theoretical_min":
            per_source_missing[source] = missing_value
        elif missing_score == "observed_min":
            per_source_missing[source] = min(scaled.values()) if scaled else 0.0
        else:
            per_source_missing[source] = 0.0

    all_ids: set[str] = set()
    for scaled in per_source_scores.values():
        all_ids |= set(scaled)

    fused_scores: dict[str, float] = {}
    for passage_id in all_ids:
        total = 0.0
        weight_seen = 0.0
        for source, scaled in per_source_scores.items():
            weight = normalized_weights[source]
            if passage_id in scaled:
                total += weight * scaled[passage_id]
                weight_seen += weight
            elif missing_score == "skip":
                continue
            else:
                total += weight * per_source_missing[source]
                weight_seen += weight
        if missing_score == "skip" and weight_seen > 0.0:
            total /= weight_seen
        fused_scores[passage_id] = total

    ordered = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    capped = ordered[:union_cap]
    method_label = f"convex_{normalization}_{missing_score}"
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
            fusion_score=score,
            fusion_method=method_label,
        )
        for index, (passage_id, score) in enumerate(capped, start=1)
    )
