"""Acceptance tests for score-preserving convex fusion."""

from __future__ import annotations

import pytest

from legal_rag.sedar_retrieval.retrieval.fusion import (
    RetrieverHit,
    convex_score_fusion,
    reciprocal_rank_fusion,
)


def _lists() -> dict[str, list[RetrieverHit]]:
    return {
        "bm25": [
            RetrieverHit("p1", 12.0, 1, "bm25"),
            RetrieverHit("p2", 8.0, 2, "bm25"),
            RetrieverHit("p3", 2.0, 3, "bm25"),
        ],
        "dense": [
            RetrieverHit("p2", 0.91, 1, "dense"),
            RetrieverHit("p4", 0.90, 2, "dense"),
            RetrieverHit("p1", 0.40, 3, "dense"),
        ],
    }


def test_convex_fusion_preserves_score_magnitude_where_rrf_cannot() -> None:
    lists = _lists()
    rrf = [c.passage_id for c in reciprocal_rank_fusion(lists)]
    convex = [
        c.passage_id
        for c in convex_score_fusion(
            lists, weights={"bm25": 0.3, "dense": 0.7}, normalization="minmax"
        )
    ]
    # p4 is dense rank 2 but scores almost exactly like dense rank 1, while p1 is
    # dense rank 3 with a much weaker score. Rank-only fusion cannot see that.
    assert rrf.index("p1") < rrf.index("p4")
    assert convex.index("p4") < convex.index("p1")


def test_weights_are_normalised_and_alpha_shifts_the_order() -> None:
    lists = _lists()
    lexical = [
        c.passage_id
        for c in convex_score_fusion(lists, weights={"bm25": 9.0, "dense": 1.0})
    ]
    semantic = [
        c.passage_id
        for c in convex_score_fusion(lists, weights={"bm25": 1.0, "dense": 9.0})
    ]
    assert lexical[0] == "p1"
    assert semantic[0] == "p2"
    # Scaling both weights by the same factor must not change anything.
    scaled = [
        c.passage_id
        for c in convex_score_fusion(lists, weights={"bm25": 18.0, "dense": 2.0})
    ]
    assert scaled == lexical


def test_theoretical_bounds_change_the_floor_not_the_top() -> None:
    lists = _lists()
    plain = convex_score_fusion(lists, weights={"bm25": 0.3, "dense": 0.7})
    bounded = convex_score_fusion(
        lists,
        weights={"bm25": 0.3, "dense": 0.7},
        normalization="theoretical_minmax",
        theoretical_bounds={"bm25": (0.0, None), "dense": (-1.0, 1.0)},
    )
    assert plain[0].passage_id == bounded[0].passage_id == "p2"
    # Under min-max the weakest candidate in each list is pinned to 0; with
    # theoretical bounds it keeps a positive score.
    assert plain[-1].fusion_score == pytest.approx(0.0)
    assert bounded[-1].fusion_score > 0.0


def test_skip_policy_averages_over_the_sources_that_returned_the_passage() -> None:
    lists = _lists()
    default = {
        c.passage_id: c.fusion_score
        for c in convex_score_fusion(lists, weights={"bm25": 0.3, "dense": 0.7})
    }
    skipped = {
        c.passage_id: c.fusion_score
        for c in convex_score_fusion(
            lists, weights={"bm25": 0.3, "dense": 0.7}, missing_score="skip"
        )
    }
    # p4 appears in the dense list only. Charging it the lexical minimum drags it
    # down; skipping does not.
    assert skipped["p4"] > default["p4"]
    assert skipped["p4"] == pytest.approx(1.0, abs=0.05)


def test_metadata_and_determinism() -> None:
    lists = _lists()
    fused = convex_score_fusion(lists, weights={"bm25": 0.3, "dense": 0.7})
    assert all(c.rrf_score is None for c in fused)
    assert all(c.fusion_method == "convex_minmax_theoretical_min" for c in fused)
    assert [c.fused_rank for c in fused] == list(range(1, len(fused) + 1))
    again = convex_score_fusion(lists, weights={"bm25": 0.3, "dense": 0.7})
    assert [c.passage_id for c in fused] == [c.passage_id for c in again]


def test_ties_break_on_passage_id() -> None:
    lists = {
        "bm25": [
            RetrieverHit("pb", 5.0, 1, "bm25"),
            RetrieverHit("pa", 5.0, 2, "bm25"),
        ]
    }
    fused = convex_score_fusion(lists, weights={"bm25": 1.0})
    assert [c.passage_id for c in fused] == ["pa", "pb"]


def test_rejects_invalid_configuration() -> None:
    lists = _lists()
    with pytest.raises(ValueError):
        convex_score_fusion(lists, weights={"bm25": 0.0, "dense": 0.0})
    with pytest.raises(ValueError):
        convex_score_fusion(lists, normalization="softmax")
    with pytest.raises(ValueError):
        convex_score_fusion(lists, missing_score="mean")
    with pytest.raises(ValueError):
        convex_score_fusion(lists, weights={"bm25": -1.0})
    with pytest.raises(ValueError):
        convex_score_fusion(lists, union_cap=0)


def test_union_cap_truncates_after_ranking() -> None:
    lists = _lists()
    fused = convex_score_fusion(lists, weights={"bm25": 0.3, "dense": 0.7}, union_cap=2)
    assert len(fused) == 2
    assert fused[0].passage_id == "p2"


def test_rrf_default_is_unchanged() -> None:
    lists = _lists()
    fused = reciprocal_rank_fusion(lists)
    assert fused[0].passage_id == "p2"
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].fusion_score is None
    assert fused[0].fusion_method is None
