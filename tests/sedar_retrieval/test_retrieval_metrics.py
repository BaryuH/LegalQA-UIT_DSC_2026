"""Unit tests for SEDAR Retrieval v3 metrics (TASK 02)."""

from __future__ import annotations

import math

from legal_rag.sedar_retrieval.eval.retrieval_metrics import (
    QueryRelevance,
    RankedList,
    evaluate_retrieval,
    metrics_to_dict,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_no_relevant_result() -> None:
    pred = RankedList("q1", ("a", "b", "c"))
    label = QueryRelevance("q1", frozenset({"z"}), provenance="gold")
    bundle = evaluate_retrieval([pred], [label], cutoffs=(5, 10))
    assert bundle.recall_at[5] == 0.0
    assert bundle.mrr_at_10 == 0.0
    assert bundle.ndcg_at_10 == 0.0


def test_relevant_at_rank_1() -> None:
    pred = RankedList("q1", ("rel", "x", "y"))
    label = QueryRelevance("q1", frozenset({"rel"}), provenance="gold")
    bundle = evaluate_retrieval([pred], [label], cutoffs=(5, 10))
    assert bundle.recall_at[5] == 1.0
    assert bundle.mrr_at_10 == 1.0
    assert math.isclose(bundle.ndcg_at_10, 1.0, abs_tol=1e-9)


def test_multiple_relevant_and_multi_hit() -> None:
    pred = RankedList("q1", ("r1", "x", "r2", "y"))
    label = QueryRelevance("q1", frozenset({"r1", "r2", "r3"}), provenance="gold")
    bundle = evaluate_retrieval([pred], [label], cutoffs=(5, 10))
    assert bundle.multi_hit_at[5] == 1.0
    # coverage = 2/3
    assert math.isclose(bundle.evidence_coverage_at[5], 2.0 / 3.0, abs_tol=1e-9)


def test_duplicate_candidates_do_not_inflate_recall() -> None:
    pred = RankedList("q1", ("rel", "rel", "rel", "x"))
    label = QueryRelevance("q1", frozenset({"rel"}), provenance="gold")
    bundle = evaluate_retrieval([pred], [label], cutoffs=(5,))
    assert bundle.recall_at[5] == 1.0
    assert math.isclose(bundle.duplicate_rate, 0.5, abs_tol=1e-9)


def test_missing_labels_are_explicit_unlabeled() -> None:
    pred = RankedList("q1", ("a", "b"))
    label = QueryRelevance("q1", frozenset(), provenance="unlabeled")
    bundle = evaluate_retrieval([pred], [label], cutoffs=(5, 10))
    assert bundle.n_unlabeled_queries == 1
    assert bundle.n_labeled_queries == 0
    assert bundle.recall_at[5] == 0.0
    assert bundle.label_provenance_counts["unlabeled"] == 1


def test_hand_computed_ndcg_toy() -> None:
    # ranks: rel at 2 => gains [0,1], DCG = 0 + 1/log2(3)
    ranked = ("x", "rel")
    relevant = frozenset({"rel"})
    dcg = 1.0 / math.log2(3)
    ideal = 1.0 / math.log2(2)
    expected = dcg / ideal
    assert math.isclose(ndcg_at_k(ranked, relevant, 10), expected, abs_tol=1e-9)
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 1.0
    assert reciprocal_rank_at_k(ranked, relevant, 10) == 0.5


def test_metrics_dict_stable_keys() -> None:
    pred = RankedList("q1", ("rel",))
    label = QueryRelevance("q1", frozenset({"rel"}), provenance="gold")
    first = metrics_to_dict(evaluate_retrieval([pred], [label]))
    second = metrics_to_dict(evaluate_retrieval([pred], [label]))
    assert first == second
