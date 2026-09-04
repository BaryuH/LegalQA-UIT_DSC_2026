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


# ── M1: level-aware relevance ──────────────────────────────────────────────
# The silver-label builder prefers article-level passage IDs, while the
# retrieval views index both "article" and "clause" levels. Callers that do not
# pass the level mappings therefore score a system that returned the correct
# clause of the correct article as a total miss, and every *_recall_at reports
# 0.0 because the code path that fills them never runs. That is what produced
# the ~0.015 Recall@20 recorded in TASK 21 against an article/provision recall
# of 0.9758 measured by audit_retrieval_recall.py on the same rankings.

_ARTICLE_LABEL = "docA::art76"
_CLAUSE_OF_SAME_ARTICLE = "docA::art76::cl1"
_SAME_ARTICLE_NUMBER_OTHER_DOCUMENT = "docZ::art76"

_TO_DOCUMENT = {
    _ARTICLE_LABEL: "docA",
    _CLAUSE_OF_SAME_ARTICLE: "docA",
    _SAME_ARTICLE_NUMBER_OTHER_DOCUMENT: "docZ",
}
# Article identity is document-scoped: never a bare article number.
_TO_ARTICLE = {
    _ARTICLE_LABEL: "docA::art::76",
    _CLAUSE_OF_SAME_ARTICLE: "docA::art::76",
    _SAME_ARTICLE_NUMBER_OTHER_DOCUMENT: "docZ::art::76",
}
_TO_CLAUSE = {_CLAUSE_OF_SAME_ARTICLE: "docA::art76::cl1"}


def _article_label() -> QueryRelevance:
    return QueryRelevance("q1", frozenset({_ARTICLE_LABEL}), provenance="silver")


def test_level_mappings_credit_the_right_clause_of_the_right_article() -> None:
    pred = RankedList("q1", (_CLAUSE_OF_SAME_ARTICLE,))
    bundle = evaluate_retrieval(
        [pred],
        [_article_label()],
        cutoffs=(4, 10),
        passage_to_document=_TO_DOCUMENT,
        passage_to_article=_TO_ARTICLE,
        passage_to_clause=_TO_CLAUSE,
    )
    # Exact passage_id relevance still misses — that is the historical number.
    assert bundle.recall_at[4] == 0.0
    # Level-aware relevance sees it.
    assert bundle.article_recall_at[4] == 1.0
    assert bundle.document_recall_at[4] == 1.0
    assert bundle.wrong_document_rate == 0.0


def test_level_recalls_silently_report_zero_without_mappings() -> None:
    """Pin the failure mode: no mappings means no level recall at all.

    Not "low" — structurally 0.0, because the branches that compute them are
    skipped and the mean of an empty list is 0.0.
    """

    pred = RankedList("q1", (_CLAUSE_OF_SAME_ARTICLE,))
    bundle = evaluate_retrieval([pred], [_article_label()], cutoffs=(4, 10))
    assert bundle.article_recall_at[4] == 0.0
    assert bundle.document_recall_at[4] == 0.0
    assert bundle.clause_recall_at[4] == 0.0


def test_same_article_number_in_another_document_is_not_a_hit() -> None:
    """'Điều 76' of a different document must not be credited.

    This is the trap the legacy silver-label builder fell into by mapping
    article numbers globally across 8,512 documents.
    """

    pred = RankedList("q1", (_SAME_ARTICLE_NUMBER_OTHER_DOCUMENT,))
    bundle = evaluate_retrieval(
        [pred],
        [_article_label()],
        cutoffs=(4, 10),
        passage_to_document=_TO_DOCUMENT,
        passage_to_article=_TO_ARTICLE,
        passage_to_clause=_TO_CLAUSE,
    )
    assert bundle.article_recall_at[4] == 0.0
    assert bundle.document_recall_at[4] == 0.0
    assert bundle.wrong_document_rate == 1.0


def test_mrr_and_ndcg_stay_exact_passage_even_with_mappings() -> None:
    """The level maps feed *_recall_at only; mrr/ndcg still use relevant_ids.

    This is why the TASK 13 exit gate ("nDCG@10 or MRR@10 improves by >= 0.01")
    reads ~0 even after the mappings are wired, and why eval_retrieval.py
    reports a second, article-expanded bundle for ranking gates.
    """

    pred = RankedList("q1", (_CLAUSE_OF_SAME_ARTICLE,))
    bundle = evaluate_retrieval(
        [pred],
        [_article_label()],
        cutoffs=(4, 10),
        passage_to_document=_TO_DOCUMENT,
        passage_to_article=_TO_ARTICLE,
        passage_to_clause=_TO_CLAUSE,
    )
    assert bundle.article_recall_at[4] == 1.0
    assert bundle.mrr_at_10 == 0.0
    assert bundle.ndcg_at_10 == 0.0


def test_article_expanded_labels_make_mrr_usable() -> None:
    """What eval_retrieval.py's article_expanded bundle does, in miniature.

    Widening relevant_ids to the article's sibling passages is what turns
    mrr/ndcg into article-level numbers a ranking gate can use.
    """

    pred = RankedList("q1", (_CLAUSE_OF_SAME_ARTICLE,))
    expanded = QueryRelevance(
        "q1",
        frozenset({_ARTICLE_LABEL, _CLAUSE_OF_SAME_ARTICLE}),
        provenance="silver",
    )
    bundle = evaluate_retrieval(
        [pred],
        [expanded],
        cutoffs=(4, 10),
        passage_to_document=_TO_DOCUMENT,
        passage_to_article=_TO_ARTICLE,
        passage_to_clause=_TO_CLAUSE,
    )
    assert bundle.recall_at[4] == 1.0
    assert bundle.mrr_at_10 == 1.0
    # nDCG is below 1.0 because the ideal ranking would also surface the
    # article passage: the expanded bundle's nDCG is coverage-flavoured.
    assert 0.0 < bundle.ndcg_at_10 < 1.0
