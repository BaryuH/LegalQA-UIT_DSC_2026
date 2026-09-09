"""Acceptance tests for the per-query sink on ``evaluate_retrieval``.

Every adoption gate in this project asks for a paired bootstrap, and a paired
bootstrap needs the per-query sample. The retrieval evaluator used to emit
aggregates only, so the gate was not satisfiable at all. These tests pin the
two properties that make the emitted rows trustworthy as that sample: the sink
changes nothing, and every aggregate is exactly the mean of the rows.
"""

from __future__ import annotations

import random

from legal_rag.sedar_retrieval.eval.retrieval_metrics import (
    QueryRelevance,
    RankedList,
    evaluate_retrieval,
    metrics_to_dict,
)

CUTOFFS = (4, 10, 20)


def _fixture(seed: int = 7) -> tuple[
    list[RankedList], list[QueryRelevance], dict[str, str], dict[str, str]
]:
    """A synthetic corpus with 2 clauses per article and 4 articles per document."""

    rng = random.Random(seed)
    passage_to_document: dict[str, str] = {}
    passage_to_article: dict[str, str] = {}
    passage_ids: list[str] = []
    for document in range(5):
        for article in range(4):
            for clause in range(2):
                passage_id = f"{document}::art::{article}::c{clause}"
                passage_ids.append(passage_id)
                passage_to_document[passage_id] = str(document)
                passage_to_article[passage_id] = f"{document}::art::{article}"

    predictions: list[RankedList] = []
    labels: list[QueryRelevance] = []
    for index in range(60):
        query_id = f"q{index}"
        predictions.append(
            RankedList(query_id=query_id, ranked_ids=tuple(rng.sample(passage_ids, 20)))
        )
        # Every 13th query is unlabeled, and unlabeled queries must not produce a
        # row: an unlabeled case contributes to no aggregate, so a row for it
        # would silently change the denominator of the paired test.
        provenance = "unlabeled" if index % 13 == 0 else ("gold" if index % 2 else "silver")
        relevant = (
            frozenset()
            if provenance == "unlabeled"
            else frozenset(rng.sample(passage_ids, rng.choice([1, 2, 3])))
        )
        labels.append(
            QueryRelevance(
                query_id=query_id,
                relevant_ids=relevant,
                provenance=provenance,  # type: ignore[arg-type]
            )
        )
    return predictions, labels, passage_to_document, passage_to_article


def _evaluate(sink: dict[str, dict[str, float]] | None = None):
    predictions, labels, to_document, to_article = _fixture()
    return evaluate_retrieval(
        predictions,
        labels,
        cutoffs=CUTOFFS,
        passage_to_document=to_document,
        passage_to_article=to_article,
        passage_to_clause=None,
        mrr_cutoff=10,
        ndcg_cutoff=10,
        per_query_sink=sink,
    )


def test_sink_does_not_change_any_aggregate() -> None:
    """Passing a sink must be observationally identical to not passing one."""

    sink: dict[str, dict[str, float]] = {}
    assert metrics_to_dict(_evaluate(sink)) == metrics_to_dict(_evaluate(None))


def test_one_row_per_labeled_query_and_none_for_unlabeled() -> None:
    sink: dict[str, dict[str, float]] = {}
    bundle = _evaluate(sink)
    assert len(sink) == bundle.n_labeled_queries
    assert bundle.n_unlabeled_queries > 0, "fixture must exercise the unlabeled path"
    assert len(sink) < bundle.n_queries


def test_every_aggregate_is_the_mean_of_the_rows() -> None:
    """The rows are the sample the aggregates are computed from, not a re-derivation."""

    sink: dict[str, dict[str, float]] = {}
    bundle = _evaluate(sink)
    rows = list(sink.values())

    def mean_of(field: str) -> float:
        return sum(row[field] for row in rows) / len(rows)

    for cutoff in CUTOFFS:
        assert bundle.recall_at[cutoff] == mean_of(f"recall_at_{cutoff}")
        assert bundle.article_recall_at[cutoff] == mean_of(f"article_at_{cutoff}")
        assert bundle.document_recall_at[cutoff] == mean_of(f"document_at_{cutoff}")
        assert bundle.evidence_coverage_at[cutoff] == mean_of(f"coverage_at_{cutoff}")
    assert bundle.mrr_at_10 == mean_of("rr_at_10")
    assert bundle.ndcg_at_10 == mean_of("ndcg_at_10")


def test_rows_are_keyed_by_the_configured_cutoffs() -> None:
    """A row must carry the cutoffs actually evaluated, not a hardcoded set."""

    predictions, labels, to_document, to_article = _fixture()
    sink: dict[str, dict[str, float]] = {}
    evaluate_retrieval(
        predictions,
        labels,
        cutoffs=(3, 7),
        passage_to_document=to_document,
        passage_to_article=to_article,
        mrr_cutoff=5,
        ndcg_cutoff=5,
        per_query_sink=sink,
    )
    row = next(iter(sink.values()))
    assert "coverage_at_3" in row and "coverage_at_7" in row
    assert "article_at_3" in row and "article_at_7" in row
    assert "rr_at_5" in row and "ndcg_at_5" in row
    assert "article_at_4" not in row
    assert "rr_at_10" not in row


def test_article_and_document_fields_absent_without_level_maps() -> None:
    """No level map means no level row - a 0.0 there would read as a miss."""

    predictions, labels, _, _ = _fixture()
    sink: dict[str, dict[str, float]] = {}
    evaluate_retrieval(
        predictions,
        labels,
        cutoffs=CUTOFFS,
        mrr_cutoff=10,
        ndcg_cutoff=10,
        per_query_sink=sink,
    )
    row = next(iter(sink.values()))
    assert "recall_at_4" in row
    assert "coverage_at_4" in row  # needs no level map
    assert "article_at_4" not in row
    assert "document_at_4" not in row
