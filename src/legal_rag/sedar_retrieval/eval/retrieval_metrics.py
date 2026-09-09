"""Deterministic retrieval metrics independent of the reader.

Metric definitions (SEDAR Retrieval v3 / TASK 02):

Recall@K
  Fraction of queries with at least one relevant passage in the top-K unique
  canonical passage IDs (duplicates collapsed).

MRR@K
  Mean reciprocal rank of the first relevant unique passage within top-K.
  Queries with no relevant hit contribute 0.

nDCG@K
  Mean normalized discounted cumulative gain over unique top-K IDs using graded
  relevance gains ``2^rel - 1``. Binary labels use rel in {0,1}.

Document / Article / Clause Recall@K
  Same as Recall@K after mapping each retrieved passage ID through the provided
  level mapping (document_id / article_id / clause_id). Missing mappings are
  ignored for that level (not counted as hits).

Multi-Hit@K
  Fraction of multi-relevant queries for which at least ``min(2, |R_q|)`` distinct
  relevant IDs appear in top-K.

All-Evidence Recall / Evidence Coverage
  Mean over queries of ``|retrieved_relevant ∩ R_q| / |R_q|`` at cutoff K
  (unique IDs). Unlabeled queries are excluded from the mean.

Wrong Document Rate
  Fraction of queries where the top-1 retrieved document ID is present and not
  in the gold/silver relevant document set. Unlabeled queries are excluded.

Duplicate Rate
  Mean over queries of ``1 - unique_count / raw_count`` for the returned ranked
  list before dedup (0 when empty).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Literal

LabelProvenance = Literal["gold", "silver", "unlabeled"]


@dataclass(frozen=True, slots=True)
class QueryRelevance:
    """Relevance labels for one query with explicit provenance."""

    query_id: str
    relevant_ids: frozenset[str]
    graded: Mapping[str, int] | None = None
    provenance: LabelProvenance = "gold"
    relevant_document_ids: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class RankedList:
    """Retrieval output for one query (may contain duplicate passage IDs)."""

    query_id: str
    ranked_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalMetricBundle:
    """Machine-readable retrieval metrics for one evaluation run."""

    n_queries: int
    n_labeled_queries: int
    n_unlabeled_queries: int
    recall_at: dict[int, float]
    mrr_at_10: float
    ndcg_at_10: float
    document_recall_at: dict[int, float]
    article_recall_at: dict[int, float]
    clause_recall_at: dict[int, float]
    multi_hit_at: dict[int, float]
    evidence_coverage_at: dict[int, float]
    wrong_document_rate: float
    duplicate_rate: float
    avg_candidates_per_query: float
    label_provenance_counts: dict[str, int]


def _unique_preserve_order(ids: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _dcg(gains: Sequence[float]) -> float:
    total = 0.0
    for index, gain in enumerate(gains, start=1):
        total += gain / math.log2(index + 1)
    return total


def _relevance_gain(rel: int) -> float:
    if rel < 0:
        raise ValueError("graded relevance must be non-negative")
    return float((1 << rel) - 1)


def recall_at_k(
    ranked_unique: Sequence[str], relevant: frozenset[str], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in ranked_unique[:k]) else 0.0


def reciprocal_rank_at_k(
    ranked_unique: Sequence[str], relevant: frozenset[str], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    for index, item in enumerate(ranked_unique[:k], start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(
    ranked_unique: Sequence[str],
    relevant: frozenset[str],
    k: int,
    graded: Mapping[str, int] | None = None,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0

    def grade(item: str) -> int:
        if graded is not None and item in graded:
            return graded[item]
        return 1 if item in relevant else 0

    gains = [_relevance_gain(grade(item)) for item in ranked_unique[:k]]
    dcg = _dcg(gains)
    if graded is not None:
        ideal_grades = sorted(
            (graded.get(item, 0) for item in relevant),
            reverse=True,
        )
    else:
        ideal_grades = [1] * len(relevant)
    ideal = _dcg([_relevance_gain(g) for g in ideal_grades[:k]])
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def evidence_coverage_at_k(
    ranked_unique: Sequence[str], relevant: frozenset[str], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    hit = sum(1 for item in ranked_unique[:k] if item in relevant)
    return hit / len(relevant)


def multi_hit_at_k(
    ranked_unique: Sequence[str], relevant: frozenset[str], k: int
) -> float | None:
    """Return 1/0 for multi-relevant queries; None when |R|<2."""

    if len(relevant) < 2:
        return None
    need = min(2, len(relevant))
    hit = sum(1 for item in ranked_unique[:k] if item in relevant)
    return 1.0 if hit >= need else 0.0


def duplicate_rate(ranked_ids: Sequence[str]) -> float:
    if not ranked_ids:
        return 0.0
    return 1.0 - (len(set(ranked_ids)) / len(ranked_ids))


def evaluate_retrieval(
    predictions: Sequence[RankedList],
    labels: Sequence[QueryRelevance],
    *,
    cutoffs: Sequence[int] = (5, 10, 20, 50),
    passage_to_document: Mapping[str, str] | None = None,
    passage_to_article: Mapping[str, str] | None = None,
    passage_to_clause: Mapping[str, str] | None = None,
    mrr_cutoff: int = 10,
    ndcg_cutoff: int = 10,
    per_query_sink: MutableMapping[str, dict[str, float]] | None = None,
) -> RetrievalMetricBundle:
    """Evaluate ranked lists against labeled relevance with stable averages.

    ``per_query_sink``, when given, is filled with one row per LABELED query:
    ``{query_id: {"recall_at_4": 0.0|1.0, "article_at_4": ..., "rr_at_10": ...}}``.
    Every aggregate this function returns is a plain mean over those rows, so the
    rows are the paired sample a paired bootstrap needs - without them, adoption
    decisions can only compare aggregates, which is the unpaired test this
    project's gates explicitly do not accept. Purely additive: the default None
    leaves every returned value bit-identical.
    """

    label_by_id = {item.query_id: item for item in labels}
    if len(label_by_id) != len(labels):
        raise ValueError("Duplicate query_id in labels")

    pred_by_id = {item.query_id: item for item in predictions}
    if len(pred_by_id) != len(predictions):
        raise ValueError("Duplicate query_id in predictions")

    missing = sorted(set(label_by_id) - set(pred_by_id))
    if missing:
        raise ValueError(f"Missing predictions for query_ids: {missing[:5]}")

    recall: dict[int, list[float]] = {k: [] for k in cutoffs}
    doc_recall: dict[int, list[float]] = {k: [] for k in cutoffs}
    art_recall: dict[int, list[float]] = {k: [] for k in cutoffs}
    clause_recall: dict[int, list[float]] = {k: [] for k in cutoffs}
    multi_hit: dict[int, list[float]] = {k: [] for k in cutoffs}
    coverage: dict[int, list[float]] = {k: [] for k in cutoffs}
    mrr_values: list[float] = []
    ndcg_values: list[float] = []
    wrong_doc: list[float] = []
    dup_rates: list[float] = []
    candidate_counts: list[float] = []
    provenance_counts = {"gold": 0, "silver": 0, "unlabeled": 0}

    ordered_query_ids = sorted(pred_by_id)
    for query_id in ordered_query_ids:
        pred = pred_by_id[query_id]
        label = label_by_id.get(query_id)
        if label is None:
            raise ValueError(f"Missing label provenance for query_id={query_id}")
        provenance_counts[label.provenance] = (
            provenance_counts.get(label.provenance, 0) + 1
        )
        ranked_unique = _unique_preserve_order(pred.ranked_ids)
        dup_rates.append(duplicate_rate(pred.ranked_ids))
        candidate_counts.append(float(len(pred.ranked_ids)))

        if label.provenance == "unlabeled" or not label.relevant_ids:
            continue

        row: dict[str, float] | None = {} if per_query_sink is not None else None

        for k in cutoffs:
            recall_k = recall_at_k(ranked_unique, label.relevant_ids, k)
            recall[k].append(recall_k)
            if row is not None:
                row[f"recall_at_{k}"] = recall_k
            coverage[k].append(
                evidence_coverage_at_k(ranked_unique, label.relevant_ids, k)
            )
            mh = multi_hit_at_k(ranked_unique, label.relevant_ids, k)
            if mh is not None:
                multi_hit[k].append(mh)

            if passage_to_document is not None:
                retrieved_docs = {
                    passage_to_document[pid]
                    for pid in ranked_unique[:k]
                    if pid in passage_to_document
                }
                gold_docs = label.relevant_document_ids or frozenset(
                    passage_to_document[pid]
                    for pid in label.relevant_ids
                    if pid in passage_to_document
                )
                doc_hit = 1.0 if gold_docs and retrieved_docs & gold_docs else 0.0
                doc_recall[k].append(doc_hit)
                if row is not None:
                    row[f"document_at_{k}"] = doc_hit
            if passage_to_article is not None:
                retrieved_arts = {
                    passage_to_article[pid]
                    for pid in ranked_unique[:k]
                    if pid in passage_to_article
                }
                gold_arts = frozenset(
                    passage_to_article[pid]
                    for pid in label.relevant_ids
                    if pid in passage_to_article
                )
                art_hit = 1.0 if gold_arts and retrieved_arts & gold_arts else 0.0
                art_recall[k].append(art_hit)
                if row is not None:
                    row[f"article_at_{k}"] = art_hit
            if passage_to_clause is not None:
                retrieved_cls = {
                    passage_to_clause[pid]
                    for pid in ranked_unique[:k]
                    if pid in passage_to_clause
                }
                gold_cls = frozenset(
                    passage_to_clause[pid]
                    for pid in label.relevant_ids
                    if pid in passage_to_clause
                )
                clause_recall[k].append(
                    1.0 if gold_cls and retrieved_cls & gold_cls else 0.0
                )

        rr_value = reciprocal_rank_at_k(
            ranked_unique, label.relevant_ids, mrr_cutoff
        )
        mrr_values.append(rr_value)
        ndcg_value = ndcg_at_k(
            ranked_unique,
            label.relevant_ids,
            ndcg_cutoff,
            graded=label.graded,
        )
        ndcg_values.append(ndcg_value)
        if row is not None:
            row[f"rr_at_{mrr_cutoff}"] = rr_value
            row[f"ndcg_at_{ndcg_cutoff}"] = ndcg_value
            assert per_query_sink is not None
            per_query_sink[query_id] = row

        if passage_to_document is not None and ranked_unique:
            top_doc = passage_to_document.get(ranked_unique[0])
            gold_docs = label.relevant_document_ids or frozenset(
                passage_to_document[pid]
                for pid in label.relevant_ids
                if pid in passage_to_document
            )
            if top_doc is not None and gold_docs:
                wrong_doc.append(0.0 if top_doc in gold_docs else 1.0)

    def _mean(values: Iterable[float]) -> float:
        items = list(values)
        if not items:
            return 0.0
        return sum(items) / len(items)

    n_queries = len(ordered_query_ids)
    n_unlabeled = provenance_counts.get("unlabeled", 0)
    return RetrievalMetricBundle(
        n_queries=n_queries,
        n_labeled_queries=n_queries - n_unlabeled,
        n_unlabeled_queries=n_unlabeled,
        recall_at={k: _mean(recall[k]) for k in cutoffs},
        mrr_at_10=_mean(mrr_values),
        ndcg_at_10=_mean(ndcg_values),
        document_recall_at={k: _mean(doc_recall[k]) for k in cutoffs},
        article_recall_at={k: _mean(art_recall[k]) for k in cutoffs},
        clause_recall_at={k: _mean(clause_recall[k]) for k in cutoffs},
        multi_hit_at={k: _mean(multi_hit[k]) for k in cutoffs},
        evidence_coverage_at={k: _mean(coverage[k]) for k in cutoffs},
        wrong_document_rate=_mean(wrong_doc),
        duplicate_rate=_mean(dup_rates),
        avg_candidates_per_query=_mean(candidate_counts),
        label_provenance_counts=dict(sorted(provenance_counts.items())),
    )


def metrics_to_dict(bundle: RetrievalMetricBundle) -> dict[str, object]:
    """Serialize metrics with stable key ordering."""

    return {
        "n_queries": bundle.n_queries,
        "n_labeled_queries": bundle.n_labeled_queries,
        "n_unlabeled_queries": bundle.n_unlabeled_queries,
        "recall_at": {str(k): v for k, v in sorted(bundle.recall_at.items())},
        "mrr_at_10": bundle.mrr_at_10,
        "ndcg_at_10": bundle.ndcg_at_10,
        "document_recall_at": {
            str(k): v for k, v in sorted(bundle.document_recall_at.items())
        },
        "article_recall_at": {
            str(k): v for k, v in sorted(bundle.article_recall_at.items())
        },
        "clause_recall_at": {
            str(k): v for k, v in sorted(bundle.clause_recall_at.items())
        },
        "multi_hit_at": {str(k): v for k, v in sorted(bundle.multi_hit_at.items())},
        "evidence_coverage_at": {
            str(k): v for k, v in sorted(bundle.evidence_coverage_at.items())
        },
        "wrong_document_rate": bundle.wrong_document_rate,
        "duplicate_rate": bundle.duplicate_rate,
        "avg_candidates_per_query": bundle.avg_candidates_per_query,
        "label_provenance_counts": bundle.label_provenance_counts,
    }
