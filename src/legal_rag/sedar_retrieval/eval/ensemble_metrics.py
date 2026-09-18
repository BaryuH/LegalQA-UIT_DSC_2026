"""Diagnostics for multi-dense retrieval ensembles."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines


class EnsembleDiagnosticError(ValueError):
    """Raised when ensemble inputs cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class EnsembleDiagnosticReport:
    """Comparable retrieval diagnostics without answer text."""

    query_count: int
    labeled_query_count: int
    unlabeled_query_count: int
    top_k: int
    cutoffs: tuple[int, ...]
    source_metrics: dict[str, dict[str, object]]
    union_metrics: dict[str, dict[str, object]]
    pairwise_metrics: dict[str, dict[str, object]]
    legal_contribution: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "sedar-retrieval-ensemble-diagnostics-v1",
            "query_count": self.query_count,
            "labeled_query_count": self.labeled_query_count,
            "unlabeled_query_count": self.unlabeled_query_count,
            "top_k": self.top_k,
            "cutoffs": list(self.cutoffs),
            "source_metrics": self.source_metrics,
            "union_metrics": self.union_metrics,
            "pairwise_metrics": self.pairwise_metrics,
            "legal_contribution": self.legal_contribution,
        }


def _unique_preserve_order(ids: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in ids:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _validate_query_id(value: object, *, field: str) -> str:
    if value is None or not str(value).strip():
        raise EnsembleDiagnosticError(f"{field} must be non-blank")
    return str(value)


def _ranked_ids_from_row(row: Mapping[str, object]) -> tuple[str, ...]:
    raw_ids = row.get("ranked_ids")
    if isinstance(raw_ids, list):
        return tuple(
            _validate_query_id(item, field="ranked_ids item") for item in raw_ids
        )

    raw_scores = row.get("scores")
    if not isinstance(raw_scores, list):
        raise EnsembleDiagnosticError(
            "Retrieval row must contain ranked_ids or a scores list"
        )
    ranked: list[tuple[int, str]] = []
    for index, item in enumerate(raw_scores, start=1):
        if not isinstance(item, Mapping):
            raise EnsembleDiagnosticError("Retrieval scores must contain objects")
        passage_id = _validate_query_id(item.get("passage_id"), field="passage_id")
        raw_rank = item.get("rank", index)
        try:
            rank = int(raw_rank)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise EnsembleDiagnosticError("Retrieval rank must be an integer") from exc
        if rank <= 0:
            raise EnsembleDiagnosticError("Retrieval rank must be positive")
        ranked.append((rank, passage_id))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(passage_id for _, passage_id in ranked)


def load_ranked_source(
    path: str | Path,
    *,
    source_name: str,
) -> dict[str, tuple[str, ...]]:
    """Load one retrieval JSONL source keyed by query ID."""

    if not source_name.strip():
        raise EnsembleDiagnosticError("source_name must be non-blank")
    result: dict[str, tuple[str, ...]] = {}
    for line in iter_jsonl_lines(path):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise EnsembleDiagnosticError("Retrieval JSONL rows must be objects")
        query_id = _validate_query_id(payload.get("query_id"), field="query_id")
        if query_id in result:
            raise EnsembleDiagnosticError(
                f"Duplicate query_id in {source_name} source: {query_id}"
            )
        result[query_id] = _ranked_ids_from_row(payload)
    if not result:
        raise EnsembleDiagnosticError(f"Retrieval source is empty: {path}")
    return result


def load_relevance_labels(
    path: str | Path,
) -> dict[str, tuple[frozenset[str], str]]:
    """Load evaluation-only relevance IDs and provenance."""

    result: dict[str, tuple[frozenset[str], str]] = {}
    for line in iter_jsonl_lines(path):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise EnsembleDiagnosticError("Label JSONL rows must be objects")
        query_id = _validate_query_id(payload.get("query_id"), field="query_id")
        if query_id in result:
            raise EnsembleDiagnosticError(f"Duplicate label query_id: {query_id}")
        raw_relevant = payload.get("relevant_ids", [])
        if not isinstance(raw_relevant, list):
            raise EnsembleDiagnosticError("relevant_ids must be a list")
        relevant = frozenset(
            _validate_query_id(item, field="relevant_ids item") for item in raw_relevant
        )
        provenance = str(payload.get("provenance", "unlabeled"))
        if provenance not in {"gold", "silver", "unlabeled"}:
            raise EnsembleDiagnosticError(f"Unsupported label provenance: {provenance}")
        result[query_id] = (relevant, provenance)
    if not result:
        raise EnsembleDiagnosticError(f"Relevance label source is empty: {path}")
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _recall(ranked_ids: Sequence[str], relevant: frozenset[str], cutoff: int) -> float:
    return 1.0 if relevant and set(ranked_ids[:cutoff]) & relevant else 0.0


def _candidate_metrics(
    ranked_by_query: Mapping[str, Sequence[str]],
    query_ids: Sequence[str],
    *,
    cutoffs: Sequence[int],
    labels: Mapping[str, tuple[frozenset[str], str]] | None,
) -> dict[str, object]:
    output: dict[str, object] = {
        "avg_candidate_count": _mean(
            [float(len(ranked_by_query[query_id])) for query_id in query_ids]
        ),
        "recall_at": {},
    }
    recall_at: dict[str, float | None] = {}
    for cutoff in cutoffs:
        if labels is None:
            recall_at[str(cutoff)] = None
            continue
        values = [
            _recall(
                _unique_preserve_order(ranked_by_query[query_id]),
                labels[query_id][0],
                cutoff,
            )
            for query_id in query_ids
            if labels[query_id][1] != "unlabeled" and labels[query_id][0]
        ]
        recall_at[str(cutoff)] = _mean(values) if values else None
    output["recall_at"] = recall_at
    return output


def _union_for_query(
    ranked_sources: Mapping[str, Mapping[str, Sequence[str]]],
    source_names: Sequence[str],
    query_id: str,
    cutoff: int,
) -> tuple[str, ...]:
    combined: list[str] = []
    for source_name in source_names:
        combined.extend(ranked_sources[source_name][query_id][:cutoff])
    return _unique_preserve_order(combined)


def _union_candidate_metrics(
    ranked_sources: Mapping[str, Mapping[str, Sequence[str]]],
    source_names: Sequence[str],
    query_ids: Sequence[str],
    *,
    cutoffs: Sequence[int],
    labels: Mapping[str, tuple[frozenset[str], str]] | None,
) -> dict[str, object]:
    """Score the full union of each source's top-K candidates at each K."""

    recall_at: dict[str, float | None] = {}
    candidate_counts = [
        float(
            len(
                _union_for_query(
                    ranked_sources,
                    source_names,
                    query_id,
                    max(cutoffs),
                )
            )
        )
        for query_id in query_ids
    ]
    for cutoff in cutoffs:
        if labels is None:
            recall_at[str(cutoff)] = None
            continue
        values = []
        for query_id in query_ids:
            relevant, provenance = labels[query_id]
            if provenance == "unlabeled" or not relevant:
                continue
            union_ids = _union_for_query(
                ranked_sources,
                source_names,
                query_id,
                cutoff,
            )
            values.append(_recall(union_ids, relevant, len(union_ids)))
        recall_at[str(cutoff)] = _mean(values) if values else None
    return {
        "avg_candidate_count": _mean(candidate_counts),
        "recall_at": recall_at,
        "union_definition": "union of each source top-K, scored over full union",
    }


def _pairwise_metrics(
    left: Mapping[str, Sequence[str]],
    right: Mapping[str, Sequence[str]],
    query_ids: Sequence[str],
    *,
    top_k: int,
) -> dict[str, object]:
    intersections: list[float] = []
    overlap_rates: list[float] = []
    jaccards: list[float] = []
    for query_id in query_ids:
        left_set = set(_unique_preserve_order(left[query_id])[:top_k])
        right_set = set(_unique_preserve_order(right[query_id])[:top_k])
        intersection = left_set & right_set
        union = left_set | right_set
        intersections.append(float(len(intersection)))
        denominator = min(len(left_set), len(right_set))
        overlap_rates.append(len(intersection) / denominator if denominator else 0.0)
        jaccards.append(len(intersection) / len(union) if union else 0.0)
    return {
        "mean_intersection_count": _mean(intersections),
        "mean_overlap_rate": _mean(overlap_rates),
        "mean_jaccard": _mean(jaccards),
        "pooled_intersection_count": sum(intersections),
    }


def diagnose_ensemble(
    sources: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    labels: Mapping[str, tuple[frozenset[str], str]] | None = None,
    top_k: int = 100,
    cutoffs: Sequence[int] = (10, 50, 100),
    primary_source: str = "dense",
    auxiliary_source: str = "legal",
) -> EnsembleDiagnosticReport:
    """Compare source quality and diversity on one fixed query set."""

    if not sources:
        raise EnsembleDiagnosticError("At least one retrieval source is required")
    if top_k <= 0:
        raise EnsembleDiagnosticError("top_k must be positive")
    normalized_cutoffs = tuple(sorted(set(int(cutoff) for cutoff in cutoffs)))
    if not normalized_cutoffs or any(cutoff <= 0 for cutoff in normalized_cutoffs):
        raise EnsembleDiagnosticError("cutoffs must contain positive integers")
    query_sets = {source: set(rows) for source, rows in sources.items()}
    query_ids = sorted(next(iter(query_sets.values())))
    expected_ids = set(query_ids)
    for source, source_ids in query_sets.items():
        if source_ids != expected_ids:
            missing = sorted(expected_ids - source_ids)
            extra = sorted(source_ids - expected_ids)
            raise EnsembleDiagnosticError(
                f"Query ID mismatch for source {source}: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
    if labels is not None:
        missing_labels = sorted(expected_ids - set(labels))
        if missing_labels:
            raise EnsembleDiagnosticError(
                f"Labels are missing query IDs: {missing_labels[:5]}"
            )

    labeled_query_count = 0
    if labels is not None:
        labeled_query_count = sum(
            1
            for query_id in query_ids
            if labels[query_id][1] != "unlabeled" and labels[query_id][0]
        )
    source_metrics = {
        source: _candidate_metrics(
            source_rows,
            query_ids,
            cutoffs=normalized_cutoffs,
            labels=labels,
        )
        for source, source_rows in sorted(sources.items())
    }

    union_metrics: dict[str, dict[str, object]] = {}
    preferred_order = [
        name for name in ("bm25", "dense", "legal", "vn_embedding") if name in sources
    ]
    if len(preferred_order) >= 2:
        combinations: list[tuple[str, ...]] = []
        for left_index in range(len(preferred_order)):
            for right_index in range(left_index + 1, len(preferred_order)):
                combinations.append(
                    (preferred_order[left_index], preferred_order[right_index])
                )
        if len(preferred_order) >= 3:
            combinations.append(tuple(preferred_order))
        for combination in combinations:
            name = "+".join(combination)
            union_metrics[name] = _union_candidate_metrics(
                sources,
                combination,
                query_ids,
                cutoffs=normalized_cutoffs,
                labels=labels,
            )

    pairwise_metrics: dict[str, dict[str, object]] = {}
    source_names = sorted(sources)
    for left_index, left_name in enumerate(source_names):
        for right_name in source_names[left_index + 1 :]:
            pairwise_metrics[f"{left_name}~{right_name}"] = _pairwise_metrics(
                sources[left_name],
                sources[right_name],
                query_ids,
                top_k=top_k,
            )

    legal_contribution: dict[str, object] | None = None
    if primary_source in sources and auxiliary_source in sources:
        unique_counts: list[float] = []
        labeled_unique_counts: list[float] = []
        labeled_relevant_counts: list[float] = []
        unique_recall_values: list[float] = []
        for query_id in query_ids:
            primary_ids = set(
                _unique_preserve_order(sources[primary_source][query_id])[:top_k]
            )
            auxiliary_ids = set(
                _unique_preserve_order(sources[auxiliary_source][query_id])[:top_k]
            )
            unique_ids = auxiliary_ids - primary_ids
            unique_counts.append(float(len(unique_ids)))
            if labels is not None and labels[query_id][1] != "unlabeled":
                relevant = labels[query_id][0]
                relevant_count = len(unique_ids & relevant)
                labeled_unique_counts.append(float(len(unique_ids)))
                labeled_relevant_counts.append(float(relevant_count))
                unique_recall_values.append(1.0 if relevant_count else 0.0)
        denominator = sum(labeled_unique_counts)
        legal_contribution = {
            "primary_source": primary_source,
            "auxiliary_source": auxiliary_source,
            "top_k": top_k,
            "mean_unique_candidate_count": _mean(unique_counts),
            "pooled_unique_candidate_count": sum(unique_counts),
            "pooled_unique_relevant_count": sum(labeled_relevant_counts),
            "unique_legal_candidate_count": sum(unique_counts),
            "unique_legal_relevant_count": sum(labeled_relevant_counts),
            "legal_only_relevant_rate": (
                sum(labeled_relevant_counts) / denominator if denominator else None
            ),
            "unique_legal_recall": (
                _mean(unique_recall_values) if unique_recall_values else None
            ),
            "labeled_unique_candidate_count": denominator,
            "labeled_unique_relevant_count": sum(labeled_relevant_counts),
        }

    return EnsembleDiagnosticReport(
        query_count=len(query_ids),
        labeled_query_count=labeled_query_count,
        unlabeled_query_count=len(query_ids) - labeled_query_count,
        top_k=top_k,
        cutoffs=normalized_cutoffs,
        source_metrics=source_metrics,
        union_metrics=union_metrics,
        pairwise_metrics=pairwise_metrics,
        legal_contribution=legal_contribution,
    )


__all__ = [
    "EnsembleDiagnosticError",
    "EnsembleDiagnosticReport",
    "diagnose_ensemble",
    "load_ranked_source",
    "load_relevance_labels",
]
