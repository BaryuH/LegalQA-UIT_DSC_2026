"""Scope validation and rank-distribution audit for SEDAR warmup retrieval."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from legal_rag.sedar_retrieval.eval.ensemble_metrics import (
    EnsembleDiagnosticError,
    load_ranked_source,
    load_relevance_labels,
)
from legal_rag.sedar_retrieval.eval.retrieval_error_analysis import (
    PassageIdentity,
    RetrievalTrace,
    _gold_evidence,
    _load_case_scores,
    _load_predictions,
    _load_retrieval_trace,
    _passage_identities,
    _read_json_object,
)

DEFAULT_CUTOFFS: tuple[int, ...] = (10, 20, 50, 100, 200, 500)
_ALLOWED_PROVENANCE = frozenset({"silver", "unlabeled"})
_LEVELS = ("exact_passage", "article", "document", "provision")


class RetrievalRecallAuditError(ValueError):
    """Raised when retrieval recall audit inputs cannot be compared safely."""


def _unique_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted(set(cutoffs)))
    if not values or any(value <= 0 for value in values):
        raise RetrievalRecallAuditError("cutoffs must contain positive integers")
    return values


def _validate_declared_count(
    payload: Mapping[str, object],
    *,
    field: str,
    actual_count: int,
) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RetrievalRecallAuditError(f"{field} must be a non-negative integer")
    if value != actual_count:
        raise RetrievalRecallAuditError(
            f"{field}={value} does not match retrieval trace count={actual_count}"
        )
    return value


def _set_scope_summary(
    required_ids: set[str],
    actual_ids: set[str],
) -> dict[str, object]:
    missing = sorted(required_ids - actual_ids)
    extra = sorted(actual_ids - required_ids)
    return {
        "required_count": len(required_ids),
        "actual_count": len(actual_ids),
        "missing_count": len(missing),
        "missing_sample": missing[:10],
        "extra_count": len(extra),
        "extra_sample": extra[:10],
        "exact_match": not missing and not extra,
    }


def _resolve_config_path(raw_path: object, *, run_dir: Path) -> Path | None:
    if raw_path is None:
        return None
    candidate = Path(str(raw_path))
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend(
            (
                run_dir / candidate,
                run_dir.parent / candidate,
                Path.cwd() / candidate,
            )
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _first_rank(
    ranked_ids: Sequence[str],
    matching_ids: frozenset[str],
) -> int | None:
    for rank, passage_id in enumerate(_unique_preserve_order(ranked_ids), start=1):
        if passage_id in matching_ids:
            return rank
    return None


def _rank_summary(ranks: Sequence[int | None]) -> dict[str, object]:
    found = sorted(rank for rank in ranks if rank is not None)
    return {
        "found_count": len(found),
        "not_found_count": len(ranks) - len(found),
        "mean_found_rank": (sum(found) / len(found) if found else None),
        "median_found_rank": median(found) if found else None,
        "p90_found_rank": (
            found[min(len(found) - 1, math.ceil(len(found) * 0.9) - 1)]
            if found
            else None
        ),
        "max_found_rank": max(found) if found else None,
    }


def _match_sets(
    relevant_ids: frozenset[str],
    passage_map: Mapping[str, PassageIdentity],
    *,
    query_id: str,
) -> dict[str, frozenset[str]]:
    gold = _gold_evidence(relevant_ids, passage_map, query_id=query_id)
    exact = gold.relevant_ids
    documents = frozenset(
        passage_id
        for passage_id, identity in passage_map.items()
        if identity.document_id in gold.document_ids
    )
    articles = frozenset(
        passage_id
        for passage_id, identity in passage_map.items()
        if identity.article_key is not None
        and identity.article_key in gold.article_keys
    )
    provision = articles if gold.has_article_labels else documents
    return {
        "exact_passage": exact,
        "article": articles,
        "document": documents,
        "provision": provision,
    }


def _curve_entry(
    *,
    ranked_by_query: Mapping[str, Sequence[str]],
    query_ids: Sequence[str],
    match_sets: Mapping[str, Mapping[str, frozenset[str]]],
    cutoff: int,
    depth_sufficient_query_count: int,
) -> dict[str, object]:
    counts = {level: 0 for level in _LEVELS}
    coverage_values: list[float] = []
    for query_id in query_ids:
        ranked = _unique_preserve_order(ranked_by_query[query_id][:cutoff])
        targets = match_sets[query_id]
        for level in _LEVELS:
            if set(ranked) & targets[level]:
                counts[level] += 1
        relevant = targets["exact_passage"]
        if relevant:
            coverage_values.append(len(set(ranked) & relevant) / len(relevant))
    denominator = len(query_ids)
    incomplete_count = denominator - depth_sufficient_query_count
    return {
        "cutoff": cutoff,
        "labeled_query_count": denominator,
        "depth_sufficient_query_count": depth_sufficient_query_count,
        "incomplete_depth_query_count": incomplete_count,
        "is_lower_bound": incomplete_count > 0,
        "exact_passage_hit_count": counts["exact_passage"],
        "exact_passage_recall": counts["exact_passage"] / denominator
        if denominator
        else 0.0,
        "article_hit_count": counts["article"],
        "article_recall": counts["article"] / denominator if denominator else 0.0,
        "document_hit_count": counts["document"],
        "document_recall": counts["document"] / denominator if denominator else 0.0,
        "provision_hit_count": counts["provision"],
        "provision_recall": counts["provision"] / denominator if denominator else 0.0,
        "exact_evidence_coverage": (
            sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
        ),
    }


def _union_for_query(
    bm25_ids: Sequence[str],
    qwen_ids: Sequence[str],
    cutoff: int,
) -> tuple[str, ...]:
    return _unique_preserve_order((*bm25_ids[:cutoff], *qwen_ids[:cutoff]))


def _source_depth_summary(
    ranked_by_query: Mapping[str, Sequence[str]],
    query_ids: Sequence[str],
) -> dict[str, object]:
    depths = [
        len(_unique_preserve_order(ranked_by_query[query_id])) for query_id in query_ids
    ]
    return {
        "query_count": len(depths),
        "min_unique_depth": min(depths) if depths else 0,
        "max_unique_depth": max(depths) if depths else 0,
        "mean_unique_depth": sum(depths) / len(depths) if depths else 0.0,
        "zero_depth_query_count": sum(depth == 0 for depth in depths),
    }


def _source_contribution(
    *,
    bm25_ids: Mapping[str, Sequence[str]],
    qwen_ids: Mapping[str, Sequence[str]],
    query_ids: Sequence[str],
    match_sets: Mapping[str, Mapping[str, frozenset[str]]],
    cutoff: int,
) -> dict[str, object]:
    counts = Counter(
        "both"
        if (
            set(bm25_ids[query_id][:cutoff]) & match_sets[query_id]["provision"]
            and set(qwen_ids[query_id][:cutoff]) & match_sets[query_id]["provision"]
        )
        else "bm25_only"
        if set(bm25_ids[query_id][:cutoff]) & match_sets[query_id]["provision"]
        else "qwen_only"
        if set(qwen_ids[query_id][:cutoff]) & match_sets[query_id]["provision"]
        else "neither"
        for query_id in query_ids
    )
    denominator = len(query_ids)
    return {
        "cutoff": cutoff,
        "labeled_query_count": denominator,
        "counts": dict(sorted(counts.items())),
        "rates": {
            key: value / denominator if denominator else 0.0
            for key, value in sorted(counts.items())
        },
    }


def _contribution_category(
    *,
    bm25_ids: Sequence[str],
    qwen_ids: Sequence[str],
    provision_ids: frozenset[str],
    cutoff: int,
) -> str:
    bm25_hit = bool(set(bm25_ids[:cutoff]) & provision_ids)
    qwen_hit = bool(set(qwen_ids[:cutoff]) & provision_ids)
    if bm25_hit and qwen_hit:
        return "both"
    if bm25_hit:
        return "bm25_only"
    if qwen_hit:
        return "qwen_only"
    return "neither"


def _trace_alignment(
    *,
    config: Mapping[str, object],
    run_dir: Path,
    trace: Mapping[str, RetrievalTrace],
) -> tuple[dict[str, object], list[str]]:
    raw_path = config.get("retrieval_path")
    resolved = _resolve_config_path(raw_path, run_dir=run_dir)
    result: dict[str, object] = {
        "configured_path": str(raw_path) if raw_path is not None else None,
        "resolved_path": str(resolved) if resolved is not None else None,
        "status": "not_checked",
        "configured_query_count": None,
        "scope": None,
        "matched_query_count": 0,
        "mismatched_query_count": 0,
        "mismatch_sample": [],
    }
    warnings: list[str] = []
    if raw_path is None:
        warnings.append("run config does not declare retrieval_path")
        return result, warnings
    if resolved is None:
        warnings.append(
            "run config retrieval_path could not be resolved from the current host"
        )
        return result, warnings

    try:
        source = load_ranked_source(resolved, source_name="configured_ltr")
    except (OSError, ValueError, EnsembleDiagnosticError) as exc:
        warnings.append(f"configured retrieval_path could not be loaded: {exc}")
        result["status"] = "load_error"
        return result, warnings

    result["configured_query_count"] = len(source)
    result["scope"] = _set_scope_summary(set(trace), set(source))
    mismatches: list[str] = []
    matched = 0
    for query_id, trace_row in trace.items():
        configured_ids = source.get(query_id)
        if configured_ids is not None and tuple(trace_row.raw_hit_ids) == tuple(
            configured_ids
        ):
            matched += 1
        else:
            mismatches.append(query_id)
    result["status"] = "matched" if not mismatches else "mismatch"
    result["matched_query_count"] = matched
    result["mismatched_query_count"] = len(mismatches)
    result["mismatch_sample"] = mismatches[:10]
    if mismatches:
        warnings.append(
            "retrieval trace raw_hit_ids do not exactly match configured LTR rankings"
        )
    if set(trace) - set(source):
        warnings.append("configured LTR rankings are missing trace query IDs")
    return result, warnings


def audit_warmup_retrieval_recall(
    *,
    run_dir: str | Path,
    metrics_path: str | Path,
    bm25_path: str | Path,
    qwen_path: str | Path,
    labels_path: str | Path,
    passages_path: str | Path,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> RetrievalRecallAuditReport:
    """Validate clean-warmup scope and produce source rank-distribution metrics."""

    cutoff_values = _validate_cutoffs(cutoffs)
    run_path = Path(run_dir)
    try:
        config = _read_json_object(run_path / "config.json")
        run_summary = _read_json_object(run_path / "run_summary.json")
    except (OSError, ValueError) as exc:
        raise RetrievalRecallAuditError(
            f"Cannot load run metadata from {run_path}"
        ) from exc
    if config.get("split") != "warmup":
        raise RetrievalRecallAuditError("recall audit requires a warmup prediction run")
    if config.get("id_source") != "clean_manifest":
        raise RetrievalRecallAuditError(
            "recall audit requires id_source='clean_manifest'"
        )
    if config.get("retrieval_variant") != "ltr_full_all":
        raise RetrievalRecallAuditError(
            "recall audit is restricted to retrieval_variant='ltr_full_all'"
        )

    try:
        trace = _load_retrieval_trace(run_path)
        scores = _load_case_scores(Path(metrics_path))
        predictions = _load_predictions(run_path)
        bm25 = load_ranked_source(bm25_path, source_name="bm25")
        qwen = load_ranked_source(qwen_path, source_name="dense")
        labels = load_relevance_labels(labels_path)
    except (OSError, ValueError, EnsembleDiagnosticError) as exc:
        raise RetrievalRecallAuditError(
            f"Cannot load retrieval audit inputs: {exc}"
        ) from exc

    run_ids = set(trace)
    declared_selected_count = _validate_declared_count(
        config,
        field="selected_ids_count",
        actual_count=len(run_ids),
    )
    declared_prediction_count = _validate_declared_count(
        run_summary,
        field="prediction_count",
        actual_count=len(run_ids),
    )
    if set(scores) != run_ids:
        raise RetrievalRecallAuditError(
            "retrieval trace and metrics IDs do not match exactly"
        )
    if set(predictions) != run_ids:
        raise RetrievalRecallAuditError(
            "retrieval trace and prediction IDs do not match exactly"
        )
    for source_name, source in (("BM25", bm25), ("Qwen", qwen)):
        missing = sorted(run_ids - set(source))
        if missing:
            raise RetrievalRecallAuditError(
                f"{source_name} source is missing clean-warmup IDs: "
                + ", ".join(missing[:5])
            )
    missing_labels = sorted(run_ids - set(labels))
    if missing_labels:
        raise RetrievalRecallAuditError(
            "Silver labels are missing clean-warmup IDs: "
            + ", ".join(missing_labels[:5])
        )

    try:
        passage_map = _passage_identities(Path(passages_path))
    except (OSError, ValueError) as exc:
        raise RetrievalRecallAuditError(
            f"Cannot load passages: {passages_path}"
        ) from exc

    provenance_counts: Counter[str] = Counter()
    invalid_provenance: list[str] = []
    labeled_ids: list[str] = []
    match_sets: dict[str, dict[str, frozenset[str]]] = {}
    for query_id in sorted(run_ids):
        relevant_ids, provenance = labels[query_id]
        provenance_counts[provenance] += 1
        if provenance not in _ALLOWED_PROVENANCE:
            invalid_provenance.append(query_id)
            continue
        if provenance == "silver" and relevant_ids:
            labeled_ids.append(query_id)
            try:
                match_sets[query_id] = _match_sets(
                    relevant_ids,
                    passage_map,
                    query_id=query_id,
                )
            except ValueError as exc:
                raise RetrievalRecallAuditError(str(exc)) from exc
    if invalid_provenance:
        raise RetrievalRecallAuditError(
            "Unsupported label provenance for query IDs: "
            + ", ".join(invalid_provenance[:5])
        )

    unknown_ids: dict[str, list[str]] = {}
    for source_name, source in (("bm25", bm25), ("qwen", qwen)):
        flattened = [
            passage_id
            for query_id in sorted(run_ids)
            for passage_id in source[query_id]
        ]
        unknown = sorted(set(flattened) - set(passage_map))
        if unknown:
            unknown_ids[source_name] = unknown[:10]
    trace_flattened = [
        passage_id
        for query_id in sorted(run_ids)
        for passage_id in trace[query_id].raw_hit_ids
    ]
    packed_flattened = [
        passage_id
        for query_id in sorted(run_ids)
        for passage_id in trace[query_id].packed_chunk_ids
    ]
    unknown_trace = sorted(set(trace_flattened) - set(passage_map))
    unknown_packed = sorted(set(packed_flattened) - set(passage_map))
    if unknown_trace:
        unknown_ids["trace"] = unknown_trace[:10]
    if unknown_packed:
        unknown_ids["packed"] = unknown_packed[:10]
    if unknown_ids:
        details = "; ".join(
            f"{name}={values}" for name, values in sorted(unknown_ids.items())
        )
        raise RetrievalRecallAuditError(
            f"Retrieval inputs contain passage IDs absent from passages: {details}"
        )

    warnings: list[str] = []
    trace_alignment, alignment_warnings = _trace_alignment(
        config=config,
        run_dir=run_path,
        trace=trace,
    )
    warnings.extend(alignment_warnings)
    non_success_trace_count = sum(
        trace[query_id].status != "success" for query_id in run_ids
    )
    packed_not_in_raw_count = sum(
        not set(trace[query_id].packed_chunk_ids).issubset(
            set(trace[query_id].raw_hit_ids)
        )
        for query_id in run_ids
    )
    if non_success_trace_count:
        warnings.append(
            f"{non_success_trace_count} retrieval traces have non-success status"
        )
    if packed_not_in_raw_count:
        warnings.append(
            f"{packed_not_in_raw_count} traces contain packed IDs absent from raw hits"
        )

    source_depths = {
        "bm25": _source_depth_summary(bm25, sorted(run_ids)),
        "qwen": _source_depth_summary(qwen, sorted(run_ids)),
        "ltr_trace": _source_depth_summary(
            {query_id: trace[query_id].raw_hit_ids for query_id in run_ids},
            sorted(run_ids),
        ),
    }

    source_curves: dict[str, list[dict[str, object]]] = {
        "bm25": [],
        "qwen": [],
        "candidate_union_bm25_qwen": [],
    }
    contribution: list[dict[str, object]] = []
    ordered_labeled_ids = tuple(sorted(labeled_ids))
    for cutoff in cutoff_values:
        bm25_depth_sufficient = sum(
            len(_unique_preserve_order(bm25[query_id])) >= cutoff
            for query_id in ordered_labeled_ids
        )
        qwen_depth_sufficient = sum(
            len(_unique_preserve_order(qwen[query_id])) >= cutoff
            for query_id in ordered_labeled_ids
        )
        union_ranked = {
            query_id: _union_for_query(
                bm25[query_id],
                qwen[query_id],
                cutoff,
            )
            for query_id in ordered_labeled_ids
        }
        union_depth_sufficient = sum(
            len(_unique_preserve_order(bm25[query_id])) >= cutoff
            and len(_unique_preserve_order(qwen[query_id])) >= cutoff
            for query_id in ordered_labeled_ids
        )
        source_curves["bm25"].append(
            _curve_entry(
                ranked_by_query=bm25,
                query_ids=ordered_labeled_ids,
                match_sets=match_sets,
                cutoff=cutoff,
                depth_sufficient_query_count=bm25_depth_sufficient,
            )
        )
        source_curves["qwen"].append(
            _curve_entry(
                ranked_by_query=qwen,
                query_ids=ordered_labeled_ids,
                match_sets=match_sets,
                cutoff=cutoff,
                depth_sufficient_query_count=qwen_depth_sufficient,
            )
        )
        source_curves["candidate_union_bm25_qwen"].append(
            _curve_entry(
                ranked_by_query=union_ranked,
                query_ids=ordered_labeled_ids,
                match_sets=match_sets,
                cutoff=cutoff,
                depth_sufficient_query_count=union_depth_sufficient,
            )
        )
        contribution.append(
            _source_contribution(
                bm25_ids=bm25,
                qwen_ids=qwen,
                query_ids=ordered_labeled_ids,
                match_sets=match_sets,
                cutoff=cutoff,
            )
        )

    max_cutoff = cutoff_values[-1]
    union_at_max = {
        query_id: _union_for_query(
            bm25[query_id],
            qwen[query_id],
            max_cutoff,
        )
        for query_id in ordered_labeled_ids
    }
    ranked_views: dict[str, Mapping[str, Sequence[str]]] = {
        "bm25": {
            query_id: bm25[query_id][:max_cutoff] for query_id in ordered_labeled_ids
        },
        "qwen": {
            query_id: qwen[query_id][:max_cutoff] for query_id in ordered_labeled_ids
        },
        "candidate_union_bm25_qwen": union_at_max,
    }
    rank_distribution: dict[str, dict[str, object]] = {}
    for source_name, ranked in ranked_views.items():
        level_summaries: dict[str, object] = {}
        for level in _LEVELS:
            ranks = [
                _first_rank(ranked[query_id], match_sets[query_id][level])
                for query_id in ordered_labeled_ids
            ]
            level_summaries[level] = _rank_summary(ranks)
        rank_distribution[source_name] = {
            "rank_cutoff": max_cutoff,
            "levels": level_summaries,
        }

    per_case: list[dict[str, object]] = []
    all_query_ids = tuple(sorted(run_ids))
    for query_id in all_query_ids:
        relevant_ids, provenance = labels[query_id]
        row: dict[str, object] = {
            "id": query_id,
            "label_provenance": provenance,
            "relevant_id_count": len(relevant_ids),
            "meteor": scores[query_id].meteor,
            "rouge_l": scores[query_id].rouge_l,
            "metric_status": scores[query_id].status,
            "retrieval_status": trace[query_id].status,
            "source_depth": {
                "bm25": len(_unique_preserve_order(bm25[query_id])),
                "qwen": len(_unique_preserve_order(qwen[query_id])),
                "ltr_trace": len(_unique_preserve_order(trace[query_id].raw_hit_ids)),
            },
        }
        if provenance != "silver" or not relevant_ids:
            row.update(
                {
                    "first_rank": None,
                    "source_contribution": {
                        str(cutoff): "UNLABELED" for cutoff in cutoff_values
                    },
                }
            )
        else:
            query_match_sets = match_sets[query_id]
            first_rank: dict[str, dict[str, int | None]] = {}
            for source_name, ranked_ids in (
                ("bm25", bm25[query_id][:max_cutoff]),
                ("qwen", qwen[query_id][:max_cutoff]),
                ("candidate_union_bm25_qwen", union_at_max[query_id]),
            ):
                first_rank[source_name] = {
                    level: _first_rank(ranked_ids, query_match_sets[level])
                    for level in _LEVELS
                }
            row["first_rank"] = first_rank
            row["source_contribution"] = {
                str(cutoff): _contribution_category(
                    bm25_ids=bm25[query_id],
                    qwen_ids=qwen[query_id],
                    provision_ids=query_match_sets["provision"],
                    cutoff=cutoff,
                )
                for cutoff in cutoff_values
            }
        per_case.append(row)

    scope_summary = {
        "trace": _set_scope_summary(run_ids, set(trace)),
        "metrics": _set_scope_summary(run_ids, set(scores)),
        "predictions": _set_scope_summary(run_ids, set(predictions)),
        "bm25": _set_scope_summary(run_ids, set(bm25)),
        "qwen": _set_scope_summary(run_ids, set(qwen)),
        "labels": _set_scope_summary(run_ids, set(labels)),
    }
    payload: dict[str, object] = {
        "schema_version": "sedar-warmup-retrieval-recall-audit-v1",
        "split": "warmup",
        "scope": "clean_warmup",
        "reference_role": "evaluation_reference_only",
        "label_provenance": "silver_only",
        "gold_answer_text_used": False,
        "gold_answer_text_written": False,
        "run_id": str(run_summary.get("run_id", "")),
        "run_dir": str(run_path),
        "metrics_path": str(metrics_path),
        "bm25_path": str(bm25_path),
        "qwen_path": str(qwen_path),
        "labels_path": str(labels_path),
        "passages_path": str(passages_path),
        "query_count": len(run_ids),
        "labeled_query_count": len(ordered_labeled_ids),
        "unlabeled_query_count": len(run_ids) - len(ordered_labeled_ids),
        "declared_counts": {
            "config_selected_ids_count": declared_selected_count,
            "run_summary_prediction_count": declared_prediction_count,
        },
        "label_provenance_counts": dict(sorted(provenance_counts.items())),
        "cutoffs": list(cutoff_values),
        "scope_validation": scope_summary,
        "trace_validation": {
            "non_success_trace_count": non_success_trace_count,
            "packed_not_in_raw_count": packed_not_in_raw_count,
            "ltr_alignment": trace_alignment,
        },
        "source_depths": source_depths,
        "rank_distribution": rank_distribution,
        "source_contribution": contribution,
        "recall_curves": source_curves,
        "per_case": per_case,
        "warnings": warnings,
    }
    return RetrievalRecallAuditReport(payload)


@dataclass(frozen=True, slots=True)
class RetrievalRecallAuditReport:
    """Serializable retrieval recall audit result."""

    payload: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-serializable report payload."""

        return dict(self.payload)


__all__ = [
    "DEFAULT_CUTOFFS",
    "RetrievalRecallAuditError",
    "RetrievalRecallAuditReport",
    "audit_warmup_retrieval_recall",
]
