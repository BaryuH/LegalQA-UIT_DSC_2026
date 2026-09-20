"""Evaluation-only audit for silver-label scope and schema integrity."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl


class SilverLabelAuditError(ValueError):
    """Raised when silver-label audit inputs are malformed."""


@dataclass(frozen=True, slots=True)
class SilverLabelAuditReport:
    """Serializable silver-label audit result without answer text."""

    payload: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-serializable report payload."""

        return dict(self.payload)


def _load_run_query_ids(run_dir: Path | None) -> set[str] | None:
    if run_dir is None:
        return None
    retrieval_path = run_dir / "retrieval.jsonl"
    if not retrieval_path.is_file():
        raise SilverLabelAuditError(
            f"Champion retrieval trace does not exist: {retrieval_path}"
        )
    query_ids: set[str] = set()
    for line in iter_jsonl_lines(retrieval_path):
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise SilverLabelAuditError("Retrieval trace row must be an object")
        query_id = row.get("query_id", row.get("id"))
        if query_id is None:
            raise SilverLabelAuditError("Retrieval trace row lacks query_id/id")
        query_ids.add(str(query_id))
    return query_ids


def audit_silver_labels(
    *,
    labels_path: str | Path,
    passages_path: str | Path,
    run_dir: str | Path | None = None,
    sample_size: int = 20,
) -> SilverLabelAuditReport:
    """Audit document/article scope of an evaluation-only labels artifact."""

    if sample_size < 0:
        raise SilverLabelAuditError("sample_size must be non-negative")
    labels_source = Path(labels_path)
    passages_source = Path(passages_path)
    run_source = Path(run_dir) if run_dir is not None else None
    clean_query_ids = _load_run_query_ids(run_source)

    passages = load_passages_jsonl(str(passages_source))
    passage_map = {passage.passage_id: passage for passage in passages}
    if len(passage_map) != len(passages):
        raise SilverLabelAuditError("Passage corpus contains duplicate passage IDs")

    article_order: dict[str, list[str]] = defaultdict(list)
    for passage in passages:
        if passage.retrieval_level == "article" and passage.article_number:
            article_order[str(passage.article_number)].append(passage.passage_id)
    first_three_by_article = {
        article: set(passage_ids[:3]) for article, passage_ids in article_order.items()
    }

    label_schema_versions: Counter[str] = Counter()
    relevant_count_histogram: Counter[int] = Counter()
    document_count_histogram: Counter[int] = Counter()
    article_count_histogram: Counter[int] = Counter()
    missing_passages: list[dict[str, str]] = []
    multi_document_sample: list[dict[str, object]] = []
    scope_violations: list[dict[str, object]] = []
    unresolved_sample: list[dict[str, object]] = []
    resolution_reason_counts: Counter[str] = Counter()
    resolution_reason_query_counts: Counter[str] = Counter()
    resolution_reason_query_id_sample: dict[str, list[str]] = defaultdict(list)
    unresolved_query_reason_counts: Counter[str] = Counter()
    unresolved_query_id_sample_by_reason: dict[str, list[str]] = defaultdict(list)
    seen_query_ids: set[str] = set()
    silver_query_count = 0
    known_label_ids = 0
    first_three_hits = 0
    selected_query_count = 0
    scope_violation_count = 0
    unresolved_resolution_metadata_missing_count = 0
    unresolved_with_article_query_count = 0

    for line in iter_jsonl_lines(labels_source):
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise SilverLabelAuditError("Label row must be an object")
        query_id_value = row.get("query_id", row.get("id"))
        if query_id_value is None:
            raise SilverLabelAuditError("Label row lacks query_id/id")
        query_id = str(query_id_value)
        if clean_query_ids is not None and query_id not in clean_query_ids:
            continue
        selected_query_count += 1
        if query_id in seen_query_ids:
            raise SilverLabelAuditError(f"Duplicate label query_id: {query_id}")
        seen_query_ids.add(query_id)
        schema_version = row.get("schema_version")
        if schema_version is not None:
            label_schema_versions[str(schema_version)] += 1
        is_unlabeled = row.get("provenance") != "silver"
        raw_reasons = row.get("resolution_reasons")
        query_reasons: set[str] = set()
        if isinstance(raw_reasons, Mapping):
            for reason, count in raw_reasons.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise SilverLabelAuditError(
                        f"resolution_reasons counts must be non-negative integers "
                        f"for query {query_id}"
                    )
                normalized_reason = str(reason)
                resolution_reason_counts[normalized_reason] += count
                if count:
                    query_reasons.add(normalized_reason)
        elif raw_reasons is not None:
            raise SilverLabelAuditError(
                f"resolution_reasons must be an object for query {query_id}"
            )

        if is_unlabeled and not query_reasons:
            note = row.get("note")
            if note is not None:
                normalized_note = str(note)
                resolution_reason_counts[normalized_note] += 1
                query_reasons.add(normalized_note)
            else:
                unresolved_resolution_metadata_missing_count += 1

        for reason in sorted(query_reasons):
            resolution_reason_query_counts[reason] += 1
            if len(resolution_reason_query_id_sample[reason]) < sample_size:
                resolution_reason_query_id_sample[reason].append(query_id)

        if is_unlabeled:
            for reason in sorted(query_reasons):
                unresolved_query_reason_counts[reason] += 1
                if len(unresolved_query_id_sample_by_reason[reason]) < sample_size:
                    unresolved_query_id_sample_by_reason[reason].append(query_id)
            if any(reason != "no_article_citation" for reason in query_reasons):
                unresolved_with_article_query_count += 1
            if len(unresolved_sample) < sample_size:
                sample_row: dict[str, object] = {
                    "query_id": query_id,
                    "note": str(row.get("note") or ""),
                }
                if isinstance(raw_reasons, Mapping):
                    sample_row["resolution_reasons"] = {
                        str(reason): int(count)
                        for reason, count in sorted(raw_reasons.items())
                    }
                raw_scopes = row.get("resolved_scopes")
                if isinstance(raw_scopes, list):
                    sample_row["resolved_scopes"] = raw_scopes
                raw_unresolved_scopes = row.get("unresolved_scopes")
                if raw_unresolved_scopes is not None:
                    if not isinstance(raw_unresolved_scopes, list):
                        raise SilverLabelAuditError(
                            f"unresolved_scopes must be a list for query {query_id}"
                        )
                    sample_row["unresolved_scopes"] = raw_unresolved_scopes
                unresolved_sample.append(sample_row)
            continue

        silver_query_count += 1
        raw_relevant_ids = row.get("relevant_ids", [])
        if not isinstance(raw_relevant_ids, list):
            raise SilverLabelAuditError(
                f"relevant_ids must be a list for query {query_id}"
            )
        relevant_ids = [str(value) for value in raw_relevant_ids]
        relevant_count_histogram[len(relevant_ids)] += 1
        infos = []
        for passage_id in relevant_ids:
            matched_passage = passage_map.get(passage_id)
            if matched_passage is None:
                missing_passages.append(
                    {"query_id": query_id, "passage_id": passage_id}
                )
                continue
            known_label_ids += 1
            article = str(matched_passage.article_number or "")
            if passage_id in first_three_by_article.get(article, set()):
                first_three_hits += 1
            infos.append(matched_passage)

        documents = {
            str(passage.document_id)
            for passage in infos
            if passage.document_id is not None
        }
        articles = {
            (str(passage.document_id), str(passage.article_number))
            for passage in infos
            if passage.document_id is not None and passage.article_number is not None
        }
        document_count_histogram[len(documents)] += 1
        article_count_histogram[len(articles)] += 1
        if len(documents) > 1 and len(multi_document_sample) < sample_size:
            multi_document_sample.append(
                {
                    "query_id": query_id,
                    "document_ids": sorted(documents),
                    "article_keys": sorted(
                        f"{document_id}::{article}" for document_id, article in articles
                    ),
                    "relevant_id_count": len(relevant_ids),
                }
            )

        raw_scopes = row.get("resolved_scopes")
        if raw_scopes is None:
            continue
        if not isinstance(raw_scopes, list):
            raise SilverLabelAuditError(
                f"resolved_scopes must be a list for query {query_id}"
            )
        declared_scopes = {
            (
                str(scope.get("document_id")),
                str(scope.get("article_number")),
            )
            for scope in raw_scopes
            if isinstance(scope, Mapping)
            and scope.get("document_id") is not None
            and scope.get("article_number") is not None
        }
        undeclared = sorted(articles - declared_scopes)
        if undeclared:
            scope_violation_count += 1
            if len(scope_violations) < sample_size:
                scope_violations.append(
                    {
                        "query_id": query_id,
                        "undeclared_article_keys": [
                            f"{document_id}::{article}"
                            for document_id, article in undeclared
                        ],
                    }
                )

    warnings: list[str] = []
    if missing_passages:
        warnings.append("labels_reference_missing_passages")
    if scope_violations:
        warnings.append("v2_labels_have_scope_violations")
    if unresolved_resolution_metadata_missing_count:
        warnings.append("unresolved_labels_missing_resolution_metadata")
    if clean_query_ids is not None and selected_query_count != len(clean_query_ids):
        warnings.append("labels_missing_clean_query_ids")
    report = {
        "schema_version": "sedar-silver-label-audit-v1",
        "gold_answer_text_written": False,
        "clean_query_count": (
            len(clean_query_ids)
            if clean_query_ids is not None
            else selected_query_count
        ),
        "selected_query_count": selected_query_count,
        "silver_query_count": silver_query_count,
        "unlabeled_query_count": selected_query_count - silver_query_count,
        "unresolved_with_article_query_count": unresolved_with_article_query_count,
        "unresolved_resolution_metadata_missing_count": (
            unresolved_resolution_metadata_missing_count
        ),
        "label_schema_versions": dict(sorted(label_schema_versions.items())),
        "resolution_reason_counts": dict(sorted(resolution_reason_counts.items())),
        "resolution_reason_query_counts": dict(
            sorted(resolution_reason_query_counts.items())
        ),
        "resolution_reason_query_id_sample": dict(
            sorted(resolution_reason_query_id_sample.items())
        ),
        "unresolved_query_reason_counts": dict(
            sorted(unresolved_query_reason_counts.items())
        ),
        "unresolved_query_id_sample_by_reason": dict(
            sorted(unresolved_query_id_sample_by_reason.items())
        ),
        "unresolved_query_sample": unresolved_sample,
        "relevant_id_count_histogram": dict(sorted(relevant_count_histogram.items())),
        "document_count_histogram": dict(sorted(document_count_histogram.items())),
        "article_count_histogram": dict(sorted(article_count_histogram.items())),
        "multi_document_query_count": sum(
            count
            for document_count, count in document_count_histogram.items()
            if document_count > 1
        ),
        "multi_document_sample": multi_document_sample,
        "missing_passage_id_count": len(missing_passages),
        "missing_passage_sample": missing_passages[:sample_size],
        "v2_scope_violation_count": scope_violation_count,
        "v2_scope_violation_sample": scope_violations,
        "selected_from_global_article_first_three_rate": (
            first_three_hits / known_label_ids if known_label_ids else None
        ),
        "warnings": warnings,
    }
    return SilverLabelAuditReport(payload=report)
