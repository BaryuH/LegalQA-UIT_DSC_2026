"""Warmup-only retrieval/evidence error analysis for SEDAR runs."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage
from legal_rag.sedar_retrieval.eval.ensemble_metrics import (
    EnsembleDiagnosticError,
    load_ranked_source,
    load_relevance_labels,
)
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines
from legal_rag.sedar_retrieval.retrieval.passage_adapter import (
    load_passages_jsonl,
)


class RetrievalErrorAnalysisError(ValueError):
    """Raised when a warmup retrieval trace cannot be analyzed safely."""


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Question-level trace written by the SEDAR E2E runner."""

    status: str
    raw_hit_ids: tuple[str, ...]
    packed_chunk_ids: tuple[str, ...]
    packed_dropped_ids: tuple[str, ...]
    packed_truncated_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaseScore:
    """Evaluation scores needed for local diagnostics."""

    meteor: float | None
    rouge_l: float | None
    status: str


@dataclass(frozen=True, slots=True)
class PassageIdentity:
    """Stable identifiers used to compare silver labels with passages."""

    passage_id: str
    document_id: str
    article_key: tuple[str, str] | None


@dataclass(frozen=True, slots=True)
class GoldEvidence:
    """Silver relevance IDs projected to document/article identities."""

    relevant_ids: frozenset[str]
    document_ids: frozenset[str]
    article_keys: frozenset[tuple[str, str]]

    @property
    def has_article_labels(self) -> bool:
        return bool(self.article_keys)


@dataclass(frozen=True, slots=True)
class RetrievalErrorAnalysisReport:
    """Content-bounded A/B/C diagnosis for one clean warmup run."""

    payload: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-serializable report payload."""

        return dict(self.payload)


_TOKEN_RE = re.compile(r"\S+", flags=re.UNICODE)


def _read_json_object(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetrievalErrorAnalysisError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RetrievalErrorAnalysisError(f"JSON artifact must be an object: {path}")
    return payload


def _read_jsonl_objects(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        records: list[Mapping[str, object]] = []
        for line_number, line in enumerate(iter_jsonl_lines(path), start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RetrievalErrorAnalysisError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise RetrievalErrorAnalysisError(
                    f"JSONL row must be an object at {path}:{line_number}"
                )
            records.append(payload)
    except OSError as exc:
        raise RetrievalErrorAnalysisError(
            f"Cannot read JSONL artifact: {path}"
        ) from exc
    return tuple(records)


def _row_id(row: Mapping[str, object], *, location: str) -> str:
    raw_id = row.get("id", row.get("query_id"))
    if raw_id is None:
        raise RetrievalErrorAnalysisError(f"{location} is missing id/query_id")
    identifier = str(raw_id)
    if not identifier.strip():
        raise RetrievalErrorAnalysisError(f"{location} has a blank id/query_id")
    return identifier


def _id_tuple(
    value: object,
    *,
    field: str,
    query_id: str,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        if required:
            raise RetrievalErrorAnalysisError(
                f"Retrieval trace {query_id} is missing {field}"
            )
        return ()
    if not isinstance(value, list):
        raise RetrievalErrorAnalysisError(
            f"Retrieval trace {query_id}.{field} must be a list"
        )
    result: list[str] = []
    seen: set[str] = set()
    for position, item in enumerate(value):
        identifier = str(item)
        if not identifier.strip():
            raise RetrievalErrorAnalysisError(
                f"Blank {field}[{position}] for retrieval trace {query_id}"
            )
        if identifier in seen:
            raise RetrievalErrorAnalysisError(
                f"Duplicate {field} ID {identifier!r} for retrieval trace {query_id}"
            )
        seen.add(identifier)
        result.append(identifier)
    return tuple(result)


def _load_retrieval_trace(run_dir: Path) -> dict[str, RetrievalTrace]:
    path = run_dir / "retrieval.jsonl"
    rows = _read_jsonl_objects(path)
    result: dict[str, RetrievalTrace] = {}
    for index, row in enumerate(rows):
        query_id = _row_id(row, location=f"retrieval[{index}]")
        if query_id in result:
            raise RetrievalErrorAnalysisError(
                f"Duplicate retrieval trace id: {query_id}"
            )
        raw_hit_ids = row.get("raw_hit_ids")
        if raw_hit_ids is None:
            raw_hit_ids = row.get("ranked_ids")
        packed_chunk_ids = _id_tuple(
            row.get("packed_chunk_ids"),
            field="packed_chunk_ids",
            query_id=query_id,
            required=True,
        )
        result[query_id] = RetrievalTrace(
            status=str(row.get("status", "success")),
            raw_hit_ids=_id_tuple(
                raw_hit_ids,
                field="raw_hit_ids",
                query_id=query_id,
                required=True,
            ),
            packed_chunk_ids=packed_chunk_ids,
            packed_dropped_ids=_id_tuple(
                row.get("packed_dropped_ids"),
                field="packed_dropped_ids",
                query_id=query_id,
            ),
            packed_truncated_ids=_id_tuple(
                row.get("packed_truncated_ids"),
                field="packed_truncated_ids",
                query_id=query_id,
            ),
        )
    if not result:
        raise RetrievalErrorAnalysisError(f"Retrieval trace is empty: {path}")
    return result


def _metric_value(value: object, *, field: str, query_id: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalErrorAnalysisError(
            f"Metric {field} for {query_id} must be numeric or null"
        )
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise RetrievalErrorAnalysisError(
            f"Metric {field} for {query_id} must be finite in [0, 1]"
        )
    return numeric


def _load_case_scores(path: Path) -> dict[str, CaseScore]:
    payload = _read_json_object(path)
    if payload.get("split") != "warmup":
        raise RetrievalErrorAnalysisError(
            "A/B/C retrieval analysis is restricted to split='warmup'"
        )
    if payload.get("reference_role") != "evaluation_reference_only":
        raise RetrievalErrorAnalysisError(
            "Metrics must be an evaluation_reference_only artifact"
        )
    raw_rows = payload.get("per_case")
    if not isinstance(raw_rows, list):
        raise RetrievalErrorAnalysisError(f"Metrics are missing per_case rows: {path}")
    result: dict[str, CaseScore] = {}
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise RetrievalErrorAnalysisError(
                f"Metric row {index} must be an object: {path}"
            )
        query_id = _row_id(raw_row, location=f"metrics[{index}]")
        if query_id in result:
            raise RetrievalErrorAnalysisError(f"Duplicate metric id: {query_id}")
        result[query_id] = CaseScore(
            meteor=_metric_value(
                raw_row.get("meteor"),
                field="meteor",
                query_id=query_id,
            ),
            rouge_l=_metric_value(
                raw_row.get("rouge_l"),
                field="rouge_l",
                query_id=query_id,
            ),
            status=str(raw_row.get("status", "unknown")),
        )
    if not result:
        raise RetrievalErrorAnalysisError(f"Metrics are empty: {path}")
    return result


def _load_predictions(run_dir: Path) -> dict[str, str]:
    path = run_dir / "predictions.jsonl"
    result: dict[str, str] = {}
    for index, row in enumerate(_read_jsonl_objects(path)):
        query_id = _row_id(row, location=f"prediction[{index}]")
        if query_id in result:
            raise RetrievalErrorAnalysisError(f"Duplicate prediction id: {query_id}")
        answer = row.get("answer")
        if not isinstance(answer, str):
            raise RetrievalErrorAnalysisError(
                f"Prediction answer for {query_id} must be a string"
            )
        result[query_id] = answer
    if not result:
        raise RetrievalErrorAnalysisError(f"Predictions are empty: {path}")
    return result


def _unique_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _article_key(passage: CanonicalPassage) -> tuple[str, str] | None:
    if passage.article_number is None:
        return None
    return (
        passage.document_id,
        passage.article_number.casefold().strip(),
    )


def _passage_identities(
    passages_path: Path,
) -> dict[str, PassageIdentity]:
    passages = load_passages_jsonl(str(passages_path))
    result: dict[str, PassageIdentity] = {}
    for passage in passages:
        if passage.passage_id in result:
            raise RetrievalErrorAnalysisError(
                f"Duplicate passage_id in corpus: {passage.passage_id}"
            )
        result[passage.passage_id] = PassageIdentity(
            passage_id=passage.passage_id,
            document_id=passage.document_id,
            article_key=_article_key(passage),
        )
    if not result:
        raise RetrievalErrorAnalysisError(f"Passage corpus is empty: {passages_path}")
    return result


def _gold_evidence(
    relevant_ids: frozenset[str],
    passage_map: Mapping[str, PassageIdentity],
    *,
    query_id: str,
) -> GoldEvidence:
    missing = sorted(set(relevant_ids) - set(passage_map))
    if missing:
        raise RetrievalErrorAnalysisError(
            f"Silver labels for {query_id} contain unknown passage IDs: "
            + ", ".join(missing[:5])
        )
    identities = [passage_map[identifier] for identifier in relevant_ids]
    return GoldEvidence(
        relevant_ids=relevant_ids,
        document_ids=frozenset(item.document_id for item in identities),
        article_keys=frozenset(
            item.article_key for item in identities if item.article_key is not None
        ),
    )


def _stage_flags(
    ids: Sequence[str],
    gold: GoldEvidence,
    passage_map: Mapping[str, PassageIdentity],
    *,
    query_id: str,
) -> dict[str, object]:
    unknown = sorted(set(ids) - set(passage_map))
    if unknown:
        raise RetrievalErrorAnalysisError(
            f"Retrieval trace for {query_id} contains unknown passage IDs: "
            + ", ".join(unknown[:5])
        )
    identities = [passage_map[identifier] for identifier in ids]
    exact_hit = bool(set(ids) & gold.relevant_ids)
    document_hit = any(item.document_id in gold.document_ids for item in identities)
    article_hit = any(
        item.article_key is not None and item.article_key in gold.article_keys
        for item in identities
    )
    provision_hit = article_hit if gold.has_article_labels else document_hit
    return {
        "exact_passage_hit": exact_hit,
        "document_hit": document_hit,
        "article_hit": article_hit,
        "provision_hit": provision_hit,
    }


def _stage_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    count = len(rows)

    def _count(field: str) -> int:
        return sum(bool(row[field]) for row in rows)

    def _rate(field: str) -> float:
        return _count(field) / count if count else 0.0

    return {
        "labeled_queries": count,
        "exact_passage_hit_count": _count("exact_passage_hit"),
        "exact_passage_hit_rate": _rate("exact_passage_hit"),
        "document_hit_count": _count("document_hit"),
        "document_hit_rate": _rate("document_hit"),
        "article_hit_count": _count("article_hit"),
        "article_hit_rate": _rate("article_hit"),
        "provision_hit_count": _count("provision_hit"),
        "provision_hit_rate": _rate("provision_hit"),
    }


def _repetition_stats(answer: str) -> dict[str, object]:
    tokens = [token.casefold() for token in _TOKEN_RE.findall(answer)]
    if not tokens:
        return {
            "token_count": 0,
            "repetition_flag": False,
            "repeated_ngram_size": 0,
            "repeated_ngram_count": 0,
            "repeated_token_ratio": 0.0,
        }
    max_count = 0
    max_size = 0
    repeated_token_count = 0
    consecutive = False
    for size in (3, 5, 7):
        if len(tokens) < size:
            continue
        ngrams = Counter(
            tuple(tokens[index : index + size])
            for index in range(len(tokens) - size + 1)
        )
        size_max = max(ngrams.values(), default=0)
        if size_max > max_count:
            max_count = size_max
            max_size = size
        repeated_token_count = max(
            repeated_token_count,
            min(
                len(tokens),
                sum(size * count for count in ngrams.values() if count > 1),
            ),
        )
        if any(
            tokens[index : index + size] == tokens[index + size : index + 2 * size]
            for index in range(len(tokens) - 2 * size + 1)
        ):
            consecutive = True
    ratio = repeated_token_count / len(tokens)
    return {
        "token_count": len(tokens),
        "repetition_flag": consecutive or max_count >= 3 or ratio >= 0.2,
        "repeated_ngram_size": max_size,
        "repeated_ngram_count": max_count,
        "repeated_token_ratio": ratio,
    }


def _require_scope_ids(
    required_ids: set[str],
    actual_ids: set[str],
    *,
    role: str,
) -> None:
    missing = sorted(required_ids - actual_ids)
    if missing:
        raise RetrievalErrorAnalysisError(
            f"{role} is missing clean-warmup IDs: " + ", ".join(missing[:5])
        )


def _load_source(
    path: Path,
    *,
    source_name: str,
) -> dict[str, tuple[str, ...]]:
    try:
        return load_ranked_source(path, source_name=source_name)
    except EnsembleDiagnosticError as exc:
        raise RetrievalErrorAnalysisError(
            f"Cannot load {source_name} retrieval source: {path}"
        ) from exc


def analyze_warmup_retrieval_errors(
    *,
    run_dir: str | Path,
    metrics_path: str | Path,
    bm25_path: str | Path,
    qwen_path: str | Path,
    labels_path: str | Path,
    passages_path: str | Path,
    candidate_cutoff: int = 100,
    ltr_cutoff: int = 20,
    packed_cutoff: int = 4,
    low_meteor_threshold: float = 0.25,
) -> RetrievalErrorAnalysisReport:
    """Classify labeled clean-warmup cases into retrieval/ranking/reader stages."""

    if candidate_cutoff <= 0 or ltr_cutoff <= 0 or packed_cutoff <= 0:
        raise RetrievalErrorAnalysisError("All cutoffs must be positive")
    if (
        not math.isfinite(low_meteor_threshold)
        or not 0.0 <= low_meteor_threshold <= 1.0
    ):
        raise RetrievalErrorAnalysisError(
            "low_meteor_threshold must be finite and within [0, 1]"
        )

    run_path = Path(run_dir)
    config = _read_json_object(run_path / "config.json")
    if config.get("split") != "warmup":
        raise RetrievalErrorAnalysisError(
            "A/B/C analysis requires a warmup prediction run"
        )
    if config.get("id_source") != "clean_manifest":
        raise RetrievalErrorAnalysisError(
            "A/B/C analysis requires id_source='clean_manifest'"
        )
    if config.get("retrieval_variant") != "ltr_full_all":
        raise RetrievalErrorAnalysisError(
            "A/B/C analysis is restricted to champion retrieval_variant="
            "'ltr_full_all'; ensemble runs are not accepted"
        )

    trace = _load_retrieval_trace(run_path)
    scores = _load_case_scores(Path(metrics_path))
    answers = _load_predictions(run_path)
    run_ids = set(trace)
    if set(scores) != run_ids:
        raise RetrievalErrorAnalysisError(
            "Retrieval trace and metrics IDs do not match exactly"
        )
    if set(answers) != run_ids:
        raise RetrievalErrorAnalysisError(
            "Retrieval trace and prediction IDs do not match exactly"
        )

    bm25 = _load_source(Path(bm25_path), source_name="bm25")
    qwen = _load_source(Path(qwen_path), source_name="dense")
    _require_scope_ids(run_ids, set(bm25), role="BM25 source")
    _require_scope_ids(run_ids, set(qwen), role="Qwen source")

    try:
        labels = load_relevance_labels(labels_path)
    except EnsembleDiagnosticError as exc:
        raise RetrievalErrorAnalysisError(
            f"Cannot load evaluation-only relevance labels: {labels_path}"
        ) from exc
    _require_scope_ids(run_ids, set(labels), role="Silver labels")

    passage_map = _passage_identities(Path(passages_path))
    per_case: list[dict[str, object]] = []
    stage_rows: dict[str, list[Mapping[str, object]]] = {
        "bm25": [],
        "qwen": [],
        "candidate_union": [],
        "ltr": [],
        "packed": [],
    }
    unlabeled_count = 0

    for query_id in sorted(run_ids):
        relevant_ids, provenance = labels[query_id]
        if provenance == "gold":
            raise RetrievalErrorAnalysisError(
                "A/B/C analysis accepts silver labels only; gold provenance "
                f"found for {query_id}"
            )
        if provenance not in {"silver", "unlabeled"}:
            raise RetrievalErrorAnalysisError(
                f"Unsupported label provenance for {query_id}: {provenance}"
            )
        is_labeled = provenance == "silver" and bool(relevant_ids)
        bm25_ids = _unique_preserve_order(bm25[query_id][:candidate_cutoff])
        qwen_ids = _unique_preserve_order(qwen[query_id][:candidate_cutoff])
        candidate_ids = _unique_preserve_order((*bm25_ids, *qwen_ids))
        ltr_ids = trace[query_id].raw_hit_ids[:ltr_cutoff]
        packed_ids = trace[query_id].packed_chunk_ids[:packed_cutoff]
        repetition = _repetition_stats(answers[query_id])
        meteor = scores[query_id].meteor
        base_row: dict[str, object] = {
            "id": query_id,
            "meteor": meteor,
            "rouge_l": scores[query_id].rouge_l,
            "metric_status": scores[query_id].status,
            "silver_relevant_ids": sorted(relevant_ids),
            "label_provenance": provenance,
            "bm25_ids": list(bm25_ids),
            "qwen_ids": list(qwen_ids),
            "candidate_union_ids": list(candidate_ids),
            "ltr_ids_at_cutoff": list(ltr_ids),
            "packed_ids_at_cutoff": list(packed_ids),
            "packed_dropped_count": len(trace[query_id].packed_dropped_ids),
            "packed_truncated_count": len(trace[query_id].packed_truncated_ids),
            "retrieval_status": trace[query_id].status,
            "repetition": repetition,
        }
        if not is_labeled:
            unlabeled_count += 1
            per_case.append(
                {
                    **base_row,
                    "silver_relevant_ids": [],
                    "failure_type": "UNLABELED",
                    "low_meteor": False,
                    "reader_issue_observed": False,
                    "bm25_flags": None,
                    "qwen_flags": None,
                    "candidate_union_flags": None,
                    "ltr_flags": None,
                    "packed_flags": None,
                }
            )
            continue

        gold = _gold_evidence(
            relevant_ids,
            passage_map,
            query_id=query_id,
        )

        candidate_flags = _stage_flags(
            candidate_ids,
            gold,
            passage_map,
            query_id=query_id,
        )
        bm25_flags = _stage_flags(
            bm25_ids,
            gold,
            passage_map,
            query_id=query_id,
        )
        qwen_flags = _stage_flags(
            qwen_ids,
            gold,
            passage_map,
            query_id=query_id,
        )
        ltr_flags = _stage_flags(
            ltr_ids,
            gold,
            passage_map,
            query_id=query_id,
        )
        packed_flags = _stage_flags(
            packed_ids,
            gold,
            passage_map,
            query_id=query_id,
        )
        low_answer_quality = meteor is not None and meteor < low_meteor_threshold
        reader_issue = bool(repetition["repetition_flag"]) or low_answer_quality
        if not bool(candidate_flags["provision_hit"]):
            failure_type = "A_RETRIEVAL_MISS"
        elif not bool(ltr_flags["provision_hit"]):
            failure_type = "B_RERANK_MISS"
        elif not bool(packed_flags["provision_hit"]):
            failure_type = "B_PACK_MISS"
        elif reader_issue:
            failure_type = "C_READER_ISSUE"
        else:
            failure_type = "C_NOT_OBSERVED"

        row = {
            **base_row,
            "bm25_flags": bm25_flags,
            "qwen_flags": qwen_flags,
            "candidate_union_flags": candidate_flags,
            "ltr_flags": ltr_flags,
            "packed_flags": packed_flags,
            "low_meteor": low_answer_quality,
            "reader_issue_observed": reader_issue,
            "failure_type": failure_type,
        }
        per_case.append(row)
        stage_rows["bm25"].append(bm25_flags)
        stage_rows["qwen"].append(qwen_flags)
        stage_rows["candidate_union"].append(candidate_flags)
        stage_rows["ltr"].append(ltr_flags)
        stage_rows["packed"].append(packed_flags)

    failure_counts = Counter(
        str(row["failure_type"])
        for row in per_case
        if row["failure_type"] != "UNLABELED"
    )
    case_counts = Counter(str(row["failure_type"]) for row in per_case)

    payload: dict[str, object] = {
        "schema_version": "sedar-warmup-retrieval-error-analysis-v1",
        "split": "warmup",
        "scope": "clean_warmup",
        "reference_role": "evaluation_reference_only",
        "label_provenance": "silver_only",
        "gold_answer_text_used": False,
        "gold_answer_text_written": False,
        "run_dir": str(run_path),
        "run_id": str(
            _read_json_object(run_path / "run_summary.json").get("run_id", "")
        ),
        "metrics_path": str(metrics_path),
        "bm25_path": str(bm25_path),
        "qwen_path": str(qwen_path),
        "labels_path": str(labels_path),
        "passages_path": str(passages_path),
        "cutoffs": {
            "candidate_source_top_k": candidate_cutoff,
            "ltr_top_k": ltr_cutoff,
            "packed_evidence_top_k": packed_cutoff,
        },
        "low_meteor_threshold": low_meteor_threshold,
        "query_count": len(run_ids),
        "labeled_query_count": len(stage_rows["candidate_union"]),
        "unlabeled_query_count": unlabeled_count,
        "source_extra_query_counts": {
            "bm25": len(set(bm25) - run_ids),
            "qwen": len(set(qwen) - run_ids),
            "labels": len(set(labels) - run_ids),
        },
        "stage_metrics": {
            f"bm25_top_{candidate_cutoff}": _stage_summary(stage_rows["bm25"]),
            f"qwen_top_{candidate_cutoff}": _stage_summary(stage_rows["qwen"]),
            "candidate_union_bm25_qwen": _stage_summary(stage_rows["candidate_union"]),
            f"ltr_top_{ltr_cutoff}": _stage_summary(stage_rows["ltr"]),
            f"packed_top_{packed_cutoff}": _stage_summary(stage_rows["packed"]),
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "case_counts": dict(sorted(case_counts.items())),
        "failure_definitions": {
            "A_RETRIEVAL_MISS": (
                "Silver provision is absent from BM25/Qwen source union."
            ),
            "B_RERANK_MISS": (
                "Provision enters source union but is absent from LTR top-k."
            ),
            "B_PACK_MISS": (
                "Provision survives LTR top-k but is absent from packed evidence."
            ),
            "C_READER_ISSUE": (
                "Provision is packed and low METEOR or repetition is observed."
            ),
            "C_NOT_OBSERVED": (
                "Provision is packed and this report found no "
                "low-score/repetition flag."
            ),
            "UNLABELED": "No silver citation label was available for this query.",
        },
        "per_case": sorted(per_case, key=lambda row: str(row["id"])),
    }
    return RetrievalErrorAnalysisReport(payload=payload)


__all__ = [
    "RetrievalErrorAnalysisError",
    "RetrievalErrorAnalysisReport",
    "analyze_warmup_retrieval_errors",
]
