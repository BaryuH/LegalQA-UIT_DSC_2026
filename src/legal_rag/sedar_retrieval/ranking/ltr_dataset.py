"""Leakage-safe LTR feature-dataset construction (TASK 12)."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from legal_rag.questions import load_inference_questions
from legal_rag.schemas import InferenceQuestion
from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines
from legal_rag.sedar_retrieval.query.citation_parser import (
    CitationMention,
    extract_years,
    parse_citations,
)

from .features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    LTRFeatureRow,
    extract_features,
)

LabelMode = Literal["binary", "graded"]
UnlabeledPolicy = Literal["fail", "skip"]
LTR_DATASET_SCHEMA_VERSION = "sedar-ltr-feature-dataset-v1"


class LTRFeatureBuildError(ValueError):
    """Raised when feature construction would violate the LTR contract."""


@dataclass(frozen=True, slots=True)
class LTRCandidate:
    """One canonical candidate with source-specific retrieval scores."""

    passage_id: str
    rank: int
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    source: str = "candidate"


@dataclass(frozen=True, slots=True)
class LTRFeatureBuildConfig:
    """Deterministic label and candidate policy for one feature build."""

    label_mode: LabelMode = "graded"
    unlabeled_policy: UnlabeledPolicy = "skip"
    max_candidates: int = 0

    def __post_init__(self) -> None:
        if self.max_candidates < 0:
            raise ValueError("max_candidates must be non-negative")


@dataclass(frozen=True, slots=True)
class LTRFeatureBuildReport:
    """Counters written to the feature manifest."""

    input_query_count: int
    output_query_count: int
    output_row_count: int
    skipped_unlabeled_query_count: int
    candidate_count_by_query: dict[str, int]
    label_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": LTR_DATASET_SCHEMA_VERSION,
            "input_query_count": self.input_query_count,
            "output_query_count": self.output_query_count,
            "output_row_count": self.output_row_count,
            "skipped_unlabeled_query_count": self.skipped_unlabeled_query_count,
            "candidate_count_by_query": dict(
                sorted(self.candidate_count_by_query.items())
            ),
            "label_counts": dict(sorted(self.label_counts.items())),
            "leakage_policy": "no_answer_text_no_gold_labels_no_reader_outputs",
        }


def schema_sha256(path: str | Path) -> str:
    """Hash the exact feature schema bytes used for a build."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_feature_schema(path: str | Path) -> str:
    """Validate the checked-in schema against the extractor feature names."""

    schema_path = Path(path)
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LTRFeatureBuildError(
            f"Cannot read LTR feature schema: {schema_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise LTRFeatureBuildError("LTR feature schema must be a JSON object")
    if payload.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise LTRFeatureBuildError(
            "LTR feature schema version does not match extractor: "
            f"{payload.get('schema_version')!r}"
        )
    names = payload.get("features")
    if not isinstance(names, list) or tuple(names) != FEATURE_NAMES:
        raise LTRFeatureBuildError(
            "LTR feature schema names do not match the extractor feature names"
        )
    if payload.get("missing_value") != -1.0:
        raise LTRFeatureBuildError("LTR schema missing_value must be -1.0")
    return schema_sha256(schema_path)


def _finite_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise LTRFeatureBuildError(f"{field} must be numeric or null")
    try:
        number = float(value)
    except ValueError as exc:
        raise LTRFeatureBuildError(f"{field} must be numeric or null") from exc
    if not math.isfinite(number):
        raise LTRFeatureBuildError(f"{field} must be finite")
    return number


def _rank(value: object, *, fallback: int, field: str) -> int:
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise LTRFeatureBuildError(f"{field} must be an integer")
    try:
        rank = int(value)
    except (TypeError, ValueError) as exc:
        raise LTRFeatureBuildError(f"{field} must be an integer") from exc
    if rank <= 0:
        raise LTRFeatureBuildError(f"{field} must be positive")
    return rank


def _candidate_from_row(
    row: Mapping[str, object],
    *,
    fallback_rank: int,
    fused: bool,
) -> LTRCandidate:
    passage_id = row.get("passage_id")
    if passage_id is None or not str(passage_id).strip():
        raise LTRFeatureBuildError("Candidate passage_id must be non-blank")
    if fused:
        rank = _rank(
            row.get("fused_rank"),
            fallback=fallback_rank,
            field="fused_rank",
        )
        source = "rrf"
        bm25_score = _finite_float(row.get("bm25_score"), field="bm25_score")
        dense_score = _finite_float(row.get("dense_score"), field="dense_score")
        rrf_score = _finite_float(row.get("rrf_score"), field="rrf_score")
        bm25_rank = (
            _rank(row.get("bm25_rank"), fallback=rank, field="bm25_rank")
            if row.get("bm25_rank") is not None
            else None
        )
        dense_rank = (
            _rank(row.get("dense_rank"), fallback=rank, field="dense_rank")
            if row.get("dense_rank") is not None
            else None
        )
    else:
        rank = _rank(row.get("rank"), fallback=fallback_rank, field="rank")
        raw_source = str(row.get("source", "candidate"))
        source = raw_source if raw_source.strip() else "candidate"
        bm25_score = _finite_float(row.get("bm25"), field="bm25")
        dense_score = _finite_float(row.get("dense"), field="dense")
        rrf_score = _finite_float(row.get("rrf"), field="rrf")
        bm25_rank = rank if bm25_score is not None else None
        dense_rank = rank if dense_score is not None else None
    return LTRCandidate(
        passage_id=str(passage_id),
        rank=rank,
        bm25_score=bm25_score,
        bm25_rank=bm25_rank,
        dense_score=dense_score,
        dense_rank=dense_rank,
        rrf_score=rrf_score,
        source=source,
    )


def load_rrf_candidates(
    path: str | Path,
) -> dict[str, tuple[LTRCandidate, ...]]:
    """Load RRF or ranked-source JSONL without reading answer fields."""

    grouped: defaultdict[str, list[LTRCandidate]] = defaultdict(list)
    for line in iter_jsonl_lines(path):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise LTRFeatureBuildError("Candidate JSONL rows must be objects")
        query_id = payload.get("query_id")
        if query_id is None or not str(query_id).strip():
            raise LTRFeatureBuildError("Candidate row query_id must be non-blank")

        raw_candidates = payload.get("candidates")
        if isinstance(raw_candidates, list) and raw_candidates:
            fused = True
            rows = [item for item in raw_candidates if isinstance(item, Mapping)]
        else:
            fused = False
            raw_scores: object = payload.get("scores", [])
            if not isinstance(raw_scores, list):
                raise LTRFeatureBuildError("Candidate scores must be a list")
            rows = [item for item in raw_scores if isinstance(item, Mapping)]
            if not rows and payload.get("ranked_ids"):
                ranked_ids = payload["ranked_ids"]
                if not isinstance(ranked_ids, list):
                    raise LTRFeatureBuildError("ranked_ids must be a list")
                rows = [{"passage_id": item} for item in ranked_ids]
        if not rows:
            raise LTRFeatureBuildError(
                f"Candidate row has no candidates: query_id={query_id}"
            )
        candidates = [
            _candidate_from_row(
                row,
                fallback_rank=index,
                fused=fused,
            )
            for index, row in enumerate(rows, start=1)
        ]
        candidates.sort(key=lambda item: (item.rank, item.passage_id))
        ids = [item.passage_id for item in candidates]
        if len(set(ids)) != len(ids):
            raise LTRFeatureBuildError(
                f"Duplicate candidate passage_id for query_id={query_id}"
            )
        grouped[str(query_id)].extend(candidates)

    result: dict[str, tuple[LTRCandidate, ...]] = {}
    for query_id, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda item: (item.rank, item.passage_id))
        result[query_id] = tuple(ordered)
    return result


def load_question_map(
    path: str | Path,
    *,
    split: str,
) -> dict[str, InferenceQuestion]:
    """Load question-only views, excluding answer fields by construction."""

    if split not in {"train", "warmup", "public", "private"}:
        raise LTRFeatureBuildError(f"Unsupported question split: {split!r}")
    try:
        questions = load_inference_questions(path, split=split)  # type: ignore[arg-type]
    except (OSError, ValueError) as exc:
        raise LTRFeatureBuildError(f"Cannot load question map: {path}") from exc
    return {question.id: question for question in questions}


def _citation_context(
    query: str,
) -> tuple[tuple[CitationMention, ...], str | None, str | None, str | None, str | None]:
    citations = parse_citations(query)
    first_article = next((item.article for item in citations if item.article), None)
    first_clause = next((item.clause for item in citations if item.clause), None)
    first_doc_number = next(
        (item.document_number for item in citations if item.document_number),
        None,
    )
    first_year = next(iter(extract_years(query)), None)
    return citations, first_article, first_clause, first_year, first_doc_number


def _citation_grade(
    citations: Sequence[CitationMention],
    passage: CanonicalPassage,
) -> int | None:
    if not citations:
        return None
    document_name = (passage.document_name or "").casefold()
    best = 0
    for citation in citations:
        if citation.article and passage.article_number:
            if citation.article.casefold() != passage.article_number.casefold():
                continue
            if citation.clause and passage.clause_number:
                if citation.clause.casefold() == passage.clause_number.casefold():
                    return 3
            best = max(best, 2)
            continue
        if (
            citation.document_number
            and citation.document_number.casefold() in document_name
        ):
            best = max(best, 1)
    return best


def _status_features(
    passage: CanonicalPassage,
) -> tuple[float | None, float | None]:
    if passage.status == "unknown":
        return None, None
    return (
        1.0 if passage.status == "effective" else 0.0,
        1.0 if passage.status in {"expired", "repealed"} else 0.0,
    )


def build_ltr_feature_rows(
    *,
    candidates: Mapping[str, Sequence[LTRCandidate]],
    questions: Mapping[str, InferenceQuestion],
    passages: Mapping[str, CanonicalPassage],
    config: LTRFeatureBuildConfig | None = None,
) -> tuple[tuple[dict[str, object], ...], LTRFeatureBuildReport]:
    """Build deterministic, citation-supervised feature rows."""

    cfg = config or LTRFeatureBuildConfig()
    output: list[dict[str, object]] = []
    candidate_counts: dict[str, int] = {}
    label_counts: Counter[str] = Counter()
    skipped_unlabeled = 0
    output_queries = 0

    for query_id in sorted(candidates):
        question = questions.get(query_id)
        if question is None:
            raise LTRFeatureBuildError(
                f"Candidate query_id is absent from question map: {query_id}"
            )
        citations, query_article, query_clause, query_year, query_doc_number = (
            _citation_context(question.question)
        )
        if not citations and cfg.unlabeled_policy == "fail":
            raise LTRFeatureBuildError(
                f"Query has no citation for label construction: {query_id}"
            )
        if not citations:
            skipped_unlabeled += 1
            continue

        selected = sorted(
            candidates[query_id],
            key=lambda item: (item.rank, item.passage_id),
        )
        if cfg.max_candidates:
            selected = selected[: cfg.max_candidates]
        if not selected:
            raise LTRFeatureBuildError(f"Query has no candidates: {query_id}")
        candidate_counts[query_id] = len(selected)
        output_queries += 1

        for candidate in selected:
            passage = passages.get(candidate.passage_id)
            if passage is None:
                raise LTRFeatureBuildError(
                    f"Candidate passage_id is absent from corpus: "
                    f"{candidate.passage_id}"
                )
            is_effective, is_expired = _status_features(passage)
            feature_row: LTRFeatureRow = extract_features(
                query_id=query_id,
                query=question.question,
                passage_id=candidate.passage_id,
                passage_text=passage.retrieval_text,
                bm25_score=candidate.bm25_score,
                bm25_rank=candidate.bm25_rank,
                dense_score=candidate.dense_score,
                dense_rank=candidate.dense_rank,
                rrf_score=candidate.rrf_score,
                document_name=passage.document_name,
                article_number=passage.article_number,
                clause_number=passage.clause_number,
                article_title=passage.article_title,
                is_effective=is_effective,
                is_expired=is_expired,
                query_article=query_article,
                query_clause=query_clause,
                query_year=query_year,
                query_doc_number=query_doc_number,
            )
            grade = _citation_grade(citations, passage)
            if grade is None:
                raise LTRFeatureBuildError(
                    f"Internal citation label error for query_id={query_id}"
                )
            label = 1 if grade > 0 else 0
            if cfg.label_mode == "graded":
                label = grade
            label_counts[str(label)] += 1
            output.append(
                {
                    "query_id": query_id,
                    "passage_id": candidate.passage_id,
                    "rank": candidate.rank,
                    "label": label,
                    "label_provenance": "query_citation_heuristic",
                    "features": feature_row.features,
                    "schema_version": FEATURE_SCHEMA_VERSION,
                }
            )

    report = LTRFeatureBuildReport(
        input_query_count=len(candidates),
        output_query_count=output_queries,
        output_row_count=len(output),
        skipped_unlabeled_query_count=skipped_unlabeled,
        candidate_count_by_query=candidate_counts,
        label_counts=dict(label_counts),
    )
    return tuple(output), report


def group_feature_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Return deterministic query groups for downstream LambdaRank."""

    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        query_id = row.get("query_id")
        if query_id is None:
            raise LTRFeatureBuildError("Feature row query_id is missing")
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, int):
            raise LTRFeatureBuildError("Feature row label must be an integer")
        grouped[str(query_id)].append(row)
    return tuple(
        {
            "query_id": query_id,
            "row_count": len(grouped[query_id]),
            "labels": [row["label"] for row in grouped[query_id]],
        }
        for query_id in sorted(grouped)
    )


__all__ = [
    "LTRCandidate",
    "LTR_DATASET_SCHEMA_VERSION",
    "LTRFeatureBuildConfig",
    "LTRFeatureBuildError",
    "LTRFeatureBuildReport",
    "build_ltr_feature_rows",
    "group_feature_rows",
    "load_question_map",
    "load_rrf_candidates",
    "schema_sha256",
    "validate_feature_schema",
]
