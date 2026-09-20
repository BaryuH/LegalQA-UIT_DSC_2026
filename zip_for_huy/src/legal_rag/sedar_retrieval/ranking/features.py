"""LTR feature extraction shared by train/inference (TASK 12)."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

FEATURE_SCHEMA_VERSION = "sedar-ltr-features-v1"
ENSEMBLE_FEATURE_SCHEMA_VERSION = "sedar-ltr-ensemble-features-v2"
FeatureProfile = Literal["baseline_v1", "ensemble_v2"]

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class LTRFeatureRow:
    query_id: str
    passage_id: str
    features: dict[str, float]
    schema_version: str = FEATURE_SCHEMA_VERSION


def _tokens(text: str) -> list[str]:
    return [tok.casefold() for tok in _TOKEN_RE.findall(text)]


def _overlap(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa)


def _bigram_overlap(a: Sequence[str], b: Sequence[str]) -> float:
    def grams(tokens: Sequence[str]) -> set[tuple[str, str]]:
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    ga, gb = grams(a), grams(b)
    if not ga:
        return 0.0
    return len(ga & gb) / len(ga)


def extract_features(
    *,
    query_id: str,
    query: str,
    passage_id: str,
    passage_text: str,
    bm25_score: float | None = None,
    bm25_rank: int | None = None,
    dense_score: float | None = None,
    dense_rank: int | None = None,
    rrf_score: float | None = None,
    document_name: str | None = None,
    article_number: str | None = None,
    clause_number: str | None = None,
    article_title: str | None = None,
    chapter_title: str | None = None,
    is_effective: float | None = None,
    is_expired: float | None = None,
    query_article: str | None = None,
    query_clause: str | None = None,
    query_year: str | None = None,
    query_doc_number: str | None = None,
) -> LTRFeatureRow:
    """Extract one feature row. Missing numeric features use -1.0 convention."""

    q_toks = _tokens(query)
    p_toks = _tokens(passage_text)
    missing = -1.0

    features = {
        "bm25_score": missing if bm25_score is None else float(bm25_score),
        "bm25_rank": missing if bm25_rank is None else float(bm25_rank),
        "dense_score": missing if dense_score is None else float(dense_score),
        "dense_rank": missing if dense_rank is None else float(dense_rank),
        "rrf_score": missing if rrf_score is None else float(rrf_score),
        "query_token_coverage": _overlap(q_toks, p_toks),
        "idf_weighted_overlap": _overlap(q_toks, p_toks),  # proxy until IDF table wired
        "bigram_overlap": _bigram_overlap(q_toks, p_toks),
        "exact_phrase_match": 1.0
        if query.strip() and query.casefold() in passage_text.casefold()
        else 0.0,
        "legal_term_overlap": _overlap(
            [t for t in q_toks if t in {"điều", "khoản", "điểm", "luật", "nghị"}],
            p_toks,
        ),
        "document_name_match": 1.0
        if document_name and document_name.casefold() in query.casefold()
        else 0.0,
        "document_number_match": 1.0
        if query_doc_number
        and query_doc_number in (passage_text + (document_name or ""))
        else 0.0,
        "year_match": 1.0 if query_year and query_year in passage_text else 0.0,
        "article_number_match": 1.0
        if query_article and article_number and query_article == article_number
        else 0.0,
        "clause_number_match": 1.0
        if query_clause and clause_number and query_clause == clause_number
        else 0.0,
        "point_match": 0.0,
        "article_title_similarity": _overlap(q_toks, _tokens(article_title or "")),
        "chapter_title_similarity": _overlap(q_toks, _tokens(chapter_title or "")),
        "is_effective": missing if is_effective is None else float(is_effective),
        "is_expired": missing if is_expired is None else float(is_expired),
    }
    # Guard against NaN/Inf.
    for key, value in list(features.items()):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite feature {key}={value}")
    return LTRFeatureRow(
        query_id=query_id,
        passage_id=passage_id,
        features=features,
    )


def _rank_difference(left: int | None, right: int | None) -> float:
    if left is None or right is None:
        return -1.0
    return float(abs(left - right))


def _rank_aggregates(ranks: Sequence[int | None]) -> dict[str, float]:
    valid = [float(rank) for rank in ranks if rank is not None]
    if not valid:
        return {
            "min_rank": -1.0,
            "max_rank": -1.0,
            "mean_rank": -1.0,
            "rank_std": -1.0,
        }
    mean = sum(valid) / len(valid)
    variance = sum((rank - mean) ** 2 for rank in valid) / len(valid)
    return {
        "min_rank": min(valid),
        "max_rank": max(valid),
        "mean_rank": mean,
        "rank_std": math.sqrt(variance),
    }


def extract_ensemble_features(
    *,
    query_id: str,
    query: str,
    passage_id: str,
    passage_text: str,
    bm25_score: float | None = None,
    bm25_rank: int | None = None,
    qwen_score: float | None = None,
    qwen_rank: int | None = None,
    legal_score: float | None = None,
    legal_rank: int | None = None,
    rrf_score: float | None = None,
    document_name: str | None = None,
    article_number: str | None = None,
    clause_number: str | None = None,
    article_title: str | None = None,
    chapter_title: str | None = None,
    is_effective: float | None = None,
    is_expired: float | None = None,
    query_article: str | None = None,
    query_clause: str | None = None,
    query_year: str | None = None,
    query_doc_number: str | None = None,
) -> LTRFeatureRow:
    """Extract v2 features for a BM25/Qwen/Legal candidate ensemble.

    ``qwen_*`` represents the existing ``dense_*`` signal in the v1
    extractor.  The v2 profile renames that signal explicitly and adds
    source-presence, agreement, rank-difference, and aggregate-rank features.
    """

    baseline = extract_features(
        query_id=query_id,
        query=query,
        passage_id=passage_id,
        passage_text=passage_text,
        bm25_score=bm25_score,
        bm25_rank=bm25_rank,
        dense_score=qwen_score,
        dense_rank=qwen_rank,
        rrf_score=rrf_score,
        document_name=document_name,
        article_number=article_number,
        clause_number=clause_number,
        article_title=article_title,
        chapter_title=chapter_title,
        is_effective=is_effective,
        is_expired=is_expired,
        query_article=query_article,
        query_clause=query_clause,
        query_year=query_year,
        query_doc_number=query_doc_number,
    )
    features = dict(baseline.features)
    features.pop("dense_score")
    features.pop("dense_rank")

    bm25_found = 1.0 if bm25_rank is not None else 0.0
    qwen_found = 1.0 if qwen_rank is not None else 0.0
    legal_found = 1.0 if legal_rank is not None else 0.0
    features.update(
        {
            "bm25_found": bm25_found,
            "qwen_score": -1.0 if qwen_score is None else float(qwen_score),
            "qwen_rank": -1.0 if qwen_rank is None else float(qwen_rank),
            "qwen_found": qwen_found,
            "legal_score": -1.0 if legal_score is None else float(legal_score),
            "legal_rank": -1.0 if legal_rank is None else float(legal_rank),
            "legal_found": legal_found,
            "bm25_qwen_both": bm25_found * qwen_found,
            "qwen_legal_both": qwen_found * legal_found,
            "bm25_legal_both": bm25_found * legal_found,
            "all_three": bm25_found * qwen_found * legal_found,
            "bm25_qwen_rank_diff": _rank_difference(bm25_rank, qwen_rank),
            "qwen_legal_rank_diff": _rank_difference(qwen_rank, legal_rank),
            "bm25_legal_rank_diff": _rank_difference(bm25_rank, legal_rank),
            **_rank_aggregates((bm25_rank, qwen_rank, legal_rank)),
        }
    )
    for key, value in features.items():
        if not math.isfinite(value):
            raise ValueError(f"Non-finite feature {key}={value}")
    return LTRFeatureRow(
        query_id=query_id,
        passage_id=passage_id,
        features=features,
        schema_version=ENSEMBLE_FEATURE_SCHEMA_VERSION,
    )


def feature_vector(
    row: LTRFeatureRow,
    names: Sequence[str] | None = None,
) -> list[float]:
    ordered = list(names) if names is not None else sorted(row.features)
    return [row.features[name] for name in ordered]


FEATURE_NAMES: tuple[str, ...] = tuple(
    sorted(
        extract_features(
            query_id="q",
            query="điều 1",
            passage_id="p",
            passage_text="điều 1 nội dung",
        ).features
    )
)


ENSEMBLE_FEATURE_NAMES: tuple[str, ...] = tuple(
    sorted(
        extract_ensemble_features(
            query_id="q",
            query="điều 1",
            passage_id="p",
            passage_text="điều 1 nội dung",
        ).features
    )
)


def assert_train_inference_parity(
    left: LTRFeatureRow, right: LTRFeatureRow, *, tol: float = 0.0
) -> None:
    if left.features.keys() != right.features.keys():
        raise AssertionError("Feature key mismatch")
    for key in left.features:
        if abs(left.features[key] - right.features[key]) > tol:
            raise AssertionError(f"Feature drift on {key}")
