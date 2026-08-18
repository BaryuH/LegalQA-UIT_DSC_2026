"""Leakage-aware hard and semi-hard negative mining (TASK 10)."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from legal_rag.schemas import CanonicalID, DomainModel, NonBlankText
from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage
from legal_rag.sedar_retrieval.query.citation_parser import parse_citations
from legal_rag.sedar_retrieval.training.synthetic_queries import (
    QueryType,
    SyntheticQueryRecord,
)

NegativeCategory = Literal["A", "B", "C", "D", "E", "F"]
MiningSource = Literal["bm25", "dense", "rrf", "candidate"]
TASK10_SCHEMA_VERSION = "sedar-retrieval-v3-hard-negative-v1"

_STOPWORDS = {
    "các",
    "cho",
    "của",
    "được",
    "là",
    "nào",
    "người",
    "những",
    "pháp",
    "theo",
    "trong",
    "và",
    "về",
}


class HardNegative(DomainModel):
    """One resolved negative plus provenance and false-negative warnings."""

    negative_passage_id: CanonicalID
    negative_document_id: CanonicalID
    negative_article_id: CanonicalID | None = None
    negative_clause_id: CanonicalID | None = None
    negative_source_hash: NonBlankText
    negative_category: NegativeCategory
    mined_from: MiningSource
    mining_sources: tuple[NonBlankText, ...] = ()
    rank: int = Field(ge=1)
    retrieval_score: float
    hardness_score: float = Field(ge=0.0, le=1.0)
    potential_false_negative: bool
    false_negative_flags: tuple[NonBlankText, ...] = ()
    resolution: Literal["resolved"] = "resolved"


class HardNegativeRecord(DomainModel):
    """One synthetic query with two to five resolved explicit negatives."""

    hard_negative_id: CanonicalID
    synthetic_id: CanonicalID
    query: NonBlankText
    query_type: QueryType
    positive_passage_id: CanonicalID
    source_document_id: CanonicalID
    positive_source_hash: NonBlankText
    negatives: tuple[HardNegative, ...] = Field(min_length=2, max_length=5)
    source_split: Literal["train"] = "train"


@dataclass(frozen=True, slots=True)
class CandidateHit:
    """A ranked candidate from BM25, dense, or an external candidate file."""

    passage_id: str
    rank: int
    score: float
    source: MiningSource
    sources: tuple[MiningSource, ...] = ()


@dataclass(frozen=True, slots=True)
class HardNegativeMiningConfig:
    """Deterministic selection and audit policy."""

    min_negatives: int = 2
    max_negatives: int = 5
    seed: int = 42
    random_rank: int = 150
    lexical_hard_threshold: float = 0.20
    lexical_false_negative_threshold: float = 0.75
    dense_false_negative_threshold: float = 0.65

    def __post_init__(self) -> None:
        if not 2 <= self.min_negatives <= self.max_negatives <= 5:
            raise ValueError("negative counts must satisfy 2 <= min <= max <= 5")
        if self.random_rank <= 0:
            raise ValueError("random_rank must be positive")
        for name in (
            "lexical_hard_threshold",
            "lexical_false_negative_threshold",
            "dense_false_negative_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class HardNegativeMiningReport:
    """Machine metrics and gate diagnostics for one mining run."""

    query_count: int
    accepted_query_count: int
    rejected_query_count: int
    negative_count: int
    unresolved_candidate_count: int
    positive_in_negative_count: int
    potential_false_negative_count: int
    rejection_counts: dict[str, int]
    category_counts: dict[str, int]
    source_counts: dict[str, int]
    hard_score_mean: float
    hard_score_p50: float
    hard_score_p95: float
    random_score_mean: float
    random_score_p50: float
    random_score_p95: float
    harder_than_random: bool
    estimated_false_negative_rate: float

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": TASK10_SCHEMA_VERSION,
            "query_count": self.query_count,
            "accepted_query_count": self.accepted_query_count,
            "rejected_query_count": self.rejected_query_count,
            "negative_count": self.negative_count,
            "unresolved_candidate_count": self.unresolved_candidate_count,
            "positive_in_negative_count": self.positive_in_negative_count,
            "potential_false_negative_count": self.potential_false_negative_count,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "category_counts": dict(sorted(self.category_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "hard_score_mean": self.hard_score_mean,
            "hard_score_p50": self.hard_score_p50,
            "hard_score_p95": self.hard_score_p95,
            "random_score_mean": self.random_score_mean,
            "random_score_p50": self.random_score_p50,
            "random_score_p95": self.random_score_p95,
            "harder_than_random": self.harder_than_random,
            "estimated_false_negative_rate": self.estimated_false_negative_rate,
            "manual_audit_required": True,
        }


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _content_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[^\W_]+", _normalize(text), flags=re.UNICODE))
    return tokens - _STOPWORDS


def _lexical_overlap(query: str, passage_text: str) -> float:
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _content_tokens(passage_text)) / len(query_tokens)


def _same_article(positive: CanonicalPassage, candidate: CanonicalPassage) -> bool:
    if positive.article_id and candidate.article_id:
        return positive.article_id == candidate.article_id
    if positive.article_number and candidate.article_number:
        return positive.article_number == candidate.article_number
    return False


def _same_document(positive: CanonicalPassage, candidate: CanonicalPassage) -> bool:
    return positive.document_id == candidate.document_id


def _citation_overlap(
    query: str,
    candidate: CanonicalPassage,
) -> bool:
    for citation in parse_citations(query):
        if citation.article and candidate.article_number:
            if citation.article.casefold() == candidate.article_number.casefold():
                if not citation.clause or (
                    candidate.clause_number
                    and citation.clause.casefold() == candidate.clause_number.casefold()
                ):
                    return True
    return False


def _classify_category(
    positive: CanonicalPassage,
    candidate: CanonicalPassage,
    *,
    source: MiningSource,
    sources: Sequence[MiningSource],
    lexical_overlap: float,
    lexical_hard_threshold: float,
) -> NegativeCategory:
    if candidate.status in {"expired", "repealed"} and candidate.source.content_hash:
        return "D"
    if _same_document(positive, candidate):
        if _same_article(positive, candidate):
            return "B"
        if positive.article_id or candidate.article_id:
            return "A"
        if positive.article_number or candidate.article_number:
            return "A"
    if lexical_overlap >= lexical_hard_threshold:
        return "C"
    if source == "dense" or "dense" in sources:
        return "E"
    return "F"


def _hardness_score(
    query: str,
    positive: CanonicalPassage,
    candidate: CanonicalPassage,
    *,
    rank: int,
) -> float:
    lexical = _lexical_overlap(query, candidate.reader_text)
    hierarchy = (
        1.0
        if _same_article(positive, candidate)
        else 0.65
        if _same_document(positive, candidate)
        else 0.0
    )
    rank_component = 1.0 / (rank + 1.0)
    return min(1.0, 0.5 * lexical + 0.35 * hierarchy + 0.15 * rank_component)


def _false_negative_flags(
    query: str,
    positive: CanonicalPassage,
    candidate: CanonicalPassage,
    *,
    source: MiningSource,
    lexical_overlap: float,
    retrieval_score: float,
    lexical_false_negative_threshold: float,
    dense_threshold: float,
) -> tuple[str, ...]:
    flags: list[str] = []
    if _same_article(positive, candidate):
        flags.append("same_article")
    if _citation_overlap(query, candidate):
        flags.append("citation_overlap")
    if lexical_overlap >= lexical_false_negative_threshold:
        flags.append("high_lexical_agreement")
    if source == "dense" and retrieval_score >= dense_threshold:
        flags.append("high_dense_agreement")
    if (
        candidate.passage_id in positive.references
        or positive.passage_id in candidate.references
    ):
        flags.append("reference_overlap")
    return tuple(flags)


def _stable_hard_negative_id(
    record: SyntheticQueryRecord,
    negative: HardNegative,
) -> str:
    payload = (
        record.synthetic_id,
        negative.negative_passage_id,
        negative.negative_category,
        negative.mined_from,
    )
    return "hn-" + hashlib.sha256("|".join(payload).encode()).hexdigest()[:24]


def _best_candidate_hits(
    hits: Sequence[CandidateHit],
) -> tuple[CandidateHit, ...]:
    """Deduplicate candidate IDs, retaining the strongest rank/source pair."""

    best: dict[str, CandidateHit] = {}
    source_priority = {"dense": 0, "bm25": 1, "rrf": 2, "candidate": 3}
    for hit in hits:
        if hit.rank <= 0 or not math.isfinite(hit.score):
            continue
        current = best.get(hit.passage_id)
        if current is None or (
            hit.rank,
            source_priority[hit.source],
            -hit.score,
        ) < (
            current.rank,
            source_priority[current.source],
            -current.score,
        ):
            best[hit.passage_id] = hit
    return tuple(sorted(best.values(), key=lambda hit: (hit.rank, hit.passage_id)))


def load_candidate_hits(
    paths: Sequence[str | Path],
) -> dict[str, tuple[CandidateHit, ...]]:
    """Load ranked JSONL from BM25, dense, or the existing RRF format."""

    def numeric_score(value: object) -> float:
        if value is None:
            return 0.0
        if not isinstance(value, (int, float, str)):
            raise ValueError(f"Candidate score is not numeric: {value!r}")
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Candidate score is not numeric: {value!r}") from exc

    grouped: defaultdict[str, list[CandidateHit]] = defaultdict(list)
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                query_id = payload.get("query_id", payload.get("synthetic_id"))
                if query_id is None:
                    raise ValueError(f"Candidate row lacks query_id: {path}")
                score_rows = [
                    item
                    for item in payload.get("scores", [])
                    if isinstance(item, Mapping) and item.get("passage_id") is not None
                ]
                fusion_rows = [
                    item
                    for item in payload.get("candidates", [])
                    if isinstance(item, Mapping) and item.get("passage_id") is not None
                ]
                is_fusion = bool(fusion_rows) and not score_rows
                rows = fusion_rows if is_fusion else score_rows
                rows_by_id = {str(item["passage_id"]): item for item in rows}
                ranked_ids = payload.get("ranked_ids", [])
                if not ranked_ids and rows:
                    ranked_ids = [
                        item["passage_id"]
                        for item in sorted(
                            rows,
                            key=lambda item: (
                                int(item.get("fused_rank", item.get("rank", 0))),
                                str(item["passage_id"]),
                            ),
                        )
                    ]
                for fallback_rank, passage_id in enumerate(ranked_ids, start=1):
                    score_row = rows_by_id.get(str(passage_id), {})
                    if is_fusion:
                        available_source_list: list[MiningSource] = []
                        if score_row.get("bm25_score") is not None:
                            available_source_list.append("bm25")
                        if score_row.get("dense_score") is not None:
                            available_source_list.append("dense")
                        available_sources = tuple(available_source_list)
                        if len(available_sources) > 1:
                            source: MiningSource = "rrf"
                            sources = available_sources
                        elif available_sources:
                            source = available_sources[0]
                            sources = available_sources
                        else:
                            source = "candidate"
                            sources = (source,)
                        raw_score = next(
                            (
                                score_row.get(field)
                                for field in (
                                    "rrf_score",
                                    "dense_score",
                                    "bm25_score",
                                )
                                if score_row.get(field) is not None
                            ),
                            0.0,
                        )
                        raw_rank = score_row.get(
                            "fused_rank",
                            score_row.get("rank", fallback_rank),
                        )
                    else:
                        raw_source = str(
                            score_row.get("source", payload.get("source", "candidate"))
                        )
                        if raw_source == "bm25":
                            source = "bm25"
                        elif raw_source == "dense":
                            source = "dense"
                        else:
                            source = "candidate"
                        sources = ()
                        raw_score = next(
                            (
                                score_row.get(name)
                                for name in ("dense", "bm25", "rrf", "score")
                                if score_row.get(name) is not None
                            ),
                            0.0,
                        )
                        raw_rank = score_row.get("rank", fallback_rank)
                    try:
                        rank = int(raw_rank)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Candidate rank is not an integer: {raw_rank!r}"
                        ) from exc
                    grouped[str(query_id)].append(
                        CandidateHit(
                            passage_id=str(passage_id),
                            rank=rank,
                            score=numeric_score(raw_score),
                            source=source,
                            sources=sources,
                        )
                    )
    return {query_id: _best_candidate_hits(hits) for query_id, hits in grouped.items()}


def _random_baseline_score(
    query: str,
    positive: CanonicalPassage,
    passages: Sequence[CanonicalPassage],
    *,
    rank: int,
    rng: random.Random,
) -> float:
    if len(passages) < 2:
        return 0.0
    candidate = positive
    for _ in range(100):
        candidate = passages[rng.randrange(len(passages))]
        if candidate.document_id != positive.document_id:
            break
    return _hardness_score(query, positive, candidate, rank=rank)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def mine_hard_negatives(
    records: Sequence[SyntheticQueryRecord],
    passages: Sequence[CanonicalPassage],
    candidate_hits: Mapping[str, Sequence[CandidateHit]],
    *,
    config: HardNegativeMiningConfig | None = None,
) -> tuple[tuple[HardNegativeRecord, ...], HardNegativeMiningReport]:
    """Mine resolved negatives and return machine gate diagnostics."""

    cfg = config or HardNegativeMiningConfig()
    passage_map = {passage.passage_id: passage for passage in passages}
    if len(passage_map) != len(passages):
        raise ValueError("Passage corpus contains duplicate canonical IDs")
    rng = random.Random(cfg.seed)
    outputs: list[HardNegativeRecord] = []
    rejection_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    unresolved_count = 0
    positive_overlap_count = 0
    potential_fn_count = 0
    hard_scores: list[float] = []
    random_scores: list[float] = []

    for record in records:
        if record.source_split != "train":
            rejection_counts["non_train_source_split"] += 1
            continue
        positive = passage_map.get(record.positive_passage_id)
        if positive is None:
            rejection_counts["missing_positive"] += 1
            continue
        if record.source_document_id != positive.document_id:
            rejection_counts["positive_document_mismatch"] += 1
            continue
        selected: list[HardNegative] = []
        seen_ids: set[str] = set()
        for hit in _best_candidate_hits(candidate_hits.get(record.synthetic_id, ())):
            if hit.passage_id == record.positive_passage_id:
                continue
            if hit.passage_id in seen_ids:
                continue
            candidate = passage_map.get(hit.passage_id)
            if candidate is None:
                unresolved_count += 1
                continue
            seen_ids.add(hit.passage_id)
            lexical_overlap = _lexical_overlap(record.query, candidate.reader_text)
            category = _classify_category(
                positive,
                candidate,
                source=hit.source,
                sources=hit.sources or (hit.source,),
                lexical_overlap=lexical_overlap,
                lexical_hard_threshold=cfg.lexical_hard_threshold,
            )
            flags = _false_negative_flags(
                record.query,
                positive,
                candidate,
                source=hit.source,
                lexical_overlap=lexical_overlap,
                retrieval_score=hit.score,
                lexical_false_negative_threshold=cfg.lexical_false_negative_threshold,
                dense_threshold=cfg.dense_false_negative_threshold,
            )
            hardness = _hardness_score(
                record.query,
                positive,
                candidate,
                rank=hit.rank,
            )
            selected.append(
                HardNegative(
                    negative_passage_id=candidate.passage_id,
                    negative_document_id=candidate.document_id,
                    negative_article_id=candidate.article_id,
                    negative_clause_id=candidate.clause_id,
                    negative_source_hash=candidate.source.content_hash,
                    negative_category=category,
                    mined_from=hit.source,
                    mining_sources=hit.sources or (hit.source,),
                    rank=hit.rank,
                    retrieval_score=hit.score,
                    hardness_score=hardness,
                    potential_false_negative=bool(flags),
                    false_negative_flags=flags,
                )
            )

        selected.sort(
            key=lambda negative: (
                -negative.hardness_score,
                negative.rank,
                negative.negative_passage_id,
            )
        )
        selected = selected[: cfg.max_negatives]
        if len(selected) < cfg.min_negatives:
            rejection_counts["insufficient_resolved_negatives"] += 1
            continue
        if any(
            negative.negative_passage_id == record.positive_passage_id
            for negative in selected
        ):
            positive_overlap_count += 1
            rejection_counts["positive_in_negatives"] += 1
            continue
        hard_id = _stable_hard_negative_id(
            record,
            selected[0],
        )
        outputs.append(
            HardNegativeRecord(
                hard_negative_id=hard_id,
                synthetic_id=record.synthetic_id,
                query=record.query,
                query_type=record.query_type,
                positive_passage_id=record.positive_passage_id,
                source_document_id=record.source_document_id,
                positive_source_hash=record.source_hash,
                negatives=tuple(selected),
                source_split="train",
            )
        )
        for negative in selected:
            category_counts[negative.negative_category] += 1
            source_counts[negative.mined_from] += 1
            hard_scores.append(negative.hardness_score)
            potential_fn_count += int(negative.potential_false_negative)
        random_scores.append(
            _random_baseline_score(
                record.query,
                positive,
                passages,
                rank=cfg.random_rank,
                rng=rng,
            )
        )

    hard_mean = sum(hard_scores) / len(hard_scores) if hard_scores else 0.0
    random_mean = sum(random_scores) / len(random_scores) if random_scores else 0.0
    negative_count = len(hard_scores)
    report = HardNegativeMiningReport(
        query_count=len(records),
        accepted_query_count=len(outputs),
        rejected_query_count=len(records) - len(outputs),
        negative_count=negative_count,
        unresolved_candidate_count=unresolved_count,
        positive_in_negative_count=positive_overlap_count,
        potential_false_negative_count=potential_fn_count,
        rejection_counts=dict(rejection_counts),
        category_counts=dict(category_counts),
        source_counts=dict(source_counts),
        hard_score_mean=hard_mean,
        hard_score_p50=_percentile(hard_scores, 0.50),
        hard_score_p95=_percentile(hard_scores, 0.95),
        random_score_mean=random_mean,
        random_score_p50=_percentile(random_scores, 0.50),
        random_score_p95=_percentile(random_scores, 0.95),
        harder_than_random=(
            bool(hard_scores)
            and hard_mean > random_mean
            and _percentile(hard_scores, 0.50) >= _percentile(random_scores, 0.50)
        ),
        estimated_false_negative_rate=(
            potential_fn_count / negative_count if negative_count else 0.0
        ),
    )
    return tuple(outputs), report


def build_audit_sample(
    records: Sequence[HardNegativeRecord],
    *,
    sample_size: int = 300,
    seed: int = 42,
) -> tuple[dict[str, object], ...]:
    """Create a deterministic pair-level audit sample."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    pairs: list[dict[str, object]] = []
    for record in records:
        for negative in record.negatives:
            pairs.append(
                {
                    "synthetic_id": record.synthetic_id,
                    "query": record.query,
                    "query_type": record.query_type,
                    "positive_passage_id": record.positive_passage_id,
                    "negative_passage_id": negative.negative_passage_id,
                    "negative_category": negative.negative_category,
                    "mined_from": negative.mined_from,
                    "hardness_score": negative.hardness_score,
                    "potential_false_negative": negative.potential_false_negative,
                    "false_negative_flags": list(negative.false_negative_flags),
                    "manual_negative_is_valid": None,
                    "manual_false_negative": None,
                    "reviewer_note": "",
                }
            )
    if len(pairs) <= sample_size:
        return tuple(pairs)
    rng = random.Random(seed)
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["negative_category"])].append(pair)
    for group in grouped.values():
        rng.shuffle(group)
    selected: list[dict[str, object]] = []
    categories = sorted(grouped)
    while len(selected) < sample_size and categories:
        progressed = False
        for category in categories:
            if grouped[category] and len(selected) < sample_size:
                selected.append(grouped[category].pop())
                progressed = True
        if not progressed:
            break
    return tuple(selected)


def write_jsonl_models(
    path: str | Path,
    records: Sequence[DomainModel | Mapping[str, object]],
    *,
    overwrite: bool = False,
) -> None:
    """Write typed records or audit dictionaries as JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
    with output.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            payload = (
                record.model_dump(mode="json")
                if isinstance(record, DomainModel)
                else dict(record)
            )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = [
    "CandidateHit",
    "HardNegative",
    "HardNegativeMiningConfig",
    "HardNegativeMiningReport",
    "HardNegativeRecord",
    "MiningSource",
    "NegativeCategory",
    "TASK10_SCHEMA_VERSION",
    "build_audit_sample",
    "load_candidate_hits",
    "mine_hard_negatives",
    "write_jsonl_models",
]
