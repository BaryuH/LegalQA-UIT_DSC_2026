"""LightGBM LambdaRank training and inference for SEDAR Retrieval TASK 13."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines
from legal_rag.sedar_retrieval.ranking.features import FEATURE_NAMES

LTR_RANKER_SCHEMA_VERSION = "sedar-retrieval-v3-ltr-ranker-v1"
FeatureGroupName = Literal[
    "all",
    "no_lexical",
    "no_citation",
    "no_hierarchy",
    "no_dense",
]

FEATURE_GROUPS: dict[str, frozenset[str]] = {
    "lexical": frozenset(
        {
            "query_token_coverage",
            "idf_weighted_overlap",
            "bigram_overlap",
            "exact_phrase_match",
            "legal_term_overlap",
        }
    ),
    "citation": frozenset(
        {
            "document_name_match",
            "document_number_match",
            "year_match",
            "article_number_match",
            "clause_number_match",
            "point_match",
        }
    ),
    "hierarchy": frozenset(
        {
            "article_title_similarity",
            "chapter_title_similarity",
            "is_effective",
            "is_expired",
        }
    ),
    "dense": frozenset({"dense_score", "dense_rank"}),
}


class LTRRankerError(ValueError):
    """Raised when LambdaRank training or inference cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class LTRRankerConfig:
    """Deterministic LightGBM LambdaRank hyperparameters."""

    feature_group: FeatureGroupName = "all"
    validation_fraction: float = 0.1
    seed: int = 42
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    max_depth: int = -1
    early_stopping_rounds: int = 30
    promotion_min_gain: float = 0.01
    recall_at_20_max_regression: float = 0.005

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.n_estimators <= 0 or self.num_leaves <= 0:
            raise ValueError("n_estimators and num_leaves must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class LTRFeatureExample:
    """One labeled candidate row used for training or offline scoring."""

    query_id: str
    passage_id: str
    label: int
    features: dict[str, float]
    rank: int | None = None


@dataclass(frozen=True, slots=True)
class RankerMatrices:
    """Dense matrices aligned with LightGBM group semantics."""

    query_ids: tuple[str, ...]
    passage_ids: tuple[str, ...]
    labels: tuple[int, ...]
    groups: tuple[int, ...]
    matrix: tuple[tuple[float, ...], ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LTRTrainSplit:
    """Train/validation matrices split by query_id."""

    train: RankerMatrices
    validation: RankerMatrices
    train_query_ids: tuple[str, ...]
    validation_query_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LTRTrainingResult:
    """Artifacts and counters for one LambdaRank training run."""

    run_id: str
    output_dir: Path
    model_path: Path
    manifest_path: Path
    metrics_path: Path
    feature_names: tuple[str, ...]
    train_query_count: int
    validation_query_count: int
    best_iteration: int | None
    validation_ndcg_at_10: float | None
    validation_mrr_at_10: float | None
    status: Literal["PASS"]

    def as_dict(self) -> dict[str, object]:
        return {
            "best_iteration": self.best_iteration,
            "feature_names": list(self.feature_names),
            "manifest_path": str(self.manifest_path),
            "metrics_path": str(self.metrics_path),
            "model_path": str(self.model_path),
            "output_dir": str(self.output_dir),
            "run_id": self.run_id,
            "status": self.status,
            "train_query_count": self.train_query_count,
            "validation_mrr_at_10": self.validation_mrr_at_10,
            "validation_ndcg_at_10": self.validation_ndcg_at_10,
            "validation_query_count": self.validation_query_count,
        }


def resolve_feature_names(
    feature_group: FeatureGroupName = "all",
) -> tuple[str, ...]:
    """Return ordered feature names for one ablation group."""

    names = list(FEATURE_NAMES)
    if feature_group == "all":
        return tuple(names)
    drop_key = feature_group.removeprefix("no_")
    drop = FEATURE_GROUPS.get(drop_key)
    if drop is None:
        raise LTRRankerError(f"Unsupported feature group: {feature_group!r}")
    selected = [name for name in names if name not in drop]
    if not selected:
        raise LTRRankerError(f"Feature group removed all features: {feature_group!r}")
    return tuple(selected)


def load_feature_examples(path: str | Path) -> tuple[LTRFeatureExample, ...]:
    """Load TASK 12 feature rows without reading answer fields."""

    examples: list[LTRFeatureExample] = []
    for line in iter_jsonl_lines(path):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise LTRRankerError("Feature rows must be JSON objects")
        query_id = payload.get("query_id")
        passage_id = payload.get("passage_id")
        label = payload.get("label")
        features = payload.get("features")
        if query_id is None or not str(query_id).strip():
            raise LTRRankerError("Feature row query_id must be non-blank")
        if passage_id is None or not str(passage_id).strip():
            raise LTRRankerError("Feature row passage_id must be non-blank")
        if isinstance(label, bool) or not isinstance(label, int):
            raise LTRRankerError("Feature row label must be an integer")
        if not isinstance(features, dict):
            raise LTRRankerError("Feature row features must be an object")
        cleaned: dict[str, float] = {}
        for name in FEATURE_NAMES:
            if name not in features:
                raise LTRRankerError(f"Feature row missing feature: {name}")
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LTRRankerError(f"Feature {name} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise LTRRankerError(f"Feature {name} must be finite")
            cleaned[name] = number
        rank_raw = payload.get("rank")
        rank: int | None = None
        if isinstance(rank_raw, int) and not isinstance(rank_raw, bool):
            rank = rank_raw
        examples.append(
            LTRFeatureExample(
                query_id=str(query_id),
                passage_id=str(passage_id),
                label=label,
                features=cleaned,
                rank=rank,
            )
        )
    if not examples:
        raise LTRRankerError("No feature rows were loaded")
    return tuple(examples)


def _sorted_query_ids(examples: Sequence[LTRFeatureExample]) -> tuple[str, ...]:
    return tuple(sorted({example.query_id for example in examples}))


def split_query_ids(
    query_ids: Sequence[str],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministically split whole query groups into train/validation."""

    if not 0.0 < validation_fraction < 1.0:
        raise LTRRankerError("validation_fraction must be in (0, 1)")
    ordered = sorted(set(query_ids))
    if len(ordered) < 2:
        raise LTRRankerError("Need at least two query groups to split")
    scored: list[tuple[str, str]] = []
    for query_id in ordered:
        digest = hashlib.sha256(f"{seed}:{query_id}".encode()).hexdigest()
        scored.append((digest, query_id))
    scored.sort()
    validation_count = max(1, int(round(len(ordered) * validation_fraction)))
    validation_count = min(validation_count, len(ordered) - 1)
    validation = tuple(query_id for _, query_id in scored[:validation_count])
    train = tuple(query_id for _, query_id in scored[validation_count:])
    if set(train) & set(validation):
        raise LTRRankerError("Train/validation query split leaked query_ids")
    return train, validation


def build_ranker_matrices(
    examples: Sequence[LTRFeatureExample],
    *,
    feature_names: Sequence[str],
    query_ids: Sequence[str] | None = None,
) -> RankerMatrices:
    """Build LightGBM-ready matrices grouped by query_id."""

    selected = set(query_ids) if query_ids is not None else None
    grouped: defaultdict[str, list[LTRFeatureExample]] = defaultdict(list)
    for example in examples:
        if selected is not None and example.query_id not in selected:
            continue
        grouped[example.query_id].append(example)
    if not grouped:
        raise LTRRankerError("No examples remained after query filtering")

    query_order = sorted(grouped)
    out_query_ids: list[str] = []
    out_passage_ids: list[str] = []
    labels: list[int] = []
    groups: list[int] = []
    rows: list[tuple[float, ...]] = []
    names = tuple(feature_names)
    for query_id in query_order:
        items = sorted(
            grouped[query_id],
            key=lambda item: (
                item.rank if item.rank is not None else 10**9,
                item.passage_id,
            ),
        )
        groups.append(len(items))
        for item in items:
            out_query_ids.append(query_id)
            out_passage_ids.append(item.passage_id)
            labels.append(item.label)
            rows.append(tuple(item.features[name] for name in names))
    return RankerMatrices(
        query_ids=tuple(out_query_ids),
        passage_ids=tuple(out_passage_ids),
        labels=tuple(labels),
        groups=tuple(groups),
        matrix=tuple(rows),
        feature_names=names,
    )


def build_train_split(
    examples: Sequence[LTRFeatureExample],
    *,
    config: LTRRankerConfig | None = None,
) -> LTRTrainSplit:
    """Split feature rows by query_id and materialize train/val matrices."""

    cfg = config or LTRRankerConfig()
    feature_names = resolve_feature_names(cfg.feature_group)
    train_ids, val_ids = split_query_ids(
        _sorted_query_ids(examples),
        validation_fraction=cfg.validation_fraction,
        seed=cfg.seed,
    )
    train = build_ranker_matrices(
        examples,
        feature_names=feature_names,
        query_ids=train_ids,
    )
    validation = build_ranker_matrices(
        examples,
        feature_names=feature_names,
        query_ids=val_ids,
    )
    return LTRTrainSplit(
        train=train,
        validation=validation,
        train_query_ids=train_ids,
        validation_query_ids=val_ids,
    )


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise LTRRankerError(
            "LambdaRank training requires lightgbm; install the sedar-ltr extra."
        ) from exc
    return lgb


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise LTRRankerError(
            "LambdaRank training requires numpy; install the sedar-ltr extra."
        ) from exc
    return np


def _ndcg_at_k(labels: Sequence[int], scores: Sequence[float], *, k: int) -> float:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    gains = [labels[index] for index in order[:k]]
    dcg = 0.0
    for rank, gain in enumerate(gains, start=1):
        dcg += (2**gain - 1) / math.log2(rank + 1)
    ideal = sorted(labels, reverse=True)[:k]
    idcg = 0.0
    for rank, gain in enumerate(ideal, start=1):
        idcg += (2**gain - 1) / math.log2(rank + 1)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def _mrr_at_k(labels: Sequence[int], scores: Sequence[float], *, k: int) -> float:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    for rank, index in enumerate(order[:k], start=1):
        if labels[index] > 0:
            return 1.0 / rank
    return 0.0


def evaluate_ranker_matrices(
    matrices: RankerMatrices,
    scores: Sequence[float],
    *,
    k: int = 10,
) -> dict[str, float]:
    """Compute mean nDCG@K and MRR@K over query groups."""

    if len(scores) != len(matrices.labels):
        raise LTRRankerError("Score count does not match label count")
    cursor = 0
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    for group_size in matrices.groups:
        group_labels = matrices.labels[cursor : cursor + group_size]
        group_scores = scores[cursor : cursor + group_size]
        ndcg_values.append(_ndcg_at_k(group_labels, group_scores, k=k))
        mrr_values.append(_mrr_at_k(group_labels, group_scores, k=k))
        cursor += group_size
    return {
        f"ndcg_at_{k}": sum(ndcg_values) / max(len(ndcg_values), 1),
        f"mrr_at_{k}": sum(mrr_values) / max(len(mrr_values), 1),
        "query_count": float(len(matrices.groups)),
    }


def train_lambdarank(
    split: LTRTrainSplit,
    *,
    output_dir: Path,
    config: LTRRankerConfig | None = None,
    feature_schema_hash: str | None = None,
    features_path: str | None = None,
) -> LTRTrainingResult:
    """Train LightGBM LGBMRanker and persist model + manifest."""

    cfg = config or LTRRankerConfig()
    lgb = _require_lightgbm()
    np = _require_numpy()
    run_id = new_run_id("ltr_lambdarank")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.txt"
    manifest_path = output_dir / "train_manifest.json"
    metrics_path = output_dir / "train_metrics.json"

    x_train = np.asarray(split.train.matrix, dtype="float32")
    y_train = np.asarray(split.train.labels, dtype="float32")
    x_val = np.asarray(split.validation.matrix, dtype="float32")
    y_val = np.asarray(split.validation.labels, dtype="float32")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        max_depth=cfg.max_depth,
        random_state=cfg.seed,
        n_jobs=1,
        importance_type="gain",
        label_gain=list(range(0, max(max(split.train.labels, default=0), 3) + 1)),
    )
    fit_kwargs: dict[str, Any] = {
        "X": x_train,
        "y": y_train,
        "group": list(split.train.groups),
        "eval_set": [(x_val, y_val)],
        "eval_group": [list(split.validation.groups)],
        "eval_at": [10],
        "feature_name": list(split.train.feature_names),
    }
    if cfg.early_stopping_rounds > 0:
        callbacks = [lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)]
        fit_kwargs["callbacks"] = callbacks
    ranker.fit(**fit_kwargs)
    ranker.booster_.save_model(str(model_path))

    val_scores = ranker.predict(x_val)
    val_metrics = evaluate_ranker_matrices(split.validation, val_scores, k=10)
    best_iteration = getattr(ranker, "best_iteration_", None)
    metrics_payload = {
        "schema_version": LTR_RANKER_SCHEMA_VERSION,
        "run_id": run_id,
        "validation": val_metrics,
        "best_iteration": best_iteration,
        "feature_group": cfg.feature_group,
        "feature_names": list(split.train.feature_names),
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": LTR_RANKER_SCHEMA_VERSION,
        "task": "TASK13",
        "run_id": run_id,
        "git_commit": git_commit_sha(Path.cwd()),
        "status": "PASS",
        "objective": "lambdarank",
        "features_path": features_path,
        "feature_schema_hash": feature_schema_hash,
        "feature_group": cfg.feature_group,
        "feature_names": list(split.train.feature_names),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "train_query_count": len(split.train_query_ids),
        "validation_query_count": len(split.validation_query_ids),
        "train_row_count": len(split.train.labels),
        "validation_row_count": len(split.validation.labels),
        "hyperparameters": {
            "seed": cfg.seed,
            "validation_fraction": cfg.validation_fraction,
            "num_leaves": cfg.num_leaves,
            "learning_rate": cfg.learning_rate,
            "n_estimators": cfg.n_estimators,
            "min_child_samples": cfg.min_child_samples,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_lambda": cfg.reg_lambda,
            "max_depth": cfg.max_depth,
            "early_stopping_rounds": cfg.early_stopping_rounds,
        },
        "promotion": {
            "promotion_min_gain": cfg.promotion_min_gain,
            "recall_at_20_max_regression": cfg.recall_at_20_max_regression,
        },
        "leakage_policy": {
            "answer_text_used": False,
            "gold_labels_used_as_features": False,
            "reader_outputs_used": False,
            "split_unit": "query_id",
        },
        "best_iteration": best_iteration,
        "validation_ndcg_at_10": val_metrics["ndcg_at_10"],
        "validation_mrr_at_10": val_metrics["mrr_at_10"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return LTRTrainingResult(
        run_id=run_id,
        output_dir=output_dir,
        model_path=model_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        feature_names=split.train.feature_names,
        train_query_count=len(split.train_query_ids),
        validation_query_count=len(split.validation_query_ids),
        best_iteration=int(best_iteration) if best_iteration is not None else None,
        validation_ndcg_at_10=float(val_metrics["ndcg_at_10"]),
        validation_mrr_at_10=float(val_metrics["mrr_at_10"]),
        status="PASS",
    )


def load_ranker_bundle(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load booster and train manifest from an output directory."""

    lgb = _require_lightgbm()
    manifest_path = model_dir / "train_manifest.json"
    model_path = model_dir / "model.txt"
    if not manifest_path.is_file() or not model_path.is_file():
        raise LTRRankerError(f"Incomplete LTR model directory: {model_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise LTRRankerError("LTR train manifest must be a JSON object")
    if manifest.get("schema_version") != LTR_RANKER_SCHEMA_VERSION:
        raise LTRRankerError(
            f"Unsupported LTR ranker schema: {manifest.get('schema_version')!r}"
        )
    booster = lgb.Booster(model_file=str(model_path))
    return booster, manifest


def score_feature_matrix(
    booster: Any,
    matrices: RankerMatrices,
) -> tuple[float, ...]:
    """Score rows with a saved booster; deterministic tie order left to caller."""

    np = _require_numpy()
    scores = booster.predict(np.asarray(matrices.matrix, dtype="float32"))
    return tuple(float(score) for score in scores)


def rerank_from_scores(
    matrices: RankerMatrices,
    scores: Sequence[float],
    *,
    top_k: int = 0,
) -> tuple[dict[str, object], ...]:
    """Emit ranked candidate JSONL rows compatible with eval_retrieval."""

    if top_k < 0:
        raise LTRRankerError("top_k must be non-negative")
    if len(scores) != len(matrices.passage_ids):
        raise LTRRankerError("Score count does not match passage count")
    cursor = 0
    output: list[dict[str, object]] = []
    for group_size in matrices.groups:
        query_id = matrices.query_ids[cursor]
        group_passages = matrices.passage_ids[cursor : cursor + group_size]
        group_scores = scores[cursor : cursor + group_size]
        ranked = sorted(
            zip(group_passages, group_scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        if top_k:
            ranked = ranked[:top_k]
        output.append(
            {
                "query_id": query_id,
                "ranked_ids": [passage_id for passage_id, _ in ranked],
                "scores": [
                    {
                        "passage_id": passage_id,
                        "ltr": score,
                        "rank": rank,
                        "source": "ltr",
                    }
                    for rank, (passage_id, score) in enumerate(ranked, start=1)
                ],
            }
        )
        cursor += group_size
    return tuple(output)


def ranked_ids_overlap(
    left: Sequence[str],
    right: Sequence[str],
    *,
    top_k: int,
) -> float:
    """Exact top-k identity overlap for online/offline parity checks."""

    left_top = list(left[:top_k])
    right_top = list(right[:top_k])
    if left_top == right_top:
        return 1.0
    return 0.0


__all__ = [
    "FEATURE_GROUPS",
    "LTRFeatureExample",
    "LTRRankerConfig",
    "LTRRankerError",
    "LTRTrainSplit",
    "LTRTrainingResult",
    "LTR_RANKER_SCHEMA_VERSION",
    "RankerMatrices",
    "build_ranker_matrices",
    "build_train_split",
    "evaluate_ranker_matrices",
    "load_feature_examples",
    "load_ranker_bundle",
    "ranked_ids_overlap",
    "rerank_from_scores",
    "resolve_feature_names",
    "score_feature_matrix",
    "split_query_ids",
    "train_lambdarank",
]
