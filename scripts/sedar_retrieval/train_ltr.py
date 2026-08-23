#!/usr/bin/env python3
"""Train LightGBM LambdaRank on TASK 12 feature rows (TASK 13)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.ranking.ltr_dataset import validate_feature_schema
from legal_rag.sedar_retrieval.ranking.ltr_ranker import (
    LTRRankerConfig,
    LTRRankerError,
    build_train_split,
    load_feature_examples,
    train_lambdarank,
)


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise SystemExit(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not force:
        raise SystemExit(
            f"Output directory is non-empty: {path}; use a new run directory "
            "or pass --force explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("configs/retrieval/ltr_feature_schema_v1.json"),
    )
    parser.add_argument(
        "--feature-group",
        choices=("all", "no_lexical", "no_citation", "no_hierarchy", "no_dense"),
        default="all",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.limit_queries < 0:
        raise SystemExit("--limit-queries must be non-negative")

    _prepare_output_dir(args.output_dir, force=args.force)
    config = LTRRankerConfig(
        feature_group=args.feature_group,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        max_depth=args.max_depth,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    try:
        schema_hash = validate_feature_schema(args.schema)
        examples = load_feature_examples(args.features)
        if args.limit_queries:
            keep = {
                query_id
                for query_id in sorted({item.query_id for item in examples})[
                    : args.limit_queries
                ]
            }
            examples = tuple(
                item for item in examples if item.query_id in keep
            )
        split = build_train_split(examples, config=config)
        result = train_lambdarank(
            split,
            output_dir=args.output_dir,
            config=config,
            feature_schema_hash=schema_hash,
            features_path=str(args.features),
        )
    except (OSError, ValueError, LTRRankerError) as exc:
        raise SystemExit(f"TASK13_LTR_TRAIN_FAILED: {exc}") from exc

    print(json.dumps(result.as_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
