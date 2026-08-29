#!/usr/bin/env python3
"""Rerank candidates with a trained TASK 13 LambdaRank model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.ranking.ltr_ranker import (
    LTRRankerError,
    build_ranker_matrices,
    load_feature_examples,
    load_ranker_bundle,
    ranked_ids_overlap,
    rerank_from_scores,
    resolve_feature_names,
    score_feature_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument(
        "--parity-features",
        type=Path,
        default=None,
        help="Optional second feature file for online/offline top-k overlap check.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.top_k < 0:
        raise SystemExit("--top-k must be non-negative")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite artifact: {args.output}")

    try:
        booster, manifest = load_ranker_bundle(args.model_dir)
        feature_group = str(manifest.get("feature_group", "all"))
        raw_feature_names = manifest.get("feature_names")
        if raw_feature_names is not None and not isinstance(raw_feature_names, list):
            raise LTRRankerError("LTR manifest feature_names must be a list")
        if raw_feature_names is not None and not all(
            isinstance(name, str) and name.strip() for name in raw_feature_names
        ):
            raise LTRRankerError("LTR manifest feature_names must contain strings")
        feature_names = tuple(raw_feature_names or ())
        if not feature_names:
            feature_names = resolve_feature_names(feature_group)  # type: ignore[arg-type]
        examples = load_feature_examples(
            args.features,
            feature_names=feature_names,
        )
        matrices = build_ranker_matrices(examples, feature_names=feature_names)
        scores = score_feature_matrix(booster, matrices)
        ranked = rerank_from_scores(matrices, scores, top_k=args.top_k)

        parity_overlap = None
        if args.parity_features is not None:
            parity_examples = load_feature_examples(
                args.parity_features,
                feature_names=feature_names,
            )
            parity_matrices = build_ranker_matrices(
                parity_examples,
                feature_names=feature_names,
            )
            parity_scores = score_feature_matrix(booster, parity_matrices)
            parity_ranked = rerank_from_scores(
                parity_matrices,
                parity_scores,
                top_k=args.top_k,
            )
            left = {row["query_id"]: row["ranked_ids"] for row in ranked}
            right = {row["query_id"]: row["ranked_ids"] for row in parity_ranked}
            if set(left) != set(right):
                raise LTRRankerError(
                    "Parity feature file query_id set does not match primary features"
                )
            overlaps = [
                ranked_ids_overlap(left[query_id], right[query_id], top_k=args.top_k)
                for query_id in sorted(left)
            ]
            parity_overlap = sum(overlaps) / max(len(overlaps), 1)
            if parity_overlap != 1.0:
                raise LTRRankerError(
                    f"Online/offline top-k overlap is {parity_overlap}, expected 1.0"
                )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            for row in ranked:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except (OSError, ValueError, LTRRankerError) as exc:
        raise SystemExit(f"TASK13_LTR_RANK_FAILED: {exc}") from exc

    print(
        json.dumps(
            {
                "status": "PASS",
                "n_queries": len(ranked),
                "top_k": args.top_k,
                "parity_topk_overlap": parity_overlap,
                "output": str(args.output),
                "model_dir": str(args.model_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
