#!/usr/bin/env python3
"""Analyze clean-warmup retrieval, reranking, and reader failure stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

from legal_rag.sedar_retrieval.eval.retrieval_error_analysis import (  # noqa: E402
    RetrievalErrorAnalysisError,
    analyze_warmup_retrieval_errors,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the warmup retrieval-error analysis CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="TASK20 clean-warmup run directory containing retrieval.jsonl.",
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--bm25",
        type=Path,
        required=True,
        help="BM25 ranked source JSONL used to build the candidate pool.",
    )
    parser.add_argument(
        "--qwen",
        type=Path,
        required=True,
        help="Qwen dense ranked source JSONL used to build the candidate pool.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Evaluation-only silver relevance labels JSONL.",
    )
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--candidate-cutoff", type=int, default=100)
    parser.add_argument("--ltr-cutoff", type=int, default=20)
    parser.add_argument("--packed-cutoff", type=int, default=4)
    parser.add_argument("--low-meteor-threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the analysis and write one evaluation-only JSON artifact."""

    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.force:
        print(f"Refusing to overwrite artifact: {args.output}", file=sys.stderr)
        return 2
    try:
        report = analyze_warmup_retrieval_errors(
            run_dir=args.run_dir,
            metrics_path=args.metrics,
            bm25_path=args.bm25,
            qwen_path=args.qwen,
            labels_path=args.labels,
            passages_path=args.passages,
            candidate_cutoff=args.candidate_cutoff,
            ltr_cutoff=args.ltr_cutoff,
            packed_cutoff=args.packed_cutoff,
            low_meteor_threshold=args.low_meteor_threshold,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                report.as_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except (RetrievalErrorAnalysisError, OSError, ValueError) as exc:
        print(f"RETRIEVAL_ERROR_ANALYSIS_FAILED: {exc}", file=sys.stderr)
        return 2

    payload = report.as_dict()
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "query_count": payload["query_count"],
                "labeled_query_count": payload["labeled_query_count"],
                "failure_counts": payload["failure_counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
