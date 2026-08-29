#!/usr/bin/env python3
"""Check frozen-reader ensemble promotion boundaries (TASK 22)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.eval.ensemble_promotion import (
    EnsemblePromotionError,
    evaluate_ensemble_promotion,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-eval-dir", type=Path, required=True)
    parser.add_argument("--candidate-eval-dir", type=Path, required=True)
    parser.add_argument("--min-gain", type=float, default=0.003)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite artifact: {args.output}")
    try:
        report = evaluate_ensemble_promotion(
            baseline_run_dir=args.baseline_run_dir,
            candidate_run_dir=args.candidate_run_dir,
            baseline_eval_dir=args.baseline_eval_dir,
            candidate_eval_dir=args.candidate_eval_dir,
            min_gain=args.min_gain,
        )
    except (EnsemblePromotionError, OSError, ValueError) as exc:
        raise SystemExit(f"ENSEMBLE_PROMOTION_CHECK_FAILED: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report.status,
                "recommendation": report.recommendation,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
