#!/usr/bin/env python3
"""Build silver evaluation labels from answer citations (evaluation-only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.eval.silver_labels import build_silver_labels_from_answers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/warmup.json"))
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sedar_retrieval/eval/warmup_silver_labels.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    stats = build_silver_labels_from_answers(
        questions_path=args.questions,
        passages_path=args.passages,
        output_path=args.output,
        limit=args.limit,
    )
    print(json.dumps({"output": str(args.output), **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
