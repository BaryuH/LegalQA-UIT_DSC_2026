#!/usr/bin/env python3
"""Measure quality and diversity of BM25/Qwen/Legal retrieval sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.eval.ensemble_metrics import (
    EnsembleDiagnosticError,
    diagnose_ensemble,
    load_ranked_source,
    load_relevance_labels,
)


def _parse_cutoffs(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise SystemExit(f"Invalid cutoff: {item!r}") from exc
        if value <= 0:
            raise SystemExit("Cutoffs must be positive integers")
        values.append(value)
    if not values:
        raise SystemExit("At least one cutoff is required")
    return tuple(sorted(set(values)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25", type=Path, default=None)
    parser.add_argument(
        "--qwen",
        "--dense",
        dest="qwen",
        type=Path,
        required=True,
        help="Qwen dense retrieval JSONL (the existing dense source).",
    )
    parser.add_argument("--legal", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional evaluation-only silver/gold relevance JSONL.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--cutoffs", default="10,50,100")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite artifact: {args.output}")

    try:
        sources = {
            "dense": load_ranked_source(args.qwen, source_name="dense"),
            "legal": load_ranked_source(args.legal, source_name="legal"),
        }
        if args.bm25 is not None:
            sources["bm25"] = load_ranked_source(args.bm25, source_name="bm25")
        labels = load_relevance_labels(args.labels) if args.labels else None
        report = diagnose_ensemble(
            sources,
            labels=labels,
            top_k=args.top_k,
            cutoffs=_parse_cutoffs(args.cutoffs),
        )
    except (OSError, ValueError, EnsembleDiagnosticError) as exc:
        raise SystemExit(f"ENSEMBLE_DIAGNOSTIC_FAILED: {exc}") from exc

    payload = report.as_dict()
    payload["inputs"] = {
        "bm25": str(args.bm25) if args.bm25 else None,
        "qwen": str(args.qwen),
        "legal": str(args.legal),
        "labels": str(args.labels) if args.labels else None,
        "labels_role": "evaluation_only" if args.labels else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "query_count": report.query_count,
                "labeled_query_count": report.labeled_query_count,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
