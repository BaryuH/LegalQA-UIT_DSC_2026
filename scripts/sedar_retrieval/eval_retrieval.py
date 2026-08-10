#!/usr/bin/env python3
"""Evaluate retrieval predictions with SEDAR Retrieval v3 metrics (TASK 02)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.eval.retrieval_metrics import (
    QueryRelevance,
    RankedList,
    evaluate_retrieval,
    metrics_to_dict,
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preds_raw = _load_jsonl(args.pred)
    labels_raw = _load_jsonl(args.labels)

    predictions = [
        RankedList(
            query_id=str(row["query_id"]),
            ranked_ids=tuple(str(x) for x in row["ranked_ids"]),  # type: ignore[index]
        )
        for row in preds_raw
    ]
    labels = []
    for row in labels_raw:
        provenance = str(row.get("provenance", "unlabeled"))
        if provenance not in {"gold", "silver", "unlabeled"}:
            raise SystemExit(f"Invalid provenance: {provenance}")
        relevant = frozenset(str(x) for x in row.get("relevant_ids", []))  # type: ignore[arg-type]
        labels.append(
            QueryRelevance(
                query_id=str(row["query_id"]),
                relevant_ids=relevant,
                provenance=provenance,  # type: ignore[arg-type]
            )
        )

    pred_ids = {item.query_id for item in predictions}
    labels = [item for item in labels if item.query_id in pred_ids]
    if len(labels) != len(predictions):
        raise SystemExit(
            "Label/prediction query_id mismatch after intersection "
            f"({len(labels)} labels vs {len(predictions)} preds)"
        )
    bundle = evaluate_retrieval(predictions, labels)
    metrics = metrics_to_dict(bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path = args.output.with_suffix(".md")
    lines = ["# Retrieval metrics", ""]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(args.output), "markdown": str(md_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
