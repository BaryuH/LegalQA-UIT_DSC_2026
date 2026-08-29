#!/usr/bin/env python3
"""Fuse BM25 (+ optional dense/legal) candidates with RRF or pure union."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines
from legal_rag.sedar_retrieval.retrieval.fusion import (
    RetrieverHit,
    candidate_union,
    reciprocal_rank_fusion,
)


def _load_source(path: Path, source: str) -> dict[str, list[RetrieverHit]]:
    by_q: dict[str, list[RetrieverHit]] = defaultdict(list)
    score_field = {
        "bm25": "bm25",
        "dense": "dense",
        "legal": "legal",
    }.get(source)
    for line in iter_jsonl_lines(path):
        row = json.loads(line)
        qid = str(row["query_id"])
        for item in row.get("scores", []):
            raw_score = item.get(score_field) if score_field else None
            if raw_score is None and source == "legal":
                raw_score = item.get("dense")
            if raw_score is None:
                raw_score = item.get("score", 0.0)
            by_q[qid].append(
                RetrieverHit(
                    passage_id=str(item["passage_id"]),
                    score=float(raw_score),
                    rank=int(item["rank"]),
                    source=source,
                )
            )
        if not row.get("scores") and row.get("ranked_ids"):
            for rank, pid in enumerate(row["ranked_ids"], start=1):
                by_q[qid].append(
                    RetrieverHit(
                        passage_id=str(pid),
                        score=0.0,
                        rank=rank,
                        source=source,
                    )
                )
    return by_q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25", type=Path, required=True)
    parser.add_argument("--dense", type=Path, default=None)
    parser.add_argument(
        "--legal",
        type=Path,
        default=None,
        help="Optional auxiliary legal-embedding ranked JSONL.",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--union-cap", type=int, default=250)
    parser.add_argument(
        "--fusion-method",
        choices=("weighted_rrf", "union"),
        default="weighted_rrf",
        help=("Rank merged candidates with weighted RRF or preserve pure union order."),
    )
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--legal-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite artifact: {args.output}")
    bm25 = _load_source(args.bm25, "bm25")
    dense = _load_source(args.dense, "dense") if args.dense else {}
    legal = _load_source(args.legal, "legal") if args.legal else {}
    query_ids = sorted(set(bm25) | set(dense) | set(legal))
    weights = {
        "bm25": args.bm25_weight,
        "dense": args.dense_weight,
        "legal": args.legal_weight,
    }
    if args.fusion_method == "union":
        if any(weight != 1.0 for weight in weights.values()):
            raise SystemExit("--*-weight options require --fusion-method=weighted_rrf")
    fusion_metadata = {
        "method": "candidate_union"
        if args.fusion_method == "union"
        else "weighted_rrf",
        "rrf_k": args.rrf_k if args.fusion_method == "weighted_rrf" else None,
        "union_cap": args.union_cap,
        "weights": weights if args.fusion_method == "weighted_rrf" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for qid in query_ids:
            lists = {}
            if qid in bm25:
                lists["bm25"] = bm25[qid]
            if qid in dense:
                lists["dense"] = dense[qid]
            if qid in legal:
                lists["legal"] = legal[qid]
            if args.fusion_method == "union":
                fused = candidate_union(lists, union_cap=args.union_cap)
            else:
                fused = reciprocal_rank_fusion(
                    lists,
                    rrf_k=args.rrf_k,
                    union_cap=args.union_cap,
                    weights=weights,
                )
            handle.write(
                json.dumps(
                    {
                        "schema_version": "sedar-retrieval-v3-fusion-v2",
                        "query_id": qid,
                        "fusion": {**fusion_metadata, "sources": sorted(lists)},
                        "ranked_ids": [c.passage_id for c in fused],
                        "candidates": [
                            {
                                "passage_id": c.passage_id,
                                "bm25_score": c.bm25_score,
                                "bm25_rank": c.bm25_rank,
                                "dense_score": c.dense_score,
                                "dense_rank": c.dense_rank,
                                "legal_score": c.legal_score,
                                "legal_rank": c.legal_rank,
                                "rrf_score": c.rrf_score,
                                "fused_rank": c.fused_rank,
                            }
                            for c in fused
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps({"n_queries": len(query_ids), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
