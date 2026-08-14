#!/usr/bin/env python3
"""Fuse BM25 (+ optional dense) candidate JSONL with RRF (TASK 08)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from legal_rag.sedar_retrieval.retrieval.fusion import RetrieverHit, reciprocal_rank_fusion
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines


def _load_source(path: Path, source: str) -> dict[str, list[RetrieverHit]]:
    by_q: dict[str, list[RetrieverHit]] = defaultdict(list)
    for line in iter_jsonl_lines(path):
        row = json.loads(line)
        qid = str(row["query_id"])
        for item in row.get("scores", []):
            by_q[qid].append(
                RetrieverHit(
                    passage_id=str(item["passage_id"]),
                    score=float(item.get("bm25") or item.get("dense") or item.get("score") or 0.0),
                    rank=int(item["rank"]),
                    source=source,
                )
            )
        if not row.get("scores") and row.get("ranked_ids"):
            for rank, pid in enumerate(row["ranked_ids"], start=1):
                by_q[qid].append(
                    RetrieverHit(passage_id=str(pid), score=0.0, rank=rank, source=source)
                )
    return by_q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25", type=Path, required=True)
    parser.add_argument("--dense", type=Path, default=None)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--union-cap", type=int, default=250)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bm25 = _load_source(args.bm25, "bm25")
    dense = _load_source(args.dense, "dense") if args.dense else {}
    query_ids = sorted(set(bm25) | set(dense))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for qid in query_ids:
            lists = {}
            if qid in bm25:
                lists["bm25"] = bm25[qid]
            if qid in dense:
                lists["dense"] = dense[qid]
            fused = reciprocal_rank_fusion(
                lists, rrf_k=args.rrf_k, union_cap=args.union_cap
            )
            handle.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "ranked_ids": [c.passage_id for c in fused],
                        "candidates": [
                            {
                                "passage_id": c.passage_id,
                                "bm25_score": c.bm25_score,
                                "bm25_rank": c.bm25_rank,
                                "dense_score": c.dense_score,
                                "dense_rank": c.dense_rank,
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
