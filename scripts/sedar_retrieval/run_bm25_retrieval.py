#!/usr/bin/env python3
"""Run BM25 retrieval over questions and emit ranked JSONL (TASK 04/06)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.retrieval.bm25 import BM25Config, load_bm25_index
from legal_rag.sedar_retrieval.retrieval.bm25_passages import (
    corpus_fingerprint,
    hits_to_ranked_ids,
    load_passages_jsonl,
    search_passages,
)
from legal_rag.sedar_retrieval.training.query_inputs import (
    load_retrieval_queries_from_json,
    load_retrieval_queries_from_synthetic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/sedar_retrieval/indexes/bm25"),
    )
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--synthetic-jsonl", type=Path, default=None)
    parser.add_argument("--split", default="warmup")
    parser.add_argument(
        "--source-split",
        default="train",
        help="Synthetic source_split filter when --synthetic-jsonl is used.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.questions is None and args.synthetic_jsonl is None:
        args.questions = Path("data/warmup.json")
    if args.questions is not None and args.synthetic_jsonl is not None:
        raise SystemExit("Provide only one of --questions or --synthetic-jsonl")

    passages = load_passages_jsonl(str(args.passages))
    fp = corpus_fingerprint(passages)
    loaded = load_bm25_index(
        args.cache_root, fp, BM25Config(version="sedar-retrieval-v3-bm25")
    )
    if loaded.index is None:
        raise SystemExit("BM25 index missing; run build_bm25_index.py first")

    if args.synthetic_jsonl is not None:
        queries = load_retrieval_queries_from_synthetic(
            args.synthetic_jsonl,
            source_split=args.source_split,
        )
    else:
        assert args.questions is not None
        queries = load_retrieval_queries_from_json(args.questions, split=args.split)
    if args.limit > 0:
        queries = queries[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for query in queries:
            hits = search_passages(loaded.index, query.question, top_k=args.top_k)
            handle.write(
                json.dumps(
                    {
                        "query_id": query.query_id,
                        "ranked_ids": list(hits_to_ranked_ids(hits)),
                        "scores": [
                            {
                                "passage_id": hit.chunk_id,
                                "bm25": hit.bm25_score,
                                "rank": hit.rank,
                                "source": "bm25",
                            }
                            for hit in hits
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps({"n_queries": len(queries), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
