#!/usr/bin/env python3
"""Run BM25 retrieval over questions and emit ranked JSONL (TASK 04/06)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.questions import load_inference_questions
from legal_rag.retrieval.bm25 import BM25Config, load_bm25_index
from legal_rag.sedar_retrieval.retrieval.bm25_passages import (
    corpus_fingerprint,
    hits_to_ranked_ids,
    load_passages_jsonl,
    search_passages,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/sedar_retrieval/indexes/bm25"),
    )
    parser.add_argument("--questions", type=Path, default=Path("data/warmup.json"))
    parser.add_argument("--split", default="warmup")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    passages = load_passages_jsonl(str(args.passages))
    fp = corpus_fingerprint(passages)
    loaded = load_bm25_index(
        args.cache_root, fp, BM25Config(version="sedar-retrieval-v3-bm25")
    )
    if loaded.index is None:
        raise SystemExit("BM25 index missing; run build_bm25_index.py first")
    questions = load_inference_questions(args.questions, split=args.split)  # type: ignore[arg-type]
    if args.limit > 0:
        questions = questions[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for question in questions:
            hits = search_passages(loaded.index, question.question, top_k=args.top_k)
            handle.write(
                json.dumps(
                    {
                        "query_id": question.id,
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
    print(json.dumps({"n_queries": len(questions), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
