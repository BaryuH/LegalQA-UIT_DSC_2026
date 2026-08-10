#!/usr/bin/env python3
"""Build BM25 index over SEDAR R1/R2a passages (TASK 06)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.retrieval.bm25_passages import (
    build_passage_bm25_index,
    corpus_fingerprint,
    load_passages_jsonl,
    search_passages,
    write_index_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/sedar_retrieval/indexes/bm25"),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
    )
    parser.add_argument("--smoke-query", default="Điều 76")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    run_id = new_run_id("bm25_legal_index")
    passages = load_passages_jsonl(str(args.passages))
    fp = corpus_fingerprint(passages)
    result = build_passage_bm25_index(passages, cache_root=args.cache_root)
    assert result.index is not None
    hits = search_passages(result.index, args.smoke_query, top_k=args.top_k)
    # Reload determinism check.
    from legal_rag.retrieval.bm25 import load_bm25_index, BM25Config

    reloaded = load_bm25_index(
        args.cache_root,
        fp,
        BM25Config(version="sedar-retrieval-v3-bm25"),
    )
    assert reloaded.index is not None
    hits2 = search_passages(reloaded.index, args.smoke_query, top_k=args.top_k)
    overlap = (
        1.0
        if [h.chunk_id for h in hits] == [h.chunk_id for h in hits2]
        else 0.0
    )

    manifest_out = args.manifest_out or (
        Path("artifacts/sedar_retrieval/indexes/manifests") / f"{run_id}.json"
    )
    write_index_manifest(
        manifest_out,
        run_id=run_id,
        corpus_hash=fp,
        passage_count=len(passages),
        index_path=str(result.cache_path),
        config={
            "version": "sedar-retrieval-v3-bm25",
            "smoke_query": args.smoke_query,
            "reload_topk_overlap": overlap,
            "git_commit": git_commit_sha(),
            "smoke_hit_ids": [h.chunk_id for h in hits],
            "index_status": result.status,
        },
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "passage_count": len(passages),
                "corpus_hash": fp,
                "reload_topk_overlap": overlap,
                "smoke_hits": len(hits),
                "manifest": str(manifest_out),
                "cache_path": str(result.cache_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if overlap == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
