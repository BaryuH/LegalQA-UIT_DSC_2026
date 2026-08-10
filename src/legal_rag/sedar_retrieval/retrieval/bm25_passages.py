"""BM25 index helpers over SEDAR Retrieval v3 canonical passages (TASK 06)."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from legal_rag.retrieval.bm25 import (
    BM25Config,
    BM25Index,
    BM25IndexResult,
    build_bm25_index,
    load_bm25_index,
    retrieve_bm25,
    write_bm25_index,
)
from legal_rag.schemas import RetrievalHit

from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage
from legal_rag.sedar_retrieval.retrieval.passage_adapter import (
    load_passages_jsonl,
    passage_to_legal_chunk,
)


def corpus_fingerprint(passages: tuple[CanonicalPassage, ...]) -> str:
    payload = [
        {
            "passage_id": p.passage_id,
            "retrieval_text": p.retrieval_text,
            "retrieval_level": p.retrieval_level,
        }
        for p in sorted(passages, key=lambda item: item.passage_id)
    ]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(blob.encode("utf-8")).hexdigest()


def build_passage_bm25_index(
    passages: tuple[CanonicalPassage, ...],
    *,
    cache_root: str | Path,
    config: BM25Config | None = None,
    overwrite_stale: bool = True,
) -> BM25IndexResult:
    chunks = [passage_to_legal_chunk(passage) for passage in passages]
    fingerprint = corpus_fingerprint(passages)
    return build_bm25_index(
        chunks,
        cache_root,
        fingerprint,
        config=config or BM25Config(version="sedar-retrieval-v3-bm25"),
        overwrite_stale=overwrite_stale,
    )


def search_passages(
    index: BM25Index,
    query: str,
    *,
    top_k: int,
) -> tuple[RetrievalHit, ...]:
    return retrieve_bm25(index, query, top_k=top_k, backend="cpu")


def hits_to_ranked_ids(hits: tuple[RetrievalHit, ...]) -> tuple[str, ...]:
    return tuple(hit.chunk_id for hit in hits)


def write_index_manifest(
    path: Path,
    *,
    run_id: str,
    corpus_hash: str,
    passage_count: int,
    index_path: str,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "corpus_hash": corpus_hash,
                "passage_count": passage_count,
                "index_path": index_path,
                "analyzer": "tokenize_legal_text",
                "config": config,
                "source": "bm25",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def load_passages_and_build(
    passages_path: Path,
    *,
    cache_root: Path,
) -> tuple[tuple[CanonicalPassage, ...], BM25IndexResult, str]:
    passages = load_passages_jsonl(str(passages_path))
    result = build_passage_bm25_index(passages, cache_root=cache_root)
    return passages, result, corpus_fingerprint(passages)


__all__ = [
    "build_passage_bm25_index",
    "corpus_fingerprint",
    "hits_to_ranked_ids",
    "load_bm25_index",
    "load_passages_and_build",
    "search_passages",
    "write_bm25_index",
    "write_index_manifest",
]
