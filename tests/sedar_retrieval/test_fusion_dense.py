"""Fusion and dense scaffold tests."""

from __future__ import annotations

from legal_rag.sedar_retrieval.retrieval.dense import build_dense_index_scaffold
from legal_rag.sedar_retrieval.retrieval.fusion import (
    RetrieverHit,
    reciprocal_rank_fusion,
)


def test_rrf_dedup_and_missing_source_none() -> None:
    bm25 = (
        RetrieverHit("p1", 10.0, 1, "bm25"),
        RetrieverHit("p2", 5.0, 2, "bm25"),
    )
    dense = (
        RetrieverHit("p2", 0.9, 1, "dense"),
        RetrieverHit("p3", 0.8, 2, "dense"),
    )
    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense}, rrf_k=60)
    ids = [item.passage_id for item in fused]
    assert len(ids) == len(set(ids))
    p1 = next(item for item in fused if item.passage_id == "p1")
    assert p1.bm25_rank == 1
    assert p1.dense_rank is None
    assert p1.dense_score is None


def test_dense_scaffold_defers_without_cuda() -> None:
    manifest = build_dense_index_scaffold()
    assert manifest.status in {"DEFERRED_GPU", "READY_FOR_ENCODE"}
    assert manifest.model.startswith("Qwen/")
