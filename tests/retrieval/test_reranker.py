"""Offline acceptance tests for the F1 reranker contract."""

from legal_rag.retrieval import MockReranker, NoOpReranker, Reranker, RerankResult
from legal_rag.schemas import RetrievalHit


def _hit(chunk_id: str, *, rank: int, bm25_score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_path="selected-contexts.zip",
        rank=rank,
        bm25_score=bm25_score,
    )


def test_noop_reranker_is_explicit_and_preserves_bm25_hits() -> None:
    hits = (
        _hit("chunk-b", rank=1, bm25_score=4.0),
        _hit("chunk-a", rank=2, bm25_score=3.0),
    )

    result = NoOpReranker().rerank("legal question", hits)

    assert isinstance(NoOpReranker(), Reranker)
    assert result == RerankResult(
        hits=hits,
        used=False,
        model="none",
        fallback_reason="reranker_disabled",
    )
    assert [hit.bm25_score for hit in result.hits] == [4.0, 3.0]
    assert all(hit.rerank_score is None for hit in result.hits)


def test_mock_reranker_separates_scores_and_uses_stable_ties() -> None:
    hits = (
        _hit("chunk-z", rank=1, bm25_score=9.0),
        _hit("chunk-a", rank=2, bm25_score=1.0),
        _hit("chunk-m", rank=3, bm25_score=5.0),
    )

    result = MockReranker({"chunk-z": 0.5, "chunk-a": 0.9, "chunk-m": 0.9}).rerank(
        "legal question", hits
    )

    assert result.used is True
    assert result.model == "mock-reranker-v1"
    assert result.fallback_reason is None
    assert [hit.chunk_id for hit in result.hits] == ["chunk-a", "chunk-m", "chunk-z"]
    assert [hit.rank for hit in result.hits] == [1, 2, 3]
    assert [hit.rerank_score for hit in result.hits] == [0.9, 0.9, 0.5]
    assert {hit.chunk_id: hit.bm25_score for hit in result.hits} == {
        "chunk-z": 9.0,
        "chunk-a": 1.0,
        "chunk-m": 5.0,
    }
    assert all(hit.rerank_score != hit.bm25_score for hit in result.hits)


def test_rerank_result_requires_explicit_reason_for_unused_reranker() -> None:
    hits = (_hit("chunk-a", rank=1, bm25_score=1.0),)

    try:
        RerankResult(hits=hits, used=False, model="none")
    except ValueError as exc:
        assert "fallback_reason" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("unused rerank result must explain its fallback")
