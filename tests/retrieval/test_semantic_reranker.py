"""Offline acceptance tests for the optional F2 semantic reranker."""

from collections.abc import Sequence

import pytest

from legal_rag.config import RerankerSection
from legal_rag.retrieval import (
    DEFAULT_SEMANTIC_RERANKER_MODEL,
    SemanticReranker,
    SemanticRerankerUnavailableError,
    create_reranker,
)
from legal_rag.schemas import RetrievalHit


def _hit(chunk_id: str, *, rank: int, bm25_score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_path="selected-contexts.zip",
        rank=rank,
        bm25_score=bm25_score,
    )


class _FakeEncoder:
    tokenizer = None

    def __init__(self) -> None:
        self.calls: list[tuple[Sequence[str], dict[str, object]]] = []
        self.max_seq_length: int | None = None

    def encode(
        self,
        texts: Sequence[str],
        **kwargs: object,
    ) -> list[list[float]]:
        self.calls.append((tuple(texts), kwargs))
        vectors: list[list[float]] = []
        for text in texts:
            if text == "query":
                vectors.append([1.0, 0.0])
            elif "candidate-a" in text:
                vectors.append([0.8, 0.6])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class _FakeTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
    ) -> list[int]:
        assert add_special_tokens is True
        assert truncation is True
        return list(range(min(len(text.split()), max_length)))

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return f"tokens={len(token_ids)}"


def test_semantic_reranker_is_offline_injectable_and_config_driven() -> None:
    encoder = _FakeEncoder()
    reranker = create_reranker(
        RerankerSection(
            enabled=True,
            required=True,
            provider="sentence_transformers",
            model=None,
            device="cpu",
            batch_size=2,
            max_length=16,
        ),
        encoder=encoder,
        model_loader=lambda *_: (_ for _ in ()).throw(
            AssertionError("unit test must not load a model")
        ),
    )

    result = reranker.rerank(
        "query",
        (
            _hit("chunk-b", rank=1, bm25_score=9.0),
            _hit("chunk-a", rank=2, bm25_score=1.0),
        ),
        candidate_texts={
            "chunk-a": "candidate-a legal passage",
            "chunk-b": "candidate-b legal passage",
        },
    )

    assert isinstance(reranker, SemanticReranker)
    assert reranker.model == DEFAULT_SEMANTIC_RERANKER_MODEL
    assert [hit.chunk_id for hit in result.hits] == ["chunk-a", "chunk-b"]
    assert [hit.bm25_score for hit in result.hits] == [1.0, 9.0]
    assert result.hits[0].rerank_score == pytest.approx(0.8)
    assert result.hits[1].rerank_score == pytest.approx(0.0)
    assert result.used is True
    assert result.fallback_reason is None
    assert result.metadata["device"] == "cpu"
    assert result.metadata["batch_size"] == 2
    assert result.metadata["max_length"] == 16
    assert isinstance(result.metadata["latency_ms"], float)
    assert float(result.metadata["latency_ms"]) >= 0.0
    assert float(result.metadata["peak_memory_mb"]) >= 0.0
    assert reranker.last_report == dict(result.metadata)
    assert encoder.max_seq_length == 16
    assert encoder.calls[0][1]["batch_size"] == 2


def test_semantic_reranker_truncates_with_tokenizer_without_mutating_source_text() -> (
    None
):
    encoder = _FakeEncoder()
    encoder.tokenizer = _FakeTokenizer()
    source_text = "one two three four five six seven"
    reranker = SemanticReranker(
        model_name="local-test-model",
        required=True,
        device="cpu",
        max_length=3,
        encoder=encoder,
    )

    reranker.rerank(
        "query text that is long",
        (_hit("chunk-a", rank=1, bm25_score=2.0),),
        candidate_texts={"chunk-a": source_text},
    )

    encoded_texts = encoder.calls[0][0]
    assert all(int(text.split("=", 1)[1]) <= 3 for text in encoded_texts)
    assert source_text == "one two three four five six seven"


def test_optional_model_failure_returns_explicit_bm25_fallback() -> None:
    def unavailable_loader(
        model_name: str,
        device: str,
        model_revision: str | None,
    ) -> object:
        raise ImportError("semantic dependency unavailable")

    reranker = SemanticReranker(
        model_name="missing-model",
        required=False,
        device="cpu",
        model_loader=unavailable_loader,
    )
    hits = (_hit("chunk-a", rank=1, bm25_score=4.0),)

    result = reranker.rerank(
        "query",
        hits,
    )

    assert result.used is False
    assert result.model == "missing-model"
    assert result.fallback_reason == "semantic_reranker_unavailable:ImportError"
    assert result.hits == hits
    assert result.metadata["provider"] == "sentence_transformers"

    with pytest.raises(SemanticRerankerUnavailableError, match="ImportError"):
        SemanticReranker(
            model_name="missing-model",
            required=True,
            device="cpu",
            model_loader=unavailable_loader,
        )


def test_cuda_request_fails_required_and_is_explicit_when_optional() -> None:
    with pytest.raises(SemanticRerankerUnavailableError, match="cuda_unavailable"):
        SemanticReranker(
            model_name="local-test-model",
            required=True,
            device="cuda",
            encoder=_FakeEncoder(),
        )

    reranker = SemanticReranker(
        model_name="local-test-model",
        required=False,
        device="cuda",
        encoder=_FakeEncoder(),
    )
    result = reranker.rerank(
        "query",
        (_hit("chunk-a", rank=1, bm25_score=1.0),),
        candidate_texts={"chunk-a": "legal passage"},
    )
    assert result.used is False
    assert result.fallback_reason == "semantic_reranker_unavailable:cuda_unavailable"
