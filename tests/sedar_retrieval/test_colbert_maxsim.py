"""Acceptance tests for late-interaction MaxSim reranking (TASK 26).

Everything here runs on random matrices: no model weights, no GPU, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from legal_rag.sedar_retrieval.ranking.colbert_maxsim import (
    COLBERT_MAXSIM_SCHEMA_VERSION,
    ColbertMaxsimConfig,
    ColbertMaxsimError,
    DocEmbeddingCache,
    dequantize_int8,
    l2_normalize,
    maxsim_score,
    maxsim_scores,
    pool_tokens_hierarchical,
    quantize_int8,
    rerank_with_maxsim,
)


def _unit(rows: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=(rows, dim)))


# --- the definition -------------------------------------------------------


def test_maxsim_is_sum_over_query_tokens_of_max_over_document_tokens() -> None:
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    document = np.array([[1.0, 0.0], [0.0, 0.5], [-1.0, 0.0]], dtype=np.float32)
    # token 0 best match 1.0 ; token 1 best match 0.5
    assert maxsim_score(query, document) == pytest.approx(1.5)
    assert maxsim_score(query, document, reduction="mean") == pytest.approx(0.75)


def test_sum_and_mean_differ_by_the_query_token_count() -> None:
    query, document = _unit(16, seed=1), _unit(40, seed=2)
    total = maxsim_score(query, document)
    mean = maxsim_score(query, document, reduction="mean")
    assert total / 16 == pytest.approx(mean)


def test_masked_document_positions_use_negative_infinity_not_zero() -> None:
    # A query token whose only genuine match is negative must score negative.
    # Multiplying the mask by zero - what PyLate does - would return 0.0 here.
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    document = np.array([[-0.5, 0.0], [0.0, 1.0]], dtype=np.float32)
    mask = np.array([True, False])
    assert maxsim_score(query, document, document_mask=mask) == pytest.approx(-0.5)


def test_query_mask_drops_tokens_from_the_reduction() -> None:
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    document = np.array([[1.0, 0.0]], dtype=np.float32)
    assert maxsim_score(query, document) == pytest.approx(1.0)
    kept = maxsim_score(query, document, query_mask=np.array([True, False]))
    assert kept == pytest.approx(1.0)
    dropped = maxsim_score(query, document, query_mask=np.array([False, True]))
    assert dropped == pytest.approx(0.0)


def test_empty_and_fully_masked_inputs_score_zero() -> None:
    query = _unit(4, seed=3)
    assert maxsim_score(query, np.zeros((0, 8), dtype=np.float32)) == 0.0
    assert maxsim_score(
        query, _unit(5, seed=4), document_mask=np.zeros(5, dtype=bool)
    ) == 0.0


def test_dimension_and_shape_errors_are_explicit() -> None:
    with pytest.raises(ColbertMaxsimError):
        maxsim_score(_unit(4, dim=8), _unit(4, dim=16))
    with pytest.raises(ColbertMaxsimError):
        maxsim_score(np.zeros(8, dtype=np.float32), _unit(4))
    with pytest.raises(ColbertMaxsimError):
        maxsim_score(_unit(4), _unit(4), document_mask=np.ones(9, dtype=bool))
    with pytest.raises(ColbertMaxsimError):
        maxsim_score(_unit(4), _unit(4), reduction="max")  # type: ignore[arg-type]


def test_scores_are_ragged_and_order_is_preserved() -> None:
    query = _unit(6, seed=5)
    documents = [_unit(3, seed=6), _unit(30, seed=7), _unit(1, seed=8)]
    scores = maxsim_scores(query, documents)
    assert len(scores) == 3
    assert all(isinstance(value, float) for value in scores)


# --- token pooling --------------------------------------------------------


def test_pooling_halves_the_vector_count_and_keeps_unit_norm() -> None:
    document = _unit(120, dim=16, seed=9)
    pooled = pool_tokens_hierarchical(document, pool_factor=2)
    assert pooled.shape[0] == pytest.approx(60, abs=2)
    assert pooled.shape[1] == 16
    # MaxSim assumes unit vectors; averaging cluster members breaks that, so the
    # pooler must restore it (PyLate does not, colpali does).
    assert np.allclose(np.linalg.norm(pooled[1:], axis=1), 1.0, atol=1e-5)


def test_pool_factor_one_is_a_no_op_but_still_normalises() -> None:
    document = _unit(10, seed=10) * 3.0
    pooled = pool_tokens_hierarchical(document, pool_factor=1)
    assert pooled.shape == document.shape
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0, atol=1e-5)


def test_protected_tokens_are_never_clustered() -> None:
    document = _unit(40, seed=11)
    pooled = pool_tokens_hierarchical(document, pool_factor=4, protected_tokens=2)
    assert np.allclose(pooled[:2], document[:2], atol=1e-6)


def test_pooling_a_single_token_document_is_safe() -> None:
    document = _unit(1, seed=12)
    assert pool_tokens_hierarchical(document, pool_factor=4).shape == (1, 8)


def test_pooling_rejects_bad_arguments() -> None:
    with pytest.raises(ColbertMaxsimError):
        pool_tokens_hierarchical(np.zeros(8, dtype=np.float32), pool_factor=2)
    with pytest.raises(ColbertMaxsimError):
        pool_tokens_hierarchical(_unit(10), pool_factor=2, protected_tokens=-1)


# --- quantization ---------------------------------------------------------


def test_int8_round_trip_is_4x_smaller_and_barely_moves_the_score() -> None:
    query, document = _unit(32, dim=128, seed=13), _unit(200, dim=128, seed=14)
    packed = quantize_int8(document)
    assert packed.dtype == np.int8
    assert packed.nbytes * 4 == document.nbytes
    exact = maxsim_score(query, document)
    approx = maxsim_score(query, dequantize_int8(packed))
    assert abs(exact - approx) / abs(exact) < 0.01


def test_quantization_clamps_and_stays_in_range() -> None:
    packed = quantize_int8(np.array([[1.0, -1.0, 0.0]], dtype=np.float32))
    assert packed.max() <= 127 and packed.min() >= -127


# --- embedding cache ------------------------------------------------------


def test_cache_stores_quantized_and_returns_unit_vectors() -> None:
    cache = DocEmbeddingCache(dim=8, quantize=True)
    document = _unit(20, seed=15)
    cache.put("u1", document)
    assert "u1" in cache and len(cache) == 1
    restored = cache.get("u1")
    assert restored is not None
    assert np.allclose(np.linalg.norm(restored, axis=1), 1.0, atol=1e-2)
    assert cache.nbytes() == 20 * 8
    assert cache.get("missing") is None


def test_cache_rejects_the_wrong_dimension() -> None:
    cache = DocEmbeddingCache(dim=8)
    with pytest.raises(ColbertMaxsimError):
        cache.put("u1", _unit(4, dim=16))
    with pytest.raises(ColbertMaxsimError):
        DocEmbeddingCache(dim=0)


def test_cache_round_trips_through_disk(tmp_path) -> None:
    cache = DocEmbeddingCache(dim=8, quantize=True)
    cache.put("u1", _unit(12, seed=16))
    cache.put("u2", _unit(5, seed=17))
    target = tmp_path / "cache.npz"
    cache.save(target)
    reloaded = DocEmbeddingCache.load(target)
    assert len(reloaded) == 2
    assert reloaded.dim == 8 and reloaded.quantize is True
    assert np.allclose(reloaded.get("u1"), cache.get("u1"), atol=1e-6)


# --- reranking with a stub backend ---------------------------------------


class StubBackend:
    """Deterministic fake: score rises with the digit in the unit text."""

    dim = 8
    native_reduction = "sum"

    def __init__(self) -> None:
        self.query_calls = 0
        self.document_calls: list[int] = []

    def encode_query(self, query: str) -> np.ndarray:
        self.query_calls += 1
        base = np.zeros((4, 8), dtype=np.float32)
        base[:, 0] = 1.0
        return base

    def encode_documents(self, texts):
        self.document_calls.append(len(texts))
        out = []
        for text in texts:
            # Unit vectors whose cosine with the query is (index + 1) / 10, so
            # the scores genuinely differ instead of collapsing under
            # normalisation.
            cosine = (float(text.split("#")[1]) + 1.0) / 10.0
            vector = np.zeros((3, 8), dtype=np.float32)
            vector[:, 0] = cosine
            vector[:, 1] = np.sqrt(max(0.0, 1.0 - cosine**2))
            out.append(vector)
        return out


def _config(**kwargs) -> ColbertMaxsimConfig:
    defaults = dict(model="local/colbert-ft", dim=8, top_k=100, device="cpu")
    defaults.update(kwargs)
    return ColbertMaxsimConfig(**defaults)


def test_rerank_orders_by_maxsim_and_returns_only_the_head() -> None:
    backend = StubBackend()
    texts = {f"u{i}": f"doc#{i}" for i in range(5)}
    ranked = rerank_with_maxsim(
        "câu hỏi",
        ["u0", "u1", "u2", "u3", "u4"],
        texts,
        backend=backend,
        config=_config(top_k=3),
    )
    assert [unit_id for unit_id, _ in ranked] == ["u2", "u1", "u0"]
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
    # 4 query tokens x cosine 0.3 for the best document.
    assert scores[0] == pytest.approx(4 * 0.3, abs=1e-5)
    assert backend.query_calls == 1


def test_rerank_uses_the_cache_and_stops_re_encoding() -> None:
    backend = StubBackend()
    texts = {f"u{i}": f"doc#{i}" for i in range(4)}
    cache = DocEmbeddingCache(dim=8)
    config = _config(top_k=4)
    rerank_with_maxsim("q", list(texts), texts, backend=backend, config=config, cache=cache)
    assert len(cache) == 4
    first_round = list(backend.document_calls)
    rerank_with_maxsim("q", list(texts), texts, backend=backend, config=config, cache=cache)
    assert backend.document_calls == first_round, "second pass must hit the cache"


def test_rerank_rejects_duplicates_and_missing_units() -> None:
    backend = StubBackend()
    texts = {"u0": "doc#1"}
    with pytest.raises(ColbertMaxsimError):
        rerank_with_maxsim("q", ["u0", "u0"], texts, backend=backend, config=_config())
    with pytest.raises(ColbertMaxsimError):
        rerank_with_maxsim("q", ["u0", "ghost"], texts, backend=backend, config=_config())


def test_rerank_rejects_a_backend_whose_dim_disagrees_with_the_config() -> None:
    backend = StubBackend()
    with pytest.raises(ColbertMaxsimError):
        rerank_with_maxsim(
            "q", ["u0"], {"u0": "doc#1"}, backend=backend, config=_config(dim=128)
        )


def test_rerank_of_an_empty_head_is_empty() -> None:
    assert rerank_with_maxsim("q", [], {}, backend=StubBackend(), config=_config()) == ()


# --- configuration guards ------------------------------------------------


def test_config_refuses_an_unnamed_checkpoint_by_default() -> None:
    # Zero-shot late interaction scores below BM25 on Vietnamese, so running a
    # base encoder by accident must be a config error, not a silent bad run.
    with pytest.raises(ValueError):
        ColbertMaxsimConfig()
    assert ColbertMaxsimConfig(require_finetuned=False).model == ""


def test_bge_m3_backend_is_pinned_to_mean_reduction() -> None:
    # M3's own colbert_score divides by the query token count; "sum" would
    # silently change the score scale by a factor of N_query_tokens.
    with pytest.raises(ValueError):
        ColbertMaxsimConfig(model="BAAI/bge-m3", backend="bge_m3", reduction="sum")
    ok = ColbertMaxsimConfig(
        model="BAAI/bge-m3", backend="bge_m3", reduction="mean", dim=1024
    )
    assert ok.score_scale == "[-1, 1]"


def test_config_reports_the_score_scale_for_the_run_artifact() -> None:
    config = _config(query_length=32)
    assert config.score_scale == "[0, 32] approx"
    payload = config.as_dict()
    assert payload["schema_version"] == COLBERT_MAXSIM_SCHEMA_VERSION
    assert payload["reduction"] == "sum"


def test_config_validates_numeric_arguments() -> None:
    with pytest.raises(ValueError):
        _config(dim=0)
    with pytest.raises(ValueError):
        _config(pool_factor=0)
    with pytest.raises(ValueError):
        _config(top_k=0)
    with pytest.raises(ValueError):
        _config(backend="colbertv2")
