"""Unit tests for the dependency-light TASK 07 dense retrieval helpers."""

from __future__ import annotations

import pytest

from legal_rag.sedar_retrieval.retrieval.dense import (
    DenseIndexError,
    dense_cache_fingerprint,
    format_instruct_query,
    format_passage_text,
    format_query_text,
    length_bucket_order,
    normalize_embedding_matrix,
    search_dense_index,
    validate_embedding_matrix,
)


def test_dense_cache_fingerprint_changes_with_encoder_config() -> None:
    base = {
        "corpus_hash": "corpus",
        "model": "Qwen/Qwen3-Embedding-4B",
        "model_revision": "revision",
        "dtype": "bf16",
        "normalized": True,
        "max_seq_length": 8192,
    }
    assert dense_cache_fingerprint(**base) != dense_cache_fingerprint(
        **{**base, "max_seq_length": 4096}
    )


def test_format_instruct_query_matches_task_contract() -> None:
    formatted = format_instruct_query("Điều kiện hưởng trợ cấp là gì?")
    assert formatted.startswith(
        "Instruct: Retrieve Vietnamese legal provisions that directly support "
        "the answer."
    )
    assert formatted.endswith("Query: Điều kiện hưởng trợ cấp là gì?")


def test_e5_format_uses_query_and_passage_prefixes() -> None:
    assert (
        format_query_text(
            "Điều kiện hưởng trợ cấp là gì?",
            input_format="e5",
        )
        == "query: Điều kiện hưởng trợ cấp là gì?"
    )
    assert (
        format_passage_text(
            "Điều kiện hưởng trợ cấp được quy định như sau.",
            input_format="e5",
        )
        == "passage: Điều kiện hưởng trợ cấp được quy định như sau."
    )


def test_dense_cache_fingerprint_changes_with_input_format() -> None:
    base = {
        "corpus_hash": "corpus",
        "model": "bqbbao6/vietnamese-legal-embedding",
        "model_revision": "7568a60f24a415e3597a74e423728272c929eb0b",
        "dtype": "bf16",
        "normalized": True,
        "max_seq_length": 512,
        "input_format": "e5",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    }
    assert dense_cache_fingerprint(**base) != dense_cache_fingerprint(
        **{**base, "passage_prefix": "passage = "}
    )


def test_length_bucket_order_is_stable() -> None:
    assert length_bucket_order(("bbb", "a", "cc", "a")) == (1, 3, 2, 0)


def test_embedding_validation_rejects_non_finite_values() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(DenseIndexError, match="NaN or Inf"):
        validate_embedding_matrix(np.array([[1.0, float("nan")]], dtype="float32"))


def test_normalize_embedding_matrix_returns_unit_rows() -> None:
    np = pytest.importorskip("numpy")
    normalized = normalize_embedding_matrix(np.array([[3.0, 4.0]], dtype="float32"))
    assert normalized[0].tolist() == pytest.approx([0.6, 0.8])


def test_search_dense_index_maps_faiss_rows_to_passage_ids() -> None:
    np = pytest.importorskip("numpy")

    class FakeIndex:
        d = 2
        ntotal = 3

        def search(self, queries, top_k):
            assert queries.shape == (1, 2)
            assert top_k == 2
            return (
                np.array([[0.91, 0.91]], dtype="float32"),
                np.array([[2, 0]], dtype="int64"),
            )

    results = search_dense_index(
        FakeIndex(),
        ("p0", "p1", "p2"),
        np.array([[1.0, 0.0]], dtype="float32"),
        top_k=2,
    )
    assert [(hit.passage_id, hit.rank) for hit in results[0]] == [
        ("p0", 1),
        ("p2", 2),
    ]
