from scripts.sedar_retrieval.train_vietnamese_reranker import _query_ids_hash


def test_query_ids_hash_is_order_independent_and_content_free() -> None:
    first = [
        {"query_id": "q2", "query": "question two"},
        {"query_id": "q1", "query": "question one"},
    ]
    second = [
        {"query_id": "q1", "query": "different text"},
        {"query_id": "q2", "query": "other text"},
    ]
    third = [*first, {"query_id": "q3", "query": "question three"}]

    assert _query_ids_hash(first) == _query_ids_hash(second)
    assert _query_ids_hash(first) != _query_ids_hash(third)
