from __future__ import annotations

import json

import pytest

from legal_rag.huy_training.common import (
    get_profile,
    load_training_pairs,
    reciprocal_rank,
    split_pairs,
)


def _write_pairs(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    rows = [
        {
            "query_id": f"q{index}",
            "query": f"question {index}",
            "positive": f"positive {index}",
            "negatives": [f"negative {index}-a", f"negative {index}-b"],
        }
        for index in range(10)
    ]
    pairs.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (tmp_path / "audit.json").write_text(
        json.dumps({"status": "PASS"}) + "\n", encoding="utf-8"
    )
    return pairs


def test_profiles_fit_declared_effective_batches() -> None:
    embedding = get_profile("rtx4090_24gb", "embedding")
    reranker = get_profile("a100_80gb", "reranker")
    assert embedding.tune_mode == "lora"
    assert embedding.effective_batch_size == 32
    assert reranker.tune_mode == "full"
    assert reranker.max_length == 2304


def test_loader_rejects_non_train_split(tmp_path) -> None:
    pairs = _write_pairs(tmp_path)
    with pytest.raises(ValueError, match="source_split='train'"):
        load_training_pairs(pairs, source_split="private", max_negatives=2)


def test_loader_requires_passed_audit(tmp_path) -> None:
    pairs = _write_pairs(tmp_path)
    (tmp_path / "audit.json").write_text(
        json.dumps({"status": "FAIL", "gate_failures": ["bad negatives"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not pass"):
        load_training_pairs(pairs, source_split="train", max_negatives=2)


def test_query_split_is_deterministic_and_disjoint(tmp_path) -> None:
    pairs_path = _write_pairs(tmp_path)
    pairs = load_training_pairs(
        pairs_path, source_split="train", max_negatives=2
    )
    first_train, first_dev = split_pairs(pairs, dev_fraction=0.2, seed=42)
    second_train, second_dev = split_pairs(pairs, dev_fraction=0.2, seed=42)
    assert first_train == second_train
    assert first_dev == second_dev
    assert {row.query_id for row in first_train}.isdisjoint(
        row.query_id for row in first_dev
    )


def test_reciprocal_rank_places_ties_before_positive() -> None:
    assert reciprocal_rank(0.5, [0.6, 0.5, 0.1]) == pytest.approx(1 / 3)
