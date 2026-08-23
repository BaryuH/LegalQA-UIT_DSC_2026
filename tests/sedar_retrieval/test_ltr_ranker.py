"""Acceptance tests for TASK 13 LambdaRank scaffolding."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from legal_rag.sedar_retrieval.ranking.features import FEATURE_NAMES
from legal_rag.sedar_retrieval.ranking.ltr_ranker import (
    LTRFeatureExample,
    LTRRankerConfig,
    LTRRankerError,
    build_ranker_matrices,
    build_train_split,
    evaluate_ranker_matrices,
    ranked_ids_overlap,
    rerank_from_scores,
    resolve_feature_names,
    split_query_ids,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _features(**overrides: float) -> dict[str, float]:
    base = {name: -1.0 for name in FEATURE_NAMES}
    base.update(overrides)
    return base


def _example(
    query_id: str,
    passage_id: str,
    label: int,
    *,
    rank: int,
    rrf: float,
) -> LTRFeatureExample:
    return LTRFeatureExample(
        query_id=query_id,
        passage_id=passage_id,
        label=label,
        rank=rank,
        features=_features(rrf_score=rrf, bm25_score=float(rank)),
    )


def test_feature_group_ablations_drop_expected_names() -> None:
    all_names = resolve_feature_names("all")
    assert all_names == FEATURE_NAMES
    no_dense = resolve_feature_names("no_dense")
    assert "dense_score" not in no_dense
    assert "dense_rank" not in no_dense
    assert "bm25_score" in no_dense


def test_split_query_ids_is_disjoint_and_deterministic() -> None:
    query_ids = tuple(f"q-{index}" for index in range(20))
    train_a, val_a = split_query_ids(
        query_ids,
        validation_fraction=0.2,
        seed=42,
    )
    train_b, val_b = split_query_ids(
        query_ids,
        validation_fraction=0.2,
        seed=42,
    )
    assert train_a == train_b
    assert val_a == val_b
    assert set(train_a).isdisjoint(val_a)
    assert len(train_a) + len(val_a) == 20


def test_build_train_split_groups_by_query_id() -> None:
    examples = [
        _example("q-1", "p-1", 1, rank=1, rrf=0.3),
        _example("q-1", "p-2", 0, rank=2, rrf=0.2),
        _example("q-2", "p-3", 1, rank=1, rrf=0.4),
        _example("q-2", "p-4", 0, rank=2, rrf=0.1),
        _example("q-3", "p-5", 0, rank=1, rrf=0.25),
        _example("q-3", "p-6", 1, rank=2, rrf=0.15),
        _example("q-4", "p-7", 0, rank=1, rrf=0.22),
        _example("q-4", "p-8", 1, rank=2, rrf=0.12),
        _example("q-5", "p-9", 1, rank=1, rrf=0.21),
        _example("q-5", "p-10", 0, rank=2, rrf=0.11),
    ]
    split = build_train_split(
        examples,
        config=LTRRankerConfig(validation_fraction=0.2, seed=7),
    )
    assert set(split.train_query_ids).isdisjoint(split.validation_query_ids)
    assert sum(split.train.groups) == len(split.train.labels)
    assert all(size > 0 for size in split.train.groups)


def test_rerank_is_deterministic_on_score_ties() -> None:
    matrices = build_ranker_matrices(
        (
            _example("q-1", "p-b", 0, rank=1, rrf=0.1),
            _example("q-1", "p-a", 1, rank=2, rrf=0.1),
        ),
        feature_names=FEATURE_NAMES,
    )
    ranked = rerank_from_scores(matrices, (0.5, 0.5), top_k=2)
    assert ranked[0]["ranked_ids"] == ["p-a", "p-b"]


def test_ranked_ids_overlap_requires_exact_topk() -> None:
    assert ranked_ids_overlap(("a", "b", "c"), ("a", "b", "d"), top_k=2) == 1.0
    assert ranked_ids_overlap(("a", "b", "c"), ("a", "c", "b"), top_k=2) == 0.0


def test_evaluate_ranker_matrices_prefers_positive_labels() -> None:
    matrices = build_ranker_matrices(
        (
            _example("q-1", "p-1", 1, rank=1, rrf=0.1),
            _example("q-1", "p-2", 0, rank=2, rrf=0.2),
        ),
        feature_names=FEATURE_NAMES,
    )
    metrics = evaluate_ranker_matrices(matrices, (0.9, 0.1), k=2)
    assert metrics["ndcg_at_2"] == pytest.approx(1.0)
    assert metrics["mrr_at_2"] == pytest.approx(1.0)


def test_train_cli_smoke_with_lightgbm(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("numpy")
    rows = []
    for query_index in range(12):
        query_id = f"q-{query_index:02d}"
        for rank, label in ((1, 1), (2, 0), (3, 0)):
            rows.append(
                {
                    "query_id": query_id,
                    "passage_id": f"{query_id}-p{rank}",
                    "rank": rank,
                    "label": label,
                    "label_provenance": "positive_passage_id",
                    "features": _features(
                        rrf_score=1.0 / rank,
                        bm25_score=float(4 - rank),
                        dense_score=float(label),
                    ),
                    "schema_version": "sedar-ltr-features-v1",
                }
            )
    features_path = tmp_path / "features.jsonl"
    features_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "model"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT))
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "sedar_retrieval" / "train_ltr.py"),
            "--features",
            str(features_path),
            "--output-dir",
            str(output_dir),
            "--n-estimators",
            "20",
            "--early-stopping-rounds",
            "5",
            "--force",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    manifest_path = output_dir / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["leakage_policy"]["split_unit"] == "query_id"
    assert (output_dir / "model.txt").is_file()

    rank_out = tmp_path / "ranked.jsonl"
    ranked = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "sedar_retrieval" / "run_ltr_rank.py"),
            "--model-dir",
            str(output_dir),
            "--features",
            str(features_path),
            "--parity-features",
            str(features_path),
            "--output",
            str(rank_out),
            "--top-k",
            "3",
            "--force",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ranked.returncode == 0, ranked.stderr
    payload = json.loads(ranked.stdout.strip().splitlines()[-1])
    assert payload["parity_topk_overlap"] == 1.0


def test_resolve_feature_group_rejects_unknown() -> None:
    with pytest.raises(LTRRankerError, match="Unsupported feature group"):
        resolve_feature_names("no_magic")  # type: ignore[arg-type]
