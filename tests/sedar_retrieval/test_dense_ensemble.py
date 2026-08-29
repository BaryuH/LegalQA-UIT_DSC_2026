"""Acceptance tests for the local dense-ensemble implementation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from legal_rag.sedar_retrieval.corpus.schema import (
    CanonicalPassage,
    SourceProvenance,
)
from legal_rag.sedar_retrieval.eval.ensemble_metrics import diagnose_ensemble
from legal_rag.sedar_retrieval.ranking.features import (
    ENSEMBLE_FEATURE_NAMES,
    ENSEMBLE_FEATURE_SCHEMA_VERSION,
    extract_ensemble_features,
)
from legal_rag.sedar_retrieval.ranking.ltr_dataset import (
    LTRCandidate,
    LTRFeatureBuildConfig,
    SyntheticQueryView,
    build_ltr_feature_rows,
    feature_profile_spec,
    validate_feature_schema,
)
from legal_rag.sedar_retrieval.retrieval.dense import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_LEGAL_MODEL,
    validate_source_model_pair,
)
from legal_rag.sedar_retrieval.retrieval.fusion import (
    RetrieverHit,
    candidate_union,
    reciprocal_rank_fusion,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _passage(passage_id: str, text: str) -> CanonicalPassage:
    return CanonicalPassage(
        passage_id=passage_id,
        document_id="law-1",
        article_id="law-1:article:1",
        clause_id="law-1:article:1:clause:1",
        retrieval_level="clause",
        document_name="Luật mẫu",
        article_number="1",
        clause_number="1",
        status="effective",
        raw_text=text,
        reader_text=text,
        retrieval_text=text,
        source=SourceProvenance(
            source_path="data/law-1.txt",
            document_id="law-1",
            content_hash=f"hash-{passage_id}",
        ),
    )


def test_weighted_rrf_tracks_legal_source_metadata() -> None:
    fused = reciprocal_rank_fusion(
        {
            "bm25": (RetrieverHit("p-b", 4.0, 1, "bm25"),),
            "dense": (RetrieverHit("p-q", 0.9, 1, "dense"),),
            "legal": (RetrieverHit("p-l", 0.8, 1, "legal"),),
        },
        rrf_k=60,
        union_cap=3,
        weights={"bm25": 1.0, "dense": 1.0, "legal": 0.2},
    )

    legal = next(item for item in fused if item.passage_id == "p-l")
    assert legal.legal_rank == 1
    assert legal.legal_score == pytest.approx(0.8)
    assert legal.rrf_score == pytest.approx(0.2 / 61)


def test_candidate_union_preserves_source_order_without_rrf_score() -> None:
    fused = candidate_union(
        {
            "legal": (RetrieverHit("p-l", 0.8, 1, "legal"),),
            "dense": (
                RetrieverHit("p-shared", 0.9, 1, "dense"),
                RetrieverHit("p-q", 0.7, 2, "dense"),
            ),
            "bm25": (
                RetrieverHit("p-shared", 4.0, 1, "bm25"),
                RetrieverHit("p-b", 3.0, 2, "bm25"),
            ),
        },
        union_cap=4,
    )

    assert [item.passage_id for item in fused] == [
        "p-shared",
        "p-b",
        "p-q",
        "p-l",
    ]
    assert fused[0].bm25_rank == 1
    assert fused[0].dense_rank == 1
    assert fused[0].rrf_score is None


def test_dense_source_model_pairing_fails_closed() -> None:
    validate_source_model_pair(
        "legal",
        DEFAULT_LEGAL_MODEL,
        input_format="e5",
    )
    validate_source_model_pair(
        "dense",
        DEFAULT_DENSE_MODEL,
        input_format="qwen_instruction",
    )

    with pytest.raises(ValueError, match="requires model"):
        validate_source_model_pair("legal", DEFAULT_DENSE_MODEL)
    with pytest.raises(ValueError, match="requires input_format"):
        validate_source_model_pair("legal", DEFAULT_LEGAL_MODEL, input_format="e5x")


def test_ensemble_features_capture_presence_agreement_and_rank_spread() -> None:
    row = extract_ensemble_features(
        query_id="q-1",
        query="Điều 1 quy định gì?",
        passage_id="p-1",
        passage_text="Điều 1 quy định nội dung.",
        bm25_score=2.0,
        bm25_rank=4,
        qwen_score=0.8,
        qwen_rank=7,
        legal_score=0.7,
        legal_rank=9,
    )

    assert row.schema_version == ENSEMBLE_FEATURE_SCHEMA_VERSION
    assert row.features["bm25_qwen_both"] == 1.0
    assert row.features["qwen_legal_both"] == 1.0
    assert row.features["all_three"] == 1.0
    assert row.features["qwen_legal_rank_diff"] == 2.0
    assert row.features["min_rank"] == 4.0
    assert row.features["max_rank"] == 9.0
    assert row.features["mean_rank"] == pytest.approx(20.0 / 3.0)


def test_ensemble_schema_matches_feature_profile() -> None:
    schema_version, feature_names = feature_profile_spec("ensemble_v2")
    schema_path = (
        REPOSITORY_ROOT / "configs" / "retrieval" / "ltr_feature_schema_v2.json"
    )

    assert schema_version == ENSEMBLE_FEATURE_SCHEMA_VERSION
    assert feature_names == ENSEMBLE_FEATURE_NAMES
    assert validate_feature_schema(
        schema_path,
        expected_schema_version=schema_version,
        expected_feature_names=feature_names,
    )


def test_ensemble_ltr_rows_preserve_legal_features() -> None:
    passages = {
        "p-1": _passage("p-1", "Điều 1 quy định nội dung."),
        "p-2": _passage("p-2", "Điều 2 quy định khác."),
    }
    rows, report = build_ltr_feature_rows(
        candidates={
            "syn-1": (
                LTRCandidate(
                    "p-1",
                    1,
                    bm25_score=2.0,
                    bm25_rank=1,
                    dense_score=0.8,
                    dense_rank=2,
                    legal_score=0.7,
                    legal_rank=1,
                    rrf_score=0.04,
                    source="rrf",
                ),
                LTRCandidate(
                    "p-2",
                    2,
                    legal_score=0.6,
                    legal_rank=2,
                    rrf_score=0.02,
                    source="rrf",
                ),
            )
        },
        synthetic_queries={
            "syn-1": SyntheticQueryView(
                synthetic_id="syn-1",
                query="Điều 1 quy định gì?",
                positive_passage_id="p-1",
            )
        },
        passages=passages,
        config=LTRFeatureBuildConfig(
            label_source="positive_passage_id",
            label_mode="binary",
            feature_profile="ensemble_v2",
        ),
    )

    assert report.output_row_count == 2
    assert rows[0]["schema_version"] == ENSEMBLE_FEATURE_SCHEMA_VERSION
    assert rows[0]["features"]["legal_found"] == 1.0
    assert rows[1]["features"]["qwen_found"] == 0.0


def test_ensemble_ltr_train_and_rank_cli(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("numpy")

    rows: list[dict[str, object]] = []
    for query_index in range(12):
        query_id = f"q-{query_index:02d}"
        positive = extract_ensemble_features(
            query_id=query_id,
            query="Điều 1 quy định gì?",
            passage_id=f"{query_id}-positive",
            passage_text="Điều 1 quy định nội dung.",
            bm25_score=4.0,
            bm25_rank=1,
            qwen_score=0.9,
            qwen_rank=1,
            legal_score=0.8,
            legal_rank=1,
            rrf_score=0.04,
        )
        negative = extract_ensemble_features(
            query_id=query_id,
            query="Điều 1 quy định gì?",
            passage_id=f"{query_id}-negative",
            passage_text="Một nội dung không liên quan.",
            bm25_score=1.0,
            bm25_rank=2,
            qwen_score=0.2,
            qwen_rank=2,
            legal_score=0.1,
            legal_rank=2,
            rrf_score=0.02,
        )
        for rank, label, feature_row in (
            (1, 1, positive),
            (2, 0, negative),
        ):
            rows.append(
                {
                    "query_id": query_id,
                    "passage_id": feature_row.passage_id,
                    "rank": rank,
                    "label": label,
                    "label_provenance": "positive_passage_id",
                    "features": feature_row.features,
                    "schema_version": ENSEMBLE_FEATURE_SCHEMA_VERSION,
                }
            )
    features_path = tmp_path / "ensemble_features.jsonl"
    features_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "ensemble_model"
    environment = dict()
    import os

    environment.update(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT))
    )
    train = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "sedar_retrieval" / "train_ltr.py"),
            "--features",
            str(features_path),
            "--feature-profile",
            "ensemble_v2",
            "--schema",
            str(
                REPOSITORY_ROOT / "configs" / "retrieval" / "ltr_feature_schema_v2.json"
            ),
            "--output-dir",
            str(model_dir),
            "--min-child-samples",
            "1",
            "--n-estimators",
            "10",
            "--early-stopping-rounds",
            "3",
            "--force",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert train.returncode == 0, train.stderr
    manifest = json.loads(
        (model_dir / "train_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["feature_schema_version"] == ENSEMBLE_FEATURE_SCHEMA_VERSION
    ranked_path = tmp_path / "ranked.jsonl"
    rank = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "sedar_retrieval" / "run_ltr_rank.py"),
            "--model-dir",
            str(model_dir),
            "--features",
            str(features_path),
            "--output",
            str(ranked_path),
            "--top-k",
            "2",
            "--force",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert rank.returncode == 0, rank.stderr
    assert ranked_path.is_file()


def test_diagnostic_reports_unique_legal_recall_and_jaccard() -> None:
    report = diagnose_ensemble(
        {
            "bm25": {
                "q-1": ("p-a", "p-b"),
                "q-2": ("p-c",),
            },
            "dense": {
                "q-1": ("p-a", "p-q"),
                "q-2": ("p-c",),
            },
            "legal": {
                "q-1": ("p-a", "p-l"),
                "q-2": ("p-l2",),
            },
        },
        labels={
            "q-1": (frozenset({"p-l"}), "silver"),
            "q-2": (frozenset({"p-l2"}), "silver"),
        },
        top_k=2,
        cutoffs=(1, 2),
    )

    assert report.union_metrics["dense+legal"]["recall_at"]["2"] == 1.0
    assert report.pairwise_metrics["dense~legal"]["mean_jaccard"] == pytest.approx(
        1.0 / 6.0
    )
    assert report.legal_contribution is not None
    assert report.legal_contribution["pooled_unique_relevant_count"] == 2.0
    assert report.legal_contribution["legal_only_relevant_rate"] == 1.0


def test_diagnostic_cli_writes_json_without_answer_text(tmp_path: Path) -> None:
    def write_source(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    bm25 = tmp_path / "bm25.jsonl"
    qwen = tmp_path / "qwen.jsonl"
    legal = tmp_path / "legal.jsonl"
    labels = tmp_path / "labels.jsonl"
    row_a = {"query_id": "q-1", "ranked_ids": ["p-a", "p-b"]}
    row_b = {"query_id": "q-2", "ranked_ids": ["p-c"]}
    write_source(bm25, [row_a, row_b])
    write_source(qwen, [row_a, row_b])
    write_source(
        legal,
        [
            {"query_id": "q-1", "ranked_ids": ["p-l"]},
            {"query_id": "q-2", "ranked_ids": ["p-c"]},
        ],
    )
    write_source(
        labels,
        [
            {
                "query_id": "q-1",
                "relevant_ids": ["p-l"],
                "provenance": "silver",
            },
            {
                "query_id": "q-2",
                "relevant_ids": ["p-c"],
                "provenance": "silver",
            },
        ],
    )
    output = tmp_path / "diagnostics.json"
    script = REPOSITORY_ROOT / "scripts" / "sedar_retrieval" / "evaluate_ensemble.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bm25",
            str(bm25),
            "--qwen",
            str(qwen),
            "--legal",
            str(legal),
            "--labels",
            str(labels),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sedar-retrieval-ensemble-diagnostics-v1"
    assert payload["legal_contribution"]["legal_only_relevant_rate"] == 1.0
    assert "answer" not in output.read_text(encoding="utf-8").casefold()


def test_fuser_cli_serializes_weighted_legal_candidates(tmp_path: Path) -> None:
    def write_source(path: Path, score_field: str, passage_id: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "query_id": "q-1",
                    "ranked_ids": [passage_id],
                    "scores": [
                        {
                            "passage_id": passage_id,
                            score_field: 0.8,
                            "rank": 1,
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    bm25 = tmp_path / "bm25.jsonl"
    qwen = tmp_path / "qwen.jsonl"
    legal = tmp_path / "legal.jsonl"
    write_source(bm25, "bm25", "p-b")
    write_source(qwen, "dense", "p-q")
    write_source(legal, "legal", "p-l")
    output = tmp_path / "fused.jsonl"
    script = REPOSITORY_ROOT / "scripts" / "sedar_retrieval" / "fuse_candidates.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bm25",
            str(bm25),
            "--dense",
            str(qwen),
            "--legal",
            str(legal),
            "--legal-weight",
            "0.2",
            "--output",
            str(output),
            "--force",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8").strip())
    legal_candidate = next(
        item for item in payload["candidates"] if item["passage_id"] == "p-l"
    )
    assert legal_candidate["legal_score"] == pytest.approx(0.8)
    assert payload["fusion"]["weights"]["legal"] == pytest.approx(0.2)

    union_output = tmp_path / "union.jsonl"
    union_completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bm25",
            str(bm25),
            "--dense",
            str(qwen),
            "--legal",
            str(legal),
            "--fusion-method",
            "union",
            "--output",
            str(union_output),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert union_completed.returncode == 0, union_completed.stderr
    union_payload = json.loads(union_output.read_text(encoding="utf-8").strip())
    assert union_payload["fusion"]["method"] == "candidate_union"
    assert union_payload["fusion"]["rrf_k"] is None
    assert union_payload["candidates"][0]["rrf_score"] is None
