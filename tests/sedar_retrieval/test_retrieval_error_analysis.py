"""Acceptance tests for clean-warmup retrieval error analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.sedar_retrieval.corpus.schema import (
    CanonicalPassage,
    SourceProvenance,
)
from legal_rag.sedar_retrieval.eval.retrieval_error_analysis import (
    analyze_warmup_retrieval_errors,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _passage(passage_id: str) -> CanonicalPassage:
    return CanonicalPassage(
        passage_id=passage_id,
        document_id=f"document-{passage_id}",
        article_id=f"article-{passage_id}",
        retrieval_level="article",
        document_name=f"Văn bản {passage_id}",
        article_number="1",
        raw_text=f"Nội dung {passage_id}.",
        reader_text=f"Nội dung {passage_id}.",
        retrieval_text=f"Nội dung {passage_id}.",
        source=SourceProvenance(
            source_path=f"data/{passage_id}.txt",
            document_id=f"document-{passage_id}",
            content_hash=f"hash-{passage_id}",
        ),
    )


def test_analyzer_classifies_retrieval_rerank_pack_and_reader_stages(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    query_ids = ["q-a", "q-b", "q-c", "q-d"]
    _write_json(
        run_dir / "config.json",
        {
            "split": "warmup",
            "id_source": "clean_manifest",
            "retrieval_variant": "ltr_full_all",
        },
    )
    _write_json(run_dir / "run_summary.json", {"run_id": "champion-run"})
    _write_jsonl(
        run_dir / "retrieval.jsonl",
        [
            {
                "id": "q-a",
                "raw_hit_ids": ["p-x"],
                "packed_chunk_ids": ["p-x"],
            },
            {
                "id": "q-b",
                "raw_hit_ids": ["p-z"],
                "packed_chunk_ids": ["p-z"],
            },
            {
                "id": "q-c",
                "raw_hit_ids": ["p-c"],
                "packed_chunk_ids": ["p-z"],
            },
            {
                "id": "q-d",
                "raw_hit_ids": ["p-d"],
                "packed_chunk_ids": ["p-d"],
            },
        ],
    )
    _write_jsonl(
        run_dir / "predictions.jsonl",
        [
            {"id": "q-a", "answer": "Sai nguồn."},
            {"id": "q-b", "answer": "Sai thứ hạng."},
            {"id": "q-c", "answer": "Sai pack."},
            {"id": "q-d", "answer": "lặp lặp lặp lặp lặp lặp lặp lặp"},
        ],
    )

    metrics = tmp_path / "metrics.json"
    _write_json(
        metrics,
        {
            "split": "warmup",
            "reference_role": "evaluation_reference_only",
            "per_case": [
                {
                    "id": query_id,
                    "status": "scored",
                    "meteor": 0.1 if query_id == "q-d" else 0.8,
                    "rouge_l": 0.1 if query_id == "q-d" else 0.8,
                }
                for query_id in query_ids
            ],
        },
    )

    bm25 = tmp_path / "bm25.jsonl"
    qwen = tmp_path / "qwen.jsonl"
    _write_jsonl(
        bm25,
        [
            {"query_id": "q-a", "ranked_ids": ["p-x"]},
            {"query_id": "q-b", "ranked_ids": ["p-b"]},
            {"query_id": "q-c", "ranked_ids": ["p-c"]},
            {"query_id": "q-d", "ranked_ids": ["p-x"]},
        ],
    )
    _write_jsonl(
        qwen,
        [
            {"query_id": "q-a", "ranked_ids": ["p-y"]},
            {"query_id": "q-b", "ranked_ids": ["p-z"]},
            {"query_id": "q-c", "ranked_ids": ["p-z"]},
            {"query_id": "q-d", "ranked_ids": ["p-d"]},
        ],
    )
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(
        labels,
        [
            {
                "query_id": query_id,
                "relevant_ids": [f"p-{query_id[-1]}"],
                "provenance": "silver",
            }
            for query_id in query_ids
        ],
    )
    passages = tmp_path / "passages.jsonl"
    _write_jsonl(
        passages,
        [
            passage.model_dump(mode="json")
            for passage in (
                _passage("p-a"),
                _passage("p-b"),
                _passage("p-c"),
                _passage("p-d"),
                _passage("p-x"),
                _passage("p-y"),
                _passage("p-z"),
            )
        ],
    )

    report = analyze_warmup_retrieval_errors(
        run_dir=run_dir,
        metrics_path=metrics,
        bm25_path=bm25,
        qwen_path=qwen,
        labels_path=labels,
        passages_path=passages,
    ).as_dict()

    assert report["query_count"] == 4
    assert report["labeled_query_count"] == 4
    assert report["failure_counts"] == {
        "A_RETRIEVAL_MISS": 1,
        "B_PACK_MISS": 1,
        "B_RERANK_MISS": 1,
        "C_READER_ISSUE": 1,
    }
    assert (
        report["stage_metrics"]["candidate_union_bm25_qwen"]["provision_hit_count"] == 3
    )
    cases = {row["id"]: row for row in report["per_case"]}
    assert cases["q-a"]["failure_type"] == "A_RETRIEVAL_MISS"
    assert cases["q-b"]["failure_type"] == "B_RERANK_MISS"
    assert cases["q-c"]["failure_type"] == "B_PACK_MISS"
    assert cases["q-d"]["failure_type"] == "C_READER_ISSUE"
    assert cases["q-d"]["repetition"]["repetition_flag"] is True

    _write_json(
        run_dir / "config.json",
        {
            "split": "warmup",
            "id_source": "clean_manifest",
            "retrieval_variant": "r8_ensemble_ltr_v2",
        },
    )
    with pytest.raises(ValueError, match="ensemble runs are not accepted"):
        analyze_warmup_retrieval_errors(
            run_dir=run_dir,
            metrics_path=metrics,
            bm25_path=bm25,
            qwen_path=qwen,
            labels_path=labels,
            passages_path=passages,
        )
