"""Acceptance tests for clean-warmup retrieval recall auditing."""

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
from legal_rag.sedar_retrieval.eval.retrieval_recall_audit import (
    RetrievalRecallAuditError,
    _curve_entry,
    _union_for_query,
    audit_warmup_retrieval_recall,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_candidate_union_preserves_both_source_cutoffs() -> None:
    union = _union_for_query(
        ("bm25-1", "bm25-2"),
        ("qwen-1", "target"),
        cutoff=2,
    )
    match_sets = {
        "q": {
            level: frozenset({"target"})
            for level in ("exact_passage", "article", "document", "provision")
        }
    }

    result = _curve_entry(
        ranked_by_query={"q": union},
        query_ids=("q",),
        match_sets=match_sets,
        cutoff=2,
        depth_sufficient_query_count=1,
        truncate_to_cutoff=False,
    )

    assert union == ("bm25-1", "bm25-2", "qwen-1", "target")
    assert result["exact_passage_hit_count"] == 1
    assert result["exact_passage_recall"] == pytest.approx(1.0)


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


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(
        run_dir / "config.json",
        {
            "split": "warmup",
            "id_source": "clean_manifest",
            "retrieval_variant": "ltr_full_all",
            "retrieval_path": "ltr.jsonl",
            "selected_ids_count": 3,
        },
    )
    _write_json(
        run_dir / "run_summary.json",
        {"run_id": "audit-run", "prediction_count": 3},
    )
    trace_rows = [
        {"id": "q-a", "raw_hit_ids": ["p-x", "p-a"], "packed_chunk_ids": ["p-x"]},
        {"id": "q-b", "raw_hit_ids": ["p-b", "p-z"], "packed_chunk_ids": ["p-b"]},
        {"id": "q-c", "raw_hit_ids": ["p-x"], "packed_chunk_ids": ["p-x"]},
    ]
    _write_jsonl(run_dir / "retrieval.jsonl", trace_rows)
    _write_jsonl(
        run_dir / "ltr.jsonl",
        [
            {"query_id": "q-a", "ranked_ids": ["p-x", "p-a"]},
            {"query_id": "q-b", "ranked_ids": ["p-b", "p-z"]},
            {"query_id": "q-c", "ranked_ids": ["p-x"]},
        ],
    )
    _write_jsonl(
        run_dir / "predictions.jsonl",
        [
            {"id": "q-a", "answer": "Đáp án a"},
            {"id": "q-b", "answer": "Đáp án b"},
            {"id": "q-c", "answer": "Đáp án c"},
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
                    "meteor": 0.8,
                    "rouge_l": 0.8,
                }
                for query_id in ("q-a", "q-b", "q-c")
            ],
        },
    )
    bm25 = tmp_path / "bm25.jsonl"
    _write_jsonl(
        bm25,
        [
            {"query_id": "q-a", "ranked_ids": ["p-x", "p-y"]},
            {"query_id": "q-b", "ranked_ids": ["p-b", "p-z"]},
            {"query_id": "q-c", "ranked_ids": ["p-x"]},
        ],
    )
    qwen = tmp_path / "qwen.jsonl"
    _write_jsonl(
        qwen,
        [
            {"query_id": "q-a", "ranked_ids": ["p-y", "p-a"]},
            {"query_id": "q-b", "ranked_ids": ["p-z", "p-y"]},
            {"query_id": "q-c", "ranked_ids": ["p-x"]},
        ],
    )
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(
        labels,
        [
            {"query_id": "q-a", "relevant_ids": ["p-a"], "provenance": "silver"},
            {"query_id": "q-b", "relevant_ids": ["p-b"], "provenance": "silver"},
            {"query_id": "q-c", "relevant_ids": [], "provenance": "unlabeled"},
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
                _passage("p-x"),
                _passage("p-y"),
                _passage("p-z"),
            )
        ],
    )
    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "bm25": bm25,
        "qwen": qwen,
        "labels": labels,
        "passages": passages,
    }


def _audit(
    fixture: dict[str, Path],
    *,
    scope_anchor_only: bool = False,
) -> dict[str, object]:
    return audit_warmup_retrieval_recall(
        run_dir=fixture["run_dir"],
        metrics_path=fixture["metrics"],
        bm25_path=fixture["bm25"],
        qwen_path=fixture["qwen"],
        labels_path=fixture["labels"],
        passages_path=fixture["passages"],
        cutoffs=(1, 2, 3),
        scope_anchor_only=scope_anchor_only,
    ).as_dict()


def test_audit_validates_scope_and_emits_recall_curves(
    tmp_path: Path,
) -> None:
    report = _audit(_build_fixture(tmp_path))

    assert report["query_count"] == 3
    assert report["labeled_query_count"] == 2
    assert report["unlabeled_query_count"] == 1
    assert report["declared_counts"] == {
        "config_selected_ids_count": 3,
        "run_summary_prediction_count": 3,
    }
    assert report["scope_validation"]["metrics"]["exact_match"] is True
    assert report["trace_validation"]["ltr_alignment"]["status"] == "matched"

    union_curves = report["recall_curves"]["candidate_union_bm25_qwen"]
    assert union_curves[0]["provision_hit_count"] == 1
    assert union_curves[0]["provision_recall"] == pytest.approx(0.5)
    assert union_curves[1]["provision_recall"] == pytest.approx(1.0)
    assert union_curves[2]["is_lower_bound"] is True

    contribution = report["source_contribution"]
    assert contribution[0]["counts"] == {"bm25_only": 1, "neither": 1}
    cases = {row["id"]: row for row in report["per_case"]}
    assert cases["q-a"]["source_contribution"]["1"] == "neither"
    assert cases["q-b"]["source_contribution"]["1"] == "bm25_only"
    assert cases["q-c"]["first_rank"] is None


def test_scope_anchor_allows_trace_from_previous_corpus(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    _write_jsonl(
        fixture["run_dir"] / "retrieval.jsonl",
        [
            {
                "id": "q-a",
                "raw_hit_ids": ["old-passage", "p-a"],
                "packed_chunk_ids": ["old-passage"],
            },
            {"id": "q-b", "raw_hit_ids": ["p-b"], "packed_chunk_ids": ["p-b"]},
            {"id": "q-c", "raw_hit_ids": ["p-x"], "packed_chunk_ids": ["p-x"]},
        ],
    )
    _write_jsonl(
        fixture["run_dir"] / "ltr.jsonl",
        [
            {"query_id": "q-a", "ranked_ids": ["old-passage", "p-a"]},
            {"query_id": "q-b", "ranked_ids": ["p-b"]},
            {"query_id": "q-c", "ranked_ids": ["p-x"]},
        ],
    )

    with pytest.raises(RetrievalRecallAuditError, match="absent from passages"):
        _audit(fixture)

    report = _audit(fixture, scope_anchor_only=True)

    assert report["scope_anchor_mode"] == "scope_only"
    assert report["trace_validation"]["trace_role"] == "clean_query_scope_anchor"
    compatibility = report["trace_validation"]["corpus_compatibility"]
    assert compatibility["status"] == "mismatch_allowed"
    assert compatibility["unknown_trace_id_count"] == 1
    assert compatibility["unknown_packed_id_count"] == 1
    assert "anchor_trace" in report["source_depths"]
    assert "ltr_trace" not in report["source_depths"]
    assert report["warnings"]


def test_audit_reports_trace_alignment_warning(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    _write_jsonl(
        fixture["run_dir"] / "ltr.jsonl",
        [
            {"query_id": "q-a", "ranked_ids": ["p-a", "p-x"]},
            {"query_id": "q-b", "ranked_ids": ["p-b", "p-z"]},
            {"query_id": "q-c", "ranked_ids": ["p-x"]},
        ],
    )

    report = _audit(fixture)

    assert report["trace_validation"]["ltr_alignment"]["status"] == "mismatch"
    assert report["trace_validation"]["ltr_alignment"]["mismatched_query_count"] == 1
    assert report["warnings"]


def test_audit_rejects_non_champion_scope(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    _write_json(
        fixture["run_dir"] / "config.json",
        {
            "split": "warmup",
            "id_source": "clean_manifest",
            "retrieval_variant": "rrf_dense",
        },
    )

    with pytest.raises(RetrievalRecallAuditError, match="ltr_full_all"):
        _audit(fixture)


def test_audit_rejects_unknown_retrieval_passage(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    _write_jsonl(
        fixture["bm25"],
        [
            {"query_id": "q-a", "ranked_ids": ["unknown"]},
            {"query_id": "q-b", "ranked_ids": ["p-b"]},
            {"query_id": "q-c", "ranked_ids": ["p-x"]},
        ],
    )

    with pytest.raises(RetrievalRecallAuditError, match="absent from passages"):
        _audit(fixture)


def test_audit_cli_writes_evaluation_only_json(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    output = tmp_path / "retrieval_recall_audit.json"
    script = (
        REPOSITORY_ROOT / "scripts" / "sedar_retrieval" / "audit_retrieval_recall.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(fixture["run_dir"]),
            "--metrics",
            str(fixture["metrics"]),
            "--bm25",
            str(fixture["bm25"]),
            "--qwen",
            str(fixture["qwen"]),
            "--labels",
            str(fixture["labels"]),
            "--passages",
            str(fixture["passages"]),
            "--cutoffs",
            "1,2,3",
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
    assert payload["schema_version"] == "sedar-warmup-retrieval-recall-audit-v1"
    assert payload["gold_answer_text_written"] is False
    assert "Đáp án" not in output.read_text(encoding="utf-8")
