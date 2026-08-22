"""Acceptance tests for TASK 12 LTR feature construction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from legal_rag.schemas import InferenceQuestion
from legal_rag.sedar_retrieval.corpus.schema import (
    CanonicalPassage,
    SourceProvenance,
)
from legal_rag.sedar_retrieval.ranking.ltr_dataset import (
    LTRCandidate,
    LTRFeatureBuildConfig,
    LTRFeatureBuildError,
    SyntheticQueryView,
    build_ltr_feature_rows,
    load_rrf_candidates,
    load_synthetic_query_map,
    validate_feature_schema,
)


def _passage(
    passage_id: str,
    article_number: str,
    clause_number: str,
    text: str,
) -> CanonicalPassage:
    document_id = "law-1"
    source = SourceProvenance(
        source_path="data/law-1.txt",
        document_id=document_id,
        content_hash=f"hash-{passage_id}",
    )
    return CanonicalPassage(
        passage_id=passage_id,
        document_id=document_id,
        article_id=f"{document_id}:article:{article_number}",
        clause_id=f"{document_id}:article:{article_number}:clause:{clause_number}",
        retrieval_level="clause",
        document_name="Bộ luật Lao động 2019",
        article_number=article_number,
        clause_number=clause_number,
        status="effective",
        raw_text=text,
        reader_text=text,
        retrieval_text=text,
        source=source,
    )


def _question() -> InferenceQuestion:
    return InferenceQuestion(
        id="101515",
        question="Theo Khoản 1 Điều 76, người lao động được nghỉ hằng năm thế nào?",
        split="warmup",
    )


def test_rrf_candidates_are_loaded_with_source_scores(tmp_path) -> None:
    path = tmp_path / "rrf.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "101515",
                "ranked_ids": ["p-1", "p-2"],
                "candidates": [
                    {
                        "passage_id": "p-1",
                        "bm25_score": 4.2,
                        "dense_score": 0.81,
                        "rrf_score": 0.03,
                        "fused_rank": 1,
                    },
                    {
                        "passage_id": "p-2",
                        "bm25_score": 2.1,
                        "dense_score": 0.70,
                        "rrf_score": 0.02,
                        "fused_rank": 2,
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    candidates = load_rrf_candidates(path)

    assert candidates["101515"][0].passage_id == "p-1"
    assert candidates["101515"][0].rrf_score == pytest.approx(0.03)
    assert candidates["101515"][0].rank == 1


def test_feature_rows_use_citation_grades_and_finite_features() -> None:
    passages = {
        "p-1": _passage(
            "p-1",
            "76",
            "1",
            "Người lao động được nghỉ hằng năm theo quy định.",
        ),
        "p-2": _passage(
            "p-2",
            "77",
            "1",
            "Người lao động được nghỉ việc trong trường hợp đặc biệt.",
        ),
    }
    rows, report = build_ltr_feature_rows(
        candidates={
            "101515": (
                LTRCandidate("p-1", 1, rrf_score=0.03, source="rrf"),
                LTRCandidate("p-2", 2, rrf_score=0.02, source="rrf"),
            )
        },
        questions={"101515": _question()},
        passages=passages,
        config=LTRFeatureBuildConfig(label_mode="graded", unlabeled_policy="fail"),
    )

    assert [row["label"] for row in rows] == [3, 0]
    assert report.output_query_count == 1
    assert report.output_row_count == 2
    assert all(
        all(
            abs(value) != float("inf") and value == value
            for value in row["features"].values()
        )
        for row in rows
    )


def test_unlabeled_queries_are_skipped_or_rejected_explicitly() -> None:
    question = InferenceQuestion(
        id="q-no-citation",
        question="Người lao động được nghỉ hằng năm thế nào?",
        split="warmup",
    )
    passages = {
        "p-1": _passage(
            "p-1",
            "76",
            "1",
            "Người lao động được nghỉ hằng năm theo quy định.",
        )
    }
    candidates = {"q-no-citation": (LTRCandidate("p-1", 1),)}

    rows, report = build_ltr_feature_rows(
        candidates=candidates,
        questions={question.id: question},
        passages=passages,
        config=LTRFeatureBuildConfig(unlabeled_policy="skip"),
    )
    assert rows == ()
    assert report.skipped_unlabeled_query_count == 1

    with pytest.raises(LTRFeatureBuildError, match="no citation"):
        build_ltr_feature_rows(
            candidates=candidates,
            questions={question.id: question},
            passages=passages,
            config=LTRFeatureBuildConfig(unlabeled_policy="fail"),
        )


def test_schema_matches_extractor() -> None:
    schema = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "retrieval"
        / "ltr_feature_schema_v1.json"
    )
    assert len(validate_feature_schema(schema)) == 64


def test_cli_writes_rows_groups_and_manifest(tmp_path) -> None:
    positive = _passage(
        "p-1",
        "76",
        "1",
        "Người lao động được nghỉ hằng năm theo quy định.",
    )
    negative = _passage(
        "p-2",
        "77",
        "1",
        "Người lao động được nghỉ việc trong trường hợp đặc biệt.",
    )
    passages_path = tmp_path / "passages.jsonl"
    passages_path.write_text(
        "\n".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
            for item in (positive, negative)
        )
        + "\n",
        encoding="utf-8",
    )
    candidates_path = tmp_path / "rrf.jsonl"
    candidates_path.write_text(
        json.dumps(
            {
                "query_id": "101515",
                "ranked_ids": ["p-1", "p-2"],
                "candidates": [
                    {"passage_id": "p-1", "rrf_score": 0.03, "fused_rank": 1},
                    {"passage_id": "p-2", "rrf_score": 0.02, "fused_rank": 2},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "101515": {
                    "question": _question().question,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "features.jsonl"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "sedar_retrieval"
        / "build_ltr_features.py"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2] / "src")
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--candidates",
            str(candidates_path),
            "--passages",
            str(passages_path),
            "--questions",
            str(questions_path),
            "--split",
            "warmup",
            "--output",
            str(output_path),
            "--unlabeled-policy",
            "fail",
        ],
        cwd=script.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "PASS"
    assert manifest["metrics"]["output_row_count"] == 2
    assert output_path.with_suffix(".groups.jsonl").is_file()


def test_load_synthetic_query_map_filters_source_split(tmp_path) -> None:
    synthetic_path = tmp_path / "synthetic.jsonl"
    rows = [
        {
            "synthetic_id": "syn-train",
            "query": "Người lao động được nghỉ hằng năm thế nào?",
            "positive_passage_id": "p-1",
            "source_document_id": "doc-1",
            "query_type": "direct",
            "legal_domain": "vietnamese_law",
            "generator_model": "test",
            "generator_revision": "test-v1",
            "prompt_version": "test-prompt-v1",
            "source_hash": "hash-1",
            "quality_flags": ["question_form_checked"],
            "raw_generation": "Người lao động được nghỉ hằng năm thế nào?",
            "source_split": "train",
        },
        {
            "synthetic_id": "syn-val",
            "query": "Thời gian nghỉ hằng năm được tính thế nào?",
            "positive_passage_id": "p-2",
            "source_document_id": "doc-2",
            "query_type": "direct",
            "legal_domain": "vietnamese_law",
            "generator_model": "test",
            "generator_revision": "test-v1",
            "prompt_version": "test-prompt-v1",
            "source_hash": "hash-2",
            "quality_flags": ["question_form_checked"],
            "raw_generation": "Thời gian nghỉ hằng năm được tính thế nào?",
            "source_split": "validation",
        },
    ]
    synthetic_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    loaded = load_synthetic_query_map(synthetic_path, source_split="train")

    assert set(loaded) == {"syn-train"}
    assert loaded["syn-train"].positive_passage_id == "p-1"


def test_positive_passage_labels_binary_and_graded() -> None:
    passages = {
        "p-pos": _passage(
            "p-pos",
            "76",
            "1",
            "Người lao động được nghỉ hằng năm theo quy định.",
        ),
        "p-neg": _passage(
            "p-neg",
            "77",
            "1",
            "Người lao động được nghỉ việc trong trường hợp đặc biệt.",
        ),
    }
    synthetic = {
        "syn-1": SyntheticQueryView(
            synthetic_id="syn-1",
            query="Người lao động được nghỉ hằng năm thế nào?",
            positive_passage_id="p-pos",
        )
    }
    candidates = {
        "syn-1": (
            LTRCandidate("p-pos", 1, rrf_score=0.03, source="rrf"),
            LTRCandidate("p-neg", 2, rrf_score=0.02, source="rrf"),
        )
    }

    binary_rows, binary_report = build_ltr_feature_rows(
        candidates=candidates,
        synthetic_queries=synthetic,
        passages=passages,
        config=LTRFeatureBuildConfig(
            label_mode="binary",
            label_source="positive_passage_id",
        ),
    )
    graded_rows, graded_report = build_ltr_feature_rows(
        candidates=candidates,
        synthetic_queries=synthetic,
        passages=passages,
        config=LTRFeatureBuildConfig(
            label_mode="graded",
            label_source="positive_passage_id",
        ),
    )

    assert [row["label"] for row in binary_rows] == [1, 0]
    assert [row["label"] for row in graded_rows] == [3, 0]
    assert binary_report.skipped_unlabeled_query_count == 0
    assert graded_report.output_query_count == 1
    assert all(row["label_provenance"] == "positive_passage_id" for row in binary_rows)


def test_synthetic_scale_build_labels_every_query() -> None:
    passages = {
        f"p-{index}": _passage(
            f"p-{index}",
            str(index),
            "1",
            f"Quy định số {index}.",
        )
        for index in range(100)
    }
    synthetic = {
        f"syn-{index:04d}": SyntheticQueryView(
            synthetic_id=f"syn-{index:04d}",
            query=f"Câu hỏi synthetic số {index}?",
            positive_passage_id=f"p-{index}",
        )
        for index in range(100)
    }
    candidates = {
        query_id: (
            LTRCandidate(f"p-{index}", 1, rrf_score=0.03, source="rrf"),
            LTRCandidate(
                f"p-{(index + 1) % 100}",
                2,
                rrf_score=0.02,
                source="rrf",
            ),
        )
        for index, query_id in enumerate(sorted(synthetic))
    }

    rows, report = build_ltr_feature_rows(
        candidates=candidates,
        synthetic_queries=synthetic,
        passages=passages,
        config=LTRFeatureBuildConfig(
            label_mode="binary",
            label_source="positive_passage_id",
            unlabeled_policy="fail",
        ),
    )

    assert report.input_query_count == 100
    assert report.output_query_count == 100
    assert report.skipped_unlabeled_query_count == 0
    assert report.output_row_count == 200
    assert len(rows) == 200


def test_cli_synthetic_positive_passage_mode(tmp_path) -> None:
    positive = _passage(
        "p-1",
        "76",
        "1",
        "Người lao động được nghỉ hằng năm theo quy định.",
    )
    negative = _passage(
        "p-2",
        "77",
        "1",
        "Người lao động được nghỉ việc trong trường hợp đặc biệt.",
    )
    passages_path = tmp_path / "passages.jsonl"
    passages_path.write_text(
        "\n".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
            for item in (positive, negative)
        )
        + "\n",
        encoding="utf-8",
    )
    candidates_path = tmp_path / "rrf.jsonl"
    candidates_path.write_text(
        json.dumps(
            {
                "query_id": "syn-1",
                "ranked_ids": ["p-1", "p-2"],
                "candidates": [
                    {"passage_id": "p-1", "rrf_score": 0.03, "fused_rank": 1},
                    {"passage_id": "p-2", "rrf_score": 0.02, "fused_rank": 2},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    synthetic_path = tmp_path / "synthetic.jsonl"
    synthetic_path.write_text(
        json.dumps(
            {
                "synthetic_id": "syn-1",
                "query": "Người lao động được nghỉ hằng năm thế nào?",
                "positive_passage_id": "p-1",
                "source_document_id": "doc-1",
                "query_type": "direct",
                "legal_domain": "vietnamese_law",
                "generator_model": "test",
                "generator_revision": "test-v1",
                "prompt_version": "test-prompt-v1",
                "source_hash": "hash-1",
                "quality_flags": ["question_form_checked"],
                "raw_generation": "Người lao động được nghỉ hằng năm thế nào?",
                "source_split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "features.jsonl"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "sedar_retrieval"
        / "build_ltr_features.py"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2] / "src")
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--candidates",
            str(candidates_path),
            "--passages",
            str(passages_path),
            "--label-source",
            "positive_passage_id",
            "--synthetic",
            str(synthetic_path),
            "--source-split",
            "train",
            "--label-mode",
            "binary",
            "--output",
            str(output_path),
        ],
        cwd=script.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        output_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "PASS"
    assert manifest["label_source"] == "positive_passage_id"
    assert manifest["metrics"]["skipped_unlabeled_query_count"] == 0
