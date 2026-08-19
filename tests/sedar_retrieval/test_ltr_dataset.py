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
    build_ltr_feature_rows,
    load_rrf_candidates,
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
