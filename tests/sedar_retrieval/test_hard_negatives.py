"""Acceptance tests for TASK 10 hard-negative contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from legal_rag.sedar_retrieval.corpus.schema import (
    CanonicalPassage,
    SourceProvenance,
)
from legal_rag.sedar_retrieval.training.hard_negatives import (
    CandidateHit,
    HardNegativeMiningConfig,
    build_audit_sample,
    load_candidate_hits,
    mine_hard_negatives,
)
from legal_rag.sedar_retrieval.training.synthetic_queries import (
    SyntheticQueryRecord,
)


def _passage(
    passage_id: str,
    document_id: str,
    article_number: str,
    clause_number: str,
    text: str,
    *,
    status: str = "effective",
) -> CanonicalPassage:
    source = SourceProvenance(
        source_path=f"data/{document_id}.txt",
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
        status=status,
        raw_text=text,
        reader_text=text,
        retrieval_text=text,
        source=source,
    )


def _record(positive: CanonicalPassage) -> SyntheticQueryRecord:
    query = "Người lao động được nghỉ hằng năm như thế nào?"
    return SyntheticQueryRecord(
        synthetic_id="syn-1",
        query=query,
        positive_passage_id=positive.passage_id,
        source_document_id=positive.document_id,
        query_type="direct",
        legal_domain="vietnamese_law",
        generator_model="test",
        generator_revision="test-v1",
        prompt_version="test-prompt-v1",
        source_hash=positive.source.content_hash,
        quality_flags=("question_form_checked",),
        raw_generation=query,
        source_split="train",
    )


def test_mining_excludes_positive_and_classifies_structural_hard_negatives() -> None:
    positive = _passage(
        "p-positive",
        "law-1",
        "76",
        "1",
        "Người lao động được nghỉ hằng năm theo quy định.",
    )
    same_article = _passage(
        "p-same-article",
        "law-1",
        "76",
        "2",
        "Người lao động phải báo trước khi nghỉ hằng năm.",
    )
    other_article = _passage(
        "p-other-article",
        "law-1",
        "77",
        "1",
        "Người lao động được nghỉ việc trong trường hợp đặc biệt.",
    )
    other_law = _passage(
        "p-other-law",
        "law-2",
        "9",
        "1",
        "Người lao động có thể nghỉ hằng năm theo điều kiện riêng.",
    )
    records, report = mine_hard_negatives(
        (_record(positive),),
        (positive, same_article, other_article, other_law),
        {
            "syn-1": (
                CandidateHit("p-positive", 1, 12.0, "bm25"),
                CandidateHit("p-same-article", 2, 10.0, "bm25"),
                CandidateHit("p-other-article", 3, 8.0, "bm25"),
                CandidateHit("p-other-law", 4, 7.0, "bm25"),
            )
        },
        config=HardNegativeMiningConfig(min_negatives=2, max_negatives=3),
    )

    assert len(records) == 1
    negatives = records[0].negatives
    assert records[0].positive_passage_id not in {
        negative.negative_passage_id for negative in negatives
    }
    assert {negative.negative_category for negative in negatives} >= {"A", "B"}
    assert report.positive_in_negative_count == 0
    assert report.harder_than_random is True
    assert any(negative.potential_false_negative for negative in negatives)


def test_unresolved_candidates_cannot_satisfy_minimum() -> None:
    positive = _passage(
        "p-positive",
        "law-1",
        "76",
        "1",
        "Người lao động được nghỉ hằng năm theo quy định.",
    )
    known = _passage(
        "p-known",
        "law-2",
        "9",
        "1",
        "Người lao động có thể nghỉ hằng năm theo điều kiện riêng.",
    )
    records, report = mine_hard_negatives(
        (_record(positive),),
        (positive, known),
        {
            "syn-1": (
                CandidateHit("missing", 1, 9.0, "dense"),
                CandidateHit("p-known", 2, 0.7, "dense"),
            )
        },
    )

    assert records == ()
    assert report.unresolved_candidate_count == 1
    assert report.rejection_counts["insufficient_resolved_negatives"] == 1


def test_candidate_jsonl_loader_preserves_source_and_scores(tmp_path) -> None:
    path = tmp_path / "dense.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "query_id": "syn-1",
                        "ranked_ids": ["p-1", "p-2"],
                        "scores": [
                            {
                                "passage_id": "p-1",
                                "dense": 0.91,
                                "source": "dense",
                            },
                            {
                                "passage_id": "p-2",
                                "dense": 0.80,
                                "source": "dense",
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "query_id": "syn-rrf",
                        "ranked_ids": ["p-3"],
                        "candidates": [
                            {
                                "passage_id": "p-3",
                                "bm25_score": 4.0,
                                "dense_score": 0.80,
                                "rrf_score": 0.03,
                                "fused_rank": 1,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_candidate_hits((path,))

    assert loaded["syn-1"][0] == CandidateHit("p-1", 1, 0.91, "dense")
    assert loaded["syn-rrf"][0] == CandidateHit(
        "p-3",
        1,
        0.03,
        "rrf",
        ("bm25", "dense"),
    )


def test_audit_sample_is_pair_level_and_bounded() -> None:
    positive = _passage(
        "p-positive",
        "law-1",
        "76",
        "1",
        "Người lao động được nghỉ hằng năm theo quy định.",
    )
    same_article = _passage(
        "p-same-article",
        "law-1",
        "76",
        "2",
        "Người lao động phải báo trước khi nghỉ hằng năm.",
    )
    other_article = _passage(
        "p-other-article",
        "law-1",
        "77",
        "1",
        "Người lao động được nghỉ việc trong trường hợp đặc biệt.",
    )
    records, _ = mine_hard_negatives(
        (_record(positive),),
        (positive, same_article, other_article),
        {
            "syn-1": (
                CandidateHit("p-same-article", 1, 10.0, "bm25"),
                CandidateHit("p-other-article", 2, 9.0, "bm25"),
            )
        },
    )

    sample = build_audit_sample(records, sample_size=1)

    assert len(sample) == 1
    assert "positive_passage_id" in sample[0]
    assert sample[0]["manual_false_negative"] is None


def test_cli_writes_machine_gate_artifacts(tmp_path) -> None:
    positive = _passage(
        "p-positive",
        "law-1",
        "76",
        "1",
        "Người lao động được nghỉ hằng năm theo quy định.",
    )
    same_article = _passage(
        "p-same-article",
        "law-1",
        "76",
        "2",
        "Người lao động phải báo trước khi nghỉ hằng năm.",
    )
    other_article = _passage(
        "p-other-article",
        "law-1",
        "77",
        "1",
        "Người lao động được nghỉ việc trong trường hợp đặc biệt.",
    )
    random_passage = _passage(
        "p-random",
        "law-2",
        "1",
        "1",
        "Quy định về thủ tục đăng ký kinh doanh.",
    )
    passages_path = tmp_path / "passages.jsonl"
    passages_path.write_text(
        "\n".join(
            json.dumps(passage.model_dump(mode="json"), ensure_ascii=False)
            for passage in (positive, same_article, other_article, random_passage)
        )
        + "\n",
        encoding="utf-8",
    )
    synthetic_path = tmp_path / "synthetic.jsonl"
    synthetic_path.write_text(
        json.dumps(_record(positive).model_dump(mode="json"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        json.dumps(
            {
                "query_id": "syn-1",
                "ranked_ids": [
                    "p-positive",
                    "p-same-article",
                    "p-other-article",
                ],
                "scores": [
                    {"passage_id": "p-positive", "bm25": 12.0, "source": "bm25"},
                    {
                        "passage_id": "p-same-article",
                        "bm25": 10.0,
                        "source": "bm25",
                    },
                    {
                        "passage_id": "p-other-article",
                        "bm25": 8.0,
                        "source": "bm25",
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "task10"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "sedar_retrieval"
        / "mine_hard_negatives.py"
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
            "--synthetic",
            str(synthetic_path),
            "--passages",
            str(passages_path),
            "--candidate-file",
            str(candidates_path),
            "--output-dir",
            str(output_dir),
            "--audit-sample-size",
            "3",
        ],
        cwd=script.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "NEEDS_MANUAL_AUDIT"
    assert (output_dir / "hard_negatives.jsonl").is_file()
    assert (output_dir / "audit_sample.jsonl").is_file()
