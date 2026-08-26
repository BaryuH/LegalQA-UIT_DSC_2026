"""Acceptance tests for SEDAR-SFT Path-B LTR-aligned dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.config import load_config
from legal_rag.finetuned_reader.dataset import (
    DatasetBuildError,
    load_prebuilt_sft_dataset,
)
from legal_rag.finetuned_reader.prompting import GenerativePromptBuilder
from legal_rag.schemas import LegalQuestion
from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage, SourceProvenance
from legal_rag.sedar_retrieval.evidence.passage_packer import (
    PassageEvidenceConfig,
    RankedPassageCandidate,
)
from legal_rag.sedar_sft.ltr_dataset import (
    LTR_EVIDENCE_SOURCE,
    LtrDatasetBuildConfig,
    build_sedar_sft_dataset_from_ltr,
    build_sft_examples_from_ltr_rankings,
)

REPO = Path(__file__).resolve().parents[1]


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


def test_ltr_pack_then_gold_join_and_missing_ranking() -> None:
    builder = GenerativePromptBuilder.from_files(
        REPO / "configs" / "prompts" / "sedar_sft_train_v1.txt",
        REPO / "configs" / "prompts" / "sedar_sft_infer_v1.txt",
        version="sedar-sft-ltr-v1",
    )
    cases = (
        LegalQuestion(
            id="t1",
            question="Người lao động được nghỉ hằng năm bao nhiêu ngày?",
            answer="Theo Điều 1, người lao động được nghỉ hằng năm.",
            split="train",
        ),
        LegalQuestion(
            id="t2",
            question="Câu hỏi không có ranking?",
            answer="Không nên vào dataset.",
            split="train",
        ),
    )
    rankings = {
        "t1": (
            RankedPassageCandidate(passage_id="p-1", rank=1, score=0.9),
            RankedPassageCandidate(passage_id="p-2", rank=2, score=0.4),
        ),
    }
    passages = {
        "p-1": _passage("p-1", "Người lao động được nghỉ hằng năm."),
        "p-2": _passage("p-2", "Thời gian nghỉ do thỏa thuận."),
    }
    examples, excluded, failures = build_sft_examples_from_ltr_rankings(
        cases,
        rankings,
        passages,
        prompt_builder=builder,
        evidence_config=PassageEvidenceConfig(evidence_top_k=2),
        retrieval_config_hash="ltr-cfg",
        index_fingerprint="idx",
    )
    assert len(examples) == 1
    assert examples[0].case_id == "t1"
    assert examples[0].target_answer.startswith("Theo Điều 1")
    assert "gold" not in examples[0].evidence.as_dict()
    assert examples[0].evidence.chunk_ids == ("p-1", "p-2")
    assert not excluded
    assert len(failures) == 1
    assert failures[0].reason_code == "MISSING_LTR_RANKING"
    assert failures[0].case_id == "t2"


def test_build_sedar_sft_dataset_from_ltr_end_to_end(tmp_path: Path) -> None:
    train_path = tmp_path / "train.json"
    train_path.write_text(
        json.dumps(
            {
                "101": {
                    "question": "Người lao động được nghỉ hằng năm?",
                    "answer": "Được nghỉ theo Điều 1.",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "warmup.json").write_text("{}", encoding="utf-8")
    (data_dir / "public-official.json").write_text("{}", encoding="utf-8")
    (data_dir / "private-official.json").write_text("{}", encoding="utf-8")

    rankings_path = tmp_path / "ltr.jsonl"
    rankings_path.write_text(
        json.dumps(
            {
                "query_id": "101",
                "ranked_ids": ["p-1"],
                "scores": [
                    {"passage_id": "p-1", "score": 0.8, "rank": 1, "source": "ltr"}
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    passages_path = tmp_path / "passages.jsonl"
    passages_path.write_text(
        _passage("p-1", "Người lao động được nghỉ hằng năm.").model_dump_json() + "\n",
        encoding="utf-8",
    )

    config = load_config(REPO / "configs" / "sedar_sft_train_ltr.yaml")
    payload = config.model_dump(mode="json")
    payload["data"]["data_dir"] = "data"
    payload["data"]["question_path"] = "train.json"
    payload["finetuned_reader"]["output"]["dataset_root"] = "artifacts/datasets"
    # relative to tmp repo root
    from legal_rag.config import ProjectConfig

    project = ProjectConfig.model_validate(payload)
    # Write config-relative paths under tmp_path as repo root
    (tmp_path / "configs" / "prompts").mkdir(parents=True)
    for name in ("sedar_sft_train_v1.txt", "sedar_sft_infer_v1.txt"):
        src = REPO / "configs" / "prompts" / name
        (tmp_path / "configs" / "prompts" / name).write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )
    # Move train.json under tmp root already; data_dir is tmp_path/data
    # question_path relative: train.json at repo root
    result = build_sedar_sft_dataset_from_ltr(
        project,
        repo_root=tmp_path,
        ltr=LtrDatasetBuildConfig(
            rankings_path=rankings_path,
            passages_path=passages_path,
            retrieval_variant="ltr_full_all",
            evidence_top_k=1,
        ),
    )
    assert result.example_count == 1
    assert result.underlying.manifest["evidence_source"] == LTR_EVIDENCE_SOURCE
    loaded = load_prebuilt_sft_dataset(
        result.underlying.output_dir,
        required_evidence_source=LTR_EVIDENCE_SOURCE,
    )
    assert len(loaded.examples) == 1
    assert loaded.examples[0].evidence.chunk_ids == ("p-1",)

    with pytest.raises(DatasetBuildError, match="evidence_source mismatch"):
        load_prebuilt_sft_dataset(
            result.underlying.output_dir,
            required_evidence_source="frozen_b2",
        )


def test_sedar_ltr_config_profile() -> None:
    config = load_config(REPO / "configs" / "sedar_sft_train_ltr.yaml")
    assert config.project.profile == "sedar_sft"
    assert config.finetuned_reader is not None
    assert config.finetuned_reader.dataset_version == "sedar-sft-ltr-v1"
    assert config.finetuned_reader.overlap_remediation_id == (
        "ftr03-train-overlap-exclusion-v1"
    )
