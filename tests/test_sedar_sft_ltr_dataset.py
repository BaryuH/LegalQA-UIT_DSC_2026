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
    assert config.finetuned_reader.model.revision == (
        "258c56ed40529cced26fa7fcc3ecc0663e914c18"
    )
    assert config.finetuned_reader.lora.target_modules == (
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    )


def _renderer_builder() -> GenerativePromptBuilder:
    return GenerativePromptBuilder.from_files(
        REPO / "configs" / "prompts" / "sedar_sft_train_v1.txt",
        REPO / "configs" / "prompts" / "sedar_sft_infer_v1.txt",
        version="sedar-sft-ltr-v1",
    )


def _renderer_case() -> LegalQuestion:
    return LegalQuestion(
        id="t1",
        question="Người lao động được nghỉ hằng năm bao nhiêu ngày?",
        answer="Theo Điều 1, người lao động được nghỉ hằng năm.",
        split="train",
    )


def test_renderer_defaults_reproduce_v1_and_record_nothing() -> None:
    """A default build must stay fingerprint-compatible with sedar-sft-ltr-v1."""

    config = LtrDatasetBuildConfig(
        rankings_path=Path("rankings.jsonl"),
        passages_path=Path("passages.jsonl"),
    )
    assert config.renderer_overrides() == {}
    assert config.renderer_settings() == {
        "candidate_window": 0,
        "body_source": "raw_text",
        "dedup_article_mode": "off",
        "include_document_name": False,
        "min_passage_chars": 0,
    }


def test_champion_renderer_is_recorded_as_an_override() -> None:
    config = LtrDatasetBuildConfig(
        rankings_path=Path("rankings.jsonl"),
        passages_path=Path("passages.jsonl"),
        evidence_top_k=6,
        max_total_chars=6000,
        max_chunks_per_document=3,
        dedup_article_mode="article",
        include_document_name=True,
    )
    assert config.renderer_overrides() == {
        "dedup_article_mode": "article",
        "include_document_name": True,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dedup_article_mode": "not-a-mode"},
        {"body_source": "not-a-source"},
        {"candidate_window": -1},
        {"min_passage_chars": -1},
    ],
)
def test_renderer_settings_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LtrDatasetBuildConfig(
            rankings_path=Path("rankings.jsonl"),
            passages_path=Path("passages.jsonl"),
            **kwargs,
        )


def test_include_document_name_reaches_the_supervised_evidence() -> None:
    """The renderer flag must change the text the reader is trained on.

    Without this the training pack shows the zip member on the ``Văn bản``
    header while champion inference shows the real document name, which is the
    train/inference mismatch READER_RETRAIN_PLAN.md section 2 identifies.
    """

    builder = _renderer_builder()
    cases = (_renderer_case(),)
    rankings = {
        "t1": (RankedPassageCandidate(passage_id="p-1", rank=1, score=0.9),),
    }
    passages = {"p-1": _passage("p-1", "Người lao động được nghỉ hằng năm.")}

    def build(*, include_document_name: bool) -> str:
        examples, _, failures = build_sft_examples_from_ltr_rankings(
            cases,
            rankings,
            passages,
            prompt_builder=builder,
            evidence_config=PassageEvidenceConfig(
                evidence_top_k=1,
                include_document_name=include_document_name,
            ),
            retrieval_config_hash="ltr-cfg",
            index_fingerprint="idx",
        )
        assert not failures
        assert len(examples) == 1
        return examples[0].evidence.rendered_text

    without_name = build(include_document_name=False)
    with_name = build(include_document_name=True)
    assert "Văn bản: Luật mẫu" in with_name
    assert "Văn bản: Luật mẫu" not in without_name
    assert with_name != without_name
