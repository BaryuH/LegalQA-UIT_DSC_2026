"""Offline tests for SEDAR-SFT SS-05..SS-07 scaffolding."""

from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.config import load_config
from legal_rag.finetuned_reader.collator import (
    TargetDoesNotFitError,
    build_answer_only_labels,
    tokenize_sft_example,
)
from legal_rag.finetuned_reader.contracts import EvidenceRecord, SFTExample
from legal_rag.finetuned_reader.prompting import GenerativePromptBuilder
from legal_rag.sedar_sft.contracts import (
    SEDAR_METHOD,
    SEDAR_TRAINING_ROLE,
    sedar_example_id,
    to_sedar_example_dict,
)
from legal_rag.sedar_sft.length_profile import (
    MockWhitespaceTokenizer,
    profile_sft_examples,
)
from legal_rag.sedar_sft.training_infra import inspect_sedar_training_infra

REPO = Path(__file__).resolve().parents[1]


def _example() -> SFTExample:
    evidence = EvidenceRecord(
        rendered_text="Điều 1. Nội dung bằng chứng đủ dài để tokenize.",
        chunk_ids=("c1",),
        document_ids=("d1",),
        retrieval_config_hash="cfg",
        index_fingerprint="idx",
        packed_evidence_hash="ev",
    )
    return SFTExample(
        example_id="train::1",
        case_id="1",
        question="Câu hỏi mẫu?",
        evidence=evidence,
        target_answer="Câu trả lời vàng đầy đủ.",
    )


def test_sedar_config_loads_and_locks_qlora_defaults() -> None:
    config = load_config(REPO / "configs" / "sedar_sft_train.yaml")
    assert config.project.profile == "sedar_sft"
    assert config.finetuned_reader is not None
    assert config.finetuned_reader.lora.adapter_type == "qlora"
    assert config.finetuned_reader.model.load_in_4bit is True
    assert config.finetuned_reader.training.packing is False
    assert config.finetuned_reader.training.gradient_checkpointing is True
    assert config.finetuned_reader.dataset_build.bm25_backend == "cuda"
    assert config.reranker.device == "cuda"


def test_sedar_example_contract_overlay() -> None:
    payload = to_sedar_example_dict(_example())
    assert payload["example_id"] == sedar_example_id("1")
    assert payload["training_role"] == SEDAR_TRAINING_ROLE
    assert payload["method"] == SEDAR_METHOD
    assert payload["target_answer"]
    assert "gold" not in payload["evidence"]


def test_sedar_prompts_reject_inference_target_placeholder() -> None:
    builder = GenerativePromptBuilder.from_files(
        REPO / "configs" / "prompts" / "sedar_sft_train_v1.txt",
        REPO / "configs" / "prompts" / "sedar_sft_infer_v1.txt",
        version="sedar-sft-v1",
    )
    assert "{question}" in builder.train_template
    assert "{evidence}" in builder.train_template
    assert "{target}" not in builder.inference_template
    assert "{answer}" not in builder.inference_template


def test_sedar_answer_only_collator_never_truncates_target() -> None:
    builder = GenerativePromptBuilder.from_files(
        REPO / "configs" / "prompts" / "sedar_sft_train_v1.txt",
        REPO / "configs" / "prompts" / "sedar_sft_infer_v1.txt",
        version="sedar-sft-v1",
    )
    tokenizer = MockWhitespaceTokenizer()
    labels = build_answer_only_labels([1, 2, 3], [4, 5], eos_token_id=1)
    assert labels[:3] == (-100, -100, -100)
    assert labels[-1] == 1
    with pytest.raises(TargetDoesNotFitError):
        tokenize_sft_example(
            _example(),
            tokenizer=tokenizer,
            prompt_builder=builder,
            max_seq_length=8,
        )


def test_sedar_length_profile_provisional_tokenizer() -> None:
    builder = GenerativePromptBuilder.from_files(
        REPO / "configs" / "prompts" / "sedar_sft_train_v1.txt",
        REPO / "configs" / "prompts" / "sedar_sft_infer_v1.txt",
        version="sedar-sft-v1",
    )
    result = profile_sft_examples(
        (_example(),),
        prompt_builder=builder,
        tokenizer=MockWhitespaceTokenizer(),
        tokenizer_mode="provisional_whitespace",
    )
    assert result.example_count == 1
    assert result.max_seq_length_selected is None
    assert "4096" in result.fit_percentages


def test_sedar_training_infra_defers_gpu_execution() -> None:
    config = load_config(REPO / "configs" / "sedar_sft_train.yaml")
    report = inspect_sedar_training_infra(
        config,
        repo_root=REPO,
        authorize_gpu_execution=False,
    )
    assert report.qlora_defaults_present is True
    assert report.ready_for_canonical_train is False
    assert "gpu_execution_deferred_by_operator" in report.blockers
