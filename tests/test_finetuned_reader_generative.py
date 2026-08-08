from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unicodedata import normalize

import pytest

from legal_rag.config import ProjectConfig, load_config
from legal_rag.finetuned_reader import dataset as dataset_module
from legal_rag.finetuned_reader import trainer as trainer_module
from legal_rag.finetuned_reader.checkpoint import (
    ValidatedCheckpoint,
    hash_directory,
    validate_checkpoint,
)
from legal_rag.finetuned_reader.collator import (
    TargetDoesNotFitError,
    build_answer_only_labels,
    tokenize_sft_example,
)
from legal_rag.finetuned_reader.contracts import EvidenceRecord, SFTExample
from legal_rag.finetuned_reader.dataset import (
    DatasetBuildError,
    FrozenRetrievalResult,
    build_sft_dataset,
    build_sft_dataset_from_config,
)
from legal_rag.finetuned_reader.inference import FineTunedReaderGenerator
from legal_rag.finetuned_reader.pipeline import run_finetuned_reader
from legal_rag.finetuned_reader.prompting import GenerativePromptBuilder
from legal_rag.finetuned_reader.training import TrainingGateError
from legal_rag.schemas import (
    InferenceQuestion,
    LegalQuestion,
    PackedEvidence,
    RetrievalHit,
)
from scripts.train_finetuned_reader import _apply_training_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]


def _evidence() -> PackedEvidence:
    hit = RetrievalHit(
        chunk_id="c1",
        document_id="d1",
        source_path="selected.zip",
        source_member="d1.json",
        section_label="Điều 1",
        rank=1,
        bm25_score=1.0,
    )
    return PackedEvidence(
        included_ids=("c1",),
        included_hits=(hit,),
        rendered_text="[TRÍCH ĐOẠN 1]\nĐiều 1. Nội dung pháp lý.",
    )


def _prompt_builder() -> GenerativePromptBuilder:
    return GenerativePromptBuilder(
        "Q: {question}\nE: {evidence}\nA:",
        "Q: {question}\nE: {evidence}\nA:",
        version="ftr-test-v1",
    )


def _example() -> SFTExample:
    evidence = EvidenceRecord.from_packed(
        _evidence(), retrieval_config_hash="cfg", index_fingerprint="idx"
    )
    return SFTExample(
        example_id="train::t1",
        case_id="t1",
        question="Câu hỏi pháp lý?",
        evidence=evidence,
        target_answer="Câu trả lời được giám sát.",
    )


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) % 97 + 3 for char in text]


class _FakeRetriever:
    def __init__(self) -> None:
        self.preparation = SimpleNamespace(
            index=SimpleNamespace(index_fingerprint="idx")
        )
        self.seen_questions: list[InferenceQuestion] = []

    def retrieve(self, question: InferenceQuestion) -> FrozenRetrievalResult:
        assert type(question) is InferenceQuestion
        self.seen_questions.append(question)
        return FrozenRetrievalResult(
            evidence=_evidence(),
            query_sha256="query-hash",
            retrieval_hits=("c1",),
            reranker={"used": False, "fallback_reason": "test"},
        )


def test_answer_only_labels_mask_prompt_and_eos() -> None:
    labels = build_answer_only_labels((10, 11), (20, 21), eos_token_id=2)

    assert labels == (-100, -100, 20, 21, 2)


def test_collator_never_silently_truncates_target() -> None:
    with_target = _example()
    try:
        tokenize_sft_example(
            with_target,
            tokenizer=_Tokenizer(),
            prompt_builder=_prompt_builder(),
            max_seq_length=3,
        )
    except TargetDoesNotFitError as exc:
        assert exc.reason_code == "TARGET_DOES_NOT_FIT"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("Expected explicit target-fit failure")


def test_sft_builder_attaches_gold_after_question_only_retrieval() -> None:
    retriever = _FakeRetriever()
    cases = (
        LegalQuestion(
            id="t1",
            question="Câu hỏi pháp lý?",
            answer="Câu trả lời được giám sát.",
            split="train",
        ),
    )

    examples, excluded, failures = build_sft_dataset(
        cases,
        retriever=retriever,  # type: ignore[arg-type]
        prompt_builder=_prompt_builder(),
        retrieval_config_hash="cfg",
    )

    assert len(examples) == 1
    assert not excluded
    assert not failures
    assert retriever.seen_questions[0].question == cases[0].question
    assert "target_answer" not in retriever.seen_questions[0].model_dump()
    assert examples[0].target_answer == cases[0].answer


def test_sft_builder_excludes_unicode_normalized_train_duplicate() -> None:
    retriever = _FakeRetriever()
    question = "Café legal question?"
    cases = (
        LegalQuestion(id="t1", question=question, answer="First.", split="train"),
        LegalQuestion(
            id="t2",
            question=normalize("NFD", question),
            answer="Second.",
            split="train",
        ),
    )

    examples, excluded, failures = build_sft_dataset(
        cases,
        retriever=retriever,  # type: ignore[arg-type]
        prompt_builder=_prompt_builder(),
        retrieval_config_hash="cfg",
    )

    assert [example.case_id for example in examples] == ["t1"]
    assert [(item.case_id, item.reason_code) for item in excluded] == [
        ("t2", "DUPLICATE_NORMALIZED_QUESTION")
    ]
    assert not failures


def test_sft_builder_smoke_limit_stops_after_requested_examples() -> None:
    retriever = _FakeRetriever()
    cases = tuple(
        LegalQuestion(
            id=f"t{index}",
            question=f"Question {index}",
            answer=f"Answer {index}",
            split="train",
        )
        for index in range(1, 4)
    )

    examples, excluded, failures = build_sft_dataset(
        cases,
        retriever=retriever,  # type: ignore[arg-type]
        prompt_builder=_prompt_builder(),
        retrieval_config_hash="cfg",
        max_examples=2,
    )

    assert [example.case_id for example in examples] == ["t1", "t2"]
    assert [question.id for question in retriever.seen_questions] == ["t1", "t2"]
    assert not excluded
    assert not failures


def test_dataset_cache_reuses_only_matching_complete_artifact(tmp_path: Path) -> None:
    example = _example()
    identity = {
        "dataset_version": "ftr-test-v1",
        "overlap_policy": "exclude_and_record",
        "overlap_remediation_id": "test-remediation",
        "profile": "finetuned_reader",
        "source_train_hash": "train-hash",
        "source_validation_hash": None,
        "retrieval_config_hash": "cfg",
        "index_fingerprint": "idx",
        "evidence_packer_hash": "packer-hash",
        "prompt_version": "ftr-test-v1",
        "prompt_hash": "prompt-hash",
        "cross_split_exclusions_hash": "exclusions-hash",
    }
    train_text = dataset_module._jsonl([example.as_dict()])
    manifest = {
        **identity,
        "dataset_scope": "full",
        "requested_max_examples": None,
        "example_count": 1,
        "excluded_count": 0,
        "retrieval_failure_count": 0,
        "examples_hash": dataset_module._hash_text(train_text),
    }
    dataset_module.write_dataset_artifacts(
        tmp_path,
        examples=(example,),
        excluded=(),
        retrieval_failures=(),
        manifest=manifest,
    )

    cached = dataset_module._load_cached_dataset(
        tmp_path,
        cache_identity=identity,
        max_examples=None,
    )

    assert cached is not None
    assert cached.examples == (example,)
    assert (
        dataset_module._load_cached_dataset(
            tmp_path,
            cache_identity={**identity, "prompt_hash": "changed"},
            max_examples=None,
        )
        is None
    )


def test_train_cli_overrides_preserve_effective_batch_size() -> None:
    config = load_config(REPO_ROOT / "configs" / "finetuned_reader_train.yaml")

    overridden = _apply_training_overrides(
        config,
        train_batch_size=2,
        gradient_accumulation_steps=8,
        bm25_backend="cuda",
    )

    assert overridden.finetuned_reader is not None
    assert overridden.finetuned_reader.training.train_batch_size == 2
    assert overridden.finetuned_reader.training.gradient_accumulation_steps == 8
    assert overridden.finetuned_reader.dataset_build.bm25_backend == "cuda"


def test_repository_dataset_build_fails_closed_on_cross_split_overlap() -> None:
    config = load_config(REPO_ROOT / "configs" / "finetuned_reader_train.yaml")
    payload = config.model_dump(mode="json")
    settings = payload["finetuned_reader"]
    assert isinstance(settings, dict)
    settings["overlap_policy"] = "fail"
    settings["overlap_remediation_id"] = None
    blocked_config = ProjectConfig.model_validate(payload)

    with pytest.raises(DatasetBuildError, match="Cross-split overlap"):
        build_sft_dataset_from_config(blocked_config, repo_root=REPO_ROOT)


def test_real_trainer_fails_closed_before_loading_model(monkeypatch) -> None:
    config = load_config(REPO_ROOT / "configs" / "finetuned_reader_train.yaml")

    def blocked_stack(*args, **kwargs):
        del args, kwargs
        raise TrainingGateError("fixture training stack is unresolved")

    monkeypatch.setattr(trainer_module, "require_training_stack", blocked_stack)
    with pytest.raises(TrainingGateError, match="fixture training stack"):
        trainer_module.run_real_sft(config, repo_root=REPO_ROOT, run_id="fixture")


def test_checkpoint_manifest_is_strict_and_hash_bound(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    adapter = checkpoint / "adapter"
    tokenizer = checkpoint / "tokenizer"
    adapter.mkdir(parents=True)
    tokenizer.mkdir()
    (adapter / "adapter.bin").write_bytes(b"adapter")
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest = {
        "profile": "finetuned_reader",
        "type": "generative_sft_reader",
        "base_model": "models/base",
        "base_revision": "local-revision",
        "tokenizer": "models/base",
        "tokenizer_path": "tokenizer",
        "adapter_path": "adapter",
        "adapter_type": "lora",
        "target_modules": ["q_proj"],
        "adapter_hash": hash_directory(adapter),
        "dataset_manifest_hash": "dataset",
        "retrieval_config_hash": "cfg",
        "index_fingerprint": "idx",
        "prompt_hash": "prompt",
        "seed": 42,
        "best_checkpoint_criterion": "meteor",
    }
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    validated = validate_checkpoint(
        checkpoint,
        expected={"retrieval_config_hash": "cfg", "index_fingerprint": "idx"},
    )

    assert isinstance(validated, ValidatedCheckpoint)
    assert validated.adapter_hash == manifest["adapter_hash"]


def test_checkpoint_rejects_unresolved_identity_fields(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    adapter = checkpoint / "adapter"
    tokenizer = checkpoint / "tokenizer"
    adapter.mkdir(parents=True)
    tokenizer.mkdir()
    (adapter / "adapter.bin").write_bytes(b"adapter")
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest = {
        "profile": "finetuned_reader",
        "type": "generative_sft_reader",
        "base_model": "UNRESOLVED",
        "base_revision": "local-revision",
        "tokenizer": "models/base",
        "tokenizer_path": "tokenizer",
        "adapter_path": "adapter",
        "adapter_type": "lora",
        "target_modules": ["q_proj"],
        "adapter_hash": hash_directory(adapter),
        "dataset_manifest_hash": "dataset",
        "retrieval_config_hash": "cfg",
        "index_fingerprint": "idx",
        "prompt_hash": "prompt",
        "seed": 42,
        "best_checkpoint_criterion": "meteor",
    }
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="base_model"):
        validate_checkpoint(checkpoint)


def test_generative_pipeline_writes_answer_free_inference_artifacts(
    tmp_path: Path,
) -> None:
    config = load_config(REPO_ROOT / "configs" / "finetuned_reader_generative.yaml")
    checkpoint = ValidatedCheckpoint(
        checkpoint_dir=tmp_path,
        manifest_path=tmp_path / "checkpoint_manifest.json",
        manifest={
            "base_model": "models/base",
            "base_revision": "local-revision",
            "adapter_path": "adapter",
            "adapter_hash": "adapter",
            "checkpoint_manifest_hash": "manifest",
            "dataset_manifest_hash": "dataset",
            "retrieval_config_hash": "cfg",
            "index_fingerprint": "idx",
        },
        manifest_hash="manifest",
        adapter_hash="adapter",
    )

    class _Backend:
        model = "mock-causal"
        model_version = "v1"

        def generate(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            stop_sequences: tuple[str, ...],
        ) -> str:
            assert "Câu hỏi" not in prompt
            assert max_new_tokens == 512
            assert not stop_sequences
            return "Câu trả lời sinh ra."

    generator = FineTunedReaderGenerator(
        backend=_Backend(),  # type: ignore[arg-type]
        prompt_builder=_prompt_builder(),
        checkpoint=checkpoint,
        max_new_tokens=512,
    )
    result = run_finetuned_reader(
        (InferenceQuestion(id="p1", question="Hỏi gì?", split="warmup"),),
        config=config,
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        generator=generator,
        output_dir=tmp_path / "runs",
        run_id="ftr-generative-fixture",
    )

    assert result.prediction_count == 1
    assert result.error_count == 0
    for path in (
        result.artifacts.predictions,
        result.artifacts.retrieval,
        result.artifacts.generation,
    ):
        assert path is not None
        content = path.read_text(encoding="utf-8").casefold()
        assert "gold_answer" not in content
        assert "reference_answer" not in content
