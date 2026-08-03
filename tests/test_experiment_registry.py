"""Acceptance tests for the I1 append-only experiment registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.artifacts import (
    ExperimentRecord,
    ExperimentRegistry,
    ExperimentRegistryError,
    current_command,
    hash_run_outputs,
)
from legal_rag.config import load_config
from legal_rag.generation import MockLLMClient, PromptBuilder
from legal_rag.pipeline import run_direct, run_hybrid_rag
from legal_rag.retrieval import (
    BM25Config,
    MockReranker,
    NoOpReranker,
    build_bm25_index,
)
from legal_rag.schemas import InferenceQuestion, LegalChunk

REPO_ROOT = Path(__file__).resolve().parents[1]


def _record() -> ExperimentRecord:
    return ExperimentRecord(
        run_id="i1-fixture",
        git_commit="abc123",
        dirty=False,
        command="python -m legal_rag.cli run",
        config_hash="config-hash",
        split="warmup",
        data_manifest_hash="manifest-hash",
        chunk_fingerprint="chunk-hash",
        index_fingerprint="index-hash",
        prompt_hash="prompt-hash",
        model="mock-model",
        seed=42,
        meteor=0.75,
        rouge_l=0.5,
        error_rate=0.25,
        latency_ms=12.5,
        reranker_fallback_rate=0.5,
        output_hash="output-hash",
        notes="fixture",
    )


def test_registry_is_typed_jsonl_append_only_and_complete(tmp_path: Path) -> None:
    registry_path = tmp_path / "artifacts" / "experiments.jsonl"
    registry = ExperimentRegistry(registry_path)

    registry.append(_record())

    rows = registry.read()
    assert rows == (_record(),)
    payload = json.loads(registry_path.read_text(encoding="utf-8").splitlines()[0])
    assert set(payload) == {
        "schema_version",
        "run_id",
        "git_commit",
        "dirty",
        "command",
        "config_hash",
        "split",
        "data_manifest_hash",
        "chunk_fingerprint",
        "index_fingerprint",
        "prompt_hash",
        "model",
        "seed",
        "meteor",
        "rouge_l",
        "error_rate",
        "latency_ms",
        "reranker_fallback_rate",
        "output_hash",
        "notes",
    }
    with pytest.raises(FileExistsError, match="i1-fixture"):
        registry.append(_record())


def test_registry_validates_rows_and_redacts_command_secrets(tmp_path: Path) -> None:
    assert "do-not-log" not in current_command(
        ["python", "run.py", "--api-key=do-not-log", "--token", "also-secret"]
    )
    assert "<redacted>" in current_command(["python", "run.py", "--api-key=do-not-log"])

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text('{"run_id": "only-id"}\n', encoding="utf-8")
    with pytest.raises(ExperimentRegistryError, match="unexpected or missing"):
        ExperimentRegistry(bad_path).read()


def test_hash_run_outputs_is_deterministic_and_content_addressed(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "b.json").write_text("b", encoding="utf-8")
    (run_dir / "a.json").write_text("a", encoding="utf-8")

    first = hash_run_outputs(run_dir)
    assert first == hash_run_outputs(run_dir)
    (run_dir / "a.json").write_text("changed", encoding="utf-8")
    assert hash_run_outputs(run_dir) != first


def test_direct_run_registers_provenance_metrics_slots_and_output_hash(
    tmp_path: Path,
) -> None:
    config = load_config(REPO_ROOT / "configs" / "direct.yaml")
    registry_path = tmp_path / "artifacts" / "experiments.jsonl"
    prompt_builder = PromptBuilder.from_config(
        config.prompts,
        prompt_dir=REPO_ROOT / "configs" / "prompts",
    )

    result = run_direct(
        (InferenceQuestion(id="i1-direct", question="A registry question"),),
        config,
        prompt_builder=prompt_builder,
        client=MockLLMClient(
            config.generation,
            response_text="A registry answer.",
            latency_ms=12.5,
        ),
        output_dir=tmp_path / "outputs",
        run_id="i1-direct-run",
        data_manifest_hash="i1-manifest",
        registry_path=registry_path,
    )

    record = ExperimentRegistry(registry_path).read()[0]
    assert result.artifacts.registry == registry_path
    assert record.run_id == result.run_id
    assert record.split == "warmup"
    assert record.data_manifest_hash == "i1-manifest"
    assert record.config_hash == config.config_hash()
    assert record.prompt_hash == prompt_builder.direct_template.sha256
    assert record.model == config.generation.model
    assert record.seed == config.runtime.seed
    assert record.chunk_fingerprint is None
    assert record.index_fingerprint is None
    assert record.meteor is None
    assert record.rouge_l is None
    assert record.error_rate == 0.0
    assert record.latency_ms == pytest.approx(12.5)
    assert record.reranker_fallback_rate == 0.0
    assert record.output_hash == hash_run_outputs(result.artifacts.run_dir)
    assert record.notes == "metrics_not_evaluated"

    summary = json.loads(result.artifacts.summary.read_text(encoding="utf-8"))
    assert summary["artifacts"]["experiment_registry"] == registry_path.name


def test_hybrid_run_registers_explicit_reranker_fallback_rate(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "configs" / "hybrid_rag.yaml")
    chunk = LegalChunk(
        chunk_id="i1-chunk:0",
        document_id="i1-document",
        source_path="i1-contexts.zip",
        source_member="context_i1-document.json",
        section_label="Article 1",
        raw_text="Annual leave is governed by the legal rule.",
        retrieval_text="Annual leave is governed by the legal rule.",
        content_hash="i1-chunk-hash",
        chunker_version="legal-chunker-v1",
    )
    built = build_bm25_index(
        (chunk,),
        tmp_path / "cache",
        "i1-cache",
        BM25Config(k1=config.retrieval.k1, b=config.retrieval.b),
    )
    assert built.index is not None
    registry_path = tmp_path / "artifacts" / "experiments.jsonl"

    run_hybrid_rag(
        (InferenceQuestion(id="i1-hybrid", question="What is annual leave?"),),
        {chunk.chunk_id: chunk},
        built.index,
        config,
        prompt_builder=PromptBuilder.from_config(
            config.prompts,
            prompt_dir=REPO_ROOT / "configs" / "prompts",
        ),
        client=MockLLMClient(config.generation, response_text="A hybrid answer."),
        reranker=NoOpReranker(model="none", fallback_reason="fixture_fallback"),
        output_dir=tmp_path / "outputs",
        run_id="i1-hybrid-run",
        data_manifest_hash="i1-manifest",
        chunk_cache_fingerprint="i1-cache",
        registry_path=registry_path,
    )

    record = ExperimentRegistry(registry_path).read()[0]
    assert record.reranker_fallback_rate == 1.0

    mock_registry_path = tmp_path / "artifacts" / "mock-registry.jsonl"
    run_hybrid_rag(
        (InferenceQuestion(id="i1-hybrid-mock", question="What is annual leave?"),),
        {chunk.chunk_id: chunk},
        built.index,
        config,
        prompt_builder=PromptBuilder.from_config(
            config.prompts,
            prompt_dir=REPO_ROOT / "configs" / "prompts",
        ),
        client=MockLLMClient(config.generation, response_text="A hybrid answer."),
        reranker=MockReranker({chunk.chunk_id: 1.0}),
        output_dir=tmp_path / "outputs",
        run_id="i1-hybrid-mock-run",
        data_manifest_hash="i1-manifest",
        chunk_cache_fingerprint="i1-cache",
        registry_path=mock_registry_path,
    )
    assert (
        ExperimentRegistry(mock_registry_path).read()[0].reranker_fallback_rate == 0.0
    )
