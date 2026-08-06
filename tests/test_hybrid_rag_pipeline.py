import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from legal_rag import cli
from legal_rag.config import load_config
from legal_rag.generation import MockLLMClient, PromptBuilder
from legal_rag.pipeline import (
    PipelineError,
    PipelineRunError,
    comparison_controls,
    run_bm25_rag,
    run_hybrid_rag,
    run_hybrid_rag_from_config,
)
from legal_rag.retrieval import BM25Config, MockReranker, NoOpReranker, build_bm25_index
from legal_rag.schemas import LegalChunk, LegalDocument, LegalQuestion


def _fixture() -> tuple[list[LegalChunk], dict[str, LegalDocument]]:
    rows = (
        ("chunk-a", "doc-a", "Article 1 annual leave legal rule."),
        ("chunk-b", "doc-b", "Article 2 annual leave legal rule."),
        ("chunk-c", "doc-c", "Article 3 salary legal rule."),
    )
    chunks = [
        LegalChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            source_path="fixture-contexts.zip",
            source_member=f"context_{document_id}.json",
            section_label="Article 1",
            raw_text=text,
            retrieval_text=text,
            content_hash=f"hash-{chunk_id}",
            chunker_version="legal-chunker-v1",
        )
        for chunk_id, document_id, text in rows
    ]
    documents = {
        chunk.document_id: LegalDocument(
            id=chunk.document_id,
            name=f"Document {chunk.document_id}",
            passage=chunk.raw_text,
            source_path=chunk.source_path,
            source_member=chunk.source_member,
            content_hash=f"source-{chunk.document_id}",
        )
        for chunk in chunks
    }
    return chunks, documents


def _question() -> LegalQuestion:
    return LegalQuestion(
        id="case-hybrid-1",
        question="What is the annual leave legal rule?",
        answer="GOLD ANSWER MUST NEVER ENTER B2",
        split="warmup",
    )


def _run_inputs(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    config_b1 = load_config(repo_root / "configs" / "bm25_rag.yaml")
    config_b2 = load_config(repo_root / "configs" / "hybrid_rag.yaml")
    chunks, documents = _fixture()
    built = build_bm25_index(
        chunks,
        tmp_path / "cache",
        "f3-fixture-cache",
        BM25Config(k1=config_b1.retrieval.k1, b=config_b1.retrieval.b),
    )
    assert built.index is not None
    prompt_builder = PromptBuilder.from_config(
        config_b1.prompts,
        prompt_dir=repo_root / "configs" / "prompts",
    )
    client = MockLLMClient(config_b1.generation, response_text="Grounded answer.")
    return (
        config_b1,
        config_b2,
        chunks,
        documents,
        built.index,
        prompt_builder,
        client,
    )


def test_hybrid_offline_e2e_changes_retrieval_order_only_and_preserves_scores(
    tmp_path: Path,
) -> None:
    (
        config_b1,
        config_b2,
        chunks,
        documents,
        index,
        prompt_builder,
        client,
    ) = _run_inputs(tmp_path)
    question = _question()

    b1 = run_bm25_rag(
        (question,),
        {chunk.chunk_id: chunk for chunk in chunks},
        index,
        config_b1,
        documents=documents,
        prompt_builder=prompt_builder,
        client=client,
        output_dir=tmp_path / "outputs",
        run_id="b1-same-split",
        chunk_cache_fingerprint="f3-fixture-cache",
    )
    b2 = run_hybrid_rag(
        (question,),
        {chunk.chunk_id: chunk for chunk in chunks},
        index,
        config_b2,
        documents=documents,
        prompt_builder=prompt_builder,
        client=client,
        reranker=MockReranker(
            {"chunk-a": 0.1, "chunk-b": 0.9, "chunk-c": 0.05},
            model="mock-reranker-f3",
        ),
        output_dir=tmp_path / "outputs",
        run_id="b2-same-split",
        chunk_cache_fingerprint="f3-fixture-cache",
    )

    b2_retrieval = json.loads(b2.artifacts.retrieval.read_text(encoding="utf-8"))
    b1_generation = json.loads(b1.artifacts.generation.read_text(encoding="utf-8"))
    b2_generation = json.loads(b2.artifacts.generation.read_text(encoding="utf-8"))

    assert b1.method == "bm25_rag"
    assert b2.method == "hybrid_rag"
    assert config_b1.data.split == config_b2.data.split == "warmup"
    assert comparison_controls(config_b1) == comparison_controls(config_b2)
    assert b1_generation["prompt"] == b2_generation["prompt"]
    assert b1_generation["provider"] == b2_generation["provider"]
    assert b1_generation["model"] == b2_generation["model"]
    assert b1_generation["temperature"] == b2_generation["temperature"]

    assert b2_retrieval["schema_version"] == "f3.retrieval.v1"
    assert b2_retrieval["method"] == "hybrid_rag"
    assert b2_retrieval["reranker"] == {
        "used": True,
        "model": "mock-reranker-f3",
        "fallback_reason": None,
        "metadata": {},
    }
    raw_hits = b2_retrieval["raw_hits"]
    reranked_hits = b2_retrieval["reranked_hits"]
    assert [hit["chunk_id"] for hit in reranked_hits] == [
        "chunk-b",
        "chunk-a",
        "chunk-c",
    ]
    assert [hit["chunk_id"] for hit in raw_hits] != [
        hit["chunk_id"] for hit in reranked_hits
    ]
    raw_scores = {hit["chunk_id"]: hit["bm25_score"] for hit in raw_hits}
    assert all(
        hit["bm25_score"] == raw_scores[hit["chunk_id"]] for hit in reranked_hits
    )
    assert [hit["rerank_score"] for hit in reranked_hits] == [0.9, 0.1, 0.05]
    assert b2_retrieval["packed_evidence"]["included_ids"][0] == "chunk-b"

    for artifact_path in (
        b2.artifacts.predictions,
        b2.artifacts.retrieval,
        b2.artifacts.generation,
        b2.artifacts.environment,
        b2.artifacts.summary,
    ):
        assert "GOLD ANSWER MUST NEVER ENTER B2" not in artifact_path.read_text(
            encoding="utf-8"
        )


def test_hybrid_optional_fallback_is_explicit_in_artifacts(tmp_path: Path) -> None:
    (
        _,
        config_b2,
        chunks,
        documents,
        index,
        prompt_builder,
        client,
    ) = _run_inputs(tmp_path)
    result = run_hybrid_rag(
        (_question(),),
        {chunk.chunk_id: chunk for chunk in chunks},
        index,
        config_b2,
        documents=documents,
        prompt_builder=prompt_builder,
        client=client,
        reranker=NoOpReranker(model="none", fallback_reason="test_disabled"),
        output_dir=tmp_path / "outputs",
        run_id="b2-explicit-fallback",
    )

    retrieval = json.loads(result.artifacts.retrieval.read_text(encoding="utf-8"))
    environment = json.loads(result.artifacts.environment.read_text(encoding="utf-8"))
    summary = json.loads(result.artifacts.summary.read_text(encoding="utf-8"))
    assert result.error_count == 0
    assert retrieval["reranker"]["used"] is False
    assert retrieval["reranker"]["fallback_reason"] == "test_disabled"
    assert environment["reranker_used"] == "false"
    assert environment["reranker_fallback_reason"] == "test_disabled"
    assert summary["reranker"]["fallback_reasons"] == ["test_disabled"]


def test_hybrid_required_fallback_fails_and_keeps_error_artifact(
    tmp_path: Path,
) -> None:
    (
        _,
        config_b2,
        chunks,
        documents,
        index,
        prompt_builder,
        client,
    ) = _run_inputs(tmp_path)
    required_config = config_b2.model_copy(
        update={"reranker": config_b2.reranker.model_copy(update={"required": True})}
    )

    with pytest.raises(PipelineRunError) as caught:
        run_hybrid_rag(
            (_question(),),
            {chunk.chunk_id: chunk for chunk in chunks},
            index,
            required_config,
            documents=documents,
            prompt_builder=prompt_builder,
            client=client,
            reranker=NoOpReranker(model="none", fallback_reason="test_required"),
            output_dir=tmp_path / "outputs",
            run_id="b2-required-failure",
        )

    result = caught.value.result
    retrieval = json.loads(result.artifacts.retrieval.read_text(encoding="utf-8"))
    errors = json.loads(
        result.artifacts.errors.read_text(encoding="utf-8").splitlines()[0]
    )
    assert result.method == "hybrid_rag"
    assert result.error_count == 1
    assert retrieval["status"] == "error"
    assert retrieval["reranker"]["fallback_reason"] == "test_required"
    assert errors["error_code"] == "RETRIEVAL_ERROR"
    assert "Required Hybrid-RAG reranker was not used" in errors["message"]


def test_hybrid_cli_routes_to_separate_method(monkeypatch, capsys) -> None:
    called: dict[str, object] = {}

    def fake_run(config_path, **kwargs):
        called["config_path"] = config_path
        called.update(kwargs)
        return SimpleNamespace(
            run_id="fake-hybrid",
            method="hybrid_rag",
            artifacts=SimpleNamespace(run_dir=Path("outputs/fake-hybrid")),
            prediction_count=1,
            error_count=0,
        )

    monkeypatch.setattr(cli, "run_hybrid_rag_from_config", fake_run)
    exit_code = cli.main(
        [
            "run",
            "--config",
            "configs/hybrid_rag.yaml",
            "--method",
            "hybrid_rag",
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    assert called["config_path"] == Path("configs/hybrid_rag.yaml")
    assert called["limit"] == 1
    assert json.loads(capsys.readouterr().out)["method"] == "hybrid_rag"


def test_real_hybrid_smoke_fails_closed_before_model_download(tmp_path: Path) -> None:
    config_text = (Path("configs/hybrid_rag.yaml")).read_text(encoding="utf-8")
    config_text = config_text.replace(
        "data/selected-contexts.zip",
        "data/missing-selected-contexts.zip",
    )
    config_path = tmp_path / "hybrid_missing_contexts.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises((PipelineError, FileNotFoundError, ValueError)):
        run_hybrid_rag_from_config(str(config_path), limit=1)
