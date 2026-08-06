import json
from pathlib import Path

import pytest

from legal_rag.config import load_config
from legal_rag.generation import MockLLMClient, PromptBuilder
from legal_rag.pipeline import PipelineError, run_bm25_rag, run_bm25_rag_from_config
from legal_rag.retrieval import BM25Config, build_bm25_index, retrieve_bm25
from legal_rag.schemas import LegalChunk, LegalDocument, LegalQuestion


def _chunk(chunk_id: str, document_id: str, text: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_path="fixture-contexts.zip",
        source_member=f"context_{document_id}.json",
        section_label="Điều 37",
        raw_text=text,
        retrieval_text=text,
        content_hash=f"hash-{chunk_id}",
        chunker_version="legal-chunker-v1",
    )


def _fixture() -> tuple[list[LegalChunk], dict[str, LegalDocument]]:
    chunks = [
        _chunk(
            "doc-a:0", "doc-a", "Điều 37. Nghỉ phép hằng năm theo quy định pháp luật."
        ),
        _chunk("doc-b:0", "doc-b", "Điều 12. Hợp đồng lao động và tiền lương."),
    ]
    documents = {
        "doc-a": LegalDocument(
            id="doc-a",
            name="Bộ luật lao động",
            passage=chunks[0].raw_text,
            source_path="fixture-contexts.zip",
            source_member="context_doc-a.json",
            content_hash="source-a",
        ),
        "doc-b": LegalDocument(
            id="doc-b",
            name="Văn bản khác",
            passage=chunks[1].raw_text,
            source_path="fixture-contexts.zip",
            source_member="context_doc-b.json",
            content_hash="source-b",
        ),
    }
    return chunks, documents


def test_bm25_retrieval_uses_question_only_and_stable_ties(tmp_path: Path) -> None:
    chunks, _ = _fixture()
    built = build_bm25_index(
        chunks,
        tmp_path / "cache",
        "e3-fixture-cache",
        BM25Config(),
    )
    assert built.index is not None

    hits = retrieve_bm25(built.index, "nghỉ phép hằng năm", top_k=5)

    assert tuple(hit.chunk_id for hit in hits) == ("doc-a:0",)
    assert hits[0].rank == 1
    assert hits[0].bm25_score > 0
    with pytest.raises(ValueError, match="non-blank"):
        retrieve_bm25(built.index, "   ", top_k=3)


def test_bm25_rag_fixture_e2e_writes_inference_artifacts_without_gold(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/bm25_rag.yaml"))
    chunks, documents = _fixture()
    built = build_bm25_index(
        chunks,
        tmp_path / "cache",
        "e3-fixture-cache",
        BM25Config(k1=config.retrieval.k1, b=config.retrieval.b),
    )
    assert built.index is not None
    prompt_builder = PromptBuilder.from_config(
        config.prompts,
        prompt_dir=Path("configs/prompts"),
    )
    client = MockLLMClient(config.generation, response_text="Có căn cứ trong evidence.")
    question = LegalQuestion(
        id="case-1",
        question="Nghỉ phép hằng năm theo quy định pháp luật là gì?",
        answer="GOLD MUST NEVER ENTER B1",
        split="warmup",
    )

    result = run_bm25_rag(
        (question,),
        {chunk.chunk_id: chunk for chunk in chunks},
        built.index,
        config,
        documents=documents,
        prompt_builder=prompt_builder,
        client=client,
        output_dir=tmp_path / "outputs",
        run_id="fixture-bm25-rag",
        data_manifest_hash="fixture-manifest",
        chunk_cache_fingerprint="e3-fixture-cache",
    )

    assert result.prediction_count == 1
    assert result.error_count == 0
    prediction_text = result.artifacts.predictions.read_text(encoding="utf-8")
    retrieval_text = result.artifacts.retrieval.read_text(encoding="utf-8")  # type: ignore[union-attr]
    generation_text = result.artifacts.generation.read_text(encoding="utf-8")
    assert "GOLD MUST NEVER ENTER B1" not in prediction_text
    assert "GOLD MUST NEVER ENTER B1" not in retrieval_text
    assert "GOLD MUST NEVER ENTER B1" not in generation_text
    prediction = json.loads(prediction_text)
    retrieval = json.loads(retrieval_text)
    generation = json.loads(generation_text)
    assert prediction["method"] == "bm25_rag"
    assert retrieval["packed_evidence"]["included_ids"][0] == "doc-a:0"
    assert generation["prompt"]["prompt_version"] == "rag-v1"
    assert generation["temperature"] == config.generation.temperature


def test_real_bm25_run_fails_closed_when_selected_contexts_are_missing(
    tmp_path: Path,
) -> None:
    config_text = (Path("configs/bm25_rag.yaml")).read_text(encoding="utf-8")
    config_text = config_text.replace(
        "data/selected-contexts.zip",
        "data/missing-selected-contexts.zip",
    )
    config_path = tmp_path / "bm25_missing_contexts.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises((PipelineError, FileNotFoundError, ValueError)):
        run_bm25_rag_from_config(str(config_path), limit=5)
