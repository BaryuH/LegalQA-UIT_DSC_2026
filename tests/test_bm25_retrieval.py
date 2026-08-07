"""Acceptance tests for deterministic BM25 retrieval and inspection."""

import hashlib
import json
from pathlib import Path

import pytest

from legal_rag import cli
from legal_rag.config import load_config
from legal_rag.generation import MockLLMClient, PromptBuilder
from legal_rag.pipeline import (
    BM25Preparation,
    inspect_bm25_retrieval,
    run_bm25_rag,
)
from legal_rag.questions import inference_view, load_questions
from legal_rag.retrieval import (
    BM25Config,
    build_bm25_index,
    build_bm25_query_cache,
    retrieve_bm25,
)
from legal_rag.schemas import LegalChunk, LegalDocument, LegalQuestion


def _chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc-1",
    section_label: str = "Điều 37",
) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_path="selected-contexts.zip",
        source_member=f"context_{document_id}.json",
        section_label=section_label,
        raw_text=f"RAW EVIDENCE {text}",
        retrieval_text=text,
        content_hash=f"hash-{chunk_id}",
        chunker_version="legal-chunker-v1",
        start_offset=0,
        end_offset=len(text),
    )


def _fixture() -> list[LegalChunk]:
    return [
        _chunk(
            "chunk-z",
            "Điều 37 xử phạt hành vi dùng 153/2020/NĐ-CP trong 30 ngày.",
        ),
        _chunk(
            "chunk-a",
            "Điều 12 quy định mức phạt 1000 đồng trong 30 ngày.",
            document_id="doc-2",
            section_label="Khoản 2 Điều 12",
        ),
        _chunk(
            "chunk-b",
            "Điều 12 quy định mức phạt 1000 đồng trong 30 ngày.",
            document_id="doc-3",
            section_label="Điểm a Khoản 2 Điều 12",
        ),
    ]


def _index(tmp_path: Path):
    result = build_bm25_index(
        list(reversed(_fixture())),
        tmp_path / "cache",
        "chunk-cache-c5",
        BM25Config(k1=1.5, b=0.75),
    )
    assert result.index is not None
    return result.index


def test_expected_hit_uses_question_query_and_preserves_full_provenance(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    hits = retrieve_bm25(index, "Hành vi xử phạt theo 153/2020/NĐ-CP", top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk-z"
    assert hits[0].document_id == "doc-1"
    assert hits[0].source_path == "selected-contexts.zip"
    assert hits[0].source_member == "context_doc-1.json"
    assert hits[0].section_label == "Điều 37"
    assert hits[0].start_offset == 0
    assert hits[0].end_offset is not None
    assert hits[0].rank == 1
    assert hits[0].bm25_score > 0


def test_legal_code_query_and_stable_ties_are_deterministic(tmp_path: Path) -> None:
    index = _index(tmp_path)

    code_hits = retrieve_bm25(index, "153/2020/NĐ-CP", top_k=10)
    tie_hits = retrieve_bm25(index, "mức 1000 đồng", top_k=10)

    assert [hit.chunk_id for hit in code_hits] == ["chunk-z"]
    assert [hit.chunk_id for hit in tie_hits] == ["chunk-a", "chunk-b"]
    assert [hit.rank for hit in tie_hits] == [1, 2]


def test_query_cache_preserves_bm25_ranking(tmp_path: Path) -> None:
    index = _index(tmp_path)
    queries = ("153/2020/NĐ-CP", "mức 1000 đồng")
    query_cache = build_bm25_query_cache(index, queries)

    for query in queries:
        uncached = retrieve_bm25(index, query, top_k=10)
        cached = retrieve_bm25(index, query, top_k=10, query_cache=query_cache)
        assert [hit.chunk_id for hit in cached] == [hit.chunk_id for hit in uncached]
        assert [hit.bm25_score for hit in cached] == pytest.approx(
            [hit.bm25_score for hit in uncached]
        )


def test_empty_query_and_top_k_larger_than_corpus_are_explicit_and_safe(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)

    with pytest.raises(ValueError, match="non-blank"):
        retrieve_bm25(index, "   ", top_k=3)
    with pytest.raises(ValueError, match="greater than zero"):
        retrieve_bm25(index, "Điều 37", top_k=0)

    hits = retrieve_bm25(index, "Điều", top_k=100)
    assert len(hits) <= len(index.documents)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_retrieval_artifact_has_schema_and_no_gold_answer(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    config = load_config(repo_root / "configs" / "bm25_rag.yaml")
    chunks = _fixture()
    index = _index(tmp_path)
    documents = {
        chunk.document_id: LegalDocument(
            id=chunk.document_id,
            name=f"Văn bản {chunk.document_id}",
            passage=chunk.raw_text,
            source_path=chunk.source_path,
            source_member=chunk.source_member,
            content_hash=f"document-{chunk.document_id}",
        )
        for chunk in chunks
    }
    question = LegalQuestion(
        id="case-c5",
        question="Hành vi xử phạt theo 153/2020/NĐ-CP là gì?",
        answer="GOLD ANSWER MUST NEVER ENTER RETRIEVAL",
        split="warmup",
    )
    prompt_builder = PromptBuilder.from_config(
        config.prompts,
        prompt_dir=repo_root / "configs" / "prompts",
    )
    result = run_bm25_rag(
        (question,),
        {chunk.chunk_id: chunk for chunk in chunks},
        index,
        config,
        documents=documents,
        prompt_builder=prompt_builder,
        client=MockLLMClient(config.generation, response_text="Evidence grounded."),
        output_dir=tmp_path / "outputs",
        run_id="c5-artifact",
    )

    artifact_text = result.artifacts.retrieval.read_text(encoding="utf-8")
    artifact = json.loads(artifact_text.splitlines()[0])
    assert artifact["schema_version"] == "e3.retrieval.v1"
    assert artifact["top_k"] == config.retrieval.rough_top_n
    assert artifact["raw_hits"][0]["source_member"] == "context_doc-1.json"
    assert "GOLD ANSWER MUST NEVER ENTER RETRIEVAL" not in artifact_text


def test_inspection_rows_and_cli_output_use_raw_preview(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    repo_root = Path(__file__).parents[1]
    config = load_config(repo_root / "configs" / "bm25_rag.yaml")
    chunks = _fixture()
    documents = {
        chunk.document_id: LegalDocument(
            id=chunk.document_id,
            name=f"Văn bản {chunk.document_id}",
            passage=chunk.raw_text,
            source_path=chunk.source_path,
            source_member=chunk.source_member,
            content_hash=f"document-{chunk.document_id}",
        )
        for chunk in chunks
    }
    preparation = BM25Preparation(
        config=config,
        chunks=tuple(chunks),
        documents=documents,
        index=_index(tmp_path),
        manifest_hash="manifest-c5",
        chunk_cache_fingerprint="chunk-cache-c5",
    )

    rows = inspect_bm25_retrieval(
        preparation,
        "Hành vi xử phạt theo 153/2020/NĐ-CP",
        top_k=1,
    )
    assert rows[0].document_name == "Văn bản doc-1"
    assert rows[0].article_clause == "Điều 37"
    assert rows[0].preview.startswith("RAW EVIDENCE")

    monkeypatch.setattr(
        cli,
        "prepare_bm25_index_from_config",
        lambda config_path, rebuild_index: preparation,
    )
    exit_code = cli.main(
        [
            "inspect-retrieval",
            "--config",
            "configs/bm25_rag.yaml",
            "--question",
            "Hành vi xử phạt theo 153/2020/NĐ-CP",
            "--top-k",
            "1",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.splitlines()[0].split("\t") == [
        "rank",
        "score",
        "chunk_id",
        "document name",
        "article/clause",
        "preview",
    ]
    assert "chunk-z" in output
    assert "RAW EVIDENCE" in output


def test_real_warmup_smoke_runs_five_question_only_queries(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    source_path = repo_root / "data" / "warmup.json"
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    records = load_questions(source_path, split="warmup")
    views = inference_view(records)[:5]
    index = _index(tmp_path)

    assert len(views) == 5
    assert all("answer" not in view.model_dump() for view in views)
    for view in views:
        hits = retrieve_bm25(index, view.question, top_k=3)
        assert len(hits) <= 3
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before
