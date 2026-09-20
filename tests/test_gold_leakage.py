"""Targeted H2 tests for keeping gold answers outside inference boundaries."""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml
from pydantic import ValidationError

from legal_rag.config import ProjectConfig, load_config
from legal_rag.generation import MockLLMClient, PromptBuilder, PromptMetadata
from legal_rag.pipeline import RunResult, run_bm25_rag, run_direct, run_hybrid_rag
from legal_rag.retrieval import (
    BM25Config,
    BM25Index,
    BM25IndexedDocument,
    MockReranker,
    build_bm25_index,
)
from legal_rag.schemas import (
    InferenceQuestion,
    LegalChunk,
    LegalDocument,
    LegalQuestion,
    PackedEvidence,
    Prediction,
    RetrievalHit,
)
from legal_rag.submission import (
    SubmissionError,
    SubmissionSpec,
    build_submission,
    create_submission,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD_SENTINEL = "H2 GOLD ANSWER MUST NEVER ENTER INFERENCE"
_FORBIDDEN_GOLD_KEYS = {
    "gold",
    "goldanswer",
    "reference",
    "referenceanswer",
}


@dataclass(frozen=True, slots=True)
class LeakageFixture:
    """Typed fixture containing gold only on the source question record."""

    question: LegalQuestion
    chunks: tuple[LegalChunk, ...]
    documents: Mapping[str, LegalDocument]
    index: BM25Index
    direct_config: ProjectConfig
    bm25_config: ProjectConfig
    hybrid_config: ProjectConfig
    prompt_builder: PromptBuilder


@dataclass(frozen=True, slots=True)
class OfflineRuns:
    direct: RunResult
    bm25: RunResult
    hybrid: RunResult


def _chunk(chunk_id: str, document_id: str, text: str) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_path="h2-fixture-contexts.zip",
        source_member=f"context_{document_id}.json",
        section_label="Article 20",
        raw_text=text,
        retrieval_text=text,
        content_hash=f"h2-hash-{chunk_id}",
        chunker_version="legal-chunker-v1",
    )


def _fixture(tmp_path: Path) -> LeakageFixture:
    chunks = (
        _chunk(
            "h2-annual:0",
            "h2-annual",
            "Article 20. Annual leave is governed by the applicable legal rule.",
        ),
        _chunk(
            "h2-salary:0",
            "h2-salary",
            "Article 30. Salary payment follows the employment agreement.",
        ),
    )
    documents = {
        chunk.document_id: LegalDocument(
            id=chunk.document_id,
            name=f"H2 document {chunk.document_id}",
            passage=chunk.raw_text,
            source_path=chunk.source_path,
            source_member=chunk.source_member,
            content_hash=f"h2-source-{chunk.document_id}",
        )
        for chunk in chunks
    }
    direct_config = load_config(REPO_ROOT / "configs" / "direct.yaml")
    bm25_config = load_config(REPO_ROOT / "configs" / "bm25_rag.yaml")
    hybrid_config = load_config(REPO_ROOT / "configs" / "hybrid_rag.yaml")
    built = build_bm25_index(
        chunks,
        tmp_path / "cache",
        "h2-fixture-cache",
        BM25Config(k1=bm25_config.retrieval.k1, b=bm25_config.retrieval.b),
    )
    assert built.index is not None
    return LeakageFixture(
        question=LegalQuestion(
            id="h2-case-1",
            question="What is the annual leave legal rule?",
            answer=GOLD_SENTINEL,
            split="warmup",
        ),
        chunks=chunks,
        documents=documents,
        index=built.index,
        direct_config=direct_config,
        bm25_config=bm25_config,
        hybrid_config=hybrid_config,
        prompt_builder=PromptBuilder.from_config(
            bm25_config.prompts,
            prompt_dir=REPO_ROOT / "configs" / "prompts",
        ),
    )


def _assert_no_gold_payload(value: object, gold: str) -> None:
    """Check values recursively while separately rejecting gold-shaped keys."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = "".join(
                character for character in str(key).casefold() if character.isalnum()
            )
            assert normalized_key not in _FORBIDDEN_GOLD_KEYS, (
                f"forbidden gold/reference key reached inference payload: {key!r}"
            )
            _assert_no_gold_payload(nested, gold)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_gold_payload(nested, gold)
    elif isinstance(value, str):
        assert gold not in value


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        decoded = json.loads(line)
        assert isinstance(decoded, dict)
        records.append(decoded)
    return tuple(records)


def _run_offline(fixture: LeakageFixture, tmp_path: Path) -> OfflineRuns:
    chunk_map = {chunk.chunk_id: chunk for chunk in fixture.chunks}
    direct = run_direct(
        (fixture.question,),
        fixture.direct_config,
        prompt_builder=fixture.prompt_builder,
        client=MockLLMClient(
            fixture.direct_config.generation,
            response_text="Generated H2 answer.",
        ),
        output_dir=tmp_path / "outputs",
        run_id="h2-direct",
        data_manifest_hash="h2-fixture-manifest",
    )
    bm25 = run_bm25_rag(
        (fixture.question,),
        chunk_map,
        fixture.index,
        fixture.bm25_config,
        documents=fixture.documents,
        prompt_builder=fixture.prompt_builder,
        client=MockLLMClient(
            fixture.bm25_config.generation,
            response_text="Generated H2 answer.",
        ),
        output_dir=tmp_path / "outputs",
        run_id="h2-bm25",
        data_manifest_hash="h2-fixture-manifest",
        chunk_cache_fingerprint="h2-fixture-cache",
    )
    hybrid = run_hybrid_rag(
        (fixture.question,),
        chunk_map,
        fixture.index,
        fixture.hybrid_config,
        documents=fixture.documents,
        prompt_builder=fixture.prompt_builder,
        client=MockLLMClient(
            fixture.hybrid_config.generation,
            response_text="Generated H2 answer.",
        ),
        reranker=MockReranker(
            {"h2-annual:0": 0.9, "h2-salary:0": 0.1},
            model="h2-mock-reranker",
        ),
        output_dir=tmp_path / "outputs",
        run_id="h2-hybrid",
        data_manifest_hash="h2-fixture-manifest",
        chunk_cache_fingerprint="h2-fixture-cache",
    )
    return OfflineRuns(direct=direct, bm25=bm25, hybrid=hybrid)


def test_inference_view_has_no_answer_field() -> None:
    question = LegalQuestion(
        id="h2-view",
        question="A question",
        answer=GOLD_SENTINEL,
        split="warmup",
    )

    view = question.inference_view()

    assert isinstance(view, InferenceQuestion)
    assert not hasattr(view, "answer")
    assert set(view.model_dump(mode="json")) == {"id", "question", "split"}
    with pytest.raises(ValidationError):
        InferenceQuestion.model_validate(
            {"id": "h2-view", "question": "A question", "answer": GOLD_SENTINEL}
        )


def test_prompt_builder_has_question_only_typed_signatures() -> None:
    direct_signature = inspect.signature(PromptBuilder.build_direct)
    rag_signature = inspect.signature(PromptBuilder.build_rag)

    assert tuple(direct_signature.parameters) == ("self", "question")
    assert tuple(rag_signature.parameters) == ("self", "question", "evidence")
    assert direct_signature.parameters["question"].annotation == "str"
    assert rag_signature.parameters["question"].annotation == "str"
    assert rag_signature.parameters["evidence"].annotation == "PackedEvidence"
    builder = PromptBuilder.from_directory(REPO_ROOT / "configs" / "prompts")
    with pytest.raises(TypeError, match="question must be a string"):
        builder.build_direct(
            LegalQuestion(id="h2-type", question="A question")  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="PackedEvidence"):
        builder.build_rag("A question", object())  # type: ignore[arg-type]


def test_prompt_text_has_no_fixture_gold(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    direct_prompt = fixture.prompt_builder.build_direct(fixture.question.question)
    hit = RetrievalHit(
        chunk_id=fixture.chunks[0].chunk_id,
        document_id=fixture.chunks[0].document_id,
        source_path=fixture.chunks[0].source_path,
        source_member=fixture.chunks[0].source_member,
        section_label=fixture.chunks[0].section_label,
        rank=1,
        bm25_score=1.0,
    )
    evidence = PackedEvidence(
        included_ids=(hit.chunk_id,),
        included_hits=(hit,),
        rendered_text=fixture.chunks[0].raw_text,
    )
    rag_prompt = fixture.prompt_builder.build_rag(fixture.question.question, evidence)

    assert GOLD_SENTINEL not in direct_prompt.text
    assert GOLD_SENTINEL not in rag_prompt.text
    _assert_no_gold_payload(direct_prompt.text, GOLD_SENTINEL)
    _assert_no_gold_payload(rag_prompt.text, GOLD_SENTINEL)


def test_bm25_index_is_typed_retrieval_corpus_without_gold(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    assert all(
        isinstance(document, BM25IndexedDocument)
        for document in fixture.index.documents
    )
    allowed_fields = {
        "chunk_id",
        "document_id",
        "document_length",
        "end_offset",
        "ordinal",
        "retrieval_text",
        "section_label",
        "source_member",
        "source_path",
        "start_offset",
        "term_frequencies",
    }
    for document in fixture.index.documents:
        payload = document.as_dict()
        assert set(payload) == allowed_fields
        assert "answer" not in payload
        assert "raw_text" not in payload
        _assert_no_gold_payload(payload, GOLD_SENTINEL)


def test_retrieval_query_receives_question_only(monkeypatch, tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    import legal_rag.pipeline as pipeline_module

    original_retrieve = pipeline_module.retrieve_bm25
    observed_queries: list[str] = []

    def spy_retrieve(
        index: BM25Index, query: str, *, top_k: int, **kwargs
    ) -> tuple[RetrievalHit, ...]:
        observed_queries.append(query)
        return original_retrieve(index, query, top_k=top_k, **kwargs)

    monkeypatch.setattr(pipeline_module, "retrieve_bm25", spy_retrieve)
    run_bm25_rag(
        (fixture.question,),
        {chunk.chunk_id: chunk for chunk in fixture.chunks},
        fixture.index,
        fixture.bm25_config,
        documents=fixture.documents,
        prompt_builder=fixture.prompt_builder,
        client=MockLLMClient(fixture.bm25_config.generation),
        output_dir=tmp_path / "outputs",
        run_id="h2-query-only",
        chunk_cache_fingerprint="h2-fixture-cache",
    )

    assert observed_queries == [fixture.question.question]
    assert all(query != GOLD_SENTINEL for query in observed_queries)


def test_hybrid_reranker_query_receives_question_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    observed_queries: list[str] = []

    def score(query: str, hit: RetrievalHit) -> float:
        observed_queries.append(query)
        return 1.0 if hit.chunk_id == "h2-annual:0" else 0.5

    run_hybrid_rag(
        (fixture.question,),
        {chunk.chunk_id: chunk for chunk in fixture.chunks},
        fixture.index,
        fixture.hybrid_config,
        documents=fixture.documents,
        prompt_builder=fixture.prompt_builder,
        client=MockLLMClient(fixture.hybrid_config.generation),
        reranker=MockReranker(score_fn=score, model="h2-query-reranker"),
        output_dir=tmp_path / "outputs",
        run_id="h2-reranker-query-only",
        data_manifest_hash="h2-fixture-manifest",
        chunk_cache_fingerprint="h2-fixture-cache",
    )

    assert observed_queries == [fixture.question.question] * len(observed_queries)
    assert observed_queries
    assert all(query != GOLD_SENTINEL for query in observed_queries)


def test_prediction_artifacts_have_typed_predictions_without_gold(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runs = _run_offline(fixture, tmp_path)

    for result in (runs.direct, runs.bm25, runs.hybrid):
        records = _jsonl(result.artifacts.predictions)
        assert len(records) == 1
        record = records[0]
        core = {
            field: record[field]
            for field in (
                "id",
                "answer",
                "method",
                "status",
                "error_code",
                "fallback_reason",
            )
            if field in record
        }
        assert isinstance(Prediction.model_validate(core), Prediction)
        assert "gold_answer" not in record
        _assert_no_gold_payload(record, GOLD_SENTINEL)


def test_retrieval_artifacts_have_typed_hits_without_gold(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    runs = _run_offline(fixture, tmp_path)

    for result in (runs.bm25, runs.hybrid):
        retrieval_path = result.artifacts.retrieval
        assert retrieval_path is not None
        record = _jsonl(retrieval_path)[0]
        for hit_key in ("raw_hits", "reranked_hits"):
            if hit_key in record:
                hits = record[hit_key]
                assert isinstance(hits, list)
                assert all(
                    isinstance(RetrievalHit.model_validate(hit), RetrievalHit)
                    for hit in hits
                )
        packed = record.get("packed_evidence")
        if packed is not None:
            assert isinstance(PackedEvidence.model_validate(packed), PackedEvidence)
        assert "query" not in record
        assert "question" not in record
        _assert_no_gold_payload(record, GOLD_SENTINEL)


def test_generation_artifacts_have_prompt_metadata_only_without_gold(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    runs = _run_offline(fixture, tmp_path)

    for result in (runs.direct, runs.bm25, runs.hybrid):
        record = _jsonl(result.artifacts.generation)[0]
        prompt = record.get("prompt")
        assert isinstance(prompt, dict)
        assert set(prompt) == {"prompt_name", "prompt_version", "prompt_sha256"}
        assert isinstance(
            PromptMetadata(
                name=prompt["prompt_name"],
                version=prompt["prompt_version"],
                sha256=prompt["prompt_sha256"],
            ),
            PromptMetadata,
        )
        assert "prompt_text" not in record
        assert "question" not in record
        _assert_no_gold_payload(record, GOLD_SENTINEL)


def test_submission_rejects_gold_metadata_and_emits_official_fields_only() -> None:
    spec = SubmissionSpec(
        container="list",
        fields=("case_id", "answer"),
        id_field="case_id",
        answer_field="answer",
        id_type="string",
        ordering="target",
    )
    payload = build_submission(
        [{"id": "h2-case-1", "answer": "Generated H2 answer."}],
        ["h2-case-1"],
        spec,
    )

    assert payload == [{"case_id": "h2-case-1", "answer": "Generated H2 answer."}]
    assert set(payload[0]) == {"case_id", "answer"}
    _assert_no_gold_payload(payload, GOLD_SENTINEL)
    with pytest.raises(SubmissionError, match="unsupported prediction field"):
        build_submission(
            [
                {
                    "id": "h2-case-1",
                    "answer": "Generated H2 answer.",
                    "gold_answer": GOLD_SENTINEL,
                }
            ],
            ["h2-case-1"],
            spec,
        )


def test_official_submission_zip_drops_internal_fields_and_rejects_gold(
    tmp_path: Path,
) -> None:
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "h2-case-1": {
                    "question": "Question text.",
                    "answer": GOLD_SENTINEL,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "h2-case-1",
                "answer": "Generated answer.",
                "method": "direct",
                "raw_answer": "Generated answer.",
                "cleaned_answer": "Generated answer.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.zip"

    result = create_submission(predictions_path, questions_path, submission_path)

    assert result.valid
    with ZipFile(submission_path) as archive:
        assert archive.namelist() == ["submission.json"]
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
    assert payload == {"h2-case-1": {"answer": "Generated answer."}}
    _assert_no_gold_payload(payload, GOLD_SENTINEL)

    gold_predictions_path = tmp_path / "gold-predictions.jsonl"
    gold_predictions_path.write_text(
        json.dumps(
            {
                "id": "h2-case-1",
                "answer": "Generated answer.",
                "gold_answer": GOLD_SENTINEL,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SubmissionError, match="unsupported prediction field"):
        create_submission(
            gold_predictions_path,
            questions_path,
            tmp_path / "rejected-submission.zip",
        )


def test_private_profile_disallows_evaluator_reference_access() -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "mock.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    private_payload = copy.deepcopy(payload)
    private_payload["data"]["split"] = "private"
    private_payload["data"]["split_policy"] = "private_final_inference"

    with pytest.raises(ValidationError, match="Private split.*reference access"):
        ProjectConfig.model_validate(private_payload)

    private_payload["evaluation"]["enabled"] = False
    private_payload["evaluation"]["reference_access"] = "none"
    private_config = ProjectConfig.model_validate(private_payload)
    assert private_config.data.split == "private"
    assert private_config.evaluation.reference_access == "none"
