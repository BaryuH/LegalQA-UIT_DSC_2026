"""Fail-fast, offline project self-check for the Legal RAG repository.

The checks use only deterministic local APIs and temporary fixture directories.
They deliberately inject ``MockLLMClient`` and ``MockReranker``; no configured
real provider or semantic model is constructed by this script.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeAlias


class SelfCheckError(RuntimeError):
    """Raised when one self-check step violates its local contract."""


CheckAction: TypeAlias = Callable[[], None]
CheckSpec: TypeAlias = tuple[str, CheckAction]
_EXPECTED_CHECK_COUNT = 13


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelfCheckError(message)


def _repository_root(root: str | Path | None) -> Path:
    selected = Path(__file__).resolve().parents[1] if root is None else Path(root)
    resolved = selected.resolve()
    root_text = str(resolved)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    source_text = str(resolved / "src")
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return resolved


def _check_manifest(root: Path) -> None:
    from scripts.verify_data_manifest import (
        DEFAULT_MANIFEST_PATH,
        verify_manifest,
    )

    count = verify_manifest(root, root / DEFAULT_MANIFEST_PATH)
    _require(count > 0, "data manifest contains no source files")


def _check_config_load(root: Path) -> None:
    from legal_rag.config import load_config

    names = (
        "default",
        "mock",
        "direct",
        "bm25_rag",
        "hybrid_rag",
        "finetuned_reader",
        "tuned_bm25_reader",
    )
    configs = [load_config(root / "configs" / f"{name}.yaml") for name in names]
    _require(len(configs) == len(names), "not all project profiles loaded")
    _require(
        configs[1].generation.provider == "mock",
        "offline mock profile is not configured with the mock provider",
    )


def _check_data_validation(root: Path) -> None:
    from legal_rag.config import load_config
    from legal_rag.data_validation import validate_data

    run = validate_data(load_config(root / "configs" / "mock.yaml"), root)
    _require(run.is_valid, "mock-profile data validation reported critical errors")


def _check_evaluator_golden() -> None:
    from legal_rag.evaluation import (
        LOCAL_SCORER_ID,
        EvaluationOptions,
        InputRecord,
        evaluate_records,
    )

    report = evaluate_records(
        (InputRecord(id="golden-1", answer="Legal answer 37."),),
        (InputRecord(id="golden-1", answer="Legal answer 37."),),
        EvaluationOptions(scorer=LOCAL_SCORER_ID),
    )
    _require(
        report.artifact["metrics"]["meteor"] == 1.0,
        "METEOR golden failed",
    )
    _require(report.artifact["metrics"]["rouge_l"] == 1.0, "ROUGE-L golden failed")


def _fixture_document() -> Any:
    from legal_rag.schemas import LegalDocument

    return LegalDocument(
        id="selfcheck-doc",
        name="Self-check document",
        passage="Article 37. Annual leave is governed by the legal rule.",
        source_path="selfcheck-contexts.zip",
        source_member="context_selfcheck-doc.json",
        content_hash="selfcheck-source-hash",
    )


def _fixture_chunks() -> tuple[Any, ...]:
    from legal_rag.schemas import LegalChunk

    rows = (
        ("selfcheck-a", "selfcheck-a-doc", "Article 37 annual leave legal rule."),
        ("selfcheck-b", "selfcheck-b-doc", "Article 12 salary legal rule."),
    )
    return tuple(
        LegalChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            source_path="selfcheck-contexts.zip",
            source_member=f"context_{document_id}.json",
            section_label="Article 37",
            raw_text=text,
            retrieval_text=text,
            content_hash=f"selfcheck-{chunk_id}",
            chunker_version="legal-chunker-v1",
        )
        for chunk_id, document_id, text in rows
    )


def _check_chunk_fixture(workspace: Path) -> None:
    from legal_rag.text import ChunkingConfig, build_chunk_cache

    result = build_chunk_cache(
        (_fixture_document(),),
        workspace / "chunk-cache",
        "selfcheck-manifest",
        ChunkingConfig(max_chars=120, overlap_chars=10, min_chars=10),
    )
    _require(result.chunks, "chunk fixture produced no chunks")
    _require(
        all(chunk.source_member for chunk in result.chunks),
        "chunk fixture lost source-member provenance",
    )


def _build_bm25_fixture(workspace: Path) -> tuple[tuple[Any, ...], Any]:
    from legal_rag.retrieval import BM25Config, build_bm25_index

    chunks = _fixture_chunks()
    built = build_bm25_index(
        chunks,
        workspace / "bm25-cache",
        "selfcheck-chunk-fingerprint",
        BM25Config(),
    )
    _require(built.index is not None, "BM25 fixture index was not built")
    return chunks, built.index


def _check_bm25_fixture(workspace: Path) -> None:
    from legal_rag.retrieval import retrieve_bm25

    _, index = _build_bm25_fixture(workspace)
    hits = retrieve_bm25(index, "annual leave", top_k=2)
    _require(hits and hits[0].chunk_id == "selfcheck-a", "BM25 fixture hit mismatch")
    _require(hits[0].bm25_score > 0, "BM25 fixture returned a non-positive score")


def _check_prompt_leakage(root: Path) -> None:
    from legal_rag.config import load_config
    from legal_rag.generation import PromptBuilder
    from legal_rag.schemas import LegalQuestion, PackedEvidence, RetrievalHit

    builder = PromptBuilder.from_config(
        load_config(root / "configs" / "mock.yaml").prompts,
        prompt_dir=root / "configs" / "prompts",
    )
    question = LegalQuestion(
        id="selfcheck-question",
        question="What is the annual leave rule?",
        answer="SELF-CHECK GOLD MUST NEVER ENTER PROMPT",
        split="warmup",
    )
    evidence = PackedEvidence(
        included_ids=("selfcheck-a",),
        included_hits=(
            RetrievalHit(
                chunk_id="selfcheck-a",
                document_id="selfcheck-a-doc",
                source_path="selfcheck-contexts.zip",
                rank=1,
                bm25_score=1.0,
            ),
        ),
        rendered_text="[1] Article 37 annual leave legal rule.",
    )
    rendered = (
        builder.build_direct(question.question).text
        + builder.build_rag(question.question, evidence).text
    ).casefold()
    _require(
        question.answer.casefold() not in rendered, "gold answer leaked into prompt"
    )
    _require(
        "chain-of-thought" not in rendered, "chain-of-thought text leaked into prompt"
    )
    _require(
        tuple(inspect.signature(PromptBuilder.build_direct).parameters)
        == ("self", "question"),
        "direct prompt accepts an unexpected record/model argument",
    )


def _question() -> Any:
    from legal_rag.schemas import LegalQuestion

    return LegalQuestion(
        id="selfcheck-case",
        question="What is the annual leave legal rule?",
        answer="SELF-CHECK GOLD MUST NEVER ENTER INFERENCE ARTIFACT",
        split="warmup",
    )


def _assert_inference_artifact(result: Any, gold: str, *, method: str) -> None:
    _require(result.method == method, f"unexpected method in {method} self-check")
    _require(result.prediction_count == 1, f"{method} produced no prediction")
    _require(result.error_count == 0, f"{method} recorded an inference error")
    for path in result.artifacts.run_dir.iterdir():
        if path.is_file():
            _require(
                gold not in path.read_text(encoding="utf-8"), "gold leaked to artifact"
            )


def _check_mock_direct(root: Path, workspace: Path) -> None:
    from legal_rag.config import load_config
    from legal_rag.generation import MockLLMClient, PromptBuilder
    from legal_rag.pipeline import run_direct

    config = load_config(root / "configs" / "direct.yaml")
    result = run_direct(
        (_question(),),
        config,
        prompt_builder=PromptBuilder.from_config(
            config.prompts,
            prompt_dir=root / "configs" / "prompts",
        ),
        client=MockLLMClient(config.generation, response_text="Self-check answer."),
        output_dir=workspace / "direct-output",
        run_id="selfcheck-direct",
        data_manifest_hash="selfcheck-manifest",
    )
    _assert_inference_artifact(
        result,
        "SELF-CHECK GOLD MUST NEVER ENTER INFERENCE ARTIFACT",
        method="direct",
    )


def _rag_inputs(root: Path, workspace: Path) -> tuple[Any, ...]:
    from legal_rag.config import load_config
    from legal_rag.generation import MockLLMClient, PromptBuilder

    chunks, index = _build_bm25_fixture(workspace)
    documents = {
        document_id: _fixture_document().model_copy(
            update={"id": document_id, "source_member": f"context_{document_id}.json"}
        )
        for document_id in {chunk.document_id for chunk in chunks}
    }
    config = load_config(root / "configs" / "bm25_rag.yaml")
    prompt_builder = PromptBuilder.from_config(
        config.prompts,
        prompt_dir=root / "configs" / "prompts",
    )
    client = MockLLMClient(
        config.generation, response_text="Self-check grounded answer."
    )
    return chunks, documents, index, config, prompt_builder, client


def _check_mock_bm25(root: Path, workspace: Path) -> None:
    from legal_rag.pipeline import run_bm25_rag

    chunks, documents, index, config, prompt_builder, client = _rag_inputs(
        root, workspace / "bm25-e2e"
    )
    result = run_bm25_rag(
        (_question(),),
        {chunk.chunk_id: chunk for chunk in chunks},
        index,
        config,
        documents=documents,
        prompt_builder=prompt_builder,
        client=client,
        output_dir=workspace / "bm25-output",
        run_id="selfcheck-bm25",
        data_manifest_hash="selfcheck-manifest",
        chunk_cache_fingerprint="selfcheck-chunk-fingerprint",
    )
    _assert_inference_artifact(
        result,
        "SELF-CHECK GOLD MUST NEVER ENTER INFERENCE ARTIFACT",
        method="bm25_rag",
    )


def _check_mock_hybrid(root: Path, workspace: Path) -> None:
    from legal_rag.config import load_config
    from legal_rag.pipeline import run_hybrid_rag
    from legal_rag.retrieval import MockReranker

    chunks, documents, index, _, prompt_builder, client = _rag_inputs(
        root, workspace / "hybrid-e2e"
    )
    config = load_config(root / "configs" / "hybrid_rag.yaml")
    result = run_hybrid_rag(
        (_question(),),
        {chunk.chunk_id: chunk for chunk in chunks},
        index,
        config,
        documents=documents,
        prompt_builder=prompt_builder,
        client=client,
        reranker=MockReranker(
            {"selfcheck-a": 0.9, "selfcheck-b": 0.1},
            model="selfcheck-reranker",
        ),
        output_dir=workspace / "hybrid-output",
        run_id="selfcheck-hybrid",
        data_manifest_hash="selfcheck-manifest",
        chunk_cache_fingerprint="selfcheck-chunk-fingerprint",
    )
    _assert_inference_artifact(
        result,
        "SELF-CHECK GOLD MUST NEVER ENTER INFERENCE ARTIFACT",
        method="hybrid_rag",
    )


def _check_mock_finetuned_reader(root: Path, workspace: Path) -> None:
    from legal_rag.config import load_config
    from legal_rag.finetuned_reader import (
        FineTunedReaderGenerator,
        FrozenRetrievalResult,
        GenerativePromptBuilder,
        ValidatedCheckpoint,
        run_finetuned_reader,
    )
    from legal_rag.schemas import InferenceQuestion, PackedEvidence, RetrievalHit

    config = load_config(root / "configs" / "finetuned_reader_generative.yaml")
    settings = config.finetuned_reader
    _require(settings is not None, "generative finetuned_reader config is incomplete")
    prompt_builder = GenerativePromptBuilder.from_files(
        root / settings.train_prompt_path,
        root / settings.inference_prompt_path,
        version=settings.dataset_version,
    )
    evidence = PackedEvidence(
        included_ids=("selfcheck-a",),
        included_hits=(
            RetrievalHit(
                chunk_id="selfcheck-a",
                document_id="selfcheck-a-doc",
                source_path="selfcheck-contexts.zip",
                source_member="context_selfcheck-a-doc.json",
                rank=1,
                bm25_score=1.0,
            ),
        ),
        rendered_text="[1] Article 37 annual leave legal rule.",
    )
    retrieval = FrozenRetrievalResult(
        evidence=evidence,
        query_sha256="selfcheck-query-hash",
        retrieval_hits=("selfcheck-a",),
        reranker={"used": False, "model": "selfcheck"},
    )

    class _Retriever:
        preparation = SimpleNamespace(
            index=SimpleNamespace(index_fingerprint="selfcheck-index")
        )

        def retrieve(self, question: InferenceQuestion) -> FrozenRetrievalResult:
            _require(
                type(question) is InferenceQuestion,
                "generative self-check passed a non-inference question",
            )
            return retrieval

    class _Backend:
        model = "selfcheck-causal"
        model_version = "selfcheck-revision"

        def generate(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            stop_sequences: tuple[str, ...],
        ) -> str:
            _require("SELF-CHECK GOLD" not in prompt, "gold reached causal prompt")
            _require(max_new_tokens > 0, "invalid self-check generation budget")
            _require(not stop_sequences, "unexpected self-check stop sequence")
            return "Self-check generated answer."

    manifest = {
        "base_model": "selfcheck-causal",
        "base_revision": "selfcheck-revision",
        "tokenizer": "selfcheck-tokenizer",
        "profile": "finetuned_reader",
        "type": "generative_sft_reader",
        "adapter_path": "adapter",
        "adapter_hash": "selfcheck-adapter-hash",
        "dataset_manifest_hash": "selfcheck-dataset-hash",
        "retrieval_config_hash": "selfcheck-retrieval-hash",
        "index_fingerprint": "selfcheck-index",
        "prompt_hash": "selfcheck-prompt-hash",
    }
    checkpoint = ValidatedCheckpoint(
        checkpoint_dir=workspace / "selfcheck-checkpoint",
        manifest_path=workspace / "selfcheck-checkpoint-manifest.json",
        manifest=manifest,
        manifest_hash="selfcheck-checkpoint-hash",
        adapter_hash="selfcheck-adapter-hash",
    )
    result = run_finetuned_reader(
        (
            InferenceQuestion(
                id="selfcheck-finetuned-reader",
                question="What is the annual leave legal rule?",
                split="warmup",
            ),
        ),
        config=config,
        retriever=_Retriever(),
        generator=FineTunedReaderGenerator(
            backend=_Backend(),
            prompt_builder=prompt_builder,
            checkpoint=checkpoint,
            max_new_tokens=settings.max_new_tokens,
        ),
        output_dir=workspace / "finetuned-reader-output",
        run_id="selfcheck-finetuned-reader",
    )
    _assert_inference_artifact(
        result,
        "SELF-CHECK GOLD MUST NEVER ENTER INFERENCE ARTIFACT",
        method="finetuned_reader",
    )


def _check_submission(workspace: Path) -> None:
    from zipfile import ZipFile

    from legal_rag.submission import (
        create_submission,
        load_submission_question_ids,
        validate_submission_file,
    )

    questions_path = workspace / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "007": {"question": "Question seven."},
                "1": {"question": "Question one."},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    predictions_path = workspace / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            (
                json.dumps({"id": "1", "answer": "Answer one."}, ensure_ascii=False),
                json.dumps({"id": "007", "answer": "Đáp án bảy."}, ensure_ascii=False),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    expected_ids = load_submission_question_ids(questions_path)
    _require(expected_ids == ("007", "1"), "submission dataset order was not preserved")
    output_path = workspace / "submission.zip"
    created = create_submission(predictions_path, questions_path, output_path)
    _require(created.valid, "official submission creation failed")
    validated = validate_submission_file(output_path, questions_path)
    _require(validated.valid, "official submission ZIP validation failed")
    with ZipFile(output_path) as archive:
        _require(archive.namelist() == ["submission.json"], "invalid ZIP member layout")
        inner = json.loads(archive.read("submission.json").decode("utf-8"))
    _require(list(inner) == ["007", "1"], "submission JSON order mismatch")
    _require(
        all(set(value) == {"answer"} for value in inner.values()), "metadata leaked"
    )


def _check_package_import() -> None:
    package = importlib.import_module("legal_rag")
    _require(bool(getattr(package, "__version__", "")), "package version is missing")
    importlib.import_module("legal_rag.cli")


def _build_checks(root: Path, workspace: Path) -> tuple[CheckSpec, ...]:
    return (
        ("data manifest verify", lambda: _check_manifest(root)),
        ("config load", lambda: _check_config_load(root)),
        ("data validation", lambda: _check_data_validation(root)),
        ("evaluator golden tests", _check_evaluator_golden),
        ("chunk fixture", lambda: _check_chunk_fixture(workspace)),
        ("BM25 fixture", lambda: _check_bm25_fixture(workspace)),
        ("prompt leakage tests", lambda: _check_prompt_leakage(root)),
        ("Mock E2E Direct", lambda: _check_mock_direct(root, workspace)),
        ("Mock E2E BM25-RAG", lambda: _check_mock_bm25(root, workspace)),
        ("Mock E2E Hybrid-RAG", lambda: _check_mock_hybrid(root, workspace)),
        (
            "Mock E2E generative finetuned_reader",
            lambda: _check_mock_finetuned_reader(root, workspace),
        ),
        ("submission validation", lambda: _check_submission(workspace)),
        ("package import", _check_package_import),
    )


def run_selfcheck(root: str | Path | None = None) -> tuple[str, ...]:
    """Run all checks in order and stop at the first failure."""

    repository_root = _repository_root(root)
    completed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="legal-rag-selfcheck-") as temporary:
        workspace = Path(temporary)
        checks = _build_checks(repository_root, workspace)
        if len(checks) != _EXPECTED_CHECK_COUNT:
            raise SelfCheckError(
                f"Expected {_EXPECTED_CHECK_COUNT} checks, got {len(checks)}"
            )
        for number, (name, action) in enumerate(checks, start=1):
            print(f"[{number}/{_EXPECTED_CHECK_COUNT}] {name}")
            action()
            completed.append(name)
            print(f"PASS {name}")
    return tuple(completed)


def main() -> int:
    """Run the self-check and return a process exit code."""

    try:
        completed = run_selfcheck()
    except Exception as exc:
        print(f"FAIL self-check: {type(exc).__name__}: {exc}")
        return 1
    print(f"SELF-CHECK PASS: {len(completed)}/{_EXPECTED_CHECK_COUNT} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
