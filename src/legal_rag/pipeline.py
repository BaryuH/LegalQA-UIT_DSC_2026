"""Inference pipelines for the Direct, BM25-RAG, and Hybrid-RAG baselines.

The pipeline module composes already-validated boundaries.  It never passes a full
question record to prompts or providers, and inference artifacts contain no gold
answer/reference fields.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .artifacts import (
    ExperimentRecord,
    ExperimentRegistry,
    RunArtifactPaths,
    RunManager,
    current_command,
    fingerprint_json,
    hash_run_outputs,
)
from .config import ProjectConfig, load_config
from .contexts import load_selected_contexts
from .evaluation import EVALUATOR_NAME, EVALUATOR_VERSION
from .evidence import (
    DeduplicationResult,
    EvidencePackingError,
    deduplicate_retrieved_chunks,
    pack_evidence,
)
from .generation import (
    LLMClient,
    LLMClientError,
    LLMResponse,
    PromptBuilder,
    PromptRender,
    create_llm_client,
    postprocess_answer,
)
from .questions import inference_view, load_questions
from .retrieval import (
    BM25Config,
    BM25Index,
    Reranker,
    RerankResult,
    create_reranker,
    load_or_build_bm25_index,
    retrieve_bm25,
)
from .schemas import (
    InferenceQuestion,
    LegalChunk,
    LegalDocument,
    LegalQuestion,
    PackedEvidence,
    Prediction,
    RetrievalHit,
)
from .text import ChunkingConfig, build_chunk_cache

PipelineMethod = Literal["direct", "bm25_rag", "hybrid_rag"]
CaseErrorType = Literal["prompt", "retrieval", "provider", "generation", "runtime"]
_RAG_METHODS = frozenset(("bm25_rag", "hybrid_rag"))


class PipelineError(RuntimeError):
    """Raised when a baseline cannot complete without violating its contract."""


class PipelineRunError(PipelineError):
    """Raised after structured case errors have been written to artifacts."""

    def __init__(self, message: str, result: RunResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True, slots=True)
class RunResult:
    """Structured result returned by Direct, BM25-RAG, and Hybrid-RAG runs."""

    method: PipelineMethod
    run_id: str
    artifacts: RunArtifactPaths
    prediction_count: int
    error_count: int
    index_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class BM25Preparation:
    """Validated source/chunk/index bundle used by B1, B2, and inspection."""

    config: ProjectConfig
    chunks: tuple[LegalChunk, ...]
    documents: Mapping[str, LegalDocument]
    index: BM25Index
    manifest_hash: str
    chunk_cache_fingerprint: str


@dataclass(frozen=True, slots=True)
class RetrievalInspectionRow:
    """Human-readable retrieval inspection row derived from raw chunk text."""

    rank: int
    score: float
    chunk_id: str
    document_name: str
    article_clause: str
    preview: str

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "article_clause": self.article_clause,
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "preview": self.preview,
            "rank": self.rank,
            "score": self.score,
        }


def inspect_bm25_retrieval(
    preparation: BM25Preparation,
    question: str,
    *,
    top_k: int | None = None,
) -> tuple[RetrievalInspectionRow, ...]:
    """Return deterministic inspection rows for a question-only query."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Retrieval question must be a non-blank string")
    requested_top_k = (
        preparation.config.retrieval.rough_top_n if top_k is None else top_k
    )
    if requested_top_k <= 0:
        raise ValueError("Retrieval top_k must be greater than zero")

    chunks = {chunk.chunk_id: chunk for chunk in preparation.chunks}
    hits = retrieve_bm25(preparation.index, question, top_k=requested_top_k)
    rows: list[RetrievalInspectionRow] = []
    for hit in hits:
        chunk = chunks.get(hit.chunk_id)
        document = preparation.documents.get(hit.document_id)
        if chunk is None or document is None:
            raise PipelineError(
                f"Retrieval hit {hit.chunk_id!r} has incomplete provenance"
            )
        preview = " ".join(chunk.raw_text.split())
        if len(preview) > 240:
            preview = preview[:237].rstrip() + "..."
        rows.append(
            RetrievalInspectionRow(
                rank=hit.rank,
                score=hit.bm25_score,
                chunk_id=hit.chunk_id,
                document_name=document.name,
                article_clause=chunk.section_label or "(unlabeled)",
                preview=preview,
            )
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class _CaseError:
    case_id: str
    error_code: str
    message: str
    stage: str
    error_type: CaseErrorType = "runtime"
    retries: int = 0
    retryable: bool = False

    def as_dict(self) -> dict[str, str | int | bool]:
        return {
            "id": self.case_id,
            "error_code": self.error_code,
            "error_type": self.error_type,
            "message": self.message,
            "stage": self.stage,
            "retries": self.retries,
            "retryable": self.retryable,
        }


def _json_hash(value: object) -> str:
    return fingerprint_json(value)


def _default_registry_path(output_dir: str | Path) -> Path:
    """Place a default registry beside the configured output directory."""

    output_root = Path(output_dir).resolve()
    return output_root.parent / "artifacts" / "experiments.jsonl"


def _mean_latency_ms(records: Sequence[Mapping[str, object]]) -> float | None:
    values = [
        float(value)
        for record in records
        for value in (record.get("latency_ms"),)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(values) / len(values) if values else None


def _manifest_hash(repo_root: Path) -> str:
    from scripts.verify_data_manifest import (
        DEFAULT_MANIFEST_PATH,
        build_manifest,
        verify_manifest,
    )

    manifest = build_manifest(repo_root)
    verify_manifest(repo_root, repo_root / DEFAULT_MANIFEST_PATH)
    return _json_hash(manifest)


def _question_view(
    questions: Sequence[LegalQuestion | InferenceQuestion],
) -> tuple[InferenceQuestion, ...]:
    views: list[InferenceQuestion] = []
    for question in questions:
        if isinstance(question, LegalQuestion):
            views.append(question.inference_view())
        elif isinstance(question, InferenceQuestion):
            views.append(question)
        else:
            raise TypeError("questions must contain LegalQuestion or InferenceQuestion")
    return tuple(sorted(views, key=lambda item: item.id))


def clean_generated_answer(text: str) -> str:
    """Return the cleaned answer while retaining the full result at the boundary."""

    return postprocess_answer(text).cleaned_answer


def comparison_controls(config: ProjectConfig) -> dict[str, str | float | int]:
    """Return controls that must match across baseline comparisons."""

    return {
        "split": config.data.split,
        "model": config.generation.model,
        "temperature": config.generation.temperature,
        "max_output_chars": config.generation.max_output_chars,
        "evaluator_name": EVALUATOR_NAME,
        "evaluator_version": EVALUATOR_VERSION,
        "direct_prompt_version": config.prompts.direct_version,
        "rag_prompt_version": config.prompts.rag_version,
    }


def _generation_record(
    question_id: str,
    config: ProjectConfig,
    prompt: PromptRender,
    client: LLMClient,
    *,
    method: PipelineMethod,
    response: LLMResponse | None = None,
    error: _CaseError | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "id": question_id,
        "method": method,
        "prompt": prompt.metadata.as_dict(),
        "provider": client.provider,
        "model": client.model,
        "temperature": config.generation.temperature,
        "max_output_chars": config.generation.max_output_chars,
        "max_output_tokens": config.generation.max_output_chars,
        "status": "error" if error else "success",
    }
    if response is not None:
        response_metadata = response.safe_metadata()
        record.update(
            {
                "latency_ms": response.latency_ms,
                "retries": response.retries,
                "metadata": response_metadata,
            }
        )
    if error is not None:
        record["error_code"] = error.error_code
        record["error_stage"] = error.stage
        record["error_type"] = error.error_type
        record["retries"] = error.retries
        record["retryable"] = error.retryable
    return record


def _retrieval_record(
    question_id: str,
    question_text: str,
    index: BM25Index,
    raw_hits: Sequence[RetrievalHit],
    deduplication: DeduplicationResult,
    packed: PackedEvidence | None,
    *,
    method: PipelineMethod = "bm25_rag",
    top_k: int,
    evidence_top_k: int | None = None,
    limited_ids: Sequence[str] = (),
    reranked_hits: Sequence[RetrievalHit] | None = None,
    rerank_result: RerankResult | None = None,
    error: _CaseError | None = None,
) -> dict[str, object]:
    schema_version = "e3.retrieval.v1" if method == "bm25_rag" else "f3.retrieval.v1"
    record: dict[str, object] = {
        "schema_version": schema_version,
        "id": question_id,
        "method": method,
        "query_sha256": hashlib.sha256(question_text.encode("utf-8")).hexdigest(),
        "index_fingerprint": index.index_fingerprint,
        "top_k": top_k,
        "raw_hits": [hit.model_dump(mode="json") for hit in raw_hits],
        "deduplication": deduplication.as_dict(),
        "top_k_limited_ids": list(limited_ids),
        "status": "error" if error else "success",
    }
    if evidence_top_k is not None:
        record["evidence_top_k"] = evidence_top_k
    if reranked_hits is not None:
        record["reranked_hits"] = [hit.model_dump(mode="json") for hit in reranked_hits]
    if rerank_result is not None:
        record["reranker"] = {
            "used": rerank_result.used,
            "model": rerank_result.model,
            "fallback_reason": rerank_result.fallback_reason,
            "metadata": dict(rerank_result.metadata),
        }
    if packed is not None:
        record["packed_evidence"] = packed.model_dump(mode="json")
    if error is not None:
        record["error_code"] = error.error_code
        record["error_stage"] = error.stage
    return record


def _empty_deduplication() -> DeduplicationResult:
    return DeduplicationResult(kept_hits=(), dropped=())


def _run_pipeline(
    questions: Sequence[LegalQuestion | InferenceQuestion],
    config: ProjectConfig,
    *,
    method: PipelineMethod,
    chunks: Mapping[str, LegalChunk] | None = None,
    documents: Mapping[str, LegalDocument] | None = None,
    index: BM25Index | None = None,
    prompt_builder: PromptBuilder | None = None,
    client: LLMClient | None = None,
    reranker: Reranker | None = None,
    output_dir: str | Path,
    run_id: str | None,
    data_manifest_hash: str = "UNRESOLVED",
    chunk_cache_fingerprint: str = "UNRESOLVED",
    fail_fast: bool | None = None,
    registry_path: str | Path | None = None,
) -> RunResult:
    if method in _RAG_METHODS and (index is None or chunks is None):
        raise PipelineError(f"{method} requires a loaded BM25 index and chunks")
    if method == "direct" and index is not None:
        raise PipelineError("Direct baseline must not receive a retrieval index")

    views = _question_view(questions)
    selected_prompt_builder = prompt_builder or PromptBuilder.from_config(
        config.prompts,
        prompt_dir=Path(__file__).resolve().parents[2] / "configs" / "prompts",
    )
    selected_client = client or create_llm_client(config.generation)
    selected_reranker: Reranker | None = None
    if method == "hybrid_rag":
        try:
            selected_reranker = reranker or create_reranker(config.reranker)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise PipelineError(f"Hybrid-RAG reranker unavailable: {exc}") from exc
    selected_fail_fast = config.runtime.fail_fast if fail_fast is None else fail_fast
    output_root = Path(output_dir)
    selected_registry_path = (
        _default_registry_path(output_root)
        if registry_path is None
        else Path(registry_path)
    )
    run_manager = RunManager.create(
        output_root,
        split=config.data.split,
        method=method,
        repo_root=Path(__file__).resolve().parents[2],
        run_id=run_id,
        registry_path=selected_registry_path,
    )
    paths = run_manager.paths
    errors: list[_CaseError] = []
    predictions: list[dict[str, object]] = []
    generation_records: list[dict[str, object]] = []
    retrieval_records: list[dict[str, object]] = []
    rerank_results: list[RerankResult] = []

    for question in views:
        retrieval_record: dict[str, object] | None = None
        rerank_result: RerankResult | None = None
        raw_hits: tuple[RetrievalHit, ...] = ()
        try:
            if method in _RAG_METHODS:
                assert index is not None
                assert chunks is not None
                raw_hits = retrieve_bm25(
                    index,
                    question.question,
                    top_k=config.retrieval.rough_top_n,
                )
                if not raw_hits:
                    raise PipelineError("No positive BM25 hits for question")
                ordered_hits: Sequence[RetrievalHit] = raw_hits
                if method == "hybrid_rag":
                    assert selected_reranker is not None
                    candidate_texts: dict[str, str] = {}
                    for hit in raw_hits:
                        chunk = chunks.get(hit.chunk_id)
                        if chunk is None:
                            raise PipelineError(
                                f"Retrieval hit {hit.chunk_id!r} has no chunk text"
                            )
                        candidate_texts[hit.chunk_id] = chunk.retrieval_text
                    rerank_result = selected_reranker.rerank(
                        question.question,
                        raw_hits,
                        candidate_texts=candidate_texts,
                    )
                    if not isinstance(rerank_result, RerankResult):
                        raise TypeError("Reranker must return a RerankResult")
                    rerank_results.append(rerank_result)
                    if not rerank_result.used and config.reranker.required:
                        reason = rerank_result.fallback_reason or "unspecified"
                        raise PipelineError(
                            f"Required Hybrid-RAG reranker was not used: {reason}"
                        )
                    ordered_hits = rerank_result.hits
                deduplication = deduplicate_retrieved_chunks(ordered_hits, chunks)
                selected_hits = deduplication.kept_hits[
                    : config.evidence.evidence_top_k
                ]
                limited_ids = deduplication.kept_hits[config.evidence.evidence_top_k :]
                packed = pack_evidence(
                    selected_hits,
                    chunks,
                    max_total_chars=config.evidence.max_total_chars,
                    max_chunks_per_document=config.evidence.max_chunks_per_document,
                    documents=documents,
                )
                prompt = selected_prompt_builder.build_rag(question.question, packed)
                retrieval_record = _retrieval_record(
                    question.id,
                    question.question,
                    index,
                    raw_hits,
                    deduplication,
                    packed,
                    method=method,
                    top_k=config.retrieval.rough_top_n,
                    evidence_top_k=config.evidence.evidence_top_k,
                    limited_ids=[hit.chunk_id for hit in limited_ids],
                    reranked_hits=(
                        rerank_result.hits if rerank_result is not None else None
                    ),
                    rerank_result=rerank_result,
                )
            else:
                prompt = selected_prompt_builder.build_direct(question.question)
            if retrieval_record is not None:
                retrieval_records.append(retrieval_record)
                retrieval_record = None
        except (
            EvidencePackingError,
            PipelineError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            error = _CaseError(
                case_id=question.id,
                error_code=(
                    "RETRIEVAL_ERROR" if method in _RAG_METHODS else "PROMPT_ERROR"
                ),
                message=str(exc),
                stage="retrieval" if method in _RAG_METHODS else "prompt",
                error_type="retrieval" if method in _RAG_METHODS else "prompt",
            )
            errors.append(error)
            if method in _RAG_METHODS:
                assert index is not None
                retrieval_records.append(
                    _retrieval_record(
                        question.id,
                        question.question,
                        index,
                        raw_hits,
                        _empty_deduplication(),
                        None,
                        method=method,
                        top_k=config.retrieval.rough_top_n,
                        evidence_top_k=config.evidence.evidence_top_k,
                        reranked_hits=(
                            rerank_result.hits if rerank_result is not None else None
                        ),
                        rerank_result=rerank_result,
                        error=error,
                    )
                )
            generation_records.append(
                {
                    "id": question.id,
                    "method": method,
                    "status": "skipped",
                    "error_code": error.error_code,
                    "error_stage": error.stage,
                    "error_type": error.error_type,
                    "retries": error.retries,
                    "retryable": error.retryable,
                }
            )
            if selected_fail_fast:
                break
            continue

        try:
            response = selected_client.generate(prompt.text, case_id=question.id)
            processed_answer = postprocess_answer(response.text)
            prediction = Prediction(
                id=question.id,
                answer=processed_answer.cleaned_answer,
                method=method,
            )
            prediction_record = prediction.model_dump(mode="json")
            prediction_record.update(
                {
                    "raw_answer": processed_answer.raw_answer,
                    "cleaned_answer": processed_answer.cleaned_answer,
                }
            )
            predictions.append(prediction_record)
            generation_records.append(
                _generation_record(
                    question.id,
                    config,
                    prompt,
                    selected_client,
                    method=method,
                    response=response,
                )
            )
        except LLMClientError as exc:
            case_error = exc.case_error
            error = _CaseError(
                case_id=question.id,
                error_code=case_error.error_code,
                message=case_error.message,
                stage="generation",
                error_type="provider",
                retries=case_error.retries,
                retryable=case_error.retryable,
            )
            errors.append(error)
            generation_records.append(
                _generation_record(
                    question.id,
                    config,
                    prompt,
                    selected_client,
                    method=method,
                    error=error,
                )
            )
            if selected_fail_fast:
                break
        except (TypeError, ValueError, RuntimeError) as exc:
            error = _CaseError(
                case_id=question.id,
                error_code="GENERATION_ERROR",
                message=str(exc),
                stage="generation",
                error_type="generation",
            )
            errors.append(error)
            generation_records.append(
                _generation_record(
                    question.id,
                    config,
                    prompt,
                    selected_client,
                    method=method,
                    error=error,
                )
            )
            if selected_fail_fast:
                break

    prompt_template = (
        selected_prompt_builder.rag_template
        if method in _RAG_METHODS
        else selected_prompt_builder.direct_template
    )
    index_fingerprint = index.index_fingerprint if index is not None else None
    model_fingerprint = _json_hash(
        {"provider": selected_client.provider, "model": selected_client.model}
    )
    fallback_reasons = tuple(
        sorted(
            {
                result.fallback_reason
                for result in rerank_results
                if result.fallback_reason is not None
            }
        )
    )
    reranker_used_values = {result.used for result in rerank_results}
    reranker_status = (
        "not_run"
        if not rerank_results
        else "true"
        if reranker_used_values == {True}
        else "false"
        if reranker_used_values == {False}
        else "mixed"
    )
    config_hash = config.config_hash()
    environment = run_manager.environment(seed=config.runtime.seed)
    environment.update(
        {
            "provider": selected_client.provider,
            "model": selected_client.model,
            "temperature": str(config.generation.temperature),
            "max_output_chars": str(config.generation.max_output_chars),
        }
    )
    if method == "hybrid_rag":
        assert selected_reranker is not None
        environment.update(
            {
                "reranker_provider": config.reranker.provider,
                "reranker_model": selected_reranker.model,
                "reranker_required": str(config.reranker.required).lower(),
                "reranker_used": reranker_status,
                "reranker_fallback_reason": (
                    ";".join(fallback_reasons) if fallback_reasons else "none"
                ),
            }
        )
    run_manager.write_json(
        paths.config,
        {
            **config.redacted_dict(),
            "config_hash": config_hash,
            "schema_version": "g1.config.v1",
        },
    )
    run_manager.write_json(paths.environment, environment)
    run_manager.write_jsonl(paths.predictions, predictions)
    run_manager.write_jsonl(paths.generation, generation_records)
    run_manager.write_jsonl(paths.errors, [error.as_dict() for error in errors])
    if paths.retrieval is not None:
        run_manager.write_jsonl(paths.retrieval, retrieval_records)
    run_manager.write_json(
        paths.metrics,
        {
            "schema_version": "g1.metrics.v1",
            "status": "not_evaluated",
            "run_id": run_manager.run_id,
            "method": method,
            "split": config.data.split,
            "data_manifest_hash": data_manifest_hash,
            "prediction_artifact": paths.predictions.name,
            "reference_role": "none",
            "evaluator_kind": "UNRESOLVED",
            "counts": {"predictions": len(predictions), "errors": len(errors)},
            "metrics": {
                "meteor": None,
                "rouge_l": None,
                "primary_metric": "meteor",
                "secondary_metric": "rouge_l",
                "official_equivalence": "UNRESOLVED",
            },
        },
    )
    run_manager.write_json(
        paths.submission,
        {
            "schema_version": "g1.submission.v1",
            "status": "not_created",
            "reason": "official_zip_requires_explicit_packaging",
            "run_id": run_manager.run_id,
            "records": [],
        },
    )
    run_manager.write_json(
        paths.summary,
        {
            "schema_version": "g1.run-summary.v1",
            "run_id": run_manager.run_id,
            "method": method,
            "split": config.data.split,
            "seed": config.runtime.seed,
            "config_hash": config_hash,
            "data_manifest_hash": data_manifest_hash,
            "model_fingerprint": model_fingerprint,
            "prompt_fingerprint": prompt_template.sha256,
            "prediction_count": len(predictions),
            "error_count": len(errors),
            "case_counts": {
                "total": len(views),
                "processed": len(generation_records),
                "succeeded": len(predictions),
                "failed": len(errors),
                "skipped": sum(
                    record.get("status") == "skipped" for record in generation_records
                ),
                "missing_predictions": len(views) - len(predictions),
            },
            "failed_ids": sorted(error.case_id for error in errors),
            "index_fingerprint": index_fingerprint,
            "chunk_fingerprint": chunk_cache_fingerprint,
            "chunk_cache_fingerprint": chunk_cache_fingerprint,
            "fingerprints": {
                "config_hash": config_hash,
                "data_manifest_hash": data_manifest_hash,
                "model_hash": model_fingerprint,
                "prompt_hash": prompt_template.sha256,
                "index_fingerprint": index_fingerprint,
                "chunk_fingerprint": chunk_cache_fingerprint,
            },
            "git": {
                "commit": environment["git_commit"],
                "dirty": environment["git_dirty"],
            },
            "comparison_controls": comparison_controls(config),
            "reranker": (
                {
                    "provider": config.reranker.provider,
                    "required": config.reranker.required,
                    "model": selected_reranker.model
                    if selected_reranker is not None
                    else None,
                    "used": reranker_status,
                    "fallback_reasons": list(fallback_reasons),
                }
                if method == "hybrid_rag"
                else None
            ),
            "artifacts": {
                "config": paths.config.name,
                "environment": paths.environment.name,
                "predictions": paths.predictions.name,
                "generation": paths.generation.name,
                "errors": paths.errors.name,
                "retrieval": paths.retrieval.name if paths.retrieval else None,
                "metrics": paths.metrics.name,
                "submission": paths.submission.name,
                "experiment_registry": (
                    paths.registry.name if paths.registry is not None else None
                ),
            },
        },
    )
    output_hash = hash_run_outputs(paths.run_dir)
    registry = ExperimentRegistry(paths.registry or selected_registry_path)
    registry.append(
        ExperimentRecord(
            run_id=run_manager.run_id,
            git_commit=str(environment["git_commit"]),
            dirty=bool(environment["git_dirty"]),
            command=current_command(),
            config_hash=config_hash,
            split=config.data.split,
            data_manifest_hash=data_manifest_hash,
            chunk_fingerprint=(
                chunk_cache_fingerprint if method in _RAG_METHODS else None
            ),
            index_fingerprint=index_fingerprint,
            prompt_hash=prompt_template.sha256,
            model=selected_client.model,
            seed=config.runtime.seed,
            meteor=None,
            rouge_l=None,
            error_rate=(len(errors) / len(views) if views else 0.0),
            latency_ms=_mean_latency_ms(generation_records),
            reranker_fallback_rate=(
                sum(not result.used for result in rerank_results) / len(rerank_results)
                if rerank_results
                else 0.0
            ),
            output_hash=output_hash,
            notes="metrics_not_evaluated",
        )
    )
    result = RunResult(
        method=method,
        run_id=run_manager.run_id,
        artifacts=paths,
        prediction_count=len(predictions),
        error_count=len(errors),
        index_fingerprint=index_fingerprint,
    )
    if errors and selected_fail_fast:
        raise PipelineRunError(
            f"{method} run failed for {len(errors)} case(s); see {paths.errors}",
            result,
        )
    return result


def run_bm25_rag(
    questions: Sequence[LegalQuestion | InferenceQuestion],
    chunks: Mapping[str, LegalChunk],
    index: BM25Index,
    config: ProjectConfig,
    *,
    documents: Mapping[str, LegalDocument] | None = None,
    prompt_builder: PromptBuilder | None = None,
    client: LLMClient | None = None,
    output_dir: str | Path = "outputs",
    run_id: str | None = None,
    data_manifest_hash: str = "fixture",
    chunk_cache_fingerprint: str = "fixture",
    fail_fast: bool | None = None,
    registry_path: str | Path | None = None,
) -> RunResult:
    """Run B1 BM25-RAG over a validated in-memory index and chunk map."""

    if config.retrieval.strategy != "bm25":
        raise PipelineError("BM25-RAG requires retrieval.strategy='bm25'")
    return _run_pipeline(
        questions,
        config,
        method="bm25_rag",
        chunks=chunks,
        documents=documents,
        index=index,
        prompt_builder=prompt_builder,
        client=client,
        output_dir=output_dir,
        run_id=run_id,
        data_manifest_hash=data_manifest_hash,
        chunk_cache_fingerprint=chunk_cache_fingerprint,
        fail_fast=fail_fast,
        registry_path=registry_path,
    )


def run_hybrid_rag(
    questions: Sequence[LegalQuestion | InferenceQuestion],
    chunks: Mapping[str, LegalChunk],
    index: BM25Index,
    config: ProjectConfig,
    *,
    documents: Mapping[str, LegalDocument] | None = None,
    prompt_builder: PromptBuilder | None = None,
    client: LLMClient | None = None,
    reranker: Reranker | None = None,
    output_dir: str | Path = "outputs",
    run_id: str | None = None,
    data_manifest_hash: str = "fixture",
    chunk_cache_fingerprint: str = "fixture",
    fail_fast: bool | None = None,
    registry_path: str | Path | None = None,
) -> RunResult:
    """Run B2 Hybrid-RAG over BM25 candidates and a configured reranker."""

    if config.retrieval.strategy != "bm25_rerank":
        raise PipelineError("Hybrid-RAG requires retrieval.strategy='bm25_rerank'")
    return _run_pipeline(
        questions,
        config,
        method="hybrid_rag",
        chunks=chunks,
        documents=documents,
        index=index,
        prompt_builder=prompt_builder,
        client=client,
        reranker=reranker,
        output_dir=output_dir,
        run_id=run_id,
        data_manifest_hash=data_manifest_hash,
        chunk_cache_fingerprint=chunk_cache_fingerprint,
        fail_fast=fail_fast,
        registry_path=registry_path,
    )


def run_direct(
    questions: Sequence[LegalQuestion | InferenceQuestion],
    config: ProjectConfig,
    *,
    prompt_builder: PromptBuilder | None = None,
    client: LLMClient | None = None,
    output_dir: str | Path = "outputs",
    run_id: str | None = None,
    data_manifest_hash: str = "fixture",
    fail_fast: bool | None = None,
    registry_path: str | Path | None = None,
) -> RunResult:
    """Run B0 Direct using the same generator controls as B1."""

    if config.retrieval.strategy != "none":
        raise PipelineError("Direct baseline requires retrieval.strategy='none'")
    return _run_pipeline(
        questions,
        config,
        method="direct",
        prompt_builder=prompt_builder,
        client=client,
        output_dir=output_dir,
        run_id=run_id,
        data_manifest_hash=data_manifest_hash,
        fail_fast=fail_fast,
        registry_path=registry_path,
    )


def _chunking_from_config(config: ProjectConfig) -> ChunkingConfig:
    return ChunkingConfig(
        max_chars=config.chunking.max_chars,
        overlap_chars=config.chunking.overlap_chars,
        min_chars=config.chunking.min_chars,
        version=config.chunking.version,
    )


def prepare_bm25_index_from_config(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    rebuild_index: bool = False,
) -> BM25Preparation:
    """Validate sources and load/build one fingerprinted BM25 index."""

    config = load_config(config_path)
    if config.retrieval.strategy not in {"bm25", "bm25_rerank"}:
        raise PipelineError(
            "BM25-backed config must set retrieval.strategy to 'bm25' or 'bm25_rerank'"
        )
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    manifest_hash = _manifest_hash(root)
    documents_tuple = load_selected_contexts(root / config.data.selected_contexts_path)
    documents = {document.id: document for document in documents_tuple}
    chunk_cache = build_chunk_cache(
        documents_tuple,
        root / config.runtime.cache_dir,
        manifest_hash,
        _chunking_from_config(config),
        data_root=root / config.data.data_dir,
    )
    index_result = load_or_build_bm25_index(
        chunk_cache.chunks,
        root / config.runtime.cache_dir,
        chunk_cache.fingerprint,
        BM25Config(k1=config.retrieval.k1, b=config.retrieval.b),
        data_root=root / config.data.data_dir,
        policy="auto_rebuild" if rebuild_index else "strict",
    )
    if index_result.index is None:
        raise PipelineError(f"BM25 index unavailable: {index_result.reason}")
    return BM25Preparation(
        config=config,
        chunks=chunk_cache.chunks,
        documents=documents,
        index=index_result.index,
        manifest_hash=manifest_hash,
        chunk_cache_fingerprint=chunk_cache.fingerprint.cache_fingerprint,
    )


def run_bm25_rag_from_config(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    rebuild_index: bool = False,
    limit: int | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> RunResult:
    """Load validated sources/indexes and run B1 with explicit rebuild policy."""

    preparation = prepare_bm25_index_from_config(
        config_path,
        repo_root=repo_root,
        rebuild_index=rebuild_index,
    )
    config = preparation.config
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    records = load_questions(
        root / config.data.question_path,
        split=config.data.split,
    )
    views = inference_view(records)
    selected = views if limit is None else views[:limit]
    selected_output = (
        root / config.runtime.outputs_dir if output_dir is None else Path(output_dir)
    )
    return run_bm25_rag(
        selected,
        {chunk.chunk_id: chunk for chunk in preparation.chunks},
        preparation.index,
        config,
        documents=preparation.documents,
        output_dir=selected_output,
        run_id=run_id,
        data_manifest_hash=preparation.manifest_hash,
        chunk_cache_fingerprint=preparation.chunk_cache_fingerprint,
        registry_path=root / config.runtime.artifacts_dir / "experiments.jsonl",
    )


def run_hybrid_rag_from_config(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    rebuild_index: bool = False,
    limit: int | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> RunResult:
    """Load validated sources/indexes and run B2 with explicit rebuild policy."""

    preparation = prepare_bm25_index_from_config(
        config_path,
        repo_root=repo_root,
        rebuild_index=rebuild_index,
    )
    config = preparation.config
    if config.retrieval.strategy != "bm25_rerank":
        raise PipelineError(
            "Hybrid-RAG config must set retrieval.strategy='bm25_rerank'"
        )
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    records = load_questions(
        root / config.data.question_path,
        split=config.data.split,
    )
    views = inference_view(records)
    selected = views if limit is None else views[:limit]
    selected_output = (
        root / config.runtime.outputs_dir if output_dir is None else Path(output_dir)
    )
    return run_hybrid_rag(
        selected,
        {chunk.chunk_id: chunk for chunk in preparation.chunks},
        preparation.index,
        config,
        documents=preparation.documents,
        output_dir=selected_output,
        run_id=run_id,
        data_manifest_hash=preparation.manifest_hash,
        chunk_cache_fingerprint=preparation.chunk_cache_fingerprint,
        registry_path=root / config.runtime.artifacts_dir / "experiments.jsonl",
    )


def run_direct_from_config(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    limit: int | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> RunResult:
    """Load inference-safe questions and run B0 Direct."""

    config = load_config(config_path)
    if config.retrieval.strategy != "none":
        raise PipelineError("Direct config must set retrieval.strategy='none'")
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    manifest_hash = _manifest_hash(root)
    records = load_questions(
        root / config.data.question_path,
        split=config.data.split,
    )
    views = inference_view(records)
    selected = views if limit is None else views[:limit]
    selected_output = (
        root / config.runtime.outputs_dir if output_dir is None else Path(output_dir)
    )
    return run_direct(
        selected,
        config,
        output_dir=selected_output,
        run_id=run_id,
        data_manifest_hash=manifest_hash,
        registry_path=root / config.runtime.artifacts_dir / "experiments.jsonl",
    )


__all__ = [
    "PipelineError",
    "PipelineRunError",
    "RunArtifactPaths",
    "RunResult",
    "BM25Preparation",
    "RetrievalInspectionRow",
    "clean_generated_answer",
    "comparison_controls",
    "run_bm25_rag",
    "run_hybrid_rag",
    "prepare_bm25_index_from_config",
    "run_bm25_rag_from_config",
    "run_hybrid_rag_from_config",
    "inspect_bm25_retrieval",
    "run_direct",
    "run_direct_from_config",
]
