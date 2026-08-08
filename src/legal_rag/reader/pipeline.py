"""Two extractive-reader profiles sharing exactly one reader backend."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..artifacts import RunArtifactPaths, RunManager
from ..config import ProjectConfig, load_config
from .backends import create_local_reader
from .bm25 import ReaderBM25Index
from .data import ReaderSplit, load_reader_dataset
from .types import (
    ExtractiveReader,
    ReaderCandidate,
    ReaderInferenceCase,
    ReaderPrediction,
    ReaderSpan,
)

ReaderMethod = Literal["finetuned_reader", "tuned_bm25_reader"]


class ReaderPipelineError(RuntimeError):
    """Raised when a reader profile cannot run without violating its contract."""


@dataclass(frozen=True, slots=True)
class ReaderRunResult:
    run_id: str
    method: ReaderMethod
    artifacts: RunArtifactPaths
    predictions: tuple[ReaderPrediction, ...]
    error_count: int = 0

    @property
    def prediction_count(self) -> int:
        return len(self.predictions)


def _method(config: ProjectConfig) -> ReaderMethod:
    mapping: dict[str, ReaderMethod] = {
        "finetuned-reader": "finetuned_reader",
        "tuned-bm25-reader": "tuned_bm25_reader",
    }
    try:
        return mapping[config.project.profile]
    except KeyError as exc:
        raise ReaderPipelineError(
            f"Not a reader profile: {config.project.profile!r}"
        ) from exc


def _candidate_record(
    case_id: str,
    candidate: ReaderCandidate,
    span: ReaderSpan,
) -> dict[str, object]:
    return {
        "id": case_id,
        "source_case_id": candidate.source_case_id,
        "source_origin": candidate.origin,
        "candidate_order": candidate.candidate_order,
        "bm25_rank": candidate.bm25_rank,
        "bm25_score": candidate.bm25_score,
        "answer": span.answer,
        "confidence": span.confidence,
        "start_char": span.start_char,
        "end_char": span.end_char,
        "impossible": span.impossible,
    }


def run_reader(
    cases: Sequence[ReaderInferenceCase],
    config: ProjectConfig,
    *,
    reader: ExtractiveReader,
    output_dir: str | Path,
    run_id: str | None = None,
    train_cases: Sequence[ReaderInferenceCase] = (),
    split: ReaderSplit = "test",
    data_manifest_hash: str = "UNRESOLVED",
    split_manifest_hash: str = "UNRESOLVED",
) -> ReaderRunResult:
    """Run either profile; only candidate-context retrieval is allowed to differ."""

    settings = config.reader
    if settings is None or not settings.enabled:
        raise ReaderPipelineError("Reader profile requires reader.enabled=true")
    method = _method(config)
    if not cases:
        raise ReaderPipelineError("Reader run requires at least one inference case")
    if any(type(case) is not ReaderInferenceCase for case in cases):
        raise ReaderPipelineError(
            "Reader inference accepts ReaderInferenceCase records only"
        )
    if any(type(case) is not ReaderInferenceCase for case in train_cases):
        raise ReaderPipelineError(
            "Reader retrieval corpus accepts ReaderInferenceCase records only"
        )
    index: ReaderBM25Index | None = None
    if method == "tuned_bm25_reader":
        index = ReaderBM25Index(
            train_cases,
            k1=settings.k1,
            b=settings.b,
            backend=settings.retrieval_backend,
        )
    elif train_cases:
        raise ReaderPipelineError(
            "finetuned_reader must not receive retrieval train cases"
        )

    manager = RunManager.create(
        output_dir,
        split=split,
        method=method,
        repo_root=Path(__file__).resolve().parents[3],
        run_id=run_id,
    )
    paths = manager.paths
    if paths.reader is None:  # pragma: no cover - RunManager contract guard
        raise ReaderPipelineError("Reader artifact path was not created")
    predictions: list[ReaderPrediction] = []
    candidate_records: list[dict[str, object]] = []
    retrieval_records: list[dict[str, object]] = []

    for case in cases:
        candidates = [
            ReaderCandidate(
                source_case_id=case.id,
                origin="original",
                context=case.context,
                candidate_order=0,
            )
        ]
        if index is not None:
            hits = index.search(case.question, top_k=settings.retrieval_top_k)
            seen_contexts = {case.context}
            for hit in hits:
                if hit.context in seen_contexts:
                    continue
                seen_contexts.add(hit.context)
                candidates.append(
                    ReaderCandidate(
                        source_case_id=hit.case_id,
                        origin="train_retrieval",
                        context=hit.context,
                        candidate_order=len(candidates),
                        bm25_rank=hit.rank,
                        bm25_score=hit.score,
                    )
                )
            retrieval_records.append(
                {
                    "id": case.id,
                    "query_role": "question_only",
                    "query_sha256": hashlib.sha256(
                        case.question.encode("utf-8")
                    ).hexdigest(),
                    "index_corpus": "train_contexts_only",
                    "index_fingerprint": index.fingerprint,
                    "hits": [
                        {
                            "source_case_id": candidate.source_case_id,
                            "rank": candidate.bm25_rank,
                            "bm25_score": candidate.bm25_score,
                        }
                        for candidate in candidates
                        if candidate.origin == "train_retrieval"
                    ],
                }
            )

        scored: list[tuple[ReaderCandidate, ReaderSpan]] = []
        for candidate in candidates:
            span = reader.predict(
                question=case.question,
                context=candidate.context,
                case_id=case.id,
            )
            scored.append((candidate, span))
            candidate_records.append(_candidate_record(case.id, candidate, span))
        selected_candidate, selected_span = max(
            scored,
            key=lambda item: (
                item[1].confidence,
                -item[0].candidate_order,
            ),
        )
        predictions.append(
            ReaderPrediction(
                id=case.id,
                answer=selected_span.answer,
                method=method,
                confidence=selected_span.confidence,
                source_case_id=selected_candidate.source_case_id,
                source_origin=selected_candidate.origin,
                model=reader.model,
                model_version=reader.model_version,
            )
        )

    config_hash = config.config_hash()
    model_hash = hashlib.sha256(
        json.dumps(
            {"model": reader.model, "version": reader.model_version},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    environment = manager.environment(seed=config.runtime.seed)
    environment.update(
        {
            "reader_model": reader.model,
            "reader_model_version": reader.model_version,
            "reader_device": settings.device,
            "retrieval_backend": settings.retrieval_backend,
            "local_files_only": "true",
        }
    )
    manager.write_json(
        paths.config,
        {
            **config.redacted_dict(),
            "config_hash": config_hash,
            "schema_version": "reader.config.v1",
        },
    )
    manager.write_json(paths.environment, environment)
    manager.write_jsonl(
        paths.predictions,
        [prediction.model_dump(mode="json") for prediction in predictions],
    )
    manager.write_jsonl(paths.reader, candidate_records)
    manager.write_jsonl(paths.errors, [])
    if paths.retrieval is not None:
        manager.write_jsonl(paths.retrieval, retrieval_records)
    manager.write_json(
        paths.metrics,
        {
            "schema_version": "reader.metrics.v1",
            "status": "not_evaluated",
            "reference_role": "none",
            "metrics": {"exact_match": None, "token_f1": None},
        },
    )
    manager.write_json(
        paths.submission,
        {
            "schema_version": "reader.submission.v1",
            "status": "not_applicable",
            "reason": "auxiliary_extractive_reader_profile",
        },
    )
    manager.write_json(
        paths.summary,
        {
            "schema_version": "reader.run-summary.v1",
            "run_id": manager.run_id,
            "method": method,
            "split": split,
            "prediction_count": len(predictions),
            "error_count": 0,
            "config_hash": config_hash,
            "data_manifest_hash": data_manifest_hash,
            "split_manifest_hash": split_manifest_hash,
            "index_fingerprint": index.fingerprint if index is not None else None,
            "model_fingerprint": model_hash,
            "model": reader.model,
            "model_version": reader.model_version,
            "same_reader_checkpoint_control": True,
            "retrieval_mode": settings.mode,
            "retrieval_backend": settings.retrieval_backend,
            "reference_role": "none",
        },
    )
    return ReaderRunResult(
        run_id=manager.run_id,
        method=method,
        artifacts=paths,
        predictions=tuple(predictions),
    )


def run_reader_from_config(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    limit: int | None = None,
) -> ReaderRunResult:
    """Load real local assets and run a configured reader profile fail-closed."""

    config = load_config(config_path)
    settings = config.reader
    if settings is None:
        raise ReaderPipelineError("Selected config is not a reader profile")
    repo_root = Path(__file__).resolve().parents[3]
    dataset = load_reader_dataset(
        repo_root / settings.dataset_path,
        repo_root / settings.split_manifest_path,
    )
    reader = create_local_reader(settings, repo_root=repo_root)
    test_cases = tuple(
        case.inference_view() for case in dataset.cases_for_split("test")
    )
    if limit is not None:
        if limit < 1:
            raise ReaderPipelineError("Reader limit must be positive")
        test_cases = test_cases[:limit]
    train_cases = (
        tuple(case.inference_view() for case in dataset.cases_for_split("train"))
        if _method(config) == "tuned_bm25_reader"
        else ()
    )
    selected_output = (
        repo_root / config.runtime.outputs_dir
        if output_dir is None
        else Path(output_dir)
    )
    return run_reader(
        test_cases,
        config,
        reader=reader,
        output_dir=selected_output,
        run_id=run_id,
        train_cases=train_cases,
        split="test",
        data_manifest_hash=dataset.dataset_sha256,
        split_manifest_hash=dataset.split_manifest_sha256,
    )


__all__ = [
    "ReaderMethod",
    "ReaderPipelineError",
    "ReaderRunResult",
    "run_reader",
    "run_reader_from_config",
]
