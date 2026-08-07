"""Generative finetuned_reader inference using the frozen B2 evidence path."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..artifacts import RunArtifactPaths, RunManager
from ..config import ProjectConfig, load_config
from ..questions import load_inference_questions
from ..schemas import InferenceQuestion
from .b2_freeze import load_b2_freeze_fingerprint, require_complete_b2_freeze
from .contracts import GENERATIVE_METHOD
from .dataset import FrozenB2EvidenceRetriever
from .inference import FineTunedReaderGenerator, load_finetuned_reader_generator
from .prompting import GenerativePromptBuilder


class FineTunedReaderPipelineError(RuntimeError):
    """Raised when the generative profile cannot run without a contract violation."""


@dataclass(frozen=True, slots=True)
class FineTunedReaderRunResult:
    run_id: str
    method: str
    artifacts: RunArtifactPaths
    prediction_count: int
    error_count: int


def _write_jsonl(
    manager: RunManager, path: Path, records: Sequence[dict[str, object]]
) -> None:
    manager.write_jsonl(path, records)


def run_finetuned_reader(
    cases: Sequence[InferenceQuestion],
    *,
    config: ProjectConfig,
    retriever: FrozenB2EvidenceRetriever,
    generator: FineTunedReaderGenerator,
    output_dir: str | Path,
    run_id: str | None = None,
    fail_fast: bool | None = None,
) -> FineTunedReaderRunResult:
    """Run one generator call per case with shared B2 evidence and artifacts."""

    settings = config.finetuned_reader
    if settings is None or not settings.enabled or not settings.required:
        raise FineTunedReaderPipelineError(
            "finetuned_reader requires enabled=true and required=true"
        )
    if any(type(case) is not InferenceQuestion for case in cases):
        raise FineTunedReaderPipelineError("Inference accepts InferenceQuestion only")
    if not cases:
        raise FineTunedReaderPipelineError(
            "finetuned_reader requires at least one case"
        )
    manager = RunManager.create(
        output_dir,
        split=config.data.split,
        method=GENERATIVE_METHOD,
        repo_root=Path(__file__).resolve().parents[3],
        run_id=run_id,
        with_retrieval=True,
        registry_path=Path(output_dir).resolve().parent
        / config.runtime.artifacts_dir
        / "experiments.jsonl",
    )
    paths = manager.paths
    if paths.retrieval is None or paths.reader is None:
        raise FineTunedReaderPipelineError(
            "Generative run artifact paths are incomplete"
        )
    predictions: list[dict[str, object]] = []
    retrieval_records: list[dict[str, object]] = []
    generation_records: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    selected_fail_fast = config.runtime.fail_fast if fail_fast is None else fail_fast

    for case in sorted(cases, key=lambda item: item.id):
        try:
            retrieved = retriever.retrieve(case)
            generated = generator.generate(case.question, retrieved.evidence)
            predictions.append(
                {
                    "id": case.id,
                    "answer": generated.cleaned_answer,
                    "method": GENERATIVE_METHOD,
                    "model": generated.model,
                    "model_version": generated.model_version,
                }
            )
            retrieval_records.append(
                {
                    "id": case.id,
                    "query_role": "question_only",
                    "query_sha256": retrieved.query_sha256,
                    "raw_hit_ids": list(retrieved.retrieval_hits),
                    "packed_chunk_ids": list(retrieved.evidence.included_ids),
                    "packed_dropped_ids": list(retrieved.evidence.dropped_ids),
                    "packed_truncated_ids": list(retrieved.evidence.truncated_ids),
                    "packed_evidence_hash": _packed_hash(retrieved.evidence),
                    "index_fingerprint": retriever.preparation.index.index_fingerprint,
                    "reranker": retrieved.reranker,
                }
            )
            generation_records.append(
                {
                    "id": case.id,
                    "method": GENERATIVE_METHOD,
                    "status": "ok",
                    "raw_answer": generated.raw_answer,
                    "cleaned_answer": generated.cleaned_answer,
                    "model": generated.model,
                    "model_version": generated.model_version,
                    "latency_ms": generated.latency_ms,
                    "prompt_hash": generated.prompt_hash,
                    "metadata": generated.metadata,
                }
            )
        except Exception as exc:
            error: dict[str, object] = {
                "id": case.id,
                "method": GENERATIVE_METHOD,
                "error_code": "FTR_GENERATION_ERROR",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "stage": "retrieval_or_generation",
            }
            errors.append(error)
            generation_records.append({**error, "status": "error"})
            if selected_fail_fast:
                break

    config_hash = config.config_hash()
    checkpoint = generator.checkpoint
    manager.write_json(
        paths.config,
        {
            **config.redacted_dict(),
            "config_hash": config_hash,
            "schema_version": "ftr10.config.v1",
        },
    )
    manager.write_json(
        paths.environment,
        {
            **manager.environment(seed=config.runtime.seed),
            "profile": "finetuned_reader",
            "type": "generative_sft_reader",
            "checkpoint_manifest_hash": checkpoint.manifest_hash,
            "base_model": checkpoint.manifest["base_model"],
            "base_revision": checkpoint.manifest["base_revision"],
        },
    )
    _write_jsonl(manager, paths.predictions, predictions)
    _write_jsonl(manager, paths.retrieval, retrieval_records)
    _write_jsonl(manager, paths.generation, generation_records)
    _write_jsonl(manager, paths.errors, errors)
    _write_jsonl(manager, paths.reader, [])
    manager.write_json(
        paths.metrics,
        {
            "schema_version": "ftr11.metrics.v1",
            "status": "not_evaluated",
            "reference_role": "none",
            "metrics": {"meteor": None, "rouge_l": None},
        },
    )
    manager.write_json(
        paths.submission,
        {
            "schema_version": "ftr11.submission-placeholder.v1",
            "status": "not_packaged",
            "reason": "Use the common official submission writer after evaluation",
        },
    )
    manager.write_json(
        paths.run_dir / "checkpoint_reference.json",
        {
            "profile": "finetuned_reader",
            "base_model": checkpoint.manifest["base_model"],
            "base_revision": checkpoint.manifest["base_revision"],
            "adapter_path": checkpoint.manifest.get("adapter_path", "adapter"),
            "adapter_hash": checkpoint.adapter_hash,
            "checkpoint_manifest_hash": checkpoint.manifest_hash,
            "training_dataset_hash": checkpoint.manifest["dataset_manifest_hash"],
            "retrieval_config_hash": checkpoint.manifest["retrieval_config_hash"],
            "index_fingerprint": checkpoint.manifest["index_fingerprint"],
        },
    )
    manager.write_json(
        paths.summary,
        {
            "schema_version": "ftr10.run-summary.v1",
            "run_id": manager.run_id,
            "method": GENERATIVE_METHOD,
            "split": config.data.split,
            "split_policy": config.data.split_policy,
            "prediction_count": len(predictions),
            "error_count": len(errors),
            "config_hash": config_hash,
            "index_fingerprint": retriever.preparation.index.index_fingerprint,
            "checkpoint_manifest_hash": checkpoint.manifest_hash,
            "packed_evidence_hashes": [
                record["packed_evidence_hash"] for record in retrieval_records
            ],
        },
    )
    return FineTunedReaderRunResult(
        run_id=manager.run_id,
        method=GENERATIVE_METHOD,
        artifacts=paths,
        prediction_count=len(predictions),
        error_count=len(errors),
    )


def _packed_hash(evidence: object) -> str:
    from ..artifacts import fingerprint_json
    from ..schemas import PackedEvidence

    if not isinstance(evidence, PackedEvidence):
        raise TypeError("Expected PackedEvidence")
    return fingerprint_json(
        {
            "included_ids": list(evidence.included_ids),
            "rendered_text": evidence.rendered_text,
        }
    )


def run_finetuned_reader_from_config(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    limit: int | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> FineTunedReaderRunResult:
    """Load/validate every FTR dependency before starting inference."""

    config = load_config(config_path)
    if config.project.profile != "finetuned_reader" or config.finetuned_reader is None:
        raise FineTunedReaderPipelineError(
            "Config must use project.profile='finetuned_reader'"
        )
    root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    freeze = load_b2_freeze_fingerprint(root)
    require_complete_b2_freeze(freeze)
    retriever = FrozenB2EvidenceRetriever.from_repo(root)
    settings = config.finetuned_reader
    prompt_builder = GenerativePromptBuilder.from_files(
        root / settings.train_prompt_path,
        root / settings.inference_prompt_path,
        version=settings.dataset_version,
    )
    generator = load_finetuned_reader_generator(
        str(root / settings.checkpoint_path),
        checkpoint_manifest=str(root / settings.checkpoint_manifest_path),
        prompt_builder=prompt_builder,
        max_new_tokens=settings.max_new_tokens,
        stop_sequences=settings.stop_sequences,
        device="auto",
        load_in_4bit=settings.model.load_in_4bit,
    )
    cases = load_inference_questions(
        root / config.data.question_path,
        split=config.data.split,
    )
    selected = cases if limit is None else cases[:limit]
    selected_output = (
        root / config.runtime.outputs_dir if output_dir is None else Path(output_dir)
    )
    return run_finetuned_reader(
        selected,
        config=config,
        retriever=retriever,
        generator=generator,
        output_dir=selected_output,
        run_id=run_id,
    )


__all__ = [
    "FineTunedReaderPipelineError",
    "FineTunedReaderRunResult",
    "run_finetuned_reader",
    "run_finetuned_reader_from_config",
]
