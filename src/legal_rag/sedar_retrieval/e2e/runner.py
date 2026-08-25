"""TASK 20 runner: SEDAR retrieval rankings -> packed evidence -> frozen reader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from legal_rag.artifacts import RunManager, fingerprint_json
from legal_rag.finetuned_reader.warmup_eval import (
    WarmupEvalError,
    load_clean_inference_questions,
    load_clean_warmup_manifest,
    select_included_ids,
)
from legal_rag.schemas import PackedEvidence
from legal_rag.sedar_retrieval.evidence.passage_packer import (
    PassageEvidenceConfig,
    PassageEvidencePackError,
    load_retrieval_rankings,
    pack_passage_retrieval_evidence,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl


class SedarE2ERunnerError(RuntimeError):
    """Raised when TASK 20 cannot run without violating the contract."""


class GeneratorFactory(Protocol):
    def __call__(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class SedarE2EConfig:
    retrieval_path: Path
    passages_path: Path
    questions_path: Path
    manifest_path: Path
    checkpoint_dir: Path
    checkpoint_manifest: Path | None
    inference_prompt_path: Path
    train_prompt_path: Path
    prompt_version: str
    output_dir: Path
    retrieval_variant: str
    evidence: PassageEvidenceConfig
    max_new_tokens: int
    stop_sequences: tuple[str, ...]
    device: str
    load_in_4bit: bool | None
    split: str = "warmup"
    limit: int | None = None
    fail_fast: bool = True
    repo_root: Path | None = None


@dataclass(frozen=True, slots=True)
class SedarE2ERunResult:
    run_id: str
    output_dir: Path
    prediction_count: int
    error_count: int
    retrieval_variant: str
    reader_adapter_hash: str
    manifest_path: Path


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _packed_hash(evidence: PackedEvidence) -> str:
    return fingerprint_json(
        {
            "included_ids": list(evidence.included_ids),
            "rendered_text": evidence.rendered_text,
        }
    )


def run_sedar_e2e(
    config: SedarE2EConfig,
    *,
    generator_factory: GeneratorFactory,
    run_id: str | None = None,
) -> SedarE2ERunResult:
    """Run frozen generative reader inference on precomputed SEDAR retrieval."""

    repo_root = (config.repo_root or Path.cwd()).resolve()
    manifest = load_clean_warmup_manifest(config.manifest_path)
    selected_ids = select_included_ids(manifest.included_ids, limit=config.limit)
    cases = load_clean_inference_questions(
        config.questions_path,
        split=config.split,
        included_ids=selected_ids,
    )
    rankings = load_retrieval_rankings(config.retrieval_path)
    passage_rows = load_passages_jsonl(str(config.passages_path))
    passages = {passage.passage_id: passage for passage in passage_rows}
    if len(passages) != len(passage_rows):
        raise SedarE2ERunnerError("Passage corpus contains duplicate passage_id values")

    missing_rankings = [case.id for case in cases if case.id not in rankings]
    if missing_rankings:
        preview = ", ".join(missing_rankings[:5])
        raise SedarE2ERunnerError(
            f"Retrieval JSONL is missing clean-warmup query IDs; examples: {preview}"
        )

    generator = generator_factory()
    manager = RunManager.create(
        config.output_dir,
        split=config.split,
        method="sedar_sft",
        repo_root=repo_root,
        run_id=run_id,
        with_retrieval=True,
        registry_path=repo_root / "artifacts" / "experiments.jsonl",
    )
    paths = manager.paths
    if paths.retrieval is None:
        raise SedarE2ERunnerError(
            "Run manager did not allocate retrieval artifact path"
        )

    predictions: list[dict[str, object]] = []
    retrieval_records: list[dict[str, object]] = []
    generation_records: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for case in sorted(cases, key=lambda item: item.id):
        try:
            packed = pack_passage_retrieval_evidence(
                rankings[case.id],
                passages,
                config=config.evidence,
            )
            generated = generator.generate(case.question, packed)
            predictions.append(
                {
                    "id": case.id,
                    "answer": generated.cleaned_answer,
                    "method": "sedar_sft",
                    "model": generated.model,
                    "model_version": generated.model_version,
                }
            )
            retrieval_records.append(
                {
                    "id": case.id,
                    "query_role": "question_only",
                    "retrieval_variant": config.retrieval_variant,
                    "retrieval_path": str(config.retrieval_path),
                    "raw_hit_ids": [
                        candidate.passage_id for candidate in rankings[case.id]
                    ],
                    "packed_chunk_ids": list(packed.included_ids),
                    "packed_dropped_ids": list(packed.dropped_ids),
                    "packed_truncated_ids": list(packed.truncated_ids),
                    "packed_evidence_hash": _packed_hash(packed),
                }
            )
            generation_records.append(
                {
                    "id": case.id,
                    "method": "sedar_sft",
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
        except (PassageEvidencePackError, Exception) as exc:
            error = {
                "id": case.id,
                "method": "sedar_sft",
                "error_code": "TASK20_E2E_ERROR",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "stage": "pack_or_generate",
            }
            errors.append(error)
            generation_records.append({**error, "status": "error"})
            if config.fail_fast:
                break

    retrieval_fingerprint = fingerprint_json(
        {
            "retrieval_variant": config.retrieval_variant,
            "retrieval_path": str(config.retrieval_path),
            "retrieval_sha256": _sha256_file(config.retrieval_path),
            "passages_path": str(config.passages_path),
            "passages_sha256": _sha256_file(config.passages_path),
            "evidence": {
                "evidence_top_k": config.evidence.evidence_top_k,
                "max_total_chars": config.evidence.max_total_chars,
                "max_chunks_per_document": config.evidence.max_chunks_per_document,
            },
        }
    )
    run_config: dict[str, Any] = {
        "schema_version": "task20.sedar_e2e.v1",
        "task": "TASK20",
        "retrieval_variant": config.retrieval_variant,
        "retrieval_path": str(config.retrieval_path),
        "passages_path": str(config.passages_path),
        "questions_path": str(config.questions_path),
        "manifest_path": str(config.manifest_path),
        "checkpoint_dir": str(config.checkpoint_dir),
        "checkpoint_manifest": str(config.checkpoint_manifest or ""),
        "inference_prompt_path": str(config.inference_prompt_path),
        "train_prompt_path": str(config.train_prompt_path),
        "prompt_version": config.prompt_version,
        "max_new_tokens": config.max_new_tokens,
        "device": config.device,
        "load_in_4bit": config.load_in_4bit,
        "split": config.split,
        "selected_ids_count": len(selected_ids),
        "reader_frozen": True,
    }
    config_hash = fingerprint_json(run_config)
    run_config["config_hash"] = config_hash
    manager.write_json(paths.config, run_config)
    manager.write_json(
        paths.environment,
        {
            **manager.environment(seed=42),
            "profile": "sedar_sft",
            "type": "generative_sft_reader",
            "task": "TASK20",
            "checkpoint_manifest_hash": generator.checkpoint.manifest_hash,
            "adapter_hash": generator.checkpoint.adapter_hash,
            "base_model": generator.checkpoint.manifest.get("base_model"),
            "base_revision": generator.checkpoint.manifest.get("base_revision"),
        },
    )
    manager.write_jsonl(paths.predictions, predictions)
    manager.write_jsonl(paths.retrieval, retrieval_records)
    manager.write_jsonl(paths.generation, generation_records)
    manager.write_jsonl(paths.errors, errors)
    manager.write_json(
        paths.metrics,
        {
            "schema_version": "task20.metrics.v1",
            "status": "not_evaluated",
            "reference_role": "none",
            "metrics": {"meteor": None, "rouge_l": None},
        },
    )
    manager.write_json(
        paths.submission,
        {
            "schema_version": "task20.submission-placeholder.v1",
            "status": "not_packaged",
            "reason": "Run VAL-01 or evaluate_predictions after TASK 20 inference",
        },
    )
    manager.write_json(
        paths.run_dir / "checkpoint_reference.json",
        {
            "profile": generator.checkpoint.manifest.get("profile", "sedar_sft"),
            "method": "sedar_sft",
            "base_model": generator.checkpoint.manifest.get("base_model"),
            "base_revision": generator.checkpoint.manifest.get("base_revision"),
            "adapter_path": generator.checkpoint.manifest.get(
                "adapter_path", "adapter"
            ),
            "adapter_hash": generator.checkpoint.adapter_hash,
            "checkpoint_manifest_hash": generator.checkpoint.manifest_hash,
            "training_dataset_hash": generator.checkpoint.manifest.get(
                "dataset_manifest_hash"
            ),
            "retrieval_config_hash": retrieval_fingerprint,
            "index_fingerprint": retrieval_fingerprint,
            "prompt_hash": generator.checkpoint.manifest.get("prompt_hash"),
            "method_version": config.prompt_version,
        },
    )
    manager.write_json(
        paths.summary,
        {
            "schema_version": "task20.run-summary.v1",
            "run_id": manager.run_id,
            "method": "sedar_sft",
            "split": config.split,
            "split_policy": "warmup_evaluation",
            "prediction_count": len(predictions),
            "error_count": len(errors),
            "config_hash": config_hash,
            "index_fingerprint": retrieval_fingerprint,
            "checkpoint_manifest_hash": generator.checkpoint.manifest_hash,
            "retrieval_variant": config.retrieval_variant,
            "packed_evidence_hashes": [
                record["packed_evidence_hash"] for record in retrieval_records
            ],
        },
    )
    manifest_out = paths.run_dir / "task20_manifest.json"
    manifest_out.write_text(
        json.dumps(
            {
                "schema_version": "task20.manifest.v1",
                "run_id": manager.run_id,
                "status": "PASS" if not errors else "FAIL",
                "retrieval_variant": config.retrieval_variant,
                "reader_adapter_hash": generator.checkpoint.adapter_hash,
                "selected_ids_count": len(selected_ids),
                "prediction_count": len(predictions),
                "error_count": len(errors),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if errors and config.fail_fast:
        raise SedarE2ERunnerError(
            f"TASK 20 stopped after first error: {errors[0]['message']}"
        )
    if len(predictions) != len(selected_ids):
        raise SedarE2ERunnerError(
            f"Expected {len(selected_ids)} predictions, got {len(predictions)}"
        )
    return SedarE2ERunResult(
        run_id=manager.run_id,
        output_dir=paths.run_dir,
        prediction_count=len(predictions),
        error_count=len(errors),
        retrieval_variant=config.retrieval_variant,
        reader_adapter_hash=generator.checkpoint.adapter_hash,
        manifest_path=manifest_out,
    )


def load_selected_case_ids(config: SedarE2EConfig) -> tuple[str, ...]:
    """Return the clean-warmup IDs that a run would score."""

    try:
        manifest = load_clean_warmup_manifest(config.manifest_path)
    except WarmupEvalError as exc:
        raise SedarE2ERunnerError(str(exc)) from exc
    return select_included_ids(manifest.included_ids, limit=config.limit)


__all__ = [
    "GeneratorFactory",
    "SedarE2EConfig",
    "SedarE2ERunResult",
    "SedarE2ERunnerError",
    "load_selected_case_ids",
    "run_sedar_e2e",
]
