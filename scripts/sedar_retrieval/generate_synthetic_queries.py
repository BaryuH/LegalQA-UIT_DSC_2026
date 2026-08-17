#!/usr/bin/env python3
"""Generate leakage-aware synthetic Vietnamese legal queries (TASK 09)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl
from legal_rag.sedar_retrieval.training.synthetic_queries import (
    DEFAULT_PROMPT_VERSION,
    QUERY_TYPES,
    SYNTHETIC_SCHEMA_VERSION,
    DocumentSplit,
    SyntheticGenerationConfig,
    SyntheticGenerationErrorRecord,
    SyntheticRejectionRecord,
    TemplateQueryGenerator,
    TransformersQueryGenerator,
    build_synthetic_records,
    synthetic_prompt_sha256,
    write_generation_errors,
    write_synthetic_records,
    write_synthetic_rejections,
)


def _parse_query_types(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(values) - set(QUERY_TYPES))
    if not values or unknown:
        raise SystemExit(
            "query types must be a non-empty subset of: " + ", ".join(QUERY_TYPES)
        )
    return values


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise SystemExit(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not force:
        raise SystemExit(
            f"Output directory is non-empty: {path}; use a new run directory "
            "or pass --force explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_document_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise SystemExit(f"Document manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        values: list[object] = []
        for line in iter_jsonl_lines(path):
            values.append(json.loads(line))
        payload = values
    if isinstance(payload, dict):
        payload = payload.get("document_ids", payload.get("ids", []))
    if not isinstance(payload, list):
        raise SystemExit(
            f"Document manifest must be a JSON list/object or JSONL: {path}"
        )
    document_ids: set[str] = set()
    for item in payload:
        if isinstance(item, dict):
            item = item.get("document_id")
        if item is None or not str(item).strip():
            raise SystemExit(f"Blank document ID in manifest: {path}")
        document_ids.add(str(item))
    return document_ids


def _external_document_splits(
    passages: tuple[Any, ...],
    *,
    validation_path: Path | None,
    test_path: Path | None,
) -> dict[str, DocumentSplit] | None:
    if validation_path is None and test_path is None:
        return None
    document_ids = {passage.document_id for passage in passages}
    validation_ids = _load_document_ids(validation_path)
    test_ids = _load_document_ids(test_path)
    overlap = validation_ids & test_ids
    unknown = (validation_ids | test_ids) - document_ids
    if overlap:
        raise SystemExit(
            "Validation/test document manifests overlap: "
            + ", ".join(sorted(overlap)[:5])
        )
    if unknown:
        raise SystemExit(
            "Document manifest contains IDs absent from passages: "
            + ", ".join(sorted(unknown)[:5])
        )
    assignments: dict[str, DocumentSplit] = {}
    for document_id in document_ids:
        assignments[document_id] = (
            "validation"
            if document_id in validation_ids
            else "test"
            if document_id in test_ids
            else "train"
        )
    return assignments


def _build_generator(args: argparse.Namespace) -> Any:
    if args.backend == "template":
        if args.generator_model != "deterministic-template":
            raise SystemExit(
                "Template backend requires --generator-model deterministic-template"
            )
        return TemplateQueryGenerator()
    if not args.generator_revision:
        raise SystemExit("Transformers backend requires a pinned --generator-revision.")
    if not args.generator_model:
        raise SystemExit("Transformers backend requires --generator-model.")
    return TransformersQueryGenerator(
        model=args.generator_model,
        revision=args.generator_revision,
        device=args.device,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("template", "transformers"),
        default="template",
    )
    parser.add_argument("--generator-model", default="deterministic-template")
    parser.add_argument("--generator-revision", default="template-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--target-accepted", type=int, default=10_000)
    parser.add_argument("--max-attempts", type=int, default=30_000)
    parser.add_argument("--query-types", default=",".join(QUERY_TYPES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--source-split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument(
        "--validation-documents",
        type=Path,
        help="Optional JSON/JSONL document-ID manifest for validation holdout.",
    )
    parser.add_argument(
        "--test-documents",
        type=Path,
        help="Optional JSON/JSONL document-ID manifest for test holdout.",
    )
    parser.add_argument("--legal-domain", default="vietnamese_law")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--min-source-token-overlap", type=float, default=0.15)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    query_types = _parse_query_types(args.query_types)
    if args.target_accepted <= 0 or args.max_attempts < args.target_accepted:
        raise SystemExit(
            "--target-accepted must be positive and --max-attempts must "
            "be at least target-accepted"
        )
    if args.source_split != "train":
        raise SystemExit(
            "TASK 09 training output must use --source-split train; "
            "validation/test documents are held out by the deterministic split."
        )
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("--max-input-tokens and --max-new-tokens must be positive")
    _prepare_output_dir(args.output_dir, force=args.force)

    passages = load_passages_jsonl(str(args.passages))
    if not passages:
        raise SystemExit("Cannot generate synthetic queries from zero passages")
    document_splits = _external_document_splits(
        passages,
        validation_path=args.validation_documents,
        test_path=args.test_documents,
    )
    generator = _build_generator(args)
    config = SyntheticGenerationConfig(
        target_accepted=args.target_accepted,
        max_attempts=args.max_attempts,
        query_types=query_types,  # type: ignore[arg-type]
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        source_split=args.source_split,
        legal_domain=args.legal_domain,
        prompt_version=args.prompt_version,
        min_source_token_overlap=args.min_source_token_overlap,
    )
    rejected_records: list[SyntheticRejectionRecord] = []
    generation_errors: list[SyntheticGenerationErrorRecord] = []
    records, report = build_synthetic_records(
        passages,
        generator=generator,
        config=config,
        document_splits=document_splits,
        document_split_source=(
            "external_manifest" if document_splits is not None else "deterministic_hash"
        ),
        rejected_records=rejected_records,
        generation_errors=generation_errors,
    )
    records_path = args.output_dir / "synthetic_queries.jsonl"
    rejected_path = args.output_dir / "rejections.jsonl"
    errors_path = args.output_dir / "generation_errors.jsonl"
    write_synthetic_records(records_path, records, overwrite=args.force)
    write_synthetic_rejections(rejected_path, rejected_records, overwrite=args.force)
    write_generation_errors(errors_path, generation_errors, overwrite=args.force)
    report_payload = report.as_dict()
    report_payload.update(
        {
            "generator_backend": args.backend,
            "generator_model": generator.model,
            "generator_revision": generator.revision,
            "reader_checkpoint_used": False,
            "gold_answers_used": False,
            "records_path": str(records_path),
            "rejections_path": str(rejected_path),
            "generation_errors_path": str(errors_path),
            "rejection_count": len(rejected_records),
            "generation_error_count": len(generation_errors),
        }
    )
    _write_json(args.output_dir / "audit.json", report_payload)

    status = (
        "NEEDS_MANUAL_AUDIT"
        if (
            report.accepted == report.target_accepted
            and report.document_isolation_ok
            and report.generation_error_count == 0
        )
        else "FAIL"
    )
    manifest = {
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "run_id": new_run_id("synthetic_queries"),
        "git_commit": git_commit_sha(),
        "status": status,
        "detail": (
            "Pilot generated; stratified manual audit of at least 300 records "
            "is required before scaling."
        ),
        "passages_path": str(args.passages),
        "passage_count": len(passages),
        "records_path": str(records_path),
        "audit_path": str(args.output_dir / "audit.json"),
        "rejections_path": str(rejected_path),
        "generation_errors_path": str(errors_path),
        "target_accepted": args.target_accepted,
        "accepted": report.accepted,
        "query_types": list(query_types),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "source_split": args.source_split,
        "document_split_source": report.document_split_source,
        "generator_backend": args.backend,
        "generator_model": generator.model,
        "generator_revision": generator.revision,
        "prompt_version": args.prompt_version,
        "prompt_sha256": synthetic_prompt_sha256(),
        "reader_checkpoint_used": False,
        "gold_answers_used": False,
        "validation_test_leakage_count": report.source_split_leakage_count,
        "generation_error_count": report.generation_error_count,
        "scale_decision": "AUDIT_300_BEFORE_SCALE",
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if status == "NEEDS_MANUAL_AUDIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
