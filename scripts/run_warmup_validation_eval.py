"""Run VAL-01 local clean-warmup evaluation (infer then eval-only gold join)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_finetuned_reader_authorized(root: Path) -> None:
    profile_path = root / "configs" / "finetuned_reader" / "model_profile.yaml"
    text = profile_path.read_text(encoding="utf-8")
    if "inference_authorized: true" not in text:
        raise RuntimeError(
            "finetuned_reader inference is blocked by "
            f"{profile_path.as_posix()} (inference_authorized is not true). "
            "Use --method hybrid_rag for the frozen B2 control path, or unlock "
            "a local base model + validated LoRA checkpoint first."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_retrieval_path(
    explicit_path: Path | None,
    prediction_run_dir: Path | None,
    *,
    required: bool,
) -> Path | None:
    """Resolve a run-local retrieval trace before opening evaluation gold."""

    if explicit_path is not None:
        if not explicit_path.is_file():
            raise RuntimeError(f"Retrieval artifact is missing: {explicit_path}")
        return explicit_path
    if prediction_run_dir is not None:
        candidate = prediction_run_dir / "retrieval.jsonl"
        if candidate.is_file():
            return candidate
    if required:
        location = (
            str(prediction_run_dir / "retrieval.jsonl")
            if prediction_run_dir is not None
            else "<prediction-run-dir>/retrieval.jsonl"
        )
        raise RuntimeError(
            "Retrieval artifact is required for the evaluation error report: "
            f"{location}. Pass --retrieval or use --skip-error-report."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from legal_rag.evaluation import EvaluationOptions
    from legal_rag.finetuned_reader.warmup_eval import (
        VAL01_POLICY_ID,
        WarmupEvalError,
        build_error_report_for_clean_eval,
        build_eval_summary,
        evaluate_clean_warmup,
        load_clean_inference_questions,
        load_clean_reference_records,
        load_clean_warmup_manifest,
        load_prediction_run_provenance,
        select_included_ids,
        write_warmup_eval_artifacts,
    )
    from legal_rag.pipeline import (
        prepare_bm25_index_from_config,
        run_hybrid_rag,
    )

    parser = argparse.ArgumentParser(
        description=(
            "VAL-01: clean-warmup IDs → question-only inference → "
            "evaluation-only METEOR/ROUGE-L join."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            root
            / "artifacts"
            / "sedar_sft"
            / "validation"
            / "clean_warmup_manifest.json"
        ),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=root / "data" / "warmup.json",
        help="Warmup question source (answers never loaded for inference).",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=root / "data" / "warmup.json",
        help="Warmup gold source opened only after predictions exist.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "frozen" / "hybrid_rag_b2.yaml",
    )
    parser.add_argument(
        "--method",
        choices=("hybrid_rag", "finetuned_reader", "sedar_sft"),
        default="hybrid_rag",
    )
    parser.add_argument(
        "--method-version",
        default="UNRESOLVED",
        help=(
            "Semantic model/method version. Required for sedar_sft and "
            "finetuned_reader."
        ),
    )
    parser.add_argument(
        "--scorer",
        choices=("btc_source_scorer_v1", "local_exact_token_metrics"),
        default="btc_source_scorer_v1",
        help="Versioned evaluation adapter; source-compatible scorer is default.",
    )
    parser.add_argument(
        "--scorer-source",
        type=Path,
        default=Path("UNRESOLVED"),
        help="Archived BTC scorer source path, when available.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="VAL-01 artifact directory (metrics/summary/error report).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Existing predictions.jsonl; skips inference when set.",
    )
    parser.add_argument(
        "--prediction-run-dir",
        type=Path,
        default=None,
        help=(
            "Immutable source inference run directory containing run_summary.json, "
            "config.json, checkpoint_reference.json and predictions.jsonl."
        ),
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=None,
        help=(
            "Evaluation-only retrieval trace; defaults to retrieval.jsonl in "
            "--prediction-run-dir."
        ),
    )
    parser.add_argument(
        "--skip-error-report",
        action="store_true",
        help="Skip Markdown/CSV error diagnostics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force BM25 rebuild before hybrid_rag inference.",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_clean_warmup_manifest(args.manifest)
        selected_ids = select_included_ids(manifest.included_ids, limit=args.limit)
        run_id = args.run_id or (f"val01_{args.method}_n{len(selected_ids)}")
        eval_dir = args.output_dir or (
            root / "artifacts" / "sedar_sft" / "validation" / "eval" / run_id
        )

        predictions_path = args.predictions
        retrieval_path: Path | None = None
        prediction_run_dir: Path | None = args.prediction_run_dir
        if predictions_path is None:
            # Inference boundary: question-only cases; no gold opened yet.
            cases = load_clean_inference_questions(
                args.questions,
                split="warmup",
                included_ids=selected_ids,
            )
            if args.method in {"finetuned_reader", "sedar_sft"}:
                _require_finetuned_reader_authorized(root)
                raise WarmupEvalError(
                    "SEDAR/FTR VAL-01 inference requires an unlocked checkpoint; "
                    "pass --predictions from a validated FTR run or use "
                    "--method hybrid_rag"
                )
            preparation = prepare_bm25_index_from_config(
                args.config,
                repo_root=root,
                rebuild_index=args.rebuild_index,
            )
            result = run_hybrid_rag(
                cases,
                {chunk.chunk_id: chunk for chunk in preparation.chunks},
                preparation.index,
                preparation.config,
                documents=preparation.documents,
                output_dir=root / preparation.config.runtime.outputs_dir,
                run_id=run_id,
                data_manifest_hash=preparation.manifest_hash,
                chunk_cache_fingerprint=preparation.chunk_cache_fingerprint,
                registry_path=root
                / preparation.config.runtime.artifacts_dir
                / "experiments.jsonl",
            )
            predictions_path = result.artifacts.predictions
            prediction_run_dir = predictions_path.parent
            if result.artifacts.retrieval is not None:
                retrieval_path = result.artifacts.retrieval
        else:
            predictions_path = Path(predictions_path)
            if not predictions_path.is_file():
                raise WarmupEvalError(f"Predictions missing: {predictions_path}")
            if prediction_run_dir is None:
                prediction_run_dir = predictions_path.parent

        if prediction_run_dir is None:
            raise WarmupEvalError("Prediction run directory is required")
        retrieval_path = _resolve_retrieval_path(
            args.retrieval,
            prediction_run_dir,
            required=not args.skip_error_report,
        )
        prediction_provenance = load_prediction_run_provenance(
            prediction_run_dir,
            method=args.method,
            method_version=args.method_version,
        )

        # Evaluation boundary: open gold only after predictions exist.
        references = load_clean_reference_records(
            args.references,
            split="warmup",
            included_ids=selected_ids,
        )
        options = EvaluationOptions(
            run_id=run_id,
            method=args.method,
            method_version=args.method_version,
            split="warmup",
            data_manifest_hash=manifest.source_warmup_sha256,
            prediction_artifact=predictions_path.as_posix(),
            command=(
                "python scripts/run_warmup_validation_eval.py " + " ".join(sys.argv[1:])
            ),
            scorer=args.scorer,
            scorer_source_path=args.scorer_source.as_posix(),
            scorer_source_sha256=(
                _sha256_file(args.scorer_source)
                if args.scorer_source.is_file()
                else "UNRESOLVED"
            ),
            pipeline_method=prediction_provenance.pipeline_method,
            prediction_run_id=prediction_provenance.source_run_id,
            prediction_source_run_dir=prediction_provenance.source_run_dir.as_posix(),
            prediction_artifact_sha256=_sha256_file(predictions_path),
            source_prediction_artifact_sha256=(
                prediction_provenance.source_prediction_artifact_sha256
            ),
            checkpoint_manifest_hash=(prediction_provenance.checkpoint_manifest_hash),
            adapter_hash=prediction_provenance.adapter_hash,
            base_model=prediction_provenance.base_model,
            base_revision=prediction_provenance.base_revision,
            inference_config_hash=prediction_provenance.config_hash,
            retrieval_config_hash=prediction_provenance.retrieval_config_hash,
            index_fingerprint=prediction_provenance.index_fingerprint,
            validation_manifest_sha256=_sha256_file(args.manifest),
            included_ids_hash=manifest.included_ids_hash,
        )
        report = evaluate_clean_warmup(
            references=references,
            predictions_path=predictions_path,
            options=options,
        )
        metrics_path = eval_dir / "metrics.json"
        summary = build_eval_summary(
            manifest=manifest,
            selected_ids=selected_ids,
            method=args.method,
            run_id=run_id,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            retrieval_path=retrieval_path,
            report=report,
            inference_used_answers=False,
            references_opened_before_predictions=False,
        )
        # Write metrics/summary/eval-only references first so error report can join.
        paths = write_warmup_eval_artifacts(
            output_dir=eval_dir,
            summary=summary,
            report=report,
            references=references,
            error_report=None,
            overwrite=args.overwrite,
        )
        if not args.skip_error_report:
            from legal_rag.evaluation.error_report import write_error_report

            error_report = build_error_report_for_clean_eval(
                predictions_path=predictions_path,
                references_eval_only_path=paths["references_eval_only"],
                metrics_path=paths["metrics"],
                retrieval_path=retrieval_path,
                retrieval_ids=selected_ids,
                split="warmup",
            )
            paths["error_report_md"] = eval_dir / "error_report.md"
            paths["error_report_csv"] = eval_dir / "error_report.csv"
            write_error_report(
                error_report,
                markdown_path=paths["error_report_md"],
                csv_path=paths["error_report_csv"],
                overwrite=args.overwrite,
            )

        print(
            json.dumps(
                {
                    "policy_id": VAL01_POLICY_ID,
                    "run_id": run_id,
                    "method": args.method,
                    "method_version": args.method_version,
                    "selected_ids_count": len(selected_ids),
                    "included_ids_hash": manifest.included_ids_hash,
                    "evaluator_name": report.artifact["evaluator_name"],
                    "evaluator_version": report.artifact["evaluator_version"],
                    "metrics": report.artifact["metrics"],
                    "predictions": predictions_path.as_posix(),
                    "artifacts": {key: path.as_posix() for key, path in paths.items()},
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (WarmupEvalError, RuntimeError, FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
