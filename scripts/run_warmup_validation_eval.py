"""Run VAL-01 local clean-warmup evaluation (infer then eval-only gold join)."""

from __future__ import annotations

import argparse
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
        choices=("hybrid_rag", "finetuned_reader"),
        default="hybrid_rag",
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
        if predictions_path is None:
            # Inference boundary: question-only cases; no gold opened yet.
            cases = load_clean_inference_questions(
                args.questions,
                split="warmup",
                included_ids=selected_ids,
            )
            if args.method == "finetuned_reader":
                _require_finetuned_reader_authorized(root)
                raise WarmupEvalError(
                    "finetuned_reader VAL-01 inference requires an unlocked "
                    "checkpoint; pass --predictions from a validated FTR run "
                    "or use --method hybrid_rag"
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
            if result.artifacts.retrieval is not None:
                retrieval_path = result.artifacts.retrieval
        else:
            predictions_path = Path(predictions_path)
            if not predictions_path.is_file():
                raise WarmupEvalError(f"Predictions missing: {predictions_path}")

        # Evaluation boundary: open gold only after predictions exist.
        references = load_clean_reference_records(
            args.references,
            split="warmup",
            included_ids=selected_ids,
        )
        options = EvaluationOptions(
            run_id=run_id,
            method=args.method,
            split="warmup",
            data_manifest_hash=manifest.source_warmup_sha256,
            prediction_artifact=predictions_path.as_posix(),
            command="python scripts/run_warmup_validation_eval.py",
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
                    "selected_ids_count": len(selected_ids),
                    "included_ids_hash": manifest.included_ids_hash,
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
