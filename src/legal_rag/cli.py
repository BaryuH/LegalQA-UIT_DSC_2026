"""Minimal command-line interface for the project scaffold."""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from zipfile import ZipFile

from .config import load_config
from .data_validation import validate_data, write_validation_report
from .pipeline import (
    PipelineError,
    PipelineRunError,
    inspect_bm25_retrieval,
    prepare_bm25_index_from_config,
    run_bm25_rag_from_config,
    run_direct_from_config,
    run_hybrid_rag_from_config,
)
from .submission import (
    SubmissionError,
    create_submission,
    load_predictions,
    validate_submission_file,
)

IMPLEMENTED_COMMANDS: dict[str, str] = {
    "build-index": "Build a fingerprinted retrieval index from selected contexts.",
    "inspect-retrieval": "Inspect retrieval evidence for a question.",
    "run": "Run an inference pipeline.",
    "create-submission": "Create an exact submission from predictions.",
    "validate-submission": "Validate an existing submission.zip against question IDs.",
}
PLACEHOLDER_COMMANDS: dict[str, str] = {
    "evaluate": "Evaluate predictions against an approved reference split.",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without executing any pipeline work."""

    parser = argparse.ArgumentParser(
        prog="legal-rag",
        description="Vietnamese legal RAG project scaffold.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to a YAML project configuration.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "check-config",
        help="Load and validate the project configuration.",
    )
    validate_parser = subparsers.add_parser(
        "validate-data",
        help="Validate configured source data and write a content-free JSON report.",
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to a YAML project configuration.",
    )
    for command, help_text in {
        **IMPLEMENTED_COMMANDS,
        **PLACEHOLDER_COMMANDS,
    }.items():
        command_parser = subparsers.add_parser(command, help=help_text)
        if command in {"build-index", "run"}:
            command_parser.add_argument(
                "--config",
                type=Path,
                default=argparse.SUPPRESS,
                help="Path to a YAML project configuration.",
            )
        if command == "build-index":
            command_parser.add_argument(
                "--rebuild-index",
                action="store_true",
                help="Explicitly rebuild a missing or stale BM25 index.",
            )
        if command == "inspect-retrieval":
            command_parser.add_argument(
                "--config",
                type=Path,
                default=argparse.SUPPRESS,
                help="Path to a BM25-RAG project configuration.",
            )
            command_parser.add_argument(
                "--question",
                default=None,
                help="Question-only retrieval query.",
            )
            command_parser.add_argument(
                "--top-k",
                type=int,
                default=None,
                help="Optional inspection limit; defaults to retrieval.rough_top_n.",
            )
            command_parser.add_argument(
                "--rebuild-index",
                action="store_true",
                help="Explicitly rebuild a missing or stale BM25 index.",
            )
        if command == "run":
            command_parser.add_argument(
                "--method",
                choices=("direct", "bm25_rag", "hybrid_rag"),
                default=None,
                help="Explicit method; must agree with the selected profile.",
            )
            command_parser.add_argument(
                "--rebuild-index",
                action="store_true",
                help="Explicitly rebuild a missing or stale BM25 index.",
            )
            command_parser.add_argument(
                "--limit",
                type=int,
                default=None,
                help="Run only the first N deterministic question IDs.",
            )
            command_parser.add_argument(
                "--run-id",
                default=None,
                help="Explicit run ID; defaults to a config/data fingerprint.",
            )
            command_parser.add_argument(
                "--output-dir",
                type=Path,
                default=None,
                help="Directory under which the run folder is written.",
            )
            command_parser.add_argument(
                "--package-submission",
                action="store_true",
                help=(
                    "After a successful batch, build submission.zip in the run "
                    "directory with the dedicated official serializer."
                ),
            )
        if command in {"create-submission", "validate-submission"}:
            command_parser.add_argument(
                "--predictions" if command == "create-submission" else "--submission",
                type=Path,
                required=True,
                help=(
                    "Prediction JSONL input."
                    if command == "create-submission"
                    else "Existing submission.zip to validate."
                ),
            )
            command_parser.add_argument(
                "--questions",
                type=Path,
                required=True,
                help="Inference/submission dataset that defines IDs and order.",
            )
            if command == "create-submission":
                command_parser.add_argument(
                    "--output",
                    type=Path,
                    default=Path("submission.zip"),
                    help=(
                        "Output path; the basename must be submission.zip and existing "
                        "files are never overwritten."
                    ),
                )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the minimal CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if args.command in {"build-index", "run"} and "--config" not in raw_argv:
        parser.error(
            f"Command '{args.command}' requires an explicit configuration; "
            "pass --config to execute it."
        )
    if args.command == "inspect-retrieval" and args.question is None:
        parser.error("Command 'inspect-retrieval' requires --question.")

    if args.command == "check-config":
        config = load_config(args.config)
        print(
            f"Loaded {args.config}: project={config.project_name!r}, "
            f"mode={config.mode!r}"
        )
        return 0

    if args.command == "validate-data":
        try:
            config = load_config(args.config)
            repo_root = Path(__file__).resolve().parents[2]
            run = validate_data(config, repo_root)
            write_validation_report(run)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        print(
            json.dumps(
                {
                    "output": run.output_path.relative_to(repo_root).as_posix(),
                    "questions": run.report["questions"]["count"],
                    "contexts": run.report["contexts"]["count"],
                    "critical_error_count": len(run.report["critical_errors"]),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if run.is_valid else 2

    if args.command == "build-index":
        try:
            preparation = prepare_bm25_index_from_config(
                args.config,
                rebuild_index=True,
            )
        except (OSError, ValueError, PipelineError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "index": preparation.index.cache_path.as_posix()
                    if preparation.index.cache_path
                    else None,
                    "index_fingerprint": preparation.index.index_fingerprint,
                    "chunks": len(preparation.chunks),
                    "documents": len(preparation.documents),
                    "manifest_hash": preparation.manifest_hash,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "inspect-retrieval":
        try:
            preparation = prepare_bm25_index_from_config(
                args.config,
                rebuild_index=args.rebuild_index,
            )
            rows = inspect_bm25_retrieval(
                preparation,
                args.question,
                top_k=args.top_k,
            )
        except (OSError, ValueError, PipelineError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        print(
            "\t".join(
                (
                    "rank",
                    "score",
                    "chunk_id",
                    "document name",
                    "article/clause",
                    "preview",
                )
            )
        )
        for row in rows:
            print(
                "\t".join(
                    (
                        str(row.rank),
                        f"{row.score:.6f}",
                        row.chunk_id,
                        row.document_name,
                        row.article_clause,
                        row.preview.replace("\t", " "),
                    )
                )
            )
        return 0

    if args.command == "run":
        try:
            config = load_config(args.config)
            profile_method = {
                "direct": "direct",
                "mock": "direct",
                "bm25-rag": "bm25_rag",
                "hybrid-rag": "hybrid_rag",
            }.get(config.project.profile)
            selected_method = args.method or profile_method
            if selected_method is None:
                raise PipelineError(
                    f"Unsupported run profile: {config.project.profile!r}"
                )
            if args.method is not None and args.method != profile_method:
                raise PipelineError(
                    f"Method {args.method!r} does not match profile "
                    f"{config.project.profile!r}"
                )
            if selected_method == "bm25_rag":
                result = run_bm25_rag_from_config(
                    args.config,
                    rebuild_index=args.rebuild_index,
                    limit=args.limit,
                    output_dir=args.output_dir,
                    run_id=args.run_id,
                )
            elif selected_method == "hybrid_rag":
                result = run_hybrid_rag_from_config(
                    args.config,
                    rebuild_index=args.rebuild_index,
                    limit=args.limit,
                    output_dir=args.output_dir,
                    run_id=args.run_id,
                )
            elif selected_method == "direct":
                result = run_direct_from_config(
                    args.config,
                    limit=args.limit,
                    output_dir=args.output_dir,
                    run_id=args.run_id,
                )
            else:
                raise PipelineError(f"Unsupported run method: {selected_method!r}")
        except PipelineRunError as exc:
            print(
                json.dumps(
                    {
                        "error": str(exc),
                        "run_dir": exc.result.artifacts.run_dir.as_posix(),
                        "errors": exc.result.error_count,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
        except (OSError, ValueError, PipelineError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        packaged_submission: Path | None = None
        if args.package_submission:
            try:
                if not config.submission.enabled:
                    raise PipelineError(
                        "Submission packaging is disabled in the selected config"
                    )
                repo_root = Path(__file__).resolve().parents[2]
                packaged_submission = (
                    result.artifacts.run_dir / config.submission.output_filename
                )
                create_submission(
                    result.artifacts.predictions,
                    repo_root / config.data.question_path,
                    packaged_submission,
                    reject_empty_answers=config.submission.reject_empty_answers,
                )
            except (OSError, SubmissionError, ValueError, PipelineError) as exc:
                print(f"ERROR: submission packaging failed: {exc}", file=sys.stderr)
                return 2
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "method": result.method,
                    "run_dir": result.artifacts.run_dir.as_posix(),
                    "predictions": result.prediction_count,
                    "errors": result.error_count,
                    "submission": (
                        packaged_submission.as_posix()
                        if packaged_submission is not None
                        else None
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "create-submission":
        try:
            prediction_count = len(load_predictions(args.predictions))
            validation = create_submission(
                args.predictions,
                args.questions,
                args.output,
            )
        except (OSError, SubmissionError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        with ZipFile(args.output) as archive:
            zip_members = archive.namelist()
        print(
            json.dumps(
                {
                    "output": args.output.as_posix(),
                    "valid": validation.valid,
                    "expected_questions": validation.expected_count,
                    "predictions": prediction_count,
                    "written_answers": validation.record_count,
                    "missing": list(validation.missing_ids),
                    "extra": list(validation.extra_ids),
                    "warnings": list(validation.warnings),
                    "json_path": "<temporary>/submission.json",
                    "zip_members": zip_members,
                    "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "validate-submission":
        try:
            validation = validate_submission_file(
                args.submission,
                args.questions,
            )
        except (OSError, SubmissionError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "submission": args.submission.as_posix(),
                    "valid": validation.valid,
                    "expected_questions": validation.expected_count,
                    "written_answers": validation.record_count,
                    "missing": list(validation.missing_ids),
                    "extra": list(validation.extra_ids),
                    "error_codes": list(validation.error_codes),
                    "warnings": list(validation.warnings),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if validation.valid else 2

    if args.command in PLACEHOLDER_COMMANDS:
        parser.error(
            f"Command '{args.command}' is a scaffold placeholder and is not "
            "implemented yet."
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
