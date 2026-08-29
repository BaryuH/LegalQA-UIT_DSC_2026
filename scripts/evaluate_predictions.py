"""Evaluate local legal-QA predictions with METEOR and ROUGE-L."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

from legal_rag.evaluation import (  # noqa: E402
    LOCAL_SCORER_ID,
    SOURCE_SCORER_ID,
    EvaluationOptions,
    SourceScorerDependencyError,
    evaluate_records,
    write_report,
)
from legal_rag.evaluation.alignment import AlignmentError  # noqa: E402
from legal_rag.evaluation.io import InputFormatError, load_records  # noqa: E402
from legal_rag.splits import SPLIT_NAMES, validate_reference_access  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(repo_root: Path, manifest: Path) -> str:
    manifest_path = manifest if manifest.is_absolute() else repo_root / manifest
    return _sha256(manifest_path) if manifest_path.is_file() else "UNRESOLVED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score a complete prediction set against references using an explicit "
            "versioned scorer adapter."
        )
    )
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", default="local")
    parser.add_argument(
        "--method-version",
        default="UNRESOLVED",
        help="Semantic model/method version recorded in the metric artifact.",
    )
    parser.add_argument(
        "--scorer",
        choices=(SOURCE_SCORER_ID, LOCAL_SCORER_ID),
        default=SOURCE_SCORER_ID,
        help="Scorer adapter; the source-compatible adapter is the default.",
    )
    parser.add_argument(
        "--scorer-source",
        default="UNRESOLVED",
        type=Path,
        help="Archived BTC scorer source path, when available.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=SPLIT_NAMES,
        help="Registered split role; only reference-enabled roles can be scored.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--manifest",
        default=Path("artifacts/data-baseline/manifest.json"),
        type=Path,
        help="Manifest path used for artifact provenance; absent means UNRESOLVED.",
    )
    parser.add_argument(
        "--data-manifest-hash",
        default=None,
        help="Explicit manifest SHA256; otherwise hash --manifest when present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the requested metric artifact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    run_id = args.run_id or args.output.stem
    try:
        # Validate before opening the reference path so private/public files
        # cannot cross the evaluation boundary through this command.
        validate_reference_access(args.split, "approved_evaluation")
        references = load_records(args.references, "Reference")
        predictions = load_records(args.predictions, "Prediction")
        manifest_hash = args.data_manifest_hash or _manifest_hash(
            repo_root, args.manifest
        )
        options = EvaluationOptions(
            run_id=run_id,
            method=args.method,
            method_version=args.method_version,
            split=args.split,
            data_manifest_hash=manifest_hash,
            prediction_artifact=args.predictions.as_posix(),
            prediction_artifact_sha256=_sha256(args.predictions),
            scorer=args.scorer,
            scorer_source_path=args.scorer_source.as_posix(),
            scorer_source_sha256=(
                _sha256(args.scorer_source)
                if args.scorer_source.is_file()
                else "UNRESOLVED"
            ),
            command="python scripts/evaluate_predictions.py " + " ".join(sys.argv[1:]),
        )
        report = evaluate_records(references, predictions, options)
        write_report(report, args.output, overwrite=args.overwrite)
    except (
        AlignmentError,
        InputFormatError,
        FileExistsError,
        OSError,
        SourceScorerDependencyError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "run_id": run_id,
                "split": report.artifact["split"],
                "split_policy": report.artifact["split_policy"],
                "counts": report.artifact["counts"],
                "metrics": report.artifact["metrics"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
