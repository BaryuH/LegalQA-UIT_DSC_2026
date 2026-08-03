"""Generate evaluation-only Markdown and CSV error reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

from legal_rag.evaluation import (  # noqa: E402
    ERROR_TYPES,
    ErrorReportError,
    generate_error_report,
    write_error_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed error-report CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Join evaluation references, predictions, metrics, and retrieval "
            "artifacts into Markdown and CSV error reports."
        )
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--retrieval", type=Path, default=None)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path, dest="csv_path")
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--error-types",
        type=Path,
        default=None,
        help="JSON object mapping case IDs to manually reviewed taxonomy codes.",
    )
    parser.add_argument(
        "--default-error-type",
        choices=tuple(sorted(ERROR_TYPES)),
        default="OTHER",
    )
    parser.add_argument("--top-evidence-k", type=int, default=3)
    parser.add_argument("--preview-chars", type=int, default=320)
    parser.add_argument(
        "--allow-private",
        action="store_true",
        help="Explicitly authorize a private evaluation report.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_manual_types(path: Path | None) -> Mapping[str, str]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ErrorReportError(f"Unable to read manual error types: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ErrorReportError("Manual error types must be a JSON object")
    result: dict[str, str] = {}
    for identifier, error_type in payload.items():
        if not isinstance(identifier, str) or not isinstance(error_type, str):
            raise ErrorReportError("Manual error type mapping must contain strings")
        result[identifier] = error_type
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Generate both report formats and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        report = generate_error_report(
            args.predictions,
            args.references,
            args.metrics,
            retrieval_path=args.retrieval,
            split=args.split,
            manual_error_types=_load_manual_types(args.error_types),
            default_error_type=args.default_error_type,
            top_evidence_k=args.top_evidence_k,
            preview_chars=args.preview_chars,
            allow_private=args.allow_private,
        )
        write_error_report(
            report,
            args.markdown,
            args.csv_path,
            overwrite=args.overwrite,
        )
    except (ErrorReportError, FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "csv": args.csv_path.as_posix(),
                "markdown": args.markdown.as_posix(),
                "run_id": report.run_id,
                "cases": len(report.cases),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
