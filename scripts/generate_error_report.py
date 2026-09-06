"""Generate evaluation-only Markdown and CSV error reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

from legal_rag.evaluation import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    ERROR_TYPES,
    ClassifierThresholds,
    ErrorReportError,
    generate_error_report,
    write_error_report,
)
from legal_rag.sedar_retrieval.eval.ensemble_metrics import (  # noqa: E402
    load_relevance_labels,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import (  # noqa: E402
    load_passages_jsonl,
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
    parser.add_argument(
        "--auto-classify",
        action="store_true",
        help="Enable deterministic automatic error taxonomy classification.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Evaluation-only silver-label JSONL.",
    )
    parser.add_argument(
        "--passages",
        type=Path,
        default=None,
        help="Canonical passages JSONL used for article mapping.",
    )
    parser.add_argument(
        "--token-counts",
        type=Path,
        default=None,
        help="JSON object mapping case IDs to model-token prediction counts.",
    )
    parser.add_argument(
        "--reference-token-counts",
        type=Path,
        default=None,
        help=(
            "JSON object mapping case IDs to model-token reference counts. "
            "Counts only; never reference text. Required for the length-ratio "
            "rules (OVER_VERBOSE / UNDER_SPECIFIED)."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--hit-depth",
        type=int,
        default=None,
        help=(
            "How many leading retrieval hits count as 'within reach of the "
            "pack'. Required with --auto-classify: a full candidate list is "
            "hundreds deep, and projecting all of it makes RETRIEVAL_MISS and "
            "RERANKING_REGRESSION meaningless."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="JSON object overriding classifier thresholds.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Flat JSON error-type count output.",
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


def _load_json_object(path: Path, *, description: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ErrorReportError(f"Unable to read {description}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ErrorReportError(f"{description} must be a JSON object")
    return payload


def _load_token_counts(path: Path | None) -> Mapping[str, int] | None:
    if path is None:
        return None
    payload = _load_json_object(path, description="token counts")
    result: dict[str, int] = {}
    for identifier, count in payload.items():
        if (
            not isinstance(identifier, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ErrorReportError(
                "Token counts must map string IDs to non-negative integers"
            )
        result[identifier] = count
    return result


def _load_thresholds(path: Path | None) -> ClassifierThresholds:
    if path is None:
        return DEFAULT_THRESHOLDS
    payload = _load_json_object(path, description="classifier thresholds")
    allowed = {
        "over_verbose_ratio",
        "under_specified_ratio",
        "rouge_gap",
        "cap_tolerance",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ErrorReportError(
            "Unknown classifier threshold(s): " + ", ".join(sorted(unknown))
        )
    try:
        return ClassifierThresholds(
            over_verbose_ratio=cast(
                float,
                payload.get(
                    "over_verbose_ratio",
                    DEFAULT_THRESHOLDS.over_verbose_ratio,
                ),
            ),
            under_specified_ratio=cast(
                float,
                payload.get(
                    "under_specified_ratio",
                    DEFAULT_THRESHOLDS.under_specified_ratio,
                ),
            ),
            rouge_gap=cast(
                float,
                payload.get("rouge_gap", DEFAULT_THRESHOLDS.rouge_gap),
            ),
            cap_tolerance=cast(
                int,
                payload.get("cap_tolerance", DEFAULT_THRESHOLDS.cap_tolerance),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ErrorReportError(f"Invalid classifier thresholds: {path}") from exc


def _load_classifier_maps(
    labels_path: Path,
    passages_path: Path,
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]], dict[str, str]]:
    try:
        passages = load_passages_jsonl(str(passages_path))
        labels = load_relevance_labels(labels_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ErrorReportError("Unable to load classifier labels/passages") from exc

    passage_by_id = {}
    passage_article_keys: dict[str, str] = {}
    for passage in passages:
        if passage.passage_id in passage_by_id:
            raise ErrorReportError(
                f"Duplicate passage_id in classifier passages: {passage.passage_id}"
            )
        passage_by_id[passage.passage_id] = passage
        if passage.article_number:
            passage_article_keys[passage.passage_id] = (
                f"{passage.document_id}::art::{passage.article_number}"
            )

    gold_article_keys: dict[str, frozenset[str]] = {}
    gold_document_ids: dict[str, frozenset[str]] = {}
    for identifier, (relevant_ids, provenance) in labels.items():
        if provenance != "silver" or not relevant_ids:
            gold_article_keys[identifier] = frozenset()
            gold_document_ids[identifier] = frozenset()
            continue
        missing = sorted(set(relevant_ids) - set(passage_by_id))
        if missing:
            raise ErrorReportError(
                f"Silver labels for {identifier} contain unknown passage IDs: "
                + ", ".join(missing[:5])
            )
        gold_article_keys[identifier] = frozenset(
            passage_article_keys[passage_id]
            for passage_id in relevant_ids
            if passage_id in passage_article_keys
        )
        gold_document_ids[identifier] = frozenset(
            passage_by_id[passage_id].document_id for passage_id in relevant_ids
        )
    return gold_article_keys, gold_document_ids, passage_article_keys


def _write_summary(
    path: Path | None,
    summary: Mapping[str, int],
    *,
    overwrite: bool,
) -> None:
    if path is None:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"Summary already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(sorted(summary.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Generate both report formats and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        if args.auto_classify:
            missing = [
                name
                for name, value in (
                    ("--labels", args.labels),
                    ("--passages", args.passages),
                    ("--hit-depth", args.hit_depth),
                )
                if value is None
            ]
            if missing:
                raise ErrorReportError(
                    "--auto-classify requires " + " and ".join(missing)
                )
            assert args.labels is not None
            assert args.passages is not None
            (
                gold_article_keys,
                gold_document_ids,
                passage_article_keys,
            ) = _load_classifier_maps(args.labels, args.passages)
            classifier_thresholds = _load_thresholds(args.thresholds)
            prediction_token_counts = _load_token_counts(args.token_counts)
            reference_token_counts = _load_token_counts(args.reference_token_counts)
        else:
            gold_article_keys = None
            gold_document_ids = None
            passage_article_keys = None
            classifier_thresholds = None
            prediction_token_counts = None
            reference_token_counts = None
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
            auto_classify=args.auto_classify,
            classifier_thresholds=classifier_thresholds,
            gold_article_keys=gold_article_keys,
            gold_document_ids=gold_document_ids,
            passage_article_keys=passage_article_keys,
            prediction_token_counts=prediction_token_counts,
            reference_token_counts=reference_token_counts,
            max_new_tokens=args.max_new_tokens,
            hit_depth=args.hit_depth,
        )
        write_error_report(
            report,
            args.markdown,
            args.csv_path,
            overwrite=args.overwrite,
        )
        _write_summary(
            args.summary_out,
            report.classification_summary,
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
