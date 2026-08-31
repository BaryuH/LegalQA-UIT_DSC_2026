#!/usr/bin/env python3
"""Validate a champion warmup run and audit retrieval recall by rank."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _SRC_ROOT = _REPO_ROOT / "src"
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

from legal_rag.sedar_retrieval.eval.retrieval_recall_audit import (  # noqa: E402
    DEFAULT_CUTOFFS,
    RetrievalRecallAuditError,
    audit_warmup_retrieval_recall,
)


def _parse_cutoffs(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"cutoff must be an integer: {item!r}"
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"cutoff must be positive: {value}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one cutoff is required")
    return tuple(sorted(set(values)))


def build_parser() -> argparse.ArgumentParser:
    """Build the retrieval recall audit CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Champion TASK20 run directory.",
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--bm25", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument(
        "--cutoffs",
        type=_parse_cutoffs,
        default=DEFAULT_CUTOFFS,
        help="Comma-separated source cutoffs; default: 10,20,50,100,200,500.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _curve_summary(payload: Mapping[str, object]) -> dict[str, object]:
    raw_curves = payload.get("recall_curves")
    if not isinstance(raw_curves, Mapping):
        return {}
    summary: dict[str, object] = {}
    for source_name, raw_entries in raw_curves.items():
        if not isinstance(source_name, str) or not isinstance(raw_entries, list):
            continue
        source_summary: dict[str, object] = {}
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                continue
            cutoff = raw_entry.get("cutoff")
            if not isinstance(cutoff, int):
                continue
            source_summary[str(cutoff)] = {
                "provision_recall": raw_entry.get("provision_recall"),
                "article_recall": raw_entry.get("article_recall"),
                "is_lower_bound": raw_entry.get("is_lower_bound"),
            }
        summary[source_name] = source_summary
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the audit and write one evaluation-only JSON artifact."""

    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.force:
        print(f"Refusing to overwrite artifact: {args.output}", file=sys.stderr)
        return 2
    try:
        report = audit_warmup_retrieval_recall(
            run_dir=args.run_dir,
            metrics_path=args.metrics,
            bm25_path=args.bm25,
            qwen_path=args.qwen,
            labels_path=args.labels,
            passages_path=args.passages,
            cutoffs=args.cutoffs,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = report.as_dict()
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RetrievalRecallAuditError) as exc:
        print(f"RETRIEVAL_RECALL_AUDIT_FAILED: {exc}", file=sys.stderr)
        return 2

    warnings = payload.get("warnings", [])
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    status = "PASS_WITH_WARNINGS" if warning_count else "PASS"
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "query_count": payload["query_count"],
                "labeled_query_count": payload["labeled_query_count"],
                "cutoffs": payload["cutoffs"],
                "warnings": warning_count,
                "recall_summary": _curve_summary(payload),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
