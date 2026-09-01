#!/usr/bin/env python3
"""Audit silver-label document/article scope without writing answer text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from legal_rag.sedar_retrieval.eval.silver_label_audit import (
    SilverLabelAuditError,
    audit_silver_labels,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional champion run used to restrict the audit to clean query IDs.",
    )
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        print(f"Refusing to overwrite artifact: {args.output}", file=sys.stderr)
        return 2
    try:
        report = audit_silver_labels(
            labels_path=args.labels,
            passages_path=args.passages,
            run_dir=args.run_dir,
            sample_size=args.sample_size,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = report.as_dict()
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, SilverLabelAuditError) as exc:
        print(f"SILVER_LABEL_AUDIT_FAILED: {exc}", file=sys.stderr)
        return 2

    warnings = payload.get("warnings", [])
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    print(
        json.dumps(
            {
                "status": "PASS_WITH_WARNINGS" if warning_count else "PASS",
                "output": str(args.output),
                "clean_query_count": payload["clean_query_count"],
                "silver_query_count": payload["silver_query_count"],
                "unlabeled_query_count": payload["unlabeled_query_count"],
                "unresolved_with_article_query_count": payload[
                    "unresolved_with_article_query_count"
                ],
                "resolution_reason_counts": payload["resolution_reason_counts"],
                "unresolved_query_reason_counts": payload[
                    "unresolved_query_reason_counts"
                ],
                "multi_document_query_count": payload["multi_document_query_count"],
                "missing_passage_id_count": payload["missing_passage_id_count"],
                "warnings": warning_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
