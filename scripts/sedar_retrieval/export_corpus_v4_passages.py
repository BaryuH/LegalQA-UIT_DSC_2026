#!/usr/bin/env python3
"""Export a corpus v4 build into the v3 passage view (TASK 27).

This is the bridge that unblocks everything downstream. Every index builder,
retrieval runner and evaluator in the repo reads a passage view through
``load_passages_jsonl``, which validates rows as ``CanonicalPassage``
(``extra="forbid"``). A v4 unit does not satisfy that model, so before this
script a v4 build could not be indexed, retrieved against, or scored.

What it produces is a drop-in replacement for
``artifacts/sedar_retrieval/views/*/passages_r2a.jsonl``: pass it to
``build_bm25_index.py --passages``, ``build_dense_index.py --passages``,
``run_*_rerank.py --passages`` and ``eval_retrieval.py --passages`` unchanged.

``--labels`` runs the check that decides whether the export is usable: what
share of the article ids the existing silver labels reference actually exist in
the exported view. Main-text articles keep the historical
``{doc}::art::{n}`` spelling precisely so that number comes out high without
re-projecting the labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from legal_rag.sedar_retrieval.corpus_v4.export import (
    ExportCounters,
    build_article_id_map,
    export_units,
)


def _iter_units(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_label_article_ids(path: Path) -> tuple[set[str], int]:
    ids: set[str] = set()
    queries = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            relevant = row.get("relevant_ids")
            if isinstance(relevant, list) and relevant:
                queries += 1
                ids.update(str(item) for item in relevant)
    return ids, queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--levels",
        default="article,article_part,block",
        help=(
            "v4 levels to export. Drop article_part for an article-only view "
            "(the arm that tests whether children help at all)."
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Silver labels JSONL: reports article-id coverage of the export.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.units.is_file():
        raise SystemExit(f"--units does not exist: {args.units}")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite artifact: {args.output}")

    keep_levels = {item.strip() for item in args.levels.split(",") if item.strip()}
    # Pre-pass: which (document, article number) pairs are ambiguous. Only those
    # get an instrument-scoped id; everything else keeps the flat spelling the
    # existing silver labels use.
    article_id_map = build_article_id_map(_iter_units(args.units))
    ambiguous = sum(1 for value in article_id_map.values() if value > 1)
    counters = ExportCounters()
    article_ids: set[str] = set()
    level_filtered = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        def source():
            nonlocal level_filtered
            for unit in _iter_units(args.units):
                if str(unit.get("level") or "") not in keep_levels:
                    level_filtered += 1
                    continue
                yield unit

        for row in export_units(
            source(), counters=counters, article_id_map=article_id_map
        ):
            if row.get("article_id"):
                article_ids.add(str(row["article_id"]))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report: dict[str, Any] = {
        "schema_version": "sedar-corpus-v4-passage-export-v1",
        "units_path": str(args.units),
        "output": str(args.output),
        "levels_kept": sorted(keep_levels),
        "units_filtered_by_level_flag": level_filtered,
        "counters": counters.as_dict(),
        "article_numbers_seen": len(article_id_map),
        "ambiguous_article_numbers": ambiguous,
        "distinct_article_ids": len(article_ids),
    }

    if args.labels is not None:
        if not args.labels.is_file():
            raise SystemExit(f"--labels does not exist: {args.labels}")
        label_ids, labeled_queries = _load_label_article_ids(args.labels)
        present = {item for item in label_ids if item in article_ids}
        missing = sorted(label_ids - present)
        report["label_compatibility"] = {
            "labels_path": str(args.labels),
            "labeled_queries": labeled_queries,
            "distinct_label_ids": len(label_ids),
            "present_in_export": len(present),
            "coverage": round(len(present) / max(1, len(label_ids)), 4),
            "missing_examples": missing[:15],
            "missing_shape": dict(
                Counter(
                    "::i" in item and "annex_scoped" or "main_text" for item in missing
                )
            ),
        }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
