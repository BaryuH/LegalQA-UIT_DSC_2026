#!/usr/bin/env python3
"""Build leakage-safe LTR feature rows for TASK 12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.ranking.features import FEATURE_NAMES
from legal_rag.sedar_retrieval.ranking.ltr_dataset import (
    LTR_DATASET_SCHEMA_VERSION,
    LTRFeatureBuildConfig,
    LTRFeatureBuildError,
    build_ltr_feature_rows,
    group_feature_rows,
    load_question_map,
    load_rrf_candidates,
    load_synthetic_query_map,
    validate_feature_schema,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: tuple[dict[str, object], ...] | tuple[dict[str, object], ...],
    *,
    force: bool,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=("train", "warmup", "public", "private"),
        default="warmup",
    )
    parser.add_argument("--synthetic", type=Path, default=None)
    parser.add_argument(
        "--label-source",
        choices=("citation", "positive_passage_id"),
        default="citation",
    )
    parser.add_argument(
        "--source-split",
        default="train",
        help="Synthetic source_split filter when label-source=positive_passage_id.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("configs/retrieval/ltr_feature_schema_v1.json"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--groups-output", type=Path, default=None)
    parser.add_argument(
        "--label-mode",
        choices=("binary", "graded"),
        default="graded",
    )
    parser.add_argument(
        "--unlabeled-policy",
        choices=("fail", "skip"),
        default="skip",
    )
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_candidates < 0:
        raise SystemExit("--max-candidates must be non-negative")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.label_source == "citation" and args.questions is None:
        raise SystemExit("--questions is required when --label-source=citation")
    if args.label_source == "positive_passage_id" and args.synthetic is None:
        raise SystemExit(
            "--synthetic is required when --label-source=positive_passage_id"
        )

    try:
        schema_hash = validate_feature_schema(args.schema)
        candidates = load_rrf_candidates(args.candidates)
        if args.limit:
            candidates = {
                query_id: candidates[query_id]
                for query_id in sorted(candidates)[: args.limit]
            }
        passages = load_passages_jsonl(str(args.passages))
        passage_map = {passage.passage_id: passage for passage in passages}
        if len(passage_map) != len(passages):
            raise LTRFeatureBuildError("Passage corpus contains duplicate IDs")

        config = LTRFeatureBuildConfig(
            label_mode=args.label_mode,
            label_source=args.label_source,
            unlabeled_policy=args.unlabeled_policy,
            max_candidates=args.max_candidates,
        )
        if args.label_source == "citation":
            assert args.questions is not None
            questions = load_question_map(args.questions, split=args.split)
            rows, report = build_ltr_feature_rows(
                candidates=candidates,
                questions=questions,
                passages=passage_map,
                config=config,
            )
            label_provenance = "query_citation_heuristic"
            leakage_source = "question_citations_only"
        else:
            assert args.synthetic is not None
            synthetic_queries = load_synthetic_query_map(
                args.synthetic,
                source_split=args.source_split,
            )
            rows, report = build_ltr_feature_rows(
                candidates=candidates,
                synthetic_queries=synthetic_queries,
                passages=passage_map,
                config=config,
            )
            label_provenance = "positive_passage_id"
            leakage_source = "synthetic_positive_passage_id_only"

        if not rows:
            raise LTRFeatureBuildError("No labeled feature rows were produced")
    except (OSError, ValueError, LTRFeatureBuildError) as exc:
        raise SystemExit(f"TASK12_FEATURE_BUILD_FAILED: {exc}") from exc

    output_path = args.output
    manifest_path = args.manifest or output_path.with_suffix(".manifest.json")
    groups_path = args.groups_output or output_path.with_suffix(".groups.jsonl")
    groups = group_feature_rows(rows)
    _write_jsonl(output_path, rows, force=args.force)
    _write_jsonl(groups_path, groups, force=args.force)
    manifest = {
        "schema_version": LTR_DATASET_SCHEMA_VERSION,
        "task": "TASK12",
        "run_id": new_run_id("ltr_features"),
        "git_commit": git_commit_sha(Path.cwd()),
        "status": "PASS",
        "candidates_path": str(args.candidates),
        "passages_path": str(args.passages),
        "questions_path": str(args.questions) if args.questions else None,
        "synthetic_path": str(args.synthetic) if args.synthetic else None,
        "question_split": args.split if args.label_source == "citation" else None,
        "source_split": args.source_split
        if args.label_source == "positive_passage_id"
        else None,
        "label_source": args.label_source,
        "label_mode": args.label_mode,
        "unlabeled_policy": args.unlabeled_policy,
        "max_candidates": args.max_candidates,
        "feature_schema_path": str(args.schema),
        "feature_schema_hash": schema_hash,
        "feature_names": list(FEATURE_NAMES),
        "artifacts": {
            "feature_rows": str(output_path),
            "feature_groups": str(groups_path),
        },
        "metrics": report.as_dict(),
        "leakage_policy": {
            "answer_text_used": False,
            "gold_labels_used_as_features": False,
            "reader_outputs_used": False,
            "source": leakage_source,
            "label_provenance": label_provenance,
        },
    }
    _write_json(manifest_path, manifest, force=args.force)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "queries": report.output_query_count,
                "rows": report.output_row_count,
                "skipped_unlabeled": report.skipped_unlabeled_query_count,
                "label_source": args.label_source,
                "output": str(output_path),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
