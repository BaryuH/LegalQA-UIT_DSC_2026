#!/usr/bin/env python3
"""SS-05B CLI: build SEDAR-SFT dataset from precomputed LTR rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.finetuned_reader.dataset import DatasetBuildError
from legal_rag.sedar_sft.ltr_dataset import (
    LTR_EVIDENCE_SOURCE,
    LtrDatasetBuildConfig,
    build_sedar_sft_dataset_from_ltr,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sedar_sft_train_ltr.yaml"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--rankings",
        type=Path,
        required=True,
        help="LTR ranked retrieval JSONL (query_id + ranked_ids).",
    )
    parser.add_argument(
        "--passages",
        type=Path,
        required=True,
        help="Canonical passages view JSONL (e.g. passages_r2a.jsonl).",
    )
    parser.add_argument(
        "--retrieval-variant",
        default="ltr_full_all",
        help="Label recorded in the dataset manifest.",
    )
    parser.add_argument("--evidence-top-k", type=int, default=4)
    parser.add_argument("--max-total-chars", type=int, default=4000)
    parser.add_argument("--max-chunks-per-document", type=int, default=2)
    # Renderer flags. Names and defaults mirror
    # scripts/sedar_retrieval/run_sedar_e2e.py exactly, so a training dataset
    # can be built with the same renderer the reader will meet at inference.
    # Defaults are the historical v1 behaviour; the deployed champion needs
    #   --dedup-article-mode article --include-document-name
    # on top of --evidence-top-k 6 --max-total-chars 6000
    # --max-chunks-per-document 3.
    parser.add_argument(
        "--candidate-window",
        type=int,
        default=0,
        help=(
            "Ranked candidates handed to the packer. 0 = same as "
            "--evidence-top-k (historical: no backfill after the per-document "
            "cap and character budget drop candidates)."
        ),
    )
    parser.add_argument(
        "--body-source",
        choices=("raw_text", "reader_text"),
        default="raw_text",
        help="Which passage field the reader is trained on.",
    )
    parser.add_argument(
        "--min-passage-chars",
        type=int,
        default=0,
        help="Drop candidate passages shorter than this before packing.",
    )
    parser.add_argument(
        "--include-document-name",
        action="store_true",
        help=(
            "Render the real document name on the 'Van ban' header line "
            "instead of the zip member. Required to match the champion."
        ),
    )
    parser.add_argument(
        "--dedup-article-mode",
        choices=("off", "article", "clause", "first"),
        default="off",
        help=(
            "Collapse candidates from the same article before packing. "
            "Champion uses 'article'."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional smoke limit; omit for full effective-train build.",
    )
    args = parser.parse_args()
    try:
        result = build_sedar_sft_dataset_from_ltr(
            args.config,
            repo_root=args.repo_root,
            ltr=LtrDatasetBuildConfig(
                rankings_path=args.rankings,
                passages_path=args.passages,
                retrieval_variant=args.retrieval_variant,
                evidence_top_k=args.evidence_top_k,
                max_total_chars=args.max_total_chars,
                max_chunks_per_document=args.max_chunks_per_document,
                candidate_window=args.candidate_window,
                body_source=args.body_source,
                dedup_article_mode=args.dedup_article_mode,
                include_document_name=args.include_document_name,
                min_passage_chars=args.min_passage_chars,
                progress_every=args.progress_every,
            ),
            max_examples=args.max_examples,
        )
    except (DatasetBuildError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "evidence_source": LTR_EVIDENCE_SOURCE,
                "example_count": result.example_count,
                "excluded_count": len(result.underlying.excluded),
                "failure_count": len(result.underlying.retrieval_failures),
                "dataset_dir": result.underlying.output_dir.as_posix(),
                "sedar_overlay_dir": result.output_dir.as_posix(),
                "manifest_path": result.manifest_path.as_posix(),
                "renderer": {
                    "candidate_window": args.candidate_window,
                    "body_source": args.body_source,
                    "dedup_article_mode": args.dedup_article_mode,
                    "include_document_name": args.include_document_name,
                    "min_passage_chars": args.min_passage_chars,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.example_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
