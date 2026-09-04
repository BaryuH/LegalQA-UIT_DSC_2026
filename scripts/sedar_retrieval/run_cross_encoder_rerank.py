#!/usr/bin/env python3
"""Rerank an existing ranking with a cross-encoder (R3).

Reads a ranking JSONL, rescores the first --top-k candidates of each query with
a cross-encoder, and writes a ranking JSONL in the same shape. Candidates past
--top-k keep their original order rather than being dropped, so deeper recall is
unchanged and the output is a drop-in replacement downstream.

The scores are also written separately so they can become LTR features
(ce_score / ce_rank) without rerunning the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.questions import load_inference_questions
from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines
from legal_rag.sedar_retrieval.ranking.cross_encoder import (
    CROSS_ENCODER_SCHEMA_VERSION,
    CrossEncoderScorer,
    rerank_rankings,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl


def _require_input_file(path: Path, flag: str) -> Path:
    if str(path) in {"", "."}:
        raise SystemExit(
            f"{flag} resolved to an empty path. An unset shell variable expands "
            f"to '' and Path('') is '.'. Export it and check with 'ls -l'."
        )
    if path.is_dir():
        raise SystemExit(f"{flag} is a directory, expected a file: {path}")
    if not path.is_file():
        raise SystemExit(f"{flag} does not exist: {path}")
    return path


def _load_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for line in iter_jsonl_lines(path):
        row = json.loads(line)
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise SystemExit(f"Ranking row without query_id in {path}")
        ranked = row.get("ranked_ids")
        if not isinstance(ranked, list) or not ranked:
            raise SystemExit(f"Query {query_id!r} has no ranked_ids in {path}")
        if query_id in rankings:
            raise SystemExit(f"Duplicate query_id {query_id!r} in {path}")
        rankings[query_id] = [str(item) for item in ranked]
    if not rankings:
        raise SystemExit(f"No ranking rows found in {path}")
    return rankings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", default="warmup")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional clean-warmup manifest: rerank only those IDs.",
    )
    parser.add_argument("--model", required=True, help="Repo id or local directory.")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--text-source",
        choices=("reader_text", "raw_text"),
        default="reader_text",
        help="Never retrieval_text: its R2a wrapper repeats the document name.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, default=None)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0 or args.batch_size <= 0 or args.max_length <= 0:
        raise SystemExit("--top-k, --batch-size and --max-length must be positive")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite artifact: {args.output}")

    _require_input_file(args.input, "--input")
    _require_input_file(args.passages, "--passages")
    _require_input_file(args.questions, "--questions")

    rankings = _load_rankings(args.input)
    if args.manifest is not None:
        _require_input_file(args.manifest, "--manifest")
        from legal_rag.finetuned_reader.warmup_eval import load_clean_warmup_manifest

        included = set(load_clean_warmup_manifest(args.manifest).included_ids)
        missing = sorted(included - rankings.keys())
        if missing:
            raise SystemExit(
                f"{len(missing)} manifest IDs are absent from --input "
                f"(e.g. {missing[:5]})."
            )
        rankings = {k: v for k, v in rankings.items() if k in included}

    questions = {
        item.id: item.question
        for item in load_inference_questions(args.questions, split=args.split)
    }
    passages = {p.passage_id: p for p in load_passages_jsonl(str(args.passages))}

    scorer = CrossEncoderScorer(
        model=args.model,
        device=args.device,
        max_length=args.max_length,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
    )
    results = rerank_rankings(
        rankings,
        questions,
        passages,
        score_fn=scorer,
        top_k=args.top_k,
        text_source=args.text_source,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(
                json.dumps(
                    {"query_id": row.query_id, "ranked_ids": list(row.ranked_ids)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    if args.scores_output is not None:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        with args.scores_output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in results:
                handle.write(
                    json.dumps(
                        {"query_id": row.query_id, "scores": list(row.scores)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    payload = {
        "schema_version": CROSS_ENCODER_SCHEMA_VERSION,
        "run_id": new_run_id("cross_encoder_rerank"),
        "git_commit": git_commit_sha(),
        "input": str(args.input),
        "passages": str(args.passages),
        "model": args.model,
        "model_revision": args.model_revision,
        "top_k": args.top_k,
        "text_source": args.text_source,
        "max_length": args.max_length,
        "device": args.device,
        "local_files_only": bool(args.local_files_only),
        "n_queries": len(results),
        "reranked_per_query": results[0].reranked_count if results else 0,
        "output": str(args.output),
    }
    if args.manifest_out is not None:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": "PASS", **payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
