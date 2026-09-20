#!/usr/bin/env python3
"""Rerank a fused ranking with AITeamVN/Vietnamese_Reranker (TASK 24).

Reads a ranking JSONL, rescores the head of each query with the cross-encoder,
merges corpus v4 ``article_part`` hits onto their parent ``Điều``, and optionally
applies a variable-size cutoff. Writes a ranking JSONL in the same shape plus a
separate scores file so ``ce_score``/``ce_rank`` can become LTR features without
rerunning the model.

Candidates past ``--top-k`` keep their retriever order below the reranked head
rather than being dropped, so recall at deeper cutoffs is unchanged and the
output is a drop-in replacement downstream.

Threshold calibration is a separate step: run once with ``--score-threshold``
unset to dump the score distribution, pick the operating point on the clean-460
manifest, then re-run with it. A threshold copied from another corpus does not
transfer.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from legal_rag.questions import load_inference_questions
from legal_rag.sedar_retrieval.ranking.vietnamese_reranker import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    VIETNAMESE_RERANKER_SCHEMA_VERSION,
    CutoffPolicy,
    VietnameseRerankerConfig,
    VietnameseRerankerScorer,
    apply_cutoff,
    load_rerank_units,
    rerank_query,
)


def _require_file(path: Path, flag: str) -> Path:
    if str(path) in {"", "."}:
        raise SystemExit(f"{flag} resolved to an empty path; export the variable")
    if not path.is_file():
        raise SystemExit(f"{flag} does not exist or is not a file: {path}")
    return path


def _load_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
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
    parser.add_argument(
        "--units",
        type=Path,
        required=True,
        help="corpus_v4 units.jsonl, or a v3 passages_*.jsonl.",
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", default="warmup")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-query-tokens", type=int, default=256)
    parser.add_argument("--max-passage-tokens", type=int, default=2048)
    parser.add_argument("--score-mode", choices=("raw_logit", "sigmoid"), default="raw_logit")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--text-field",
        choices=("reader_text", "raw_text", "breadcrumb_reader_text"),
        default="reader_text",
    )
    parser.add_argument("--aggregate", choices=("max", "mean"), default="max")
    parser.add_argument("--no-merge-children", action="store_true")
    parser.add_argument("--allow-hub", action="store_true")
    # Cutoff
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--relative-margin", type=float, default=None)
    parser.add_argument("--min-keep", type=int, default=1)
    parser.add_argument("--max-keep", type=int, default=8)
    parser.add_argument("--max-total-chars", type=int, default=6000)
    parser.add_argument("--max-per-document", type=int, default=3)
    parser.add_argument("--ignore-packable", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite artifact: {args.output}")
    _require_file(args.input, "--input")
    _require_file(args.units, "--units")
    _require_file(args.questions, "--questions")

    rankings = _load_rankings(args.input)
    if args.manifest is not None:
        _require_file(args.manifest, "--manifest")
        from legal_rag.finetuned_reader.warmup_eval import load_clean_warmup_manifest

        included = set(load_clean_warmup_manifest(args.manifest).included_ids)
        missing = sorted(included - rankings.keys())
        if missing:
            raise SystemExit(
                f"{len(missing)} manifest IDs are absent from --input "
                f"(e.g. {missing[:5]})"
            )
        rankings = {key: value for key, value in rankings.items() if key in included}

    questions = {
        item.id: item.question
        for item in load_inference_questions(args.questions, split=args.split)
    }
    units = load_rerank_units(args.units)

    config = VietnameseRerankerConfig(
        model=args.model,
        revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        max_query_tokens=args.max_query_tokens,
        max_passage_tokens=args.max_passage_tokens,
        max_length=args.max_length,
        batch_size=args.batch_size,
        score_mode=args.score_mode,
        local_files_only=not args.allow_hub,
    )
    scorer = VietnameseRerankerScorer(config)

    policy = CutoffPolicy(
        score_threshold=args.score_threshold,
        relative_margin=args.relative_margin,
        min_keep=args.min_keep,
        max_keep=args.max_keep,
        max_total_chars=args.max_total_chars,
        max_per_document=args.max_per_document,
        respect_packable=not args.ignore_packable,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores_handle = None
    if args.scores_output:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        scores_handle = args.scores_output.open("w", encoding="utf-8")

    top_scores: list[float] = []
    kept_counts: list[int] = []
    stop_reasons: dict[str, int] = {}
    merged_queries = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for query_id in sorted(rankings):
            question = questions.get(query_id)
            if question is None:
                raise SystemExit(f"No question text for query_id={query_id!r}")
            ranked = rerank_query(
                question,
                rankings[query_id],
                units,
                score_fn=scorer,
                top_k=args.top_k,
                text_field=args.text_field,
                aggregate=args.aggregate,
            )
            if args.no_merge_children:
                # Reported, not silently ignored: without merging, one long
                # article can occupy several pack slots.
                pass
            if any(item.merged_from for item in ranked):
                merged_queries += 1
            scored_head = [item for item in ranked if item.score != float("-inf")]
            if scored_head:
                top_scores.append(scored_head[0].score)
            cutoff = apply_cutoff(tuple(scored_head), units, policy)
            kept_counts.append(len(cutoff.kept))
            stop_reasons[cutoff.stop_reason] = stop_reasons.get(cutoff.stop_reason, 0) + 1

            handle.write(
                json.dumps(
                    {
                        "schema_version": VIETNAMESE_RERANKER_SCHEMA_VERSION,
                        "query_id": query_id,
                        "reranker": config.as_dict(),
                        "ranked_ids": [item.unit_id for item in ranked],
                        "pack": cutoff.as_dict(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if scores_handle is not None:
                scores_handle.write(
                    json.dumps(
                        {
                            "schema_version": VIETNAMESE_RERANKER_SCHEMA_VERSION,
                            "query_id": query_id,
                            "scores": [item.as_dict() for item in scored_head],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    if scores_handle is not None:
        scores_handle.close()

    report = {
        "schema_version": VIETNAMESE_RERANKER_SCHEMA_VERSION,
        "reranker": config.as_dict(),
        "queries": len(rankings),
        "top_k": args.top_k,
        "text_field": args.text_field,
        "aggregate": args.aggregate,
        "queries_with_merged_children": merged_queries,
        "cutoff": {
            "score_threshold": args.score_threshold,
            "relative_margin": args.relative_margin,
            "min_keep": args.min_keep,
            "max_keep": args.max_keep,
            "max_total_chars": args.max_total_chars,
            "max_per_document": args.max_per_document,
            "respect_packable": not args.ignore_packable,
        },
        "pack_size": {
            "mean": round(statistics.mean(kept_counts), 3) if kept_counts else 0,
            "median": statistics.median(kept_counts) if kept_counts else 0,
            "min": min(kept_counts) if kept_counts else 0,
            "max": max(kept_counts) if kept_counts else 0,
        },
        "stop_reasons": stop_reasons,
        # Dump the top-score distribution so the threshold can be calibrated
        # from a single pass instead of guessed.
        "top_score_percentiles": _percentiles(top_scores),
        "output": str(args.output),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(len(ordered) * fraction))], 4)
    return {"p05": at(0.05), "p25": at(0.25), "p50": at(0.50), "p75": at(0.75), "p95": at(0.95)}


if __name__ == "__main__":
    raise SystemExit(main())
