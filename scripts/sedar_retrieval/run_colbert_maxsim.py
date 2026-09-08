#!/usr/bin/env python3
"""Rerank a hybrid top-100 with late-interaction MaxSim (TASK 26).

Reads a ranking JSONL (the fused list, or the cross-encoder's output if you are
stacking the two), rescores the head with the ColBERT MaxSim score, merges
corpus v4 ``article_part`` hits onto their parent ``Điều``, and writes a ranking
in the same shape plus a separate scores file.

Candidates past ``--top-k`` keep their retriever order below the reranked head
rather than being dropped, so recall at deeper cutoffs is unchanged.

Two things the report is for. First, the **score scale**: a summed ColBERT score
lives on roughly ``[0, query_length]`` while a BGE-M3-style averaged score lives
on ``[-1, 1]``. Any downstream threshold (the adaptive pack's ``score_floor``) or
fixed-weight fusion has to be recalibrated per scale, and the artifact records
which one produced the file. Second, the **document embedding cache**: pass
``--cache`` to persist per-unit token vectors so a sweep re-encodes nothing.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from legal_rag.questions import load_inference_questions
from legal_rag.sedar_retrieval.ranking.colbert_maxsim import (
    COLBERT_MAXSIM_SCHEMA_VERSION,
    BgeM3ColbertBackend,
    ColbertMaxsimConfig,
    DocEmbeddingCache,
    PylateColbertBackend,
    rerank_with_maxsim,
)
from legal_rag.sedar_retrieval.ranking.vietnamese_reranker import (
    RerankedCandidate,
    load_rerank_units,
    merge_children_to_parent,
    unit_text,
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


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(len(ordered) * fraction))], 4)

    return {"p05": at(0.05), "p25": at(0.25), "p50": at(0.5), "p75": at(0.75), "p95": at(0.95)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", default="warmup")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--backend", choices=("pylate", "bge_m3"), default="pylate")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--document-length", type=int, default=512)
    parser.add_argument("--reduction", choices=("sum", "mean"), default="sum")
    parser.add_argument(
        "--pool-factor",
        type=int,
        default=1,
        help=(
            "Token pooling at encode time. Factor 2 measured at 100.62%% of "
            "unpooled quality, i.e. free; 3 holds 99.03%%."
        ),
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-field", choices=("reader_text", "raw_text", "breadcrumb_reader_text"), default="reader_text")
    parser.add_argument("--aggregate", choices=("max", "mean"), default="max")
    parser.add_argument("--no-merge-children", action="store_true")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--no-quantize-cache", action="store_true")
    parser.add_argument("--allow-hub", action="store_true")
    parser.add_argument("--allow-base-encoder", action="store_true")
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
                f"{len(missing)} manifest IDs are absent from --input (e.g. {missing[:5]})"
            )
        rankings = {key: value for key, value in rankings.items() if key in included}

    questions = {
        item.id: item.question
        for item in load_inference_questions(args.questions, split=args.split)
    }
    units = load_rerank_units(args.units)
    texts = {
        unit_id: unit_text(unit, text_field=args.text_field)
        for unit_id, unit in units.items()
    }

    config = ColbertMaxsimConfig(
        model=args.model,
        revision=args.model_revision,
        backend=args.backend,
        dim=args.dim,
        query_length=args.query_length,
        document_length=args.document_length,
        reduction=args.reduction,
        pool_factor=args.pool_factor,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
        quantize_cache=not args.no_quantize_cache,
        require_finetuned=not args.allow_base_encoder,
        local_files_only=not args.allow_hub,
    )
    backend = (
        PylateColbertBackend(config)
        if args.backend == "pylate"
        else BgeM3ColbertBackend(config)
    )

    cache: DocEmbeddingCache | None = None
    if args.cache is not None:
        cache = (
            DocEmbeddingCache.load(args.cache)
            if args.cache.is_file()
            else DocEmbeddingCache(dim=args.dim, quantize=config.quantize_cache)
        )
        if cache.dim != args.dim:
            raise SystemExit(
                f"Cache dim {cache.dim} does not match --dim {args.dim}; "
                "delete the cache or fix the flag"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores_handle = None
    if args.scores_output:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        scores_handle = args.scores_output.open("w", encoding="utf-8")

    top_scores: list[float] = []
    merged_queries = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for query_id in sorted(rankings):
            question = questions.get(query_id)
            if question is None:
                raise SystemExit(f"No question text for query_id={query_id!r}")
            candidate_ids = [uid for uid in rankings[query_id] if uid in units]
            if not candidate_ids:
                raise SystemExit(
                    f"Query {query_id!r} has no candidate present in --units"
                )
            scored = rerank_with_maxsim(
                question,
                candidate_ids,
                texts,
                backend=backend,
                config=config,
                cache=cache,
            )
            head = [
                RerankedCandidate(
                    unit_id=unit_id,
                    document_id=units[unit_id].document_id,
                    score=score,
                    rank=0,
                    retriever_rank=candidate_ids.index(unit_id) + 1,
                )
                for unit_id, score in scored
            ]
            if args.no_merge_children:
                ranked = tuple(
                    RerankedCandidate(
                        unit_id=item.unit_id,
                        document_id=item.document_id,
                        score=item.score,
                        rank=index,
                        retriever_rank=item.retriever_rank,
                    )
                    for index, item in enumerate(head, start=1)
                )
            else:
                ranked = merge_children_to_parent(head, units, aggregate=args.aggregate)
            if any(item.merged_from for item in ranked):
                merged_queries += 1
            if ranked:
                top_scores.append(ranked[0].score)

            tail = [
                unit_id
                for unit_id in candidate_ids[args.top_k :]
                if unit_id not in {item.unit_id for item in ranked}
            ]
            handle.write(
                json.dumps(
                    {
                        "schema_version": COLBERT_MAXSIM_SCHEMA_VERSION,
                        "query_id": query_id,
                        "reranker": config.as_dict(),
                        "ranked_ids": [item.unit_id for item in ranked] + tail,
                        "reranked_count": len(ranked),
                        "tail_count": len(tail),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if scores_handle is not None:
                scores_handle.write(
                    json.dumps(
                        {
                            "schema_version": COLBERT_MAXSIM_SCHEMA_VERSION,
                            "query_id": query_id,
                            "scores": [item.as_dict() for item in ranked],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    if scores_handle is not None:
        scores_handle.close()
    if cache is not None and args.cache is not None:
        cache.save(args.cache)

    report = {
        "schema_version": COLBERT_MAXSIM_SCHEMA_VERSION,
        "reranker": config.as_dict(),
        "queries": len(rankings),
        "text_field": args.text_field,
        "aggregate": args.aggregate,
        "merge_children": not args.no_merge_children,
        "queries_with_merged_children": merged_queries,
        "top_score": {
            "mean": round(statistics.mean(top_scores), 4) if top_scores else 0.0,
            **_percentiles(top_scores),
        },
        # Reminder for whoever sets the adaptive pack's score_floor next: it is
        # scale-specific, and this file's scale is recorded above.
        "score_scale": config.score_scale,
        "cache": {
            "path": str(args.cache) if args.cache else None,
            "units": len(cache) if cache else 0,
            "bytes": cache.nbytes() if cache else 0,
        },
        "output": str(args.output),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
