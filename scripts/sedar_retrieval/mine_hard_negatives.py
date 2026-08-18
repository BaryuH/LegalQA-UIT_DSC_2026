#!/usr/bin/env python3
"""Mine leakage-aware hard/semi-hard negatives for TASK 10."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from legal_rag.retrieval.bm25 import BM25Config, load_bm25_index
from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.retrieval.bm25_passages import (
    corpus_fingerprint,
    search_passages,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl
from legal_rag.sedar_retrieval.training.hard_negatives import (
    TASK10_SCHEMA_VERSION,
    CandidateHit,
    HardNegativeMiningConfig,
    build_audit_sample,
    load_candidate_hits,
    mine_hard_negatives,
    write_jsonl_models,
)
from legal_rag.sedar_retrieval.training.synthetic_queries import (
    load_synthetic_records,
)


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise SystemExit(f"--output-dir is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not force:
        raise SystemExit(
            f"Output directory is not empty: {path}; use a new run directory "
            "or pass --force explicitly"
        )
    path.mkdir(parents=True, exist_ok=True)


def _add_bm25_candidates(
    candidate_map: defaultdict[str, list[CandidateHit]],
    records: tuple[Any, ...],
    passages: tuple[Any, ...],
    *,
    cache_root: Path,
    corpus_hash: str,
    top_k: int,
) -> None:
    loaded = load_bm25_index(
        cache_root,
        corpus_hash,
        BM25Config(version="sedar-retrieval-v3-bm25"),
    )
    if loaded.index is None:
        raise SystemExit(
            "BM25 index missing for this corpus; run build_bm25_index.py first"
        )
    for record in records:
        hits = search_passages(loaded.index, record.query, top_k=top_k)
        candidate_map[record.synthetic_id].extend(
            CandidateHit(
                passage_id=hit.chunk_id,
                rank=hit.rank,
                score=float(hit.bm25_score),
                source="bm25",
            )
            for hit in hits
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bm25-cache-root",
        type=Path,
        default=None,
        help=(
            "Optional BM25 index root; candidates are mined directly "
            "for synthetic queries."
        ),
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        action="append",
        default=[],
        help="Existing ranked JSONL candidate file; repeat for dense/RRF sources.",
    )
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--min-negatives", type=int, default=2)
    parser.add_argument("--max-negatives", type=int, default=5)
    parser.add_argument("--audit-sample-size", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.audit_sample_size <= 0:
        raise SystemExit("--audit-sample-size must be positive")

    _prepare_output_dir(args.output_dir, force=args.force)
    passages = load_passages_jsonl(str(args.passages))
    records = load_synthetic_records(args.synthetic)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("Synthetic input contains no records")

    candidate_map: defaultdict[str, list[CandidateHit]] = defaultdict(list)
    candidate_sources: list[str] = []
    corpus_hash = corpus_fingerprint(passages)
    if args.candidate_file:
        loaded_candidates = load_candidate_hits(args.candidate_file)
        for query_id, hits in loaded_candidates.items():
            candidate_map[query_id].extend(hits)
        candidate_sources.extend(str(path) for path in args.candidate_file)
    if args.bm25_cache_root is not None:
        _add_bm25_candidates(
            candidate_map,
            records,
            passages,
            cache_root=args.bm25_cache_root,
            corpus_hash=corpus_hash,
            top_k=args.top_k,
        )
        candidate_sources.append(f"bm25:{args.bm25_cache_root}")
    if not candidate_sources:
        raise SystemExit("Provide --bm25-cache-root or at least one --candidate-file")

    config = HardNegativeMiningConfig(
        min_negatives=args.min_negatives,
        max_negatives=args.max_negatives,
        seed=args.seed,
        random_rank=args.top_k,
    )
    config_payload = {
        "min_negatives": config.min_negatives,
        "max_negatives": config.max_negatives,
        "seed": config.seed,
        "random_rank": config.random_rank,
        "lexical_hard_threshold": config.lexical_hard_threshold,
        "lexical_false_negative_threshold": (config.lexical_false_negative_threshold),
        "dense_false_negative_threshold": config.dense_false_negative_threshold,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    mined, report = mine_hard_negatives(
        records,
        passages,
        candidate_map,
        config=config,
    )
    hard_path = args.output_dir / "hard_negatives.jsonl"
    rejection_path = args.output_dir / "rejections.jsonl"
    audit_path = args.output_dir / "audit_sample.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    write_jsonl_models(hard_path, mined, overwrite=args.force)

    rejection_rows: list[dict[str, object]] = []
    accepted_ids = {record.synthetic_id for record in mined}
    for record in records:
        if record.synthetic_id not in accepted_ids:
            rejection_rows.append(
                {
                    "synthetic_id": record.synthetic_id,
                    "reason": "not_accepted_by_miner",
                    "candidate_count": len(candidate_map.get(record.synthetic_id, ())),
                }
            )
    write_jsonl_models(rejection_path, rejection_rows, overwrite=args.force)
    audit_rows = build_audit_sample(
        mined,
        sample_size=args.audit_sample_size,
        seed=args.seed,
    )
    write_jsonl_models(audit_path, audit_rows, overwrite=args.force)

    gate_ready = (
        report.accepted_query_count == report.query_count
        and report.unresolved_candidate_count == 0
        and report.positive_in_negative_count == 0
        and report.harder_than_random
    )
    status = "NEEDS_MANUAL_AUDIT" if gate_ready else "FAIL"
    manifest = {
        "schema_version": TASK10_SCHEMA_VERSION,
        "task": "TASK10",
        "run_id": new_run_id("task10_hard_negatives"),
        "git_commit": git_commit_sha(Path.cwd()),
        "status": status,
        "manual_audit_required": True,
        "manual_audit_sample_size": len(audit_rows),
        "synthetic_path": str(args.synthetic),
        "passages_path": str(args.passages),
        "corpus_hash": corpus_hash,
        "candidate_sources": candidate_sources,
        "config": config_payload,
        "config_hash": config_hash,
        "top_k": args.top_k,
        "min_negatives": args.min_negatives,
        "max_negatives": args.max_negatives,
        "seed": args.seed,
        "artifacts": {
            "hard_negatives": str(hard_path),
            "rejections": str(rejection_path),
            "audit_sample": str(audit_path),
        },
        "metrics": report.as_dict(),
        "exit_gate": {
            "all_negatives_resolved": report.unresolved_candidate_count == 0,
            "positive_excluded": report.positive_in_negative_count == 0,
            "leakage_zero": report.positive_in_negative_count == 0,
            "harder_than_random": report.harder_than_random,
            "estimated_false_negative_rate": report.estimated_false_negative_rate,
            "estimated_false_negative_rate_target": 0.03,
            "manual_audit_required": True,
        },
    }
    _write_json(manifest_path, manifest, overwrite=args.force)
    print(
        json.dumps(
            {
                "status": status,
                "queries": report.query_count,
                "accepted_queries": report.accepted_query_count,
                "negatives": report.negative_count,
                "estimated_false_negative_rate": (report.estimated_false_negative_rate),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if status != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
