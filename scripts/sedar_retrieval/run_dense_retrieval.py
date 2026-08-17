#!/usr/bin/env python3
"""Run Qwen3 dense retrieval over SEDAR questions (TASK 07)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from legal_rag.questions import load_inference_questions
from legal_rag.sedar_retrieval.retrieval.bm25_passages import corpus_fingerprint
from legal_rag.sedar_retrieval.retrieval.dense import (
    DEFAULT_QUERY_INSTRUCTION,
    DENSE_INDEX_TYPE,
    SentenceTransformerEncoder,
    format_instruct_query,
    load_dense_index,
    normalize_embedding_matrix,
    require_dense_encode,
    search_dense_index,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _write_latency(path: Path, values: list[float], *, n_queries: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_queries": n_queries,
        "unit": "milliseconds",
        "measurement": "amortized_batch_encode_plus_faiss_search",
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_predictions(
    path: Path,
    questions: tuple[Any, ...],
    hits_by_query: tuple[tuple[Any, ...], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for question, hits in zip(questions, hits_by_query, strict=True):
            handle.write(
                json.dumps(
                    {
                        "query_id": question.id,
                        "ranked_ids": [hit.passage_id for hit in hits],
                        "scores": [
                            {
                                "passage_id": hit.passage_id,
                                "dense": hit.score,
                                "rank": hit.rank,
                                "source": "dense",
                            }
                            for hit in hits
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/warmup.json"))
    parser.add_argument("--split", default="warmup")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.top_k <= 0:
        raise SystemExit("--batch-size and --top-k must be positive")
    if args.max_seq_length <= 0:
        raise SystemExit("--max-seq-length must be positive")

    passages = load_passages_jsonl(str(args.passages))
    corpus_hash = corpus_fingerprint(passages)
    loaded = load_dense_index(args.index_dir)
    expected_hash = loaded.manifest.get("corpus_hash")
    if expected_hash != corpus_hash:
        raise SystemExit(
            "Dense index corpus hash mismatch: "
            f"index={expected_hash!r}, passages={corpus_hash!r}"
        )
    require_dense_encode()

    if len(passages) != len(loaded.passage_ids):
        raise SystemExit(
            "Dense index/passages count mismatch: "
            f"index={len(loaded.passage_ids)}, passages={len(passages)}"
        )
    if loaded.manifest.get("index_type") != DENSE_INDEX_TYPE:
        raise SystemExit(
            f"Unsupported dense index type: {loaded.manifest.get('index_type')!r}"
        )
    model = loaded.manifest.get("model")
    if not isinstance(model, str) or not model.strip():
        raise SystemExit("Dense manifest does not contain a valid model name")
    if args.model is not None and args.model != model:
        raise SystemExit(
            f"Requested model {args.model!r} does not match index model {model!r}"
        )
    revision = loaded.manifest.get("model_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise SystemExit("Dense manifest must contain a pinned model_revision")
    instruction = loaded.manifest.get("query_instruction")
    if instruction != DEFAULT_QUERY_INSTRUCTION:
        raise SystemExit("Dense manifest query instruction does not match the contract")
    dtype = str(loaded.manifest.get("dtype", "bf16"))
    normalized_value = loaded.manifest.get("normalized")
    if not isinstance(normalized_value, bool):
        raise SystemExit("Dense manifest normalized must be boolean")
    normalized = normalized_value
    encoder = SentenceTransformerEncoder(
        model=model,
        device=args.device,
        dtype=dtype,
        revision=revision,
        max_seq_length=args.max_seq_length,
        local_files_only=args.local_files_only,
    )

    questions = load_inference_questions(args.questions, split=args.split)
    if args.limit > 0:
        questions = questions[: args.limit]
    query_texts = [
        format_instruct_query(question.question, instruction=instruction)
        for question in questions
    ]
    all_hits: list[tuple[Any, ...]] = []
    latency_values: list[float] = []
    for start in range(0, len(query_texts), args.batch_size):
        batch_started = time.perf_counter()
        query_vectors = encoder.encode(
            query_texts[start : start + args.batch_size],
            batch_size=args.batch_size,
        )
        if normalized:
            query_vectors = normalize_embedding_matrix(query_vectors)
        batch_hits = search_dense_index(
            loaded.index,
            loaded.passage_ids,
            query_vectors,
            top_k=args.top_k,
        )
        all_hits.extend(batch_hits)
        batch_latency_ms = (
            (time.perf_counter() - batch_started) * 1000.0 / max(len(batch_hits), 1)
        )
        latency_values.extend([batch_latency_ms] * len(batch_hits))

    _write_predictions(args.output, questions, tuple(all_hits))
    latency_path = args.output.with_name(f"{args.output.stem}_latency.json")
    _write_latency(latency_path, latency_values, n_queries=len(questions))
    print(
        json.dumps(
            {
                "n_queries": len(questions),
                "output": str(args.output),
                "corpus_hash": corpus_hash,
                "model": model,
                "model_revision": revision,
                "top_k": args.top_k,
                "latency": str(latency_path),
                "latency_p50_ms": _percentile(latency_values, 0.50),
                "latency_p95_ms": _percentile(latency_values, 0.95),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
