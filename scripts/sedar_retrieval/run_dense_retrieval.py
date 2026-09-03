#!/usr/bin/env python3
"""Run dense retrieval over SEDAR questions (TASK 07)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from legal_rag.sedar_retrieval.retrieval.bm25_passages import corpus_fingerprint
from legal_rag.sedar_retrieval.retrieval.dense import (
    DEFAULT_E5_QUERY_PREFIX,
    DEFAULT_INPUT_FORMAT,
    DEFAULT_QUERY_INSTRUCTION,
    DENSE_INDEX_TYPE,
    DENSE_INPUT_FORMATS,
    SentenceTransformerEncoder,
    format_query_text,
    load_dense_index,
    normalize_embedding_matrix,
    require_dense_encode,
    search_dense_index,
    validate_source_model_pair,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl
from legal_rag.sedar_retrieval.training.query_inputs import (
    RetrievalQuery,
    load_retrieval_queries_from_json,
    load_retrieval_queries_from_synthetic,
)


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


def _apply_lora_adapter_if_present(
    encoder: SentenceTransformerEncoder, manifest: dict[str, object]
) -> None:
    adapter_dir = manifest.get("adapter_dir")
    if not adapter_dir:
        return
    if not isinstance(adapter_dir, str) or not adapter_dir.strip():
        raise SystemExit("Dense manifest adapter_dir must be a non-empty string")
    try:
        from peft import PeftModel
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "LoRA dense index requires peft to load the TASK 11 adapter."
        ) from exc
    try:
        encoder.model[0].auto_model = PeftModel.from_pretrained(
            encoder.model[0].auto_model,
            adapter_dir,
            is_trainable=False,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"DENSE_RETRIEVAL_LORA_FAILED: {exc}") from exc


def _write_predictions(
    path: Path,
    queries: tuple[RetrievalQuery, ...],
    hits_by_query: tuple[tuple[Any, ...], ...],
    *,
    source_name: str = "dense",
) -> None:
    if source_name not in {"dense", "legal"}:
        raise ValueError("source_name must be 'dense' or 'legal'")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for query, hits in zip(queries, hits_by_query, strict=True):
            handle.write(
                json.dumps(
                    {
                        "query_id": query.query_id,
                        "retrieval_source": source_name,
                        "ranked_ids": [hit.passage_id for hit in hits],
                        "scores": [
                            {
                                "passage_id": hit.passage_id,
                                source_name: hit.score,
                                "rank": hit.rank,
                                "source": source_name,
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
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--synthetic-jsonl", type=Path, default=None)
    parser.add_argument("--split", default="warmup")
    parser.add_argument(
        "--source-split",
        default="train",
        help="Synthetic source_split filter when --synthetic-jsonl is used.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Optional local snapshot directory to load the query encoder from. "
            "The manifest's model name and pinned revision are still validated "
            "and recorded; only the loader path differs. Use the same directory "
            "that built the index."
        ),
    )
    parser.add_argument(
        "--source-name",
        choices=("dense", "legal"),
        default="dense",
        help="Logical source name in the output artifact.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override the index manifest sequence length.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.questions is None and args.synthetic_jsonl is None:
        args.questions = Path("data/warmup.json")
    if args.questions is not None and args.synthetic_jsonl is not None:
        raise SystemExit("Provide only one of --questions or --synthetic-jsonl")
    if args.batch_size <= 0 or args.top_k <= 0:
        raise SystemExit("--batch-size and --top-k must be positive")
    if args.max_seq_length is not None and args.max_seq_length <= 0:
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
    input_format = loaded.manifest.get("input_format", DEFAULT_INPUT_FORMAT)
    if not isinstance(input_format, str) or input_format not in DENSE_INPUT_FORMATS:
        raise SystemExit(
            f"Dense manifest input_format is unsupported: {input_format!r}"
        )
    try:
        validate_source_model_pair(
            args.source_name,
            model,
            input_format=input_format,
        )
    except ValueError as exc:
        raise SystemExit(f"DENSE_SOURCE_MODEL_MISMATCH: {exc}") from exc
    raw_instruction = loaded.manifest.get("query_instruction")
    if input_format == "qwen_instruction":
        if raw_instruction != DEFAULT_QUERY_INSTRUCTION:
            raise SystemExit(
                "Dense manifest query instruction does not match the contract"
            )
        instruction = DEFAULT_QUERY_INSTRUCTION
        query_prefix = DEFAULT_E5_QUERY_PREFIX
    else:
        instruction = DEFAULT_QUERY_INSTRUCTION
        raw_query_prefix = loaded.manifest.get("query_prefix")
        raw_passage_prefix = loaded.manifest.get("passage_prefix")
        if (
            not isinstance(raw_query_prefix, str)
            or not raw_query_prefix.strip()
            or not isinstance(raw_passage_prefix, str)
            or not raw_passage_prefix.strip()
        ):
            raise SystemExit(
                "Dense manifest E5 query_prefix/passage_prefix must be "
                "non-blank strings"
            )
        query_prefix = raw_query_prefix
    manifest_max_seq_length = loaded.manifest.get("max_seq_length", 8192)
    if (
        isinstance(manifest_max_seq_length, bool)
        or not isinstance(manifest_max_seq_length, int)
        or manifest_max_seq_length <= 0
    ):
        raise SystemExit("Dense manifest max_seq_length must be a positive integer")
    max_seq_length = (
        args.max_seq_length
        if args.max_seq_length is not None
        else manifest_max_seq_length
    )
    dtype = str(loaded.manifest.get("dtype", "bf16"))
    normalized_value = loaded.manifest.get("normalized")
    if not isinstance(normalized_value, bool):
        raise SystemExit("Dense manifest normalized must be boolean")
    normalized = normalized_value
    if args.model_path is not None:
        if not args.model_path.is_dir():
            raise SystemExit(f"--model-path is not a directory: {args.model_path}")
        load_target = str(args.model_path)
        # Local directory: load from disk and do not re-enter hub resolution.
        load_revision = None
    else:
        load_target = model
        load_revision = revision

    encoder = SentenceTransformerEncoder(
        model=load_target,
        device=args.device,
        dtype=dtype,
        revision=load_revision,
        max_seq_length=max_seq_length,
        local_files_only=args.local_files_only,
    )
    _apply_lora_adapter_if_present(encoder, loaded.manifest)

    if args.synthetic_jsonl is not None:
        queries = load_retrieval_queries_from_synthetic(
            args.synthetic_jsonl,
            source_split=args.source_split,
        )
    else:
        assert args.questions is not None
        queries = load_retrieval_queries_from_json(args.questions, split=args.split)
    if args.limit > 0:
        queries = queries[: args.limit]
    query_texts = [
        format_query_text(
            query.question,
            input_format=input_format,
            instruction=instruction,
            query_prefix=query_prefix,
        )
        for query in queries
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

    _write_predictions(
        args.output,
        queries,
        tuple(all_hits),
        source_name=args.source_name,
    )
    latency_path = args.output.with_name(f"{args.output.stem}_latency.json")
    _write_latency(latency_path, latency_values, n_queries=len(queries))
    print(
        json.dumps(
            {
                "n_queries": len(queries),
                "output": str(args.output),
                "corpus_hash": corpus_hash,
                "model": model,
                "model_revision": revision,
                "source_name": args.source_name,
                "input_format": input_format,
                "max_seq_length": max_seq_length,
                "adapter_dir": loaded.manifest.get("adapter_dir"),
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
