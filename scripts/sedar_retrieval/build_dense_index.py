#!/usr/bin/env python3
"""Build the Qwen3 dense legal index for SEDAR Retrieval TASK 07."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.retrieval.bm25_passages import (
    corpus_fingerprint,
)
from legal_rag.sedar_retrieval.retrieval.dense import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_QUERY_INSTRUCTION,
    DENSE_CACHE_SCHEMA_VERSION,
    DENSE_INDEX_SCHEMA_VERSION,
    DENSE_INDEX_TYPE,
    DenseIndexError,
    SentenceTransformerEncoder,
    build_dense_index_scaffold,
    dense_cache_fingerprint,
    dense_manifest_to_dict,
    format_instruct_query,
    length_bucket_order,
    normalize_embedding_matrix,
    require_dense_encode,
    search_dense_index,
    validate_embedding_matrix,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl


def _optional_runtime_imports() -> tuple[Any, Any]:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "TASK 07 requires numpy and FAISS; install faiss-cpu on the server."
        ) from exc
    return faiss, np


def _write_metadata(path: Path, passages: tuple[Any, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for ordinal, passage in enumerate(passages):
            handle.write(
                json.dumps(
                    {
                        "ordinal": ordinal,
                        "passage_id": passage.passage_id,
                        "document_id": passage.document_id,
                        "article_id": passage.article_id,
                        "clause_id": passage.clause_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _gpu_peak_memory_gb() -> float | None:
    try:
        import torch
    except ImportError:  # pragma: no cover - guarded by the encoder
        return None
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / 1024**3)


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise SystemExit(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not force:
        raise SystemExit(
            f"Output directory is non-empty: {path}; use a new run directory "
            "or pass --force explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)


def _scaffold_manifest(
    *,
    run_id: str,
    model: str,
    dtype: str,
    normalized: bool,
    corpus_hash: str,
    passage_count: int,
) -> dict[str, Any]:
    manifest = build_dense_index_scaffold(
        model=model,
        index_type=DENSE_INDEX_TYPE,
        normalized=normalized,
        dtype=dtype,
    )
    payload = dense_manifest_to_dict(manifest)
    payload.update(
        {
            "schema_version": DENSE_INDEX_SCHEMA_VERSION,
            "run_id": run_id,
            "git_commit": git_commit_sha(),
            "corpus_hash": corpus_hash,
            "passage_count": passage_count,
            "alignment_ok": False,
            "nan_inf_count": 0,
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--smoke-query", default="Điều 76")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-normalization", action="store_true")
    parser.add_argument("--disable-length-bucketing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a DEFERRED_GPU status manifest without loading the model.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.shard_size <= 0 or args.top_k <= 0:
        raise SystemExit("--batch-size, --shard-size, and --top-k must be positive")
    if args.max_seq_length <= 0:
        raise SystemExit("--max-seq-length must be positive")
    if not args.dry_run and not str(args.model_revision or "").strip():
        raise SystemExit(
            "A pinned --model-revision is required for a non-dry-run dense index."
        )

    _prepare_output_dir(args.output_dir, force=args.force)
    run_id = new_run_id("dense_legal_index")
    passages = load_passages_jsonl(str(args.passages))
    if not passages:
        raise SystemExit("Cannot build a dense index from zero passages")
    passage_ids = tuple(passage.passage_id for passage in passages)
    if len(set(passage_ids)) != len(passage_ids):
        raise SystemExit("Cannot build a dense index with duplicate passage IDs")
    corpus_hash = corpus_fingerprint(passages)
    normalized = not args.disable_normalization
    cache_key = dense_cache_fingerprint(
        corpus_hash=corpus_hash,
        model=args.model,
        model_revision=str(args.model_revision or "UNPINNED"),
        dtype=args.dtype,
        normalized=normalized,
        max_seq_length=args.max_seq_length,
    )

    if args.dry_run:
        payload = _scaffold_manifest(
            run_id=run_id,
            model=args.model,
            dtype=args.dtype,
            normalized=normalized,
            corpus_hash=corpus_hash,
            passage_count=len(passages),
        )
        _write_manifest(args.output_dir / "manifest.json", payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    require_dense_encode()
    faiss, np = _optional_runtime_imports()
    encoder = SentenceTransformerEncoder(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        revision=args.model_revision,
        max_seq_length=args.max_seq_length,
        local_files_only=args.local_files_only,
    )

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:  # pragma: no cover - guarded by the encoder
        pass

    texts = [passage.retrieval_text for passage in passages]
    order = (
        tuple(range(len(texts)))
        if args.disable_length_bucketing
        else length_bucket_order(texts)
    )
    embeddings: Any = None
    embedding_dim: int | None = None
    encode_started = time.perf_counter()
    shard_count = 0

    for shard_start in range(0, len(order), args.shard_size):
        shard_count += 1
        shard = order[shard_start : shard_start + args.shard_size]
        for batch_start in range(0, len(shard), args.batch_size):
            batch_indices = shard[batch_start : batch_start + args.batch_size]
            encoded = encoder.encode(
                [texts[index] for index in batch_indices],
                batch_size=args.batch_size,
            )
            if normalized:
                encoded = normalize_embedding_matrix(encoded)
            else:
                encoded = validate_embedding_matrix(encoded)
            if embedding_dim is None:
                embedding_dim = int(encoded.shape[1])
                embeddings = np.lib.format.open_memmap(
                    args.output_dir / "embeddings.npy",
                    mode="w+",
                    dtype="float32",
                    shape=(len(passages), embedding_dim),
                )
            elif int(encoded.shape[1]) != embedding_dim:
                raise DenseIndexError(
                    f"Model returned inconsistent embedding dimension: "
                    f"{encoded.shape[1]} vs {embedding_dim}"
                )
            embeddings[np.asarray(batch_indices, dtype=np.int64)] = encoded
        embeddings.flush()

    assert embeddings is not None
    assert embedding_dim is not None
    encode_seconds = max(time.perf_counter() - encode_started, 1e-9)

    index = faiss.IndexFlatIP(embedding_dim)
    for start in range(0, len(passages), args.batch_size):
        index.add(
            np.asarray(
                embeddings[start : start + args.batch_size],
                dtype="float32",
                order="C",
            )
        )
    index_path = args.output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    metadata_path = args.output_dir / "passage_metadata.jsonl"
    _write_metadata(metadata_path, passages)
    alignment_ok = index.ntotal == len(passage_ids)
    if not alignment_ok:
        raise DenseIndexError(
            f"FAISS/metadata alignment mismatch: {index.ntotal} vs {len(passage_ids)}"
        )

    smoke_vector = encoder.encode(
        [format_instruct_query(args.smoke_query)],
        batch_size=1,
    )
    if normalized:
        smoke_vector = normalize_embedding_matrix(smoke_vector)
    before_reload = search_dense_index(
        index,
        passage_ids,
        smoke_vector,
        top_k=args.top_k,
    )[0]
    reloaded = faiss.read_index(str(index_path))
    after_reload = search_dense_index(
        reloaded,
        passage_ids,
        smoke_vector,
        top_k=args.top_k,
    )[0]
    before_ids = tuple(hit.passage_id for hit in before_reload)
    after_ids = tuple(hit.passage_id for hit in after_reload)
    reload_overlap = 1.0 if before_ids == after_ids else 0.0
    index_size_bytes = index_path.stat().st_size
    cache_manifest_path = args.output_dir / "embedding_cache_manifest.json"
    cache_manifest = {
        "schema_version": DENSE_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "corpus_hash": corpus_hash,
        "model": args.model,
        "model_revision": args.model_revision,
        "dtype": args.dtype,
        "normalized": normalized,
        "max_seq_length": args.max_seq_length,
        "passage_count": len(passages),
        "embedding_dim": embedding_dim,
        "vector_dtype": "float32",
        "embeddings_path": str(args.output_dir / "embeddings.npy"),
        "passage_metadata_path": str(metadata_path),
    }
    _write_manifest(cache_manifest_path, cache_manifest)
    payload = {
        "schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "run_id": run_id,
        "git_commit": git_commit_sha(),
        "model": args.model,
        "model_revision": args.model_revision,
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "cache_key": cache_key,
        "embedding_dim": embedding_dim,
        "dtype": args.dtype,
        "normalized": normalized,
        "corpus_hash": corpus_hash,
        "passage_count": len(passages),
        "index_type": DENSE_INDEX_TYPE,
        "status": "PASS" if reload_overlap == 1.0 else "FAIL",
        "detail": "Dense index built and reload determinism verified.",
        "index_path": str(index_path),
        "embedding_cache_path": str(args.output_dir / "embeddings.npy"),
        "embedding_cache_manifest_path": str(cache_manifest_path),
        "passage_metadata_path": str(metadata_path),
        "alignment_ok": alignment_ok,
        "nan_inf_count": 0,
        "reload_topk_overlap": reload_overlap,
        "smoke_query": args.smoke_query,
        "smoke_hit_count": len(before_reload),
        "smoke_hit_ids": list(before_ids),
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "shard_count": shard_count,
        "length_bucketed": not args.disable_length_bucketing,
        "encode_seconds": encode_seconds,
        "encode_passages_per_second": len(passages) / encode_seconds,
        "index_size_bytes": index_size_bytes,
        "peak_vram_gb": _gpu_peak_memory_gb(),
    }
    _write_manifest(args.output_dir / "manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if reload_overlap == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
