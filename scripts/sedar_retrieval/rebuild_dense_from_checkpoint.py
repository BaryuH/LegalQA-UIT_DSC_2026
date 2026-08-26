#!/usr/bin/env python3
"""Rebuild a dense index from a TASK 11 LoRA adapter checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.retrieval.bm25_passages import corpus_fingerprint
from legal_rag.sedar_retrieval.retrieval.dense import (
    DEFAULT_E5_PASSAGE_PREFIX,
    DEFAULT_E5_QUERY_PREFIX,
    DEFAULT_INPUT_FORMAT,
    DEFAULT_QUERY_INSTRUCTION,
    DENSE_CACHE_SCHEMA_VERSION,
    DENSE_INDEX_SCHEMA_VERSION,
    DENSE_INDEX_TYPE,
    DenseIndexError,
    SentenceTransformerEncoder,
    dense_cache_fingerprint,
    format_passage_text,
    format_query_text,
    length_bucket_order,
    load_dense_index,
    normalize_embedding_matrix,
    require_dense_encode,
    search_dense_index,
    validate_embedding_matrix,
)
from legal_rag.sedar_retrieval.retrieval.passage_adapter import load_passages_jsonl
from legal_rag.sedar_retrieval.training.retriever_lora import load_adapter_manifest


def _optional_runtime_imports() -> tuple[Any, Any]:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "TASK 11 rebuild requires numpy and FAISS; install faiss-cpu on the server."
        ) from exc
    return faiss, np


def _prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise SystemExit(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not force:
        raise SystemExit(
            f"Output directory is non-empty: {path}; use a new run directory "
            "or pass --force explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument(
        "--embedding-storage",
        choices=("auto", "memmap", "sharded"),
        default="auto",
        help="Use mmap, .npy shards, or mmap with automatic fallback to shards.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override the base index manifest sequence length.",
    )
    parser.add_argument("--smoke-query", default="Điều 76")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.shard_size <= 0 or args.top_k <= 0:
        raise SystemExit("--batch-size, --shard-size, and --top-k must be positive")
    if args.max_seq_length is not None and args.max_seq_length <= 0:
        raise SystemExit("--max-seq-length must be positive")

    _prepare_output_dir(args.output_dir, force=args.force)
    require_dense_encode()
    faiss, np = _optional_runtime_imports()

    base_loaded = load_dense_index(args.base_index_dir)
    adapter_manifest = load_adapter_manifest(args.adapter_dir)
    model = str(adapter_manifest.get("base_model", base_loaded.manifest.get("model")))
    revision = str(
        adapter_manifest.get(
            "model_revision",
            base_loaded.manifest.get("model_revision"),
        )
    )
    if not revision.strip():
        raise SystemExit("Adapter/base manifest must contain a model revision")
    input_format = str(
        adapter_manifest.get(
            "input_format",
            base_loaded.manifest.get("input_format", DEFAULT_INPUT_FORMAT),
        )
    )
    raw_instruction = adapter_manifest.get(
        "query_instruction",
        base_loaded.manifest.get("query_instruction"),
    )
    instruction = (
        str(raw_instruction)
        if isinstance(raw_instruction, str) and raw_instruction.strip()
        else DEFAULT_QUERY_INSTRUCTION
    )
    if input_format == "e5":
        raw_query_prefix = adapter_manifest.get(
            "query_prefix",
            base_loaded.manifest.get("query_prefix"),
        )
        raw_passage_prefix = adapter_manifest.get(
            "passage_prefix",
            base_loaded.manifest.get("passage_prefix"),
        )
        if (
            not isinstance(raw_query_prefix, str)
            or not raw_query_prefix.strip()
            or not isinstance(raw_passage_prefix, str)
            or not raw_passage_prefix.strip()
        ):
            raise SystemExit("E5 dense manifest prefixes must be non-blank strings")
        query_prefix = raw_query_prefix
        passage_prefix = raw_passage_prefix
    elif input_format == "qwen_instruction":
        query_prefix = None
        passage_prefix = None
    else:
        raise SystemExit(f"Unsupported dense input format: {input_format!r}")
    manifest_max_seq_length = adapter_manifest.get(
        "max_seq_length",
        base_loaded.manifest.get("max_seq_length", 8192),
    )
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
    dtype = str(base_loaded.manifest.get("dtype", "bf16"))
    normalized = bool(base_loaded.manifest.get("normalized", True))

    passages = load_passages_jsonl(str(args.passages))
    corpus_hash = corpus_fingerprint(passages)
    if base_loaded.manifest.get("corpus_hash") != corpus_hash:
        raise SystemExit(
            "Corpus hash mismatch between base index and passages: "
            f"{base_loaded.manifest.get('corpus_hash')!r} vs {corpus_hash!r}"
        )

    encoder = SentenceTransformerEncoder(
        model=model,
        device=args.device,
        dtype=dtype,  # type: ignore[arg-type]
        revision=revision,
        max_seq_length=max_seq_length,
        local_files_only=args.local_files_only,
    )
    try:
        from peft import PeftModel

        encoder.model[0].auto_model = PeftModel.from_pretrained(
            encoder.model[0].auto_model,
            str(args.adapter_dir),
            is_trainable=False,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TASK 11 rebuild requires peft to load the LoRA adapter."
        ) from exc
    except (OSError, ValueError) as exc:
        raise SystemExit(f"TASK11_REBUILD_FAILED: {exc}") from exc

    texts = [
        format_passage_text(
            passage.retrieval_text,
            input_format=input_format,
            passage_prefix=passage_prefix or DEFAULT_E5_PASSAGE_PREFIX,
        )
        for passage in passages
    ]
    order = length_bucket_order(texts)
    embeddings: Any = None
    embedding_dim: int | None = None
    embedding_path = args.output_dir / "embeddings.npy"
    shard_dir = args.output_dir / "embedding_shards"
    shard_paths: list[Path] = []
    storage_mode = "sharded" if args.embedding_storage == "sharded" else "memmap"
    storage_fallback_reason: str | None = None
    encode_started = time.perf_counter()
    shard_count = 0

    for shard_start in range(0, len(order), args.shard_size):
        shard_count += 1
        shard = order[shard_start : shard_start + args.shard_size]
        shard_vectors: list[Any] = []
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
                if storage_mode == "memmap":
                    try:
                        embeddings = np.lib.format.open_memmap(
                            embedding_path,
                            mode="w+",
                            dtype="float32",
                            shape=(len(passages), embedding_dim),
                        )
                    except OSError as exc:
                        if args.embedding_storage == "memmap":
                            raise DenseIndexError(
                                "Embedding memmap is unsupported on this "
                                "filesystem; use --embedding-storage sharded."
                            ) from exc
                        storage_mode = "sharded"
                        storage_fallback_reason = str(exc)
                        embedding_path.unlink(missing_ok=True)
            elif int(encoded.shape[1]) != embedding_dim:
                raise DenseIndexError(
                    f"Model returned inconsistent embedding dimension: "
                    f"{encoded.shape[1]} vs {embedding_dim}"
                )
            if storage_mode == "memmap":
                assert embeddings is not None
                embeddings[np.asarray(batch_indices, dtype=np.int64)] = encoded
            else:
                shard_vectors.append(encoded)
        if storage_mode == "memmap":
            assert embeddings is not None
            embeddings.flush()
        else:
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / f"embeddings-{shard_count:05d}.npy"
            shard_matrix = np.concatenate(shard_vectors, axis=0)
            np.save(shard_path, shard_matrix, allow_pickle=False)
            shard_paths.append(shard_path)

    if storage_mode == "memmap":
        assert embeddings is not None
    elif not shard_paths:
        raise DenseIndexError("No embedding shards were written")
    assert embedding_dim is not None
    encode_seconds = max(time.perf_counter() - encode_started, 1e-9)

    index = faiss.IndexFlatIP(embedding_dim)
    if storage_mode == "memmap":
        for start in range(0, len(passages), args.batch_size):
            index.add(
                np.asarray(
                    embeddings[start : start + args.batch_size],
                    dtype="float32",
                    order="C",
                )
            )
        index_passage_indices = tuple(range(len(passages)))
        embedding_cache_path = embedding_path
    else:
        for shard_path in shard_paths:
            shard_matrix = validate_embedding_matrix(
                np.load(shard_path, allow_pickle=False),
                expected_dim=embedding_dim,
            )
            index.add(np.asarray(shard_matrix, dtype="float32", order="C"))
        index_passage_indices = order
        embedding_cache_path = shard_dir

    index_path = args.output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    metadata_path = args.output_dir / "passage_metadata.jsonl"
    index_passages = tuple(passages[index] for index in index_passage_indices)
    _write_metadata(metadata_path, index_passages)
    index_passage_ids = tuple(passage.passage_id for passage in index_passages)
    if index.ntotal != len(index_passage_ids):
        raise DenseIndexError(
            "FAISS/metadata alignment mismatch after rebuild: "
            f"{index.ntotal} vs {len(index_passage_ids)}"
        )

    smoke_vector = encoder.encode(
        [
            format_query_text(
                args.smoke_query,
                input_format=input_format,
                instruction=instruction,
                query_prefix=query_prefix or DEFAULT_E5_QUERY_PREFIX,
            )
        ],
        batch_size=1,
    )
    if normalized:
        smoke_vector = normalize_embedding_matrix(smoke_vector)
    smoke_hits = search_dense_index(
        index,
        index_passage_ids,
        smoke_vector,
        top_k=args.top_k,
    )[0]

    cache_key = dense_cache_fingerprint(
        corpus_hash=corpus_hash,
        model=model,
        model_revision=revision,
        dtype=dtype,
        normalized=normalized,
        max_seq_length=max_seq_length,
        input_format=input_format,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
    run_id = new_run_id("dense_lora_rebuild")
    payload = {
        "schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "run_id": run_id,
        "git_commit": git_commit_sha(),
        "model": model,
        "model_revision": revision,
        "query_instruction": (
            instruction if input_format == "qwen_instruction" else None
        ),
        "input_format": input_format,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "max_seq_length": max_seq_length,
        "cache_key": cache_key,
        "embedding_dim": embedding_dim,
        "dtype": dtype,
        "normalized": normalized,
        "corpus_hash": corpus_hash,
        "passage_count": len(passages),
        "index_type": DENSE_INDEX_TYPE,
        "status": "PASS",
        "detail": "Dense index rebuilt from TASK 11 LoRA adapter.",
        "parent_index_dir": str(args.base_index_dir),
        "adapter_dir": str(args.adapter_dir),
        "adapter_manifest": {
            "run_id": adapter_manifest.get("run_id"),
            "lora_config_hash": adapter_manifest.get("lora_config_hash"),
            "corpus_hash": adapter_manifest.get("corpus_hash"),
        },
        "index_path": str(index_path),
        "embedding_cache_path": str(embedding_cache_path),
        "embedding_storage": storage_mode,
        "embedding_storage_requested": args.embedding_storage,
        "embedding_storage_fallback_reason": storage_fallback_reason,
        "passage_metadata_path": str(metadata_path),
        "alignment_ok": True,
        "nan_inf_count": 0,
        "length_bucketed": True,
        "smoke_query": args.smoke_query,
        "smoke_hit_count": len(smoke_hits),
        "smoke_hit_ids": [hit.passage_id for hit in smoke_hits],
        "encode_seconds": encode_seconds,
        "encode_passages_per_second": len(passages) / encode_seconds,
    }
    cache_manifest = {
        "schema_version": DENSE_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "corpus_hash": corpus_hash,
        "model": model,
        "model_revision": revision,
        "dtype": dtype,
        "normalized": normalized,
        "max_seq_length": max_seq_length,
        "input_format": input_format,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "passage_count": len(passages),
        "embedding_dim": embedding_dim,
        "vector_dtype": "float32",
        "embeddings_path": str(embedding_cache_path),
        "storage_mode": storage_mode,
        "shard_paths": [str(path) for path in shard_paths],
        "passage_metadata_path": str(metadata_path),
        "adapter_dir": str(args.adapter_dir),
    }
    _write_manifest(args.output_dir / "embedding_cache_manifest.json", cache_manifest)
    _write_manifest(args.output_dir / "manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
