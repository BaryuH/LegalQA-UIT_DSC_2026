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
    DENSE_CACHE_SCHEMA_VERSION,
    DENSE_INDEX_SCHEMA_VERSION,
    DENSE_INDEX_TYPE,
    DenseIndexError,
    SentenceTransformerEncoder,
    dense_cache_fingerprint,
    format_instruct_query,
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
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--smoke-query", default="Điều 76")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.shard_size <= 0 or args.top_k <= 0:
        raise SystemExit("--batch-size, --shard-size, and --top-k must be positive")

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
    instruction = str(
        adapter_manifest.get(
            "query_instruction",
            base_loaded.manifest.get("query_instruction"),
        )
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
        max_seq_length=args.max_seq_length,
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

    texts = [passage.retrieval_text for passage in passages]
    order = length_bucket_order(texts)
    embeddings: Any = None
    embedding_dim: int | None = None
    embedding_path = args.output_dir / "embeddings.npy"
    encode_started = time.perf_counter()

    for shard_start in range(0, len(order), args.shard_size):
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
                    embedding_path,
                    mode="w+",
                    dtype="float32",
                    shape=(len(passages), embedding_dim),
                )
            assert embeddings is not None
            embeddings[np.asarray(batch_indices, dtype=np.int64)] = encoded

    assert embeddings is not None and embedding_dim is not None
    embeddings.flush()
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
    index_passage_ids = tuple(passage.passage_id for passage in passages)
    if index.ntotal != len(index_passage_ids):
        raise DenseIndexError(
            "FAISS/metadata alignment mismatch after rebuild: "
            f"{index.ntotal} vs {len(index_passage_ids)}"
        )

    smoke_vector = encoder.encode(
        [format_instruct_query(args.smoke_query, instruction=instruction)],
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
        max_seq_length=args.max_seq_length,
    )
    run_id = new_run_id("dense_lora_rebuild")
    payload = {
        "schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "run_id": run_id,
        "git_commit": git_commit_sha(),
        "model": model,
        "model_revision": revision,
        "query_instruction": instruction,
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
        "embedding_cache_path": str(embedding_path),
        "passage_metadata_path": str(metadata_path),
        "alignment_ok": True,
        "nan_inf_count": 0,
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
        "max_seq_length": args.max_seq_length,
        "passage_count": len(passages),
        "embedding_dim": embedding_dim,
        "vector_dtype": "float32",
        "embeddings_path": str(embedding_path),
        "storage_mode": "memmap",
        "passage_metadata_path": str(metadata_path),
        "adapter_dir": str(args.adapter_dir),
    }
    _write_manifest(args.output_dir / "embedding_cache_manifest.json", cache_manifest)
    _write_manifest(args.output_dir / "manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
