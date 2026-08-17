"""Dense retrieval utilities for the SEDAR Retrieval v3 TASK 07 pipeline.

The module keeps heavyweight dependencies optional.  Importing the package and
running local unit tests therefore does not require CUDA, FAISS, or a model
download.  The executable build/search paths fail closed when those runtime
dependencies or live CUDA are missing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from legal_rag.sedar_retrieval.cuda_policy import CudaDeferredError, probe_cuda

DEFAULT_DENSE_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_QUERY_INSTRUCTION = (
    "Retrieve Vietnamese legal provisions that directly support the answer. "
    "Prioritize applicable rules, conditions, exceptions, definitions and "
    "referenced provisions."
)
DENSE_INDEX_SCHEMA_VERSION = "sedar-retrieval-v3-dense-index-v1"
DENSE_CACHE_SCHEMA_VERSION = "sedar-retrieval-v3-dense-cache-v1"
DENSE_INDEX_TYPE = "faiss.IndexFlatIP"


class DenseDependencyError(RuntimeError):
    """Raised when an optional dense-retrieval dependency is unavailable."""


class DenseIndexError(RuntimeError):
    """Raised when a dense index or its alignment metadata is invalid."""


@dataclass(frozen=True, slots=True)
class DenseIndexManifest:
    model: str
    model_revision: str | None
    embedding_dim: int | None
    dtype: str
    normalized: bool
    corpus_hash: str | None
    index_type: str
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class DenseSearchHit:
    """One dense result with an index-aligned passage ID."""

    passage_id: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class LoadedDenseIndex:
    """FAISS index plus immutable passage metadata used for retrieval."""

    index: Any
    passage_ids: tuple[str, ...]
    manifest: dict[str, Any]


def build_dense_index_scaffold(
    *,
    model: str = DEFAULT_DENSE_MODEL,
    index_type: str = DENSE_INDEX_TYPE,
    normalized: bool = True,
    dtype: str = "bf16",
) -> DenseIndexManifest:
    """Create a local status manifest without attempting model inference."""

    status = probe_cuda()
    if not status.can_run_gpu_stage:
        return DenseIndexManifest(
            model=model,
            model_revision=None,
            embedding_dim=None,
            dtype=dtype,
            normalized=normalized,
            corpus_hash=None,
            index_type=index_type,
            status="DEFERRED_GPU",
            detail=status.message,
        )
    return DenseIndexManifest(
        model=model,
        model_revision=None,
        embedding_dim=None,
        dtype=dtype,
        normalized=normalized,
        corpus_hash=None,
        index_type=index_type,
        status="READY_FOR_ENCODE",
        detail="CUDA live; run build_dense_index.py",
    )


def require_dense_encode() -> None:
    """Fail closed unless live CUDA is available for dense encoding."""

    status = probe_cuda()
    if not status.can_run_gpu_stage:
        raise CudaDeferredError(
            "Dense corpus encoding requires live CUDA on the GPU server."
        )


def format_instruct_query(
    question: str, instruction: str = DEFAULT_QUERY_INSTRUCTION
) -> str:
    """Format the exact Qwen3 retrieval instruction required by TASK 07."""

    if not question.strip():
        raise ValueError("question must not be blank")
    if not instruction.strip():
        raise ValueError("instruction must not be blank")
    return f"Instruct: {instruction}\nQuery: {question}"


def length_bucket_order(texts: Sequence[str]) -> tuple[int, ...]:
    """Return a deterministic length-bucket order while preserving tie order."""

    if any(not isinstance(text, str) for text in texts):
        raise TypeError("texts must contain only strings")
    return tuple(
        sorted(range(len(texts)), key=lambda index: (len(texts[index]), index))
    )


def dense_cache_fingerprint(
    *,
    corpus_hash: str,
    model: str,
    model_revision: str,
    dtype: str,
    normalized: bool,
    max_seq_length: int,
) -> str:
    """Create a stable cache key for passage IDs and encoder configuration."""

    payload = {
        "corpus_hash": corpus_hash,
        "dtype": dtype,
        "max_seq_length": max_seq_length,
        "model": model,
        "model_revision": model_revision,
        "normalized": normalized,
        "schema_version": DENSE_CACHE_SCHEMA_VERSION,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DenseDependencyError(
            "Dense indexing requires numpy; install the SEDAR retrieval runtime."
        ) from exc
    return np


def _require_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DenseDependencyError(
            "Dense indexing requires FAISS; install faiss-cpu on the server."
        ) from exc
    return faiss


def validate_embedding_matrix(
    vectors: Any,
    *,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
) -> Any:
    """Validate and return a float32 2-D embedding matrix."""

    np = _require_numpy()
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise DenseIndexError(f"Embeddings must be 2-D, got shape {matrix.shape}")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise DenseIndexError(
            f"Embedding row mismatch: expected {expected_rows}, got {matrix.shape[0]}"
        )
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise DenseIndexError(
            f"Embedding dimension mismatch: expected {expected_dim}, "
            f"got {matrix.shape[1]}"
        )
    if not bool(np.isfinite(matrix).all()):
        raise DenseIndexError("Embeddings contain NaN or Inf values")
    if matrix.shape[1] == 0:
        raise DenseIndexError("Embedding dimension must be greater than zero")
    return matrix


def normalize_embedding_matrix(vectors: Any) -> Any:
    """L2-normalize embeddings for cosine-equivalent inner-product search."""

    np = _require_numpy()
    matrix = validate_embedding_matrix(vectors)
    norms = np.linalg.norm(matrix, axis=1)
    if bool((norms <= 0).any()):
        raise DenseIndexError("Embeddings contain a zero-norm row")
    normalized = matrix / norms[:, None]
    return validate_embedding_matrix(normalized, expected_dim=matrix.shape[1])


def _load_sentence_transformer() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DenseDependencyError(
            "Dense encoding requires sentence-transformers; install the SEDAR "
            "retrieval runtime."
        ) from exc
    return SentenceTransformer


class SentenceTransformerEncoder:
    """Thin, auditable adapter around SentenceTransformers for Qwen embeddings."""

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        dtype: str = "bf16",
        revision: str | None = None,
        max_seq_length: int | None = None,
        local_files_only: bool = False,
    ) -> None:
        if device.startswith("cuda"):
            require_dense_encode()
        if dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be one of: bf16, fp16, fp32")

        kwargs: dict[str, Any] = {
            "device": device,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        if revision:
            kwargs["revision"] = revision
        if dtype != "fp32":
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise DenseDependencyError(
                    "Non-fp32 dense encoding requires torch."
                ) from exc
            if dtype == "bf16":
                if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
                    raise DenseIndexError("Requested bf16 but CUDA bf16 is unsupported")
                kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
            else:
                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

        sentence_transformer = _load_sentence_transformer()
        self.model = sentence_transformer(model, **kwargs)
        if max_seq_length is not None:
            if max_seq_length <= 0:
                raise ValueError("max_seq_length must be positive")
            self.model.max_seq_length = max_seq_length

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool = False,
    ) -> Any:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain blank strings")
        encoded = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=show_progress_bar,
        )
        return validate_embedding_matrix(encoded, expected_rows=len(texts))


def write_passage_metadata(path: Path, metadata: Sequence[dict[str, Any]]) -> None:
    """Write index-row metadata without raw text or gold answers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for ordinal, row in enumerate(metadata):
            payload = dict(row)
            payload["ordinal"] = ordinal
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_passage_ids(path: Path) -> tuple[str, ...]:
    """Load and validate passage IDs in FAISS row order."""

    passage_ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for ordinal, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("ordinal") != ordinal:
                raise DenseIndexError(
                    f"Invalid metadata ordinal at physical row {ordinal}"
                )
            passage_id = row.get("passage_id")
            if not isinstance(passage_id, str) or not passage_id.strip():
                raise DenseIndexError(f"Blank passage_id at metadata row {ordinal}")
            passage_ids.append(passage_id)
    if len(set(passage_ids)) != len(passage_ids):
        raise DenseIndexError("Dense metadata contains duplicate passage IDs")
    return tuple(passage_ids)


def load_dense_index(index_dir: Path) -> LoadedDenseIndex:
    """Load FAISS index and enforce row-to-passage alignment."""

    faiss = _require_faiss()
    manifest_path = index_dir / "manifest.json"
    metadata_path = index_dir / "passage_metadata.jsonl"
    index_path = index_dir / "index.faiss"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_path))
        passage_ids = load_passage_ids(metadata_path)
    except FileNotFoundError as exc:
        raise DenseIndexError(f"Incomplete dense index directory: {index_dir}") from exc
    if not isinstance(manifest, dict):
        raise DenseIndexError("Dense manifest must be a JSON object")
    if manifest.get("schema_version") != DENSE_INDEX_SCHEMA_VERSION:
        raise DenseIndexError(
            f"Unsupported dense index schema: {manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "PASS":
        raise DenseIndexError(
            f"Dense index is not loadable; status={manifest.get('status')!r}"
        )
    if manifest.get("index_type") != DENSE_INDEX_TYPE:
        raise DenseIndexError(
            f"Unsupported dense index type: {manifest.get('index_type')!r}"
        )
    if index.ntotal != len(passage_ids):
        raise DenseIndexError(
            f"FAISS/metadata alignment mismatch: {index.ntotal} vs {len(passage_ids)}"
        )
    expected_count = manifest.get("passage_count")
    if expected_count is not None and int(expected_count) != len(passage_ids):
        raise DenseIndexError(
            f"Manifest/metadata count mismatch: {expected_count} vs {len(passage_ids)}"
        )
    if manifest.get("alignment_ok") is not True:
        raise DenseIndexError("Dense manifest does not confirm alignment_ok=true")
    expected_dim = manifest.get("embedding_dim")
    if expected_dim is not None and index.d != int(expected_dim):
        raise DenseIndexError(
            f"FAISS dimension mismatch: {index.d} vs manifest {expected_dim}"
        )
    return LoadedDenseIndex(index=index, passage_ids=passage_ids, manifest=manifest)


def search_dense_index(
    index: Any,
    passage_ids: Sequence[str],
    query_vectors: Any,
    *,
    top_k: int,
) -> tuple[tuple[DenseSearchHit, ...], ...]:
    """Search a FAISS-compatible index and return deterministic hit records."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    matrix = validate_embedding_matrix(query_vectors, expected_dim=int(index.d))
    scores, indices = index.search(matrix, min(top_k, int(index.ntotal)))
    results: list[tuple[DenseSearchHit, ...]] = []
    for score_row, index_row in zip(scores, indices, strict=True):
        hits: list[DenseSearchHit] = []
        for rank, (score, ordinal) in enumerate(
            zip(score_row, index_row, strict=True), start=1
        ):
            ordinal_int = int(ordinal)
            if ordinal_int < 0:
                continue
            if ordinal_int >= len(passage_ids):
                raise DenseIndexError(f"FAISS returned invalid row {ordinal_int}")
            hits.append(
                DenseSearchHit(
                    passage_id=passage_ids[ordinal_int],
                    score=float(score),
                    rank=rank,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.passage_id))
        results.append(
            tuple(
                DenseSearchHit(
                    passage_id=hit.passage_id,
                    score=hit.score,
                    rank=rank,
                )
                for rank, hit in enumerate(hits, start=1)
            )
        )
    return tuple(results)


def dense_manifest_to_dict(manifest: DenseIndexManifest) -> dict[str, Any]:
    return {
        "model": manifest.model,
        "model_revision": manifest.model_revision,
        "embedding_dim": manifest.embedding_dim,
        "dtype": manifest.dtype,
        "normalized": manifest.normalized,
        "corpus_hash": manifest.corpus_hash,
        "index_type": manifest.index_type,
        "status": manifest.status,
        "detail": manifest.detail,
    }


__all__ = [
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_QUERY_INSTRUCTION",
    "DENSE_CACHE_SCHEMA_VERSION",
    "DENSE_INDEX_SCHEMA_VERSION",
    "DENSE_INDEX_TYPE",
    "DenseDependencyError",
    "DenseIndexError",
    "DenseIndexManifest",
    "DenseSearchHit",
    "LoadedDenseIndex",
    "SentenceTransformerEncoder",
    "build_dense_index_scaffold",
    "dense_manifest_to_dict",
    "dense_cache_fingerprint",
    "format_instruct_query",
    "length_bucket_order",
    "load_dense_index",
    "load_passage_ids",
    "normalize_embedding_matrix",
    "require_dense_encode",
    "search_dense_index",
    "validate_embedding_matrix",
    "write_passage_metadata",
]
