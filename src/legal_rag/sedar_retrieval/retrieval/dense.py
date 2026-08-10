"""Dense legal index scaffold (TASK 07). Live encode/search requires CUDA server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from legal_rag.sedar_retrieval.cuda_policy import CudaDeferredError, probe_cuda

DEFAULT_DENSE_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_QUERY_INSTRUCTION = (
    "Retrieve Vietnamese legal provisions that directly support the answer. "
    "Prioritize applicable rules, conditions, exceptions, definitions and "
    "referenced provisions."
)


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


def build_dense_index_scaffold(
    *,
    model: str = DEFAULT_DENSE_MODEL,
    index_type: str = "faiss.IndexFlatIP",
    normalized: bool = True,
    dtype: str = "bf16",
) -> DenseIndexManifest:
    """Create a dense-index manifest. Actual encoding is deferred without CUDA."""

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
    # Live path is intentionally not implemented on this local scaffold pass;
    # server follow-up fills embeddings/index bytes.
    return DenseIndexManifest(
        model=model,
        model_revision=None,
        embedding_dim=None,
        dtype=dtype,
        normalized=normalized,
        corpus_hash=None,
        index_type=index_type,
        status="READY_FOR_ENCODE",
        detail="CUDA live; run server encode job",
    )


def require_dense_encode() -> None:
    status = probe_cuda()
    if not status.can_run_gpu_stage:
        raise CudaDeferredError(
            "Dense corpus encoding requires live CUDA on the GPU server."
        )


def format_instruct_query(question: str, instruction: str = DEFAULT_QUERY_INSTRUCTION) -> str:
    return f"Instruct: {instruction}\nQuery: {question}"


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
