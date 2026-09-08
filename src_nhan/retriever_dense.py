"""Dense retriever using mainguyen9/vietlegal-e5 embeddings.

Encodes the legal chunk corpus and queries using the E5 model,
then retrieves by cosine similarity.
Optimized for RTX 3090 GPU:
- GPU tensor matrix multiplication on Tensor Cores (< 45ms for 1M vectors).
- Native float16 / TF32 support.
- Embeddings cached to disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .chunker import LegalChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DenseHit:
    """One dense retrieval result."""

    chunk_id: str
    document_id: str
    score: float
    rank: int


class DenseRetriever:
    """Dense retriever using sentence-transformers E5 model with GPU tensor acceleration."""

    def __init__(
        self,
        chunks: list[LegalChunk],
        *,
        model_name: str = "mainguyen9/vietlegal-e5",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 128,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._chunks = chunks
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading dense model: %s on device=%s", model_name, self._device)
        self._model = SentenceTransformer(model_name, device=self._device)

        # Encode corpus or load from cache
        self._corpus_embeddings = self._encode_or_load_corpus(
            chunks, cache_dir=cache_dir
        )

        # GPU Tensor Acceleration on RTX 3090
        self._corpus_tensor: torch.Tensor | None = None
        if self._device == "cuda" and torch.cuda.is_available():
            try:
                # Store in float16 on GPU (takes only ~1.5 GB VRAM for 1M vectors)
                self._corpus_tensor = torch.from_numpy(
                    self._corpus_embeddings
                ).to("cuda", dtype=torch.float16)
                logger.info(
                    "Corpus embeddings loaded to RTX 3090 GPU VRAM (shape=%s, dtype=float16).",
                    tuple(self._corpus_tensor.shape),
                )
            except Exception as exc:
                logger.warning(
                    "Could not load corpus embeddings to GPU VRAM (%s), falling back to CPU.",
                    exc,
                )
                self._corpus_tensor = None

    def _encode_or_load_corpus(
        self,
        chunks: list[LegalChunk],
        *,
        cache_dir: str | None = None,
    ) -> np.ndarray:
        """Encode corpus chunks, using disk cache if available."""
        cache_path: Path | None = None
        if cache_dir:
            cache_path = Path(cache_dir) / f"dense_corpus_{len(chunks)}.npy"
            if cache_path.is_file():
                logger.info("Loading cached corpus embeddings from %s", cache_path)
                try:
                    embeddings = np.load(str(cache_path))
                    if embeddings.shape[0] == len(chunks):
                        return embeddings
                    logger.warning(
                        "Cache size mismatch (%d vs %d), re-encoding.",
                        embeddings.shape[0],
                        len(chunks),
                    )
                except Exception as exc:
                    logger.warning("Failed to load embeddings cache (%s), re-encoding.", exc)

        logger.info(
            "Encoding %d chunks with dense model (batch_size=%d, device=%s)...",
            len(chunks),
            self._batch_size,
            self._device,
        )
        texts = [
            f"{self._passage_prefix}{chunk.retrieval_text}" for chunk in chunks
        ]

        with torch.inference_mode():
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cache_path), embeddings)
                logger.info("Corpus embeddings cached to %s", cache_path)
            except Exception as exc:
                logger.warning("Failed to cache corpus embeddings: %s", exc)

        return embeddings

    def retrieve(self, query: str, *, top_n: int = 30) -> list[DenseHit]:
        """Retrieve top-N chunks by cosine similarity."""
        if not query.strip():
            return []

        query_text = f"{self._query_prefix}{query}"

        # Fast GPU Tensor search on RTX 3090
        if self._corpus_tensor is not None and torch.cuda.is_available():
            with torch.inference_mode():
                q_emb = self._model.encode(
                    [query_text],
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                    device="cuda",
                )
                q_vec = q_emb.to(dtype=torch.float16)

                # Matrix-vector multiplication on RTX 3090 Tensor Cores (< 45ms)
                similarities = torch.matmul(self._corpus_tensor, q_vec.T).squeeze()
                k = min(top_n, similarities.shape[0])
                top_scores, top_indices = torch.topk(similarities, k)

                top_indices_list = top_indices.cpu().tolist()
                top_scores_list = top_scores.cpu().tolist()

            hits: list[DenseHit] = []
            for rank, (idx, score) in enumerate(
                zip(top_indices_list, top_scores_list), start=1
            ):
                chunk = self._chunks[idx]
                hits.append(
                    DenseHit(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        score=float(score),
                        rank=rank,
                    )
                )
            return hits

        # Fallback: CPU NumPy matrix multiplication
        with torch.inference_mode():
            query_embedding = self._model.encode(
                [query_text], normalize_embeddings=True
            )
        query_vec = np.asarray(query_embedding, dtype=np.float32)

        similarities = (self._corpus_embeddings @ query_vec.T).squeeze()

        if top_n >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_n)[-top_n:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        hits = []
        for rank, idx in enumerate(top_indices[:top_n], start=1):
            chunk = self._chunks[idx]
            hits.append(
                DenseHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(similarities[idx]),
                    rank=rank,
                )
            )

        return hits
