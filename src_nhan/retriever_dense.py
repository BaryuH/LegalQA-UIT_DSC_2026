"""Dense retriever using mainguyen9/vietlegal-e5 embeddings.

Encodes the legal chunk corpus and queries using the E5 model,
then retrieves by cosine similarity. Embeddings are cached to disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    """Dense retriever using sentence-transformers E5 model."""

    def __init__(
        self,
        chunks: list[LegalChunk],
        *,
        model_name: str = "mainguyen9/vietlegal-e5",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 64,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._chunks = chunks
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size

        logger.info("Loading dense model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)

        # Encode corpus or load from cache
        self._corpus_embeddings = self._encode_or_load_corpus(
            chunks, cache_dir=cache_dir
        )

    def _encode_or_load_corpus(
        self,
        chunks: list[LegalChunk],
        *,
        cache_dir: str | None = None,
    ) -> np.ndarray:
        """Encode corpus chunks, using disk cache if available."""
        cache_path: Path | None = None
        if cache_dir:
            cache_path = Path(cache_dir) / "dense_corpus_embeddings.npy"
            if cache_path.is_file():
                logger.info("Loading cached corpus embeddings from %s", cache_path)
                embeddings = np.load(str(cache_path))
                if embeddings.shape[0] == len(chunks):
                    return embeddings
                logger.warning(
                    "Cache size mismatch (%d vs %d), re-encoding.",
                    embeddings.shape[0],
                    len(chunks),
                )

        logger.info("Encoding %d chunks with dense model...", len(chunks))
        texts = [
            f"{self._passage_prefix}{chunk.retrieval_text}" for chunk in chunks
        ]
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), embeddings)
            logger.info("Corpus embeddings cached to %s", cache_path)

        return embeddings

    def retrieve(self, query: str, *, top_n: int = 30) -> list[DenseHit]:
        """Retrieve top-N chunks by cosine similarity."""
        if not query.strip():
            return []

        query_text = f"{self._query_prefix}{query}"
        query_embedding = self._model.encode(
            [query_text], normalize_embeddings=True
        )
        query_vec = np.asarray(query_embedding, dtype=np.float32)

        # Cosine similarity (embeddings are already normalized)
        similarities = (self._corpus_embeddings @ query_vec.T).squeeze()

        # Top-N indices
        if top_n >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_n)[-top_n:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        hits: list[DenseHit] = []
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
