"""BM25 lexical retriever with Vietnamese word segmentation and multi-core acceleration.

Uses underthesea tokenization for Vietnamese-aware BM25 indexing.
Accelerated with multi-core multiprocessing on AMD EPYC and persistent disk/RAM caching.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunker import LegalChunk
from .tokenizer_vi import segment

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BM25Hit:
    """One BM25 retrieval result."""

    chunk_id: str
    document_id: str
    score: float
    rank: int


class BM25Retriever:
    """BM25 retriever with Vietnamese tokenization, parallel build, and caching."""

    def __init__(
        self,
        chunks: list[LegalChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        cache_dir: str | Path | None = None,
        num_workers: int = 30,
    ) -> None:
        self._chunks = chunks
        self._chunk_index: dict[int, LegalChunk] = {
            i: c for i, c in enumerate(chunks)
        }

        # Check disk cache first
        cache_path: Path | None = None
        if cache_dir:
            cache_path = Path(cache_dir) / f"bm25_index_{len(chunks)}.pkl"
            if cache_path.is_file():
                logger.info("Loading cached BM25 index from %s...", cache_path)
                try:
                    loaded = self.load_index(cache_path)
                    if len(loaded._chunks) == len(chunks):
                        self._chunks = loaded._chunks
                        self._chunk_index = loaded._chunk_index
                        self._tokenized_corpus = loaded._tokenized_corpus
                        self._bm25 = loaded._bm25
                        logger.info("BM25 index successfully restored from cache.")
                        return
                    logger.warning("Cache size mismatch, re-building index.")
                except Exception as exc:
                    logger.warning("Failed to load BM25 cache (%s), re-building.", exc)

        # Multi-core tokenization on AMD EPYC
        texts = [chunk.retrieval_text for chunk in chunks]
        workers = max(1, min(num_workers, mp.cpu_count() or 1))

        if workers > 1 and len(texts) > 200:
            logger.info(
                "Tokenizing %d chunks for BM25 index in parallel across %d CPU workers (underthesea)...",
                len(texts),
                workers,
            )
            ctx = mp.get_context("fork")
            chunksize = max(50, len(texts) // (workers * 8))
            with ctx.Pool(processes=workers) as pool:
                self._tokenized_corpus = pool.map(
                    segment, texts, chunksize=chunksize
                )
        else:
            logger.info(
                "Tokenizing %d chunks for BM25 index sequentially (underthesea)...",
                len(texts),
            )
            self._tokenized_corpus = [segment(t) for t in texts]

        logger.info("Building BM25Okapi index (k1=%.2f, b=%.2f)...", k1, b)
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=k1, b=b)
        logger.info("BM25 index built with %d documents.", len(chunks))

        # Persist to disk cache
        if cache_path:
            try:
                self.save_index(cache_path)
            except Exception as exc:
                logger.warning("Failed to save BM25 cache: %s", exc)

    def retrieve(self, query: str, *, top_n: int = 30) -> list[BM25Hit]:
        """Retrieve top-N chunks for a query."""
        if not query.strip():
            return []

        tokenized_query = segment(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Get top-N indices sorted by score descending
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_n]

        hits: list[BM25Hit] = []
        for rank, idx in enumerate(scored_indices, start=1):
            if scores[idx] <= 0:
                break
            chunk = self._chunk_index[idx]
            hits.append(
                BM25Hit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        return hits

    def save_index(self, path: str | Path) -> None:
        """Persist BM25 index to disk."""
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "chunks": self._chunks,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("BM25 index saved to %s", cache_path)

    @classmethod
    def load_index(cls, path: str | Path) -> BM25Retriever:
        """Load a persisted BM25 index."""
        cache_path = Path(path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"BM25 index not found: {cache_path}")
        with cache_path.open("rb") as fh:
            data = pickle.load(fh)  # noqa: S301
        instance = cls.__new__(cls)
        instance._chunks = data["chunks"]
        instance._chunk_index = {i: c for i, c in enumerate(data["chunks"])}
        instance._tokenized_corpus = data["tokenized_corpus"]
        instance._bm25 = data["bm25"]
        logger.info("BM25 index loaded from %s", cache_path)
        return instance
