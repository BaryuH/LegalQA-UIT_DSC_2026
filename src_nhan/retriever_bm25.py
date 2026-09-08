"""BM25 lexical retriever with Vietnamese word segmentation.

Uses underthesea tokenization for Vietnamese-aware BM25 indexing,
which correctly handles compound legal terms like "quyền_sử_dụng_đất".
"""

from __future__ import annotations

import logging
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
    """BM25 retriever with Vietnamese tokenization."""

    def __init__(
        self,
        chunks: list[LegalChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._chunks = chunks
        self._chunk_index: dict[int, LegalChunk] = {
            i: c for i, c in enumerate(chunks)
        }

        logger.info(
            "Tokenizing %d chunks for BM25 index (underthesea)...", len(chunks)
        )
        self._tokenized_corpus = [
            segment(chunk.retrieval_text) for chunk in chunks
        ]

        logger.info("Building BM25 index (k1=%.2f, b=%.2f)...", k1, b)
        self._bm25 = BM25Okapi(
            self._tokenized_corpus, k1=k1, b=b
        )
        logger.info("BM25 index built with %d documents.", len(chunks))

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
