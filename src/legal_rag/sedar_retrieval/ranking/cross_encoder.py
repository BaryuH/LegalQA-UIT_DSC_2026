"""Cross-encoder reranking over an existing ranking (R3).

Nothing in the SEDAR path reads a (query, passage) pair jointly. BM25 and the
dense retriever score the two sides independently, and LambdaRank combines rank
lists using cheap lexical features. ``retrieval/reranker.py`` is a bi-encoder
(``BAAI/bge-m3`` embeddings plus cosine), i.e. a second dense retriever, not a
cross-encoder.

That gap is what the measured numbers point at. On clean-460 the candidate set
holds the right article for 97.5% of labeled queries at rank 500, while the
top-4 the reader actually receives holds it for 78%. Recall is close to its
ceiling; precision at the pack cutoff is untouched. A perfect reranker over
top-100 would be worth about +0.16 article recall@4.

The reordering is a pure function of a score callable, so it is testable
without a model. ``CrossEncoderScorer`` is the only part that needs weights.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage

TextSource = Literal["reader_text", "raw_text"]
ScoreFn = Callable[[str, Sequence[str]], Sequence[float]]

CROSS_ENCODER_SCHEMA_VERSION = "sedar-cross-encoder-rerank-v1"


class CrossEncoderError(RuntimeError):
    """Raised when reranking cannot proceed without violating the contract."""


@dataclass(frozen=True, slots=True)
class RerankedQuery:
    query_id: str
    ranked_ids: tuple[str, ...]
    scores: tuple[dict[str, object], ...]
    reranked_count: int
    tail_count: int


def passage_text(
    passage: CanonicalPassage,
    *,
    text_source: TextSource = "reader_text",
) -> str:
    """Return the text handed to the cross-encoder.

    Never ``retrieval_text``: for the R2a view that field is wrapped in
    ``[DOCUMENT CONTEXT]`` / ``[HIERARCHY]`` markers and repeats the document
    name, which spends the encoder's 512-token budget on boilerplate that is
    identical for every passage of a document.
    """

    if text_source == "raw_text":
        return passage.raw_text
    return passage.reader_text


def rerank_rankings(
    rankings: Mapping[str, Sequence[str]],
    questions: Mapping[str, str],
    passages: Mapping[str, CanonicalPassage],
    *,
    score_fn: ScoreFn,
    top_k: int = 100,
    text_source: TextSource = "reader_text",
) -> tuple[RerankedQuery, ...]:
    """Rerank the first ``top_k`` candidates of each query, keeping the tail.

    Candidates beyond ``top_k`` are appended in their original order rather than
    dropped, so recall at deeper cutoffs is unchanged and the output stays a
    drop-in replacement for the input ranking.
    """

    if top_k <= 0:
        raise CrossEncoderError("top_k must be positive")

    results: list[RerankedQuery] = []
    for query_id in sorted(rankings):
        question = questions.get(query_id)
        if question is None:
            raise CrossEncoderError(f"No question text for query_id={query_id!r}")
        candidates = list(rankings[query_id])
        if len(set(candidates)) != len(candidates):
            raise CrossEncoderError(f"Duplicate passage_id for query {query_id!r}")

        head, tail = candidates[:top_k], candidates[top_k:]
        texts: list[str] = []
        for passage_id in head:
            passage = passages.get(passage_id)
            if passage is None:
                raise CrossEncoderError(
                    f"Passage {passage_id!r} is absent from the corpus view"
                )
            texts.append(passage_text(passage, text_source=text_source))

        raw_scores = list(score_fn(question, texts)) if texts else []
        if len(raw_scores) != len(head):
            raise CrossEncoderError(
                f"Scorer returned {len(raw_scores)} scores for {len(head)} "
                f"candidates on query {query_id!r}"
            )

        # Sort by descending score, passage_id as a deterministic tiebreak.
        order = sorted(
            zip(head, (float(value) for value in raw_scores), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        results.append(
            RerankedQuery(
                query_id=query_id,
                ranked_ids=tuple(passage_id for passage_id, _ in order) + tuple(tail),
                scores=tuple(
                    {"passage_id": passage_id, "ce_score": score, "ce_rank": rank}
                    for rank, (passage_id, score) in enumerate(order, start=1)
                ),
                reranked_count=len(head),
                tail_count=len(tail),
            )
        )
    return tuple(results)


class CrossEncoderScorer:
    """Thin, auditable adapter around a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        max_length: int = 512,
        revision: str | None = None,
        local_files_only: bool = True,
        batch_size: int = 16,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if max_length <= 0 or batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")
        self.batch_size = batch_size
        if model_factory is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise CrossEncoderError(
                    "Cross-encoder reranking requires sentence-transformers."
                ) from exc
            model_factory = CrossEncoder
        kwargs: dict[str, Any] = {
            "device": device,
            "max_length": max_length,
            "local_files_only": local_files_only,
        }
        # A local directory is read straight from disk; passing a revision
        # alongside it sends the loader back through hub resolution, which is
        # exactly how the Qwen snapshot blocked the dense rebuild.
        if revision and not _looks_like_path(model):
            kwargs["revision"] = revision
        self.model = model_factory(model, **kwargs)

    def __call__(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        if not texts:
            return ()
        pairs = [(query, text) for text in texts]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in scores]


def _looks_like_path(model: str) -> bool:
    """An existing directory is a local snapshot, whatever its spelling."""

    from pathlib import Path

    try:
        return Path(model).expanduser().is_dir()
    except OSError:  # pragma: no cover - defensive on odd path input
        return False


__all__ = [
    "CROSS_ENCODER_SCHEMA_VERSION",
    "CrossEncoderError",
    "CrossEncoderScorer",
    "RerankedQuery",
    "ScoreFn",
    "TextSource",
    "passage_text",
    "rerank_rankings",
]
