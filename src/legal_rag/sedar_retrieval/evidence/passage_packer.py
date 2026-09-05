"""Pack SEDAR passage rankings into the shared ``PackedEvidence`` contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from legal_rag.evidence import pack_evidence
from legal_rag.schemas import LegalChunk, PackedEvidence, RetrievalHit
from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines
from legal_rag.sedar_retrieval.retrieval.passage_adapter import (
    BodySource,
    passage_to_legal_chunk,
)

ArticleDedupMode = Literal["off", "article", "clause", "first"]


class PassageEvidencePackError(ValueError):
    """Raised when ranked passages cannot be packed for reader inference."""


@dataclass(frozen=True, slots=True)
class RankedPassageCandidate:
    passage_id: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class PassageEvidenceConfig:
    """Evidence budget for the SEDAR passage path.

    ``evidence_top_k`` is the number of blocks wanted; ``candidate_window`` is
    how many ranked candidates are offered to the packer. They used to be the
    same number, which is the P1 defect: the candidate list was truncated to
    ``evidence_top_k`` *before* ``max_chunks_per_document`` and the character
    budget were applied, and anything those constraints dropped was not backfilled.
    Four candidates where three share a document therefore produced a two-block
    pack, silently — ``pack_evidence`` only raises when *nothing* fits.

    Defaults keep the historical behaviour exactly: ``candidate_window=0`` means
    "same as ``evidence_top_k``", and no ``target_blocks`` is passed downstream.
    Set ``candidate_window`` to something wider (16 is a reasonable start) to
    enable backfill.

    ``include_document_name`` also defaults to the old behaviour even though the
    old behaviour is wrong (the header shows the zip member instead of the
    document name). It has to: the frozen reader was fine-tuned on evidence
    rendered the old way, so turning this on changes the reader's input
    distribution. It must be measured as its own ablation arm against a control
    that reproduces the champion, not switched on silently underneath one.
    """

    evidence_top_k: int = 4
    max_total_chars: int = 4000
    max_chunks_per_document: int = 2
    candidate_window: int = 0
    body_source: BodySource = "raw_text"
    dedup_article_mode: ArticleDedupMode = "off"
    include_document_name: bool = False

    def __post_init__(self) -> None:
        if self.evidence_top_k <= 0:
            raise ValueError("evidence_top_k must be positive")
        if self.max_total_chars <= 0:
            raise ValueError("max_total_chars must be positive")
        if self.max_chunks_per_document <= 0:
            raise ValueError("max_chunks_per_document must be positive")
        if self.candidate_window < 0:
            raise ValueError("candidate_window must be non-negative")
        if 0 < self.candidate_window < self.evidence_top_k:
            raise ValueError("candidate_window must be zero or at least evidence_top_k")
        if self.body_source not in {"raw_text", "reader_text"}:
            raise ValueError(f"Unsupported body_source: {self.body_source!r}")
        if self.dedup_article_mode not in {"off", "article", "clause", "first"}:
            raise ValueError(
                f"Unsupported dedup_article_mode: {self.dedup_article_mode!r}"
            )

    @property
    def effective_window(self) -> int:
        """Number of ranked candidates handed to the packer."""

        return self.candidate_window or self.evidence_top_k

    @property
    def backfill_enabled(self) -> bool:
        return self.effective_window > self.evidence_top_k


def _score_from_payload(payload: Mapping[str, object], *, rank: int) -> float:
    scores = payload.get("scores")
    if isinstance(scores, list) and scores:
        for item in scores:
            if not isinstance(item, Mapping):
                continue
            item_rank = item.get("rank")
            if item_rank == rank:
                for key in ("dense", "bm25", "rrf", "score"):
                    value = item.get(key)
                    if isinstance(value, (int, float)):
                        return float(value)
        first = scores[0]
        if isinstance(first, Mapping):
            for key in ("dense", "bm25", "rrf", "score"):
                value = first.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
    return 1.0 / float(rank)


def load_retrieval_rankings(
    path: str | Path,
) -> dict[str, tuple[RankedPassageCandidate, ...]]:
    """Load query_id -> ranked passage candidates from retrieval JSONL."""

    rankings: dict[str, tuple[RankedPassageCandidate, ...]] = {}
    for line in iter_jsonl_lines(path):
        payload = json.loads(line)
        query_id = str(payload.get("query_id", "")).strip()
        if not query_id:
            raise PassageEvidencePackError("Retrieval row is missing query_id")
        ranked_ids = payload.get("ranked_ids")
        if not isinstance(ranked_ids, list) or not ranked_ids:
            raise PassageEvidencePackError(
                f"Retrieval row for {query_id!r} has no ranked_ids"
            )
        candidates: list[RankedPassageCandidate] = []
        for rank, passage_id in enumerate(ranked_ids, start=1):
            normalized = str(passage_id).strip()
            if not normalized:
                raise PassageEvidencePackError(
                    f"Blank passage_id at rank {rank} for query {query_id!r}"
                )
            candidates.append(
                RankedPassageCandidate(
                    passage_id=normalized,
                    rank=rank,
                    score=_score_from_payload(payload, rank=rank),
                )
            )
        rankings[query_id] = tuple(candidates)
    if not rankings:
        raise PassageEvidencePackError(f"No retrieval rows found in {path}")
    return dict(sorted(rankings.items()))


def dedup_candidates_by_article(
    candidates: Sequence[RankedPassageCandidate],
    passages: Mapping[str, CanonicalPassage],
    *,
    mode: ArticleDedupMode = "off",
) -> tuple[RankedPassageCandidate, ...]:
    """Collapse candidates that belong to the same article (P4).

    The retrieval views index both ``article`` and ``clause`` levels
    (``hierarchy.nodes_to_passages`` defaults to ``("article", "clause")``), so
    ``Điều 76`` and ``Khoản 1 Điều 76`` are two competing candidates whose text
    is nested. A top-4 holding both spends two slots on the same provision and
    burns the ``max_chunks_per_document`` cap, which then blocks a second
    document — the ``RIGHT_DOCUMENT_WRONG_CHUNK`` failure in
    ``docs/ERROR_TAXONOMY.md``.

    Modes:

    ``off``
        no change (default; preserves the historical pack).
    ``article``
        keep the article-level passage of each article, drop its clauses.
    ``clause``
        keep clause-level passages, drop the whole-article passage.
    ``first``
        keep whichever ranked higher.

    Article identity is document-scoped: the key falls back to
    ``{document_id}::art::{article_number}``, never a bare article number, so
    two documents that share ``Điều 76`` are never collapsed together.
    """

    if mode == "off":
        return tuple(candidates)

    def article_key(passage: CanonicalPassage) -> str | None:
        if passage.article_id:
            return str(passage.article_id)
        if passage.article_number:
            return f"{passage.document_id}::art::{passage.article_number}"
        return None

    def rank_of(level: str) -> int:
        if mode == "article":
            return 0 if level == "article" else 1
        if mode == "clause":
            return 0 if level == "clause" else 1
        return 0

    best: dict[str, RankedPassageCandidate] = {}
    best_score: dict[str, tuple[int, int]] = {}
    passthrough: list[RankedPassageCandidate] = []

    for candidate in candidates:
        passage = passages.get(candidate.passage_id)
        if passage is None:
            raise PassageEvidencePackError(
                f"Passage {candidate.passage_id!r} is absent from the corpus view"
            )
        key = article_key(passage)
        if key is None:
            passthrough.append(candidate)
            continue
        preference = (rank_of(passage.retrieval_level), candidate.rank)
        if key not in best or preference < best_score[key]:
            best[key] = candidate
            best_score[key] = preference

    kept = list(best.values()) + passthrough
    return tuple(sorted(kept, key=lambda item: (item.rank, item.passage_id)))


def ranked_passages_to_hits(
    candidates: Sequence[RankedPassageCandidate],
    passages: Mapping[str, CanonicalPassage],
    *,
    limit: int | None = None,
    body_source: BodySource = "raw_text",
    evidence_top_k: int | None = None,
) -> tuple[RetrievalHit, ...]:
    """Convert ranked passage IDs into ``RetrievalHit`` rows for ``pack_evidence``.

    ``limit`` is how many candidates to hand over, which is the candidate window
    rather than the number of blocks wanted — ``pack_evidence`` decides the block
    count via its own ``target_blocks``. ``evidence_top_k`` is kept as a
    backward-compatible alias for callers written before the two were separated.
    """

    if limit is None and evidence_top_k is None:
        raise TypeError("ranked_passages_to_hits requires limit or evidence_top_k")
    window = limit if limit is not None else evidence_top_k
    assert window is not None  # narrowed by the check above
    if window <= 0:
        raise ValueError("limit must be positive")

    selected = list(candidates[:window])
    hits: list[RetrievalHit] = []
    for candidate in selected:
        passage = passages.get(candidate.passage_id)
        if passage is None:
            raise PassageEvidencePackError(
                f"Passage {candidate.passage_id!r} is absent from the corpus view"
            )
        chunk = passage_to_legal_chunk(passage, body_source=body_source)
        hits.append(
            RetrievalHit(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_path=chunk.source_path,
                source_member=chunk.source_member,
                section_label=chunk.section_label,
                rank=candidate.rank,
                bm25_score=candidate.score,
                rerank_score=None,
            )
        )
    return tuple(hits)


def pack_passage_retrieval_evidence(
    candidates: Sequence[RankedPassageCandidate],
    passages: Mapping[str, CanonicalPassage],
    *,
    config: PassageEvidenceConfig,
) -> PackedEvidence:
    """Pack ranked SEDAR passages using the same budget contract as R0."""

    selected = dedup_candidates_by_article(
        candidates,
        passages,
        mode=config.dedup_article_mode,
    )
    hits = ranked_passages_to_hits(
        selected,
        passages,
        limit=config.effective_window,
        body_source=config.body_source,
    )
    chunks: dict[str, LegalChunk] = {}
    document_names: dict[str, str] = {}
    for hit in hits:
        passage = passages[hit.chunk_id]
        chunks[hit.chunk_id] = passage_to_legal_chunk(
            passage,
            body_source=config.body_source,
        )
        if config.include_document_name and passage.document_name:
            document_names.setdefault(passage.document_id, passage.document_name)
    return pack_evidence(
        hits,
        chunks,
        max_total_chars=config.max_total_chars,
        max_chunks_per_document=config.max_chunks_per_document,
        document_names=document_names or None,
        target_blocks=config.evidence_top_k if config.backfill_enabled else None,
    )


__all__ = [
    "ArticleDedupMode",
    "PassageEvidenceConfig",
    "PassageEvidencePackError",
    "RankedPassageCandidate",
    "dedup_candidates_by_article",
    "load_retrieval_rankings",
    "pack_passage_retrieval_evidence",
    "ranked_passages_to_hits",
]
