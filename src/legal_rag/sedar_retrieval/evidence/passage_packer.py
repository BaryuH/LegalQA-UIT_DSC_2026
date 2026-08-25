"""Pack SEDAR passage rankings into the shared ``PackedEvidence`` contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from legal_rag.evidence import pack_evidence
from legal_rag.schemas import LegalChunk, PackedEvidence, RetrievalHit
from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines
from legal_rag.sedar_retrieval.retrieval.passage_adapter import passage_to_legal_chunk


class PassageEvidencePackError(ValueError):
    """Raised when ranked passages cannot be packed for reader inference."""


@dataclass(frozen=True, slots=True)
class RankedPassageCandidate:
    passage_id: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class PassageEvidenceConfig:
    evidence_top_k: int = 4
    max_total_chars: int = 4000
    max_chunks_per_document: int = 2

    def __post_init__(self) -> None:
        if self.evidence_top_k <= 0:
            raise ValueError("evidence_top_k must be positive")
        if self.max_total_chars <= 0:
            raise ValueError("max_total_chars must be positive")
        if self.max_chunks_per_document <= 0:
            raise ValueError("max_chunks_per_document must be positive")


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


def ranked_passages_to_hits(
    candidates: Sequence[RankedPassageCandidate],
    passages: Mapping[str, CanonicalPassage],
    *,
    evidence_top_k: int,
) -> tuple[RetrievalHit, ...]:
    """Convert ranked passage IDs into ``RetrievalHit`` rows for ``pack_evidence``."""

    selected = list(candidates[:evidence_top_k])
    hits: list[RetrievalHit] = []
    for candidate in selected:
        passage = passages.get(candidate.passage_id)
        if passage is None:
            raise PassageEvidencePackError(
                f"Passage {candidate.passage_id!r} is absent from the corpus view"
            )
        chunk = passage_to_legal_chunk(passage)
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

    hits = ranked_passages_to_hits(
        candidates,
        passages,
        evidence_top_k=config.evidence_top_k,
    )
    chunks: dict[str, LegalChunk] = {}
    for hit in hits:
        passage = passages[hit.chunk_id]
        chunks[hit.chunk_id] = passage_to_legal_chunk(passage)
    return pack_evidence(
        hits,
        chunks,
        max_total_chars=config.max_total_chars,
        max_chunks_per_document=config.max_chunks_per_document,
    )


__all__ = [
    "PassageEvidenceConfig",
    "PassageEvidencePackError",
    "RankedPassageCandidate",
    "load_retrieval_rankings",
    "pack_passage_retrieval_evidence",
    "ranked_passages_to_hits",
]
