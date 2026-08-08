"""Auditable BM25 indexing over legal-chunk retrieval views."""

from __future__ import annotations

import heapq
import json
import math
import os
import tempfile
from array import array
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from ..schemas import LegalChunk, RetrievalHit
from ..text.cache import ChunkCacheFingerprint
from ..text.normalize import tokenize_legal_text

try:  # Optional acceleration; the exact Python path remains the compatibility path.
    import numpy as _numpy
except ImportError:  # pragma: no cover - exercised only in minimal installations
    _numpy = None  # type: ignore[assignment]

BM25_INDEX_SCHEMA_VERSION = "c4.bm25-index.v1"
BM25_INDEX_VERSION = "legal-bm25-v1"
BM25_CUDA_SCORER_VERSION = "legal-bm25-cuda-v1-fp32"
IndexStatus = Literal["hit", "miss", "stale"]
IndexLoadPolicy = Literal["strict", "auto_rebuild"]
BM25Backend = Literal["cpu", "cuda"]


class BM25IndexError(RuntimeError):
    """Base error for unsafe or invalid BM25-index operations."""


class BM25IndexConflictError(BM25IndexError):
    """Raised when a cache path already contains different index content."""


class BM25IndexMissingError(BM25IndexError):
    """Raised when strict loading cannot find the requested index."""


class BM25IndexStaleError(BM25IndexError):
    """Raised when an index cannot satisfy its expected fingerprint."""


class BM25CudaUnavailableError(BM25IndexError):
    """Raised when an explicitly requested CUDA BM25 backend is unavailable."""


def _sha256_json(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _require_identity_part(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be blank")
    if any(character in value for character in "/\\"):
        raise ValueError(f"{name} must not contain path separators")
    return value


@dataclass(frozen=True, slots=True)
class BM25Config:
    """Validated BM25 hyperparameters and algorithm version."""

    k1: float = 1.5
    b: float = 0.75
    version: str = BM25_INDEX_VERSION

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and greater than zero")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must be finite and between zero and one")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must not be blank")

    def as_dict(self) -> dict[str, float | str]:
        return {"b": self.b, "k1": self.k1, "version": self.version}


def _chunk_cache_identity(value: str | ChunkCacheFingerprint) -> str:
    if isinstance(value, ChunkCacheFingerprint):
        return value.cache_fingerprint
    return _require_identity_part(value, "chunk_cache_fingerprint")


@dataclass(frozen=True, slots=True)
class BM25IndexFingerprint:
    """Complete identity for an index derived from one chunk cache."""

    chunk_cache_fingerprint: str
    bm25_config_hash: str
    index_version: str = BM25_INDEX_VERSION
    schema_version: str = BM25_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identity_part(self.chunk_cache_fingerprint, "chunk_cache_fingerprint")
        _require_identity_part(self.bm25_config_hash, "bm25_config_hash")
        _require_identity_part(self.index_version, "index_version")
        _require_identity_part(self.schema_version, "schema_version")

    @classmethod
    def from_config(
        cls,
        chunk_cache_fingerprint: str | ChunkCacheFingerprint,
        config: BM25Config,
        index_version: str | None = None,
    ) -> BM25IndexFingerprint:
        return cls(
            chunk_cache_fingerprint=_chunk_cache_identity(chunk_cache_fingerprint),
            bm25_config_hash=_sha256_json(config.as_dict()),
            index_version=config.version if index_version is None else index_version,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "bm25_config_hash": self.bm25_config_hash,
            "chunk_cache_fingerprint": self.chunk_cache_fingerprint,
            "index_version": self.index_version,
            "schema_version": self.schema_version,
        }

    @property
    def index_fingerprint(self) -> str:
        """Return a stable, path-safe identity for this index."""

        return _sha256_json(self.as_dict())


@dataclass(frozen=True, slots=True)
class BM25IndexedDocument:
    """One indexed chunk with only retrieval-view text and provenance."""

    ordinal: int
    chunk_id: str
    document_id: str
    source_path: str
    retrieval_text: str
    term_frequencies: tuple[tuple[str, int], ...]
    document_length: int
    source_member: str | None = None
    section_label: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must not be negative")
        for value, name in (
            (self.chunk_id, "chunk_id"),
            (self.document_id, "document_id"),
            (self.source_path, "source_path"),
            (self.retrieval_text, "retrieval_text"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-blank")
        if self.document_length < 0:
            raise ValueError("document_length must not be negative")
        if tuple(sorted(self.term_frequencies)) != self.term_frequencies:
            raise ValueError("term_frequencies must be sorted")
        if any(
            not isinstance(term, str) or not term or count <= 0
            for term, count in self.term_frequencies
        ):
            raise ValueError("term_frequencies must contain positive counts")
        if len({term for term, _ in self.term_frequencies}) != len(
            self.term_frequencies
        ):
            raise ValueError("term_frequencies must not contain duplicate terms")
        if sum(count for _, count in self.term_frequencies) != self.document_length:
            raise ValueError("document_length must equal term-frequency total")
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("Document offsets must be provided together")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError("Document end_offset must be greater than start_offset")

    def as_dict(self) -> dict[str, Any]:
        """Serialize the retrieval view without ``raw_text`` or answer fields."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_length": self.document_length,
            "end_offset": self.end_offset,
            "ordinal": self.ordinal,
            "retrieval_text": self.retrieval_text,
            "section_label": self.section_label,
            "source_member": self.source_member,
            "source_path": self.source_path,
            "start_offset": self.start_offset,
            "term_frequencies": {term: count for term, count in self.term_frequencies},
        }

    @classmethod
    def from_dict(cls, value: object) -> BM25IndexedDocument:
        if not isinstance(value, dict):
            raise TypeError("BM25 document record must be an object")
        allowed = {
            "chunk_id",
            "document_id",
            "document_length",
            "end_offset",
            "ordinal",
            "retrieval_text",
            "section_label",
            "source_member",
            "source_path",
            "start_offset",
            "term_frequencies",
        }
        if set(value) != allowed:
            raise ValueError("BM25 document record contains unexpected fields")
        term_frequencies = value["term_frequencies"]
        if not isinstance(term_frequencies, dict):
            raise TypeError("term_frequencies must be an object")
        parsed_terms = tuple(
            sorted((str(term), int(count)) for term, count in term_frequencies.items())
        )
        document = cls(
            ordinal=int(value["ordinal"]),
            chunk_id=str(value["chunk_id"]),
            document_id=str(value["document_id"]),
            source_path=str(value["source_path"]),
            retrieval_text=str(value["retrieval_text"]),
            term_frequencies=parsed_terms,
            document_length=int(value["document_length"]),
            source_member=(
                None if value["source_member"] is None else str(value["source_member"])
            ),
            section_label=(
                None if value["section_label"] is None else str(value["section_label"])
            ),
            start_offset=(
                None if value["start_offset"] is None else int(value["start_offset"])
            ),
            end_offset=(
                None if value["end_offset"] is None else int(value["end_offset"])
            ),
        )
        expected_terms = Counter(tokenize_legal_text(document.retrieval_text))
        if document.term_frequencies != tuple(sorted(expected_terms.items())):
            raise ValueError("term_frequencies do not match retrieval_text")
        return document


@dataclass(frozen=True, slots=True)
class BM25Summary:
    """Corpus statistics recorded in the auditable index metadata."""

    corpus_size: int
    vocabulary_size: int
    total_tokens: int
    average_document_length: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "average_document_length": self.average_document_length,
            "corpus_size": self.corpus_size,
            "total_tokens": self.total_tokens,
            "vocabulary_size": self.vocabulary_size,
        }


@dataclass(frozen=True, slots=True)
class BM25Index:
    """In-memory BM25 index over deterministic retrieval-view documents."""

    fingerprint: BM25IndexFingerprint
    config: BM25Config
    documents: tuple[BM25IndexedDocument, ...]
    vocabulary: tuple[str, ...]
    document_frequencies: tuple[tuple[str, int], ...]
    summary: BM25Summary
    cache_path: Path | None = None

    def __post_init__(self) -> None:
        if tuple(range(len(self.documents))) != tuple(
            document.ordinal for document in self.documents
        ):
            raise ValueError("BM25 documents must have deterministic ordinals")
        if tuple(sorted(self.vocabulary)) != self.vocabulary:
            raise ValueError("BM25 vocabulary must be sorted")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("BM25 vocabulary must not contain duplicates")
        if tuple(sorted(self.document_frequencies)) != self.document_frequencies:
            raise ValueError("BM25 document frequencies must be sorted")
        if tuple(term for term, _ in self.document_frequencies) != self.vocabulary:
            raise ValueError("BM25 document frequencies must cover the vocabulary")
        expected_frequencies = _document_frequencies(self.documents)
        if self.document_frequencies != expected_frequencies:
            raise ValueError("BM25 document frequencies do not match documents")
        if self.summary != _summary(self.documents, self.vocabulary):
            raise ValueError("BM25 summary does not match documents")

    @property
    def index_fingerprint(self) -> str:
        return self.fingerprint.index_fingerprint


@dataclass(frozen=True, slots=True)
class BM25IndexResult:
    """Explicit result for BM25 cache load/build operations."""

    status: IndexStatus
    cache_path: Path
    fingerprint: BM25IndexFingerprint
    index: BM25Index | None
    summary: BM25Summary
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BM25PostingList:
    """Compact ordinal/term-frequency postings for one query term."""

    ordinals: array[int]
    term_frequencies: array[int]

    def __post_init__(self) -> None:
        if len(self.ordinals) != len(self.term_frequencies):
            raise ValueError("BM25 posting arrays must have equal lengths")
        if tuple(sorted(self.ordinals)) != tuple(self.ordinals):
            raise ValueError("BM25 posting ordinals must be sorted")
        if any(term_frequency <= 0 for term_frequency in self.term_frequencies):
            raise ValueError("BM25 posting term frequencies must be positive")


@dataclass(frozen=True, slots=True)
class BM25QueryCache:
    """In-memory postings limited to the terms used by one inference batch."""

    index_fingerprint: str
    postings: Mapping[str, BM25PostingList]
    document_frequencies: Mapping[str, int]
    document_lengths: array[int]

    def __post_init__(self) -> None:
        if not isinstance(self.index_fingerprint, str) or not self.index_fingerprint:
            raise ValueError("index_fingerprint must not be blank")
        if set(self.postings) != set(self.document_frequencies):
            raise ValueError(
                "BM25 query cache postings and document frequencies must cover "
                "the same terms"
            )
        for term, posting in self.postings.items():
            if not isinstance(term, str) or not term:
                raise ValueError("BM25 query cache terms must be non-blank")
            if any(ordinal < 0 for ordinal in posting.ordinals):
                raise ValueError("BM25 query cache ordinals must be non-negative")
        if any(length < 0 for length in self.document_lengths):
            raise ValueError("BM25 query cache document lengths must be non-negative")


@dataclass(frozen=True, slots=True)
class BM25CudaQueryCache:
    """CUDA-resident derived postings for one immutable :class:`BM25QueryCache`.

    This is derived state only: the persisted BM25 index remains byte-identical and
    the backend never contains question answers or references.
    """

    index_fingerprint: str
    device: str
    document_lengths: Any
    postings: Mapping[str, tuple[Any, Any]]


def _require_cuda_torch() -> Any:
    """Load PyTorch lazily and fail closed when CUDA scoring was requested."""

    try:
        torch = import_module("torch")
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise BM25CudaUnavailableError(
            "BM25 CUDA backend requires the optional 'torch' package"
        ) from exc
    if not torch.cuda.is_available():
        raise BM25CudaUnavailableError(
            "BM25 CUDA backend was requested but torch.cuda.is_available() is false"
        )
    return torch


def build_bm25_query_cache(
    index: BM25Index,
    queries: Iterable[str],
) -> BM25QueryCache:
    """Build exact BM25 postings for the terms used by a query batch.

    The persisted index remains unchanged.  Restricting the derived postings to
    the batch's query vocabulary avoids scanning every corpus document for every
    question while preserving the original BM25 scoring formula.
    """

    query_terms: set[str] = set()
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("BM25 query must be a non-blank string")
        query_terms.update(tokenize_legal_text(query))

    postings_lists: dict[str, tuple[array, array]] = {
        term: (array("I"), array("I")) for term in query_terms
    }
    if postings_lists:
        for ordinal, document in enumerate(index.documents):
            for term, term_frequency in document.term_frequencies:
                entries = postings_lists.get(term)
                if entries is not None:
                    entries[0].append(ordinal)
                    entries[1].append(term_frequency)

    document_frequencies = dict(index.document_frequencies)
    return BM25QueryCache(
        index_fingerprint=index.index_fingerprint,
        postings={
            term: BM25PostingList(
                ordinals=entries[0],
                term_frequencies=entries[1],
            )
            for term, entries in postings_lists.items()
        },
        document_frequencies={
            term: document_frequencies.get(term, 0) for term in query_terms
        },
        document_lengths=array(
            "I", (document.document_length for document in index.documents)
        ),
    )


def build_bm25_cuda_query_cache(
    query_cache: BM25QueryCache,
    *,
    device: str = "cuda",
) -> BM25CudaQueryCache:
    """Transfer one derived query cache to CUDA once for repeated exact scoring.

    CUDA uses the explicitly versioned float32 scorer for RTX-class throughput.
    Dataset manifests bind this backend version so its output is never confused
    with the compatibility CPU scorer.
    """

    torch = _require_cuda_torch()
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("BM25 CUDA cache device must resolve to a CUDA device")
    return BM25CudaQueryCache(
        index_fingerprint=query_cache.index_fingerprint,
        device=str(resolved_device),
        document_lengths=torch.tensor(
            list(query_cache.document_lengths),
            dtype=torch.float32,
            device=resolved_device,
        ),
        postings={
            term: (
                torch.tensor(
                    list(posting.ordinals), dtype=torch.long, device=resolved_device
                ),
                torch.tensor(
                    list(posting.term_frequencies),
                    dtype=torch.float32,
                    device=resolved_device,
                ),
            )
            for term, posting in query_cache.postings.items()
        },
    )


def _retrieve_bm25_cuda(
    index: BM25Index,
    *,
    query_terms: set[str],
    top_k: int,
    query_cache: BM25QueryCache,
    cuda_query_cache: BM25CudaQueryCache,
) -> list[tuple[float, BM25IndexedDocument]]:
    """Score one query using CUDA-resident postings and deterministic CPU ordering."""

    if cuda_query_cache.index_fingerprint != index.index_fingerprint:
        raise ValueError("BM25 CUDA query cache does not match the loaded index")
    missing_terms = query_terms.difference(cuda_query_cache.postings)
    if missing_terms:
        raise ValueError(
            "BM25 CUDA query cache does not cover query terms: "
            + ", ".join(sorted(missing_terms))
        )
    torch = _require_cuda_torch()
    corpus_size = len(index.documents)
    scores = torch.zeros(
        corpus_size, dtype=torch.float32, device=cuda_query_cache.device
    )
    average_length = index.summary.average_document_length
    for term in sorted(query_terms):
        frequency = query_cache.document_frequencies[term]
        if frequency <= 0:
            continue
        idf = math.log(1.0 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
        ordinals, term_frequencies = cuda_query_cache.postings[term]
        length_ratio = (
            cuda_query_cache.document_lengths[ordinals] / average_length
            if average_length
            else 0.0
        )
        denominator = term_frequencies + index.config.k1 * (
            1.0 - index.config.b + index.config.b * length_ratio
        )
        scores.index_add_(
            0,
            ordinals,
            idf * (term_frequencies * (index.config.k1 + 1.0) / denominator),
        )
    candidate_ordinals = torch.nonzero(scores > 0.0, as_tuple=False).flatten()
    if candidate_ordinals.numel() == 0:
        return []
    if candidate_ordinals.numel() > top_k:
        threshold = torch.topk(scores[candidate_ordinals], top_k).values[-1]
        candidate_ordinals = torch.nonzero(
            scores >= threshold, as_tuple=False
        ).flatten()
    candidate_scores = scores[candidate_ordinals].cpu().tolist()
    return [
        (float(score), index.documents[int(ordinal)])
        for ordinal, score in zip(
            candidate_ordinals.cpu().tolist(), candidate_scores, strict=True
        )
    ]


def retrieve_bm25(
    index: BM25Index,
    query: str,
    *,
    top_k: int,
    query_cache: BM25QueryCache | None = None,
    backend: BM25Backend = "cpu",
    cuda_query_cache: BM25CudaQueryCache | None = None,
) -> tuple[RetrievalHit, ...]:
    """Retrieve positive-scoring chunks with deterministic BM25 ranking.

    Only ``query`` is tokenized.  The index stores retrieval views and provenance,
    never raw answers or reference text.  Results are sorted by descending score and
    ascending chunk ID, then assigned one-based ranks.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("BM25 query must be a non-blank string")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if backend not in {"cpu", "cuda"}:
        raise ValueError("BM25 backend must be 'cpu' or 'cuda'")

    query_terms = set(tokenize_legal_text(query))
    if not query_terms:
        return ()

    if backend == "cuda":
        if query_cache is None:
            query_cache = build_bm25_query_cache(index, (query,))
        if cuda_query_cache is None:
            cuda_query_cache = build_bm25_cuda_query_cache(query_cache)
        try:
            scored = _retrieve_bm25_cuda(
                index,
                query_terms=query_terms,
                top_k=top_k,
                query_cache=query_cache,
                cuda_query_cache=cuda_query_cache,
            )
        except BM25CudaUnavailableError:
            raise
        except RuntimeError as exc:
            raise BM25CudaUnavailableError(
                f"BM25 CUDA scoring failed: {type(exc).__name__}: {exc}"
            ) from exc
    elif query_cache is not None:
        if query_cache.index_fingerprint != index.index_fingerprint:
            raise ValueError("BM25 query cache does not match the loaded index")
        missing_terms = query_terms.difference(query_cache.postings)
        if missing_terms:
            raise ValueError(
                "BM25 query cache does not cover query terms: "
                + ", ".join(sorted(missing_terms))
            )
        document_frequency = query_cache.document_frequencies
        scored_by_ordinal: dict[int, float] = {}
        average_length = index.summary.average_document_length
        corpus_size = len(index.documents)
        if _numpy is not None:
            scores = _numpy.zeros(corpus_size, dtype=_numpy.float64)
            document_lengths = _numpy.frombuffer(
                query_cache.document_lengths,
                dtype=_numpy.uint32,
                count=corpus_size,
            )
            for term in sorted(query_terms):
                frequency = document_frequency[term]
                if frequency <= 0:
                    continue
                idf = math.log(
                    1.0 + (corpus_size - frequency + 0.5) / (frequency + 0.5)
                )
                posting = query_cache.postings[term]
                ordinals = _numpy.frombuffer(posting.ordinals, dtype=_numpy.uint32)
                term_frequencies = _numpy.frombuffer(
                    posting.term_frequencies,
                    dtype=_numpy.uint32,
                )
                length_ratio = (
                    document_lengths[ordinals] / average_length
                    if average_length
                    else 0.0
                )
                denominator = term_frequencies + index.config.k1 * (
                    1.0 - index.config.b + index.config.b * length_ratio
                )
                scores[ordinals] += idf * (
                    term_frequencies * (index.config.k1 + 1.0) / denominator
                )
            candidate_ordinals = _numpy.flatnonzero(scores > 0.0)
            if len(candidate_ordinals) > top_k:
                threshold = _numpy.partition(scores[candidate_ordinals], -top_k)[-top_k]
                candidate_ordinals = candidate_ordinals[
                    scores[candidate_ordinals] >= threshold
                ]
            scored = [
                (float(scores[ordinal]), index.documents[int(ordinal)])
                for ordinal in candidate_ordinals
            ]
        else:
            for term in sorted(query_terms):
                frequency = document_frequency[term]
                if frequency <= 0:
                    continue
                idf = math.log(
                    1.0 + (corpus_size - frequency + 0.5) / (frequency + 0.5)
                )
                posting = query_cache.postings[term]
                for ordinal, term_frequency in zip(
                    posting.ordinals,
                    posting.term_frequencies,
                    strict=True,
                ):
                    document = index.documents[ordinal]
                    length_ratio = (
                        document.document_length / average_length
                        if average_length
                        else 0.0
                    )
                    denominator = term_frequency + index.config.k1 * (
                        1.0 - index.config.b + index.config.b * length_ratio
                    )
                    scored_by_ordinal[ordinal] = scored_by_ordinal.get(
                        ordinal, 0.0
                    ) + idf * (term_frequency * (index.config.k1 + 1.0) / denominator)
            scored = [
                (score, index.documents[ordinal])
                for ordinal, score in scored_by_ordinal.items()
                if score > 0.0
            ]
    else:
        document_frequency = dict(index.document_frequencies)
        average_length = index.summary.average_document_length
        scored = []
        for document in index.documents:
            frequencies = dict(document.term_frequencies)
            score = 0.0
            for term in sorted(query_terms):
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                frequency = document_frequency[term]
                idf = math.log(
                    1.0 + (len(index.documents) - frequency + 0.5) / (frequency + 0.5)
                )
                document_length_ratio = (
                    document.document_length / average_length if average_length else 0.0
                )
                document_denominator = term_frequency + index.config.k1 * (
                    1.0 - index.config.b + index.config.b * document_length_ratio
                )
                score += idf * (
                    term_frequency * (index.config.k1 + 1.0) / document_denominator
                )
            if score > 0.0:
                scored.append((score, document))

    if len(scored) > top_k:
        scored = heapq.nsmallest(
            top_k,
            scored,
            key=lambda item: (-item[0], item[1].chunk_id),
        )
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return tuple(
        RetrievalHit(
            chunk_id=document.chunk_id,
            document_id=document.document_id,
            source_path=document.source_path,
            source_member=document.source_member,
            section_label=document.section_label,
            start_offset=document.start_offset,
            end_offset=document.end_offset,
            rank=rank,
            bm25_score=score,
        )
        for rank, (score, document) in enumerate(scored[:top_k], start=1)
    )


def _chunk_sort_key(chunk: LegalChunk) -> tuple[str, str, str, int, int, str]:
    return (
        chunk.document_id,
        chunk.source_path,
        chunk.source_member or "",
        chunk.start_offset if chunk.start_offset is not None else -1,
        chunk.end_offset if chunk.end_offset is not None else -1,
        chunk.chunk_id,
    )


def _ordered_chunks(chunks: Iterable[LegalChunk]) -> tuple[LegalChunk, ...]:
    ordered = tuple(sorted(chunks, key=_chunk_sort_key))
    chunk_ids = [chunk.chunk_id for chunk in ordered]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise BM25IndexError("BM25 index cannot contain duplicate chunk IDs")
    return ordered


def _summary(
    documents: tuple[BM25IndexedDocument, ...], vocabulary: Iterable[str]
) -> BM25Summary:
    total_tokens = sum(document.document_length for document in documents)
    corpus_size = len(documents)
    return BM25Summary(
        corpus_size=corpus_size,
        vocabulary_size=len(tuple(vocabulary)),
        total_tokens=total_tokens,
        average_document_length=(total_tokens / corpus_size if corpus_size else 0.0),
    )


def _document_frequencies(
    documents: tuple[BM25IndexedDocument, ...],
) -> tuple[tuple[str, int], ...]:
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(term for term, _ in document.term_frequencies)
    return tuple(sorted(frequencies.items()))


def _empty_result(
    status: IndexStatus,
    path: Path,
    fingerprint: BM25IndexFingerprint,
    reason: str | None = None,
) -> BM25IndexResult:
    summary = _summary((), ())
    return BM25IndexResult(status, path, fingerprint, None, summary, reason)


def _validate_cache_root(cache_root: str | Path, data_root: str | Path | None) -> Path:
    resolved_cache = Path(cache_root).resolve()
    if data_root is not None:
        resolved_data = Path(data_root).resolve()
        try:
            resolved_cache.relative_to(resolved_data)
        except ValueError:
            pass
        else:
            raise BM25IndexError(
                f"BM25 cache must be outside source data directory: {resolved_cache}"
            )
    if any(part.casefold() == "data" for part in resolved_cache.parts):
        raise BM25IndexError(
            f"BM25 cache path appears inside a data directory: {resolved_cache}"
        )
    return resolved_cache


def bm25_cache_path(
    cache_root: str | Path,
    fingerprint: BM25IndexFingerprint,
    data_root: str | Path | None = None,
) -> Path:
    """Return a deterministic safe-cache path for one chunk/config identity."""

    root = _validate_cache_root(cache_root, data_root)
    return (
        root
        / "indexes"
        / "bm25"
        / fingerprint.chunk_cache_fingerprint
        / fingerprint.index_fingerprint
        / "index.jsonl"
    )


def _build_index(
    chunks: Iterable[LegalChunk],
    fingerprint: BM25IndexFingerprint,
    config: BM25Config,
    cache_path: Path | None = None,
) -> BM25Index:
    ordered_chunks = _ordered_chunks(chunks)
    documents: list[BM25IndexedDocument] = []
    vocabulary: set[str] = set()
    for ordinal, chunk in enumerate(ordered_chunks):
        tokens = tokenize_legal_text(chunk.retrieval_text)
        frequencies = Counter(tokens)
        vocabulary.update(frequencies)
        documents.append(
            BM25IndexedDocument(
                ordinal=ordinal,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_path=chunk.source_path,
                retrieval_text=chunk.retrieval_text,
                term_frequencies=tuple(sorted(frequencies.items())),
                document_length=len(tokens),
                source_member=chunk.source_member,
                section_label=chunk.section_label,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
            )
        )
    ordered_documents = tuple(documents)
    ordered_vocabulary = tuple(sorted(vocabulary))
    return BM25Index(
        fingerprint=fingerprint,
        config=config,
        documents=ordered_documents,
        vocabulary=ordered_vocabulary,
        document_frequencies=_document_frequencies(ordered_documents),
        summary=_summary(ordered_documents, ordered_vocabulary),
        cache_path=cache_path,
    )


def _serialize_index(index: BM25Index) -> str:
    records: list[dict[str, Any]] = [
        {
            "record_type": "metadata",
            "schema_version": BM25_INDEX_SCHEMA_VERSION,
            "fingerprint": index.fingerprint.as_dict(),
            "index_fingerprint": index.index_fingerprint,
            "config": index.config.as_dict(),
            "summary": index.summary.as_dict(),
            "vocabulary": list(index.vocabulary),
            "document_frequencies": {
                term: count for term, count in index.document_frequencies
            },
        }
    ]
    records.extend(
        {"record_type": "document", "document": document.as_dict()}
        for document in index.documents
    )
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )


def read_bm25_index(
    path: str | Path,
    fingerprint: BM25IndexFingerprint,
    config: BM25Config,
    data_root: str | Path | None = None,
) -> BM25IndexResult:
    """Read a BM25 JSONL cache without rebuilding or mutating it."""

    resolved_path = Path(path).resolve()
    _validate_cache_root(resolved_path.parent, data_root)
    if not resolved_path.is_file():
        return _empty_result("miss", resolved_path, fingerprint, "index_missing")

    try:
        with resolved_path.open("r", encoding="utf-8", newline="") as handle:
            metadata_line = handle.readline()
            if not metadata_line:
                raise ValueError("index is empty")
            metadata = json.loads(metadata_line)
        if not isinstance(metadata, dict):
            raise ValueError("metadata record must be an object")
        if metadata.get("record_type") != "metadata":
            raise ValueError("first record must be metadata")
        if metadata.get("schema_version") != BM25_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported BM25 index schema version")
        if metadata.get("fingerprint") != fingerprint.as_dict():
            return _empty_result(
                "stale", resolved_path, fingerprint, "fingerprint_mismatch"
            )
        if metadata.get("index_fingerprint") != fingerprint.index_fingerprint:
            return _empty_result(
                "stale", resolved_path, fingerprint, "index_key_mismatch"
            )
        if metadata.get("config") != config.as_dict():
            return _empty_result("stale", resolved_path, fingerprint, "config_mismatch")
        vocabulary = metadata.get("vocabulary")
        if not isinstance(vocabulary, list) or any(
            not isinstance(term, str) for term in vocabulary
        ):
            raise ValueError("vocabulary metadata must be a string list")
        ordered_vocabulary = tuple(vocabulary)
        if ordered_vocabulary != tuple(sorted(set(ordered_vocabulary))):
            raise ValueError("vocabulary metadata is not deterministic")
        document_frequencies = metadata.get("document_frequencies")
        if not isinstance(document_frequencies, dict):
            raise ValueError("document_frequencies metadata must be an object")
        ordered_document_frequencies = tuple(
            sorted(
                (str(term), int(count)) for term, count in document_frequencies.items()
            )
        )

        documents: list[BM25IndexedDocument] = []
        with resolved_path.open("r", encoding="utf-8", newline="") as handle:
            handle.readline()
            for line_number, line in enumerate(handle, start=2):
                if not line.strip():
                    raise ValueError(f"line {line_number} is blank")
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or record.get("record_type") != "document"
                ):
                    raise ValueError(f"line {line_number} is not a document record")
                documents.append(BM25IndexedDocument.from_dict(record.get("document")))
        ordered_documents = tuple(documents)
        if ordered_documents != tuple(
            sorted(ordered_documents, key=lambda document: document.ordinal)
        ):
            raise ValueError("document records are not in deterministic order")
        if tuple(document.ordinal for document in ordered_documents) != tuple(
            range(len(ordered_documents))
        ):
            raise ValueError("document ordinals are not contiguous")
        actual_vocabulary = tuple(
            sorted(
                {
                    term
                    for document in ordered_documents
                    for term, _ in document.term_frequencies
                }
            )
        )
        if actual_vocabulary != ordered_vocabulary:
            raise ValueError("vocabulary does not match document records")
        actual_summary = _summary(ordered_documents, ordered_vocabulary)
        if metadata.get("summary") != actual_summary.as_dict():
            raise ValueError("summary does not match document records")
        index = BM25Index(
            fingerprint=fingerprint,
            config=config,
            documents=ordered_documents,
            vocabulary=ordered_vocabulary,
            document_frequencies=ordered_document_frequencies,
            summary=actual_summary,
            cache_path=resolved_path,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        return _empty_result("stale", resolved_path, fingerprint, str(exc))

    return BM25IndexResult(
        status="hit",
        cache_path=resolved_path,
        fingerprint=fingerprint,
        index=index,
        summary=index.summary,
    )


def load_bm25_index(
    cache_root: str | Path,
    chunk_cache_fingerprint: str | ChunkCacheFingerprint,
    config: BM25Config,
    data_root: str | Path | None = None,
) -> BM25IndexResult:
    """Load only the exact index identified by the chunk cache and config."""

    fingerprint = BM25IndexFingerprint.from_config(chunk_cache_fingerprint, config)
    return read_bm25_index(
        bm25_cache_path(cache_root, fingerprint, data_root),
        fingerprint,
        config,
        data_root,
    )


def write_bm25_index(
    cache_root: str | Path,
    index: BM25Index,
    data_root: str | Path | None = None,
    *,
    overwrite_stale: bool = False,
) -> BM25IndexResult:
    """Atomically persist an index without replacing valid unrelated caches."""

    if index.cache_path is not None:
        expected_path = bm25_cache_path(cache_root, index.fingerprint, data_root)
    else:
        expected_path = bm25_cache_path(cache_root, index.fingerprint, data_root)
        index = BM25Index(
            fingerprint=index.fingerprint,
            config=index.config,
            documents=index.documents,
            vocabulary=index.vocabulary,
            document_frequencies=index.document_frequencies,
            summary=index.summary,
            cache_path=expected_path,
        )
    existing = read_bm25_index(
        expected_path, index.fingerprint, index.config, data_root
    )
    if existing.status == "hit":
        if existing.index != index:
            raise BM25IndexConflictError(
                f"BM25 cache already contains different content: {expected_path}"
            )
        return existing
    if expected_path.exists() and existing.status == "stale" and not overwrite_stale:
        raise BM25IndexStaleError(
            f"Refusing to overwrite stale BM25 index {expected_path}: {existing.reason}"
        )

    serialized = _serialize_index(index)
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=expected_path.parent,
            prefix=f".{expected_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if expected_path.exists() and not overwrite_stale:
            raise BM25IndexConflictError(
                f"BM25 cache appeared during write: {expected_path}"
            )
        os.replace(temporary_path, expected_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    verified = read_bm25_index(
        expected_path, index.fingerprint, index.config, data_root
    )
    if verified.status != "hit":
        raise BM25IndexError(
            f"Atomic BM25-index write could not be verified: {verified.reason}"
        )
    return BM25IndexResult(
        status="miss",
        cache_path=verified.cache_path,
        fingerprint=verified.fingerprint,
        index=verified.index,
        summary=verified.summary,
        reason="index_built",
    )


def build_bm25_index(
    chunks: Iterable[LegalChunk],
    cache_root: str | Path,
    chunk_cache_fingerprint: str | ChunkCacheFingerprint,
    config: BM25Config | None = None,
    data_root: str | Path | None = None,
    *,
    overwrite_stale: bool = False,
) -> BM25IndexResult:
    """Explicitly build and cache an index from LegalChunk retrieval views."""

    effective_config = BM25Config() if config is None else config
    fingerprint = BM25IndexFingerprint.from_config(
        chunk_cache_fingerprint, effective_config
    )
    path = bm25_cache_path(cache_root, fingerprint, data_root)
    index = _build_index(chunks, fingerprint, effective_config, path)
    return write_bm25_index(
        cache_root,
        index,
        data_root,
        overwrite_stale=overwrite_stale,
    )


def load_or_build_bm25_index(
    chunks: Iterable[LegalChunk],
    cache_root: str | Path,
    chunk_cache_fingerprint: str | ChunkCacheFingerprint,
    config: BM25Config | None = None,
    data_root: str | Path | None = None,
    *,
    policy: IndexLoadPolicy = "strict",
) -> BM25IndexResult:
    """Load strictly, or rebuild only when ``policy='auto_rebuild'`` is explicit."""

    if policy not in {"strict", "auto_rebuild"}:
        raise ValueError(f"Unsupported BM25 index load policy: {policy}")
    effective_config = BM25Config() if config is None else config
    loaded = load_bm25_index(
        cache_root, chunk_cache_fingerprint, effective_config, data_root
    )
    if loaded.status == "hit":
        return loaded
    if policy == "strict":
        error_type = (
            BM25IndexStaleError if loaded.status == "stale" else BM25IndexMissingError
        )
        raise error_type(
            f"BM25 index unavailable under strict policy: {loaded.cache_path} "
            f"({loaded.reason})"
        )
    return build_bm25_index(
        chunks,
        cache_root,
        chunk_cache_fingerprint,
        effective_config,
        data_root,
        overwrite_stale=loaded.status == "stale",
    )


__all__ = [
    "BM25Config",
    "BM25_INDEX_SCHEMA_VERSION",
    "BM25_INDEX_VERSION",
    "BM25Index",
    "BM25IndexConflictError",
    "BM25IndexError",
    "BM25IndexFingerprint",
    "BM25IndexMissingError",
    "BM25IndexResult",
    "BM25IndexStaleError",
    "BM25IndexedDocument",
    "BM25Summary",
    "IndexLoadPolicy",
    "IndexStatus",
    "bm25_cache_path",
    "build_bm25_index",
    "load_bm25_index",
    "load_or_build_bm25_index",
    "read_bm25_index",
    "write_bm25_index",
    "retrieve_bm25",
]
