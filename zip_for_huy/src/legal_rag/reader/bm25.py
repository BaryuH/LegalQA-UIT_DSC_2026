"""Small deterministic BM25 index for train-case contexts only."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Final, Literal

from ..retrieval.bm25 import resolve_bm25_backend
from .types import ReaderInferenceCase

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)
ReaderBM25Backend = Literal["auto", "cpu", "cuda"]


class ReaderBM25CudaUnavailableError(RuntimeError):
    """Raised when the extractive reader explicitly requests CUDA BM25."""


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


@dataclass(frozen=True, slots=True)
class ReaderBM25Hit:
    case_id: str
    context: str
    rank: int
    score: float


class ReaderBM25Index:
    """In-memory index whose records cannot contain reference-answer fields."""

    def __init__(
        self,
        cases: Sequence[ReaderInferenceCase],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        backend: ReaderBM25Backend = "cpu",
    ) -> None:
        if not cases:
            raise ValueError("Reader BM25 requires at least one train case")
        if any(type(case) is not ReaderInferenceCase for case in cases):
            raise TypeError(
                "Reader BM25 accepts ReaderInferenceCase records only; "
                "reference-bearing cases are forbidden"
            )
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("Reader BM25 train case IDs must be unique")
        self._cases = tuple(cases)
        self._k1 = k1
        self._b = b
        self.backend = resolve_bm25_backend(backend)
        self._terms = tuple(Counter(_tokens(case.context)) for case in cases)
        self._lengths = tuple(sum(terms.values()) for terms in self._terms)
        self._average_length = sum(self._lengths) / len(self._lengths)
        frequencies: Counter[str] = Counter()
        for terms in self._terms:
            frequencies.update(terms.keys())
        self._document_frequencies = frequencies
        self._torch: Any | None = None
        self._cuda_document_lengths: Any | None = None
        self._cuda_postings: dict[str, tuple[Any, Any]] = {}
        if self.backend == "cuda":
            self._prepare_cuda_state()
        serialized = json.dumps(
            [{"id": case.id, "context": case.context} for case in cases],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _prepare_cuda_state(self) -> None:
        try:
            torch = import_module("torch")
        except ImportError as exc:
            raise ReaderBM25CudaUnavailableError(
                "Reader BM25 CUDA requires the optional 'torch' package"
            ) from exc
        if not torch.cuda.is_available():
            raise ReaderBM25CudaUnavailableError(
                "Reader BM25 CUDA was requested but torch.cuda.is_available() is false"
            )
        device = torch.device("cuda")
        self._torch = torch
        self._cuda_document_lengths = torch.tensor(
            self._lengths, dtype=torch.float64, device=device
        )
        for term in self._document_frequencies:
            ordinals: list[int] = []
            term_frequencies: list[float] = []
            for ordinal, terms in enumerate(self._terms):
                frequency = terms.get(term, 0)
                if frequency:
                    ordinals.append(ordinal)
                    term_frequencies.append(float(frequency))
            self._cuda_postings[term] = (
                torch.tensor(ordinals, dtype=torch.long, device=device),
                torch.tensor(term_frequencies, dtype=torch.float64, device=device),
            )

    def _search_cpu(self, question: str, *, top_k: int) -> tuple[ReaderBM25Hit, ...]:
        query_terms = _tokens(question)
        document_count = len(self._cases)
        scored: list[tuple[int, float]] = []
        for index, terms in enumerate(self._terms):
            score = 0.0
            length = self._lengths[index]
            for term in query_terms:
                term_frequency = terms.get(term, 0)
                if term_frequency == 0:
                    continue
                document_frequency = self._document_frequencies[term]
                inverse_document_frequency = math.log(
                    1.0
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = term_frequency + self._k1 * (
                    1.0 - self._b + self._b * length / max(self._average_length, 1.0)
                )
                score += inverse_document_frequency * (
                    term_frequency * (self._k1 + 1.0) / denominator
                )
            scored.append((index, score))
        return self._hits_from_scores(scored, top_k=top_k)

    def _search_cuda(self, question: str, *, top_k: int) -> tuple[ReaderBM25Hit, ...]:
        if self._torch is None or self._cuda_document_lengths is None:
            raise ReaderBM25CudaUnavailableError("Reader BM25 CUDA state is missing")
        query_terms = _tokens(question)
        scores = self._torch.zeros_like(self._cuda_document_lengths)
        document_count = len(self._cases)
        for term in sorted(query_terms):
            document_frequency = self._document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue
            ordinals, term_frequencies = self._cuda_postings[term]
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_ratio = self._cuda_document_lengths[ordinals] / max(
                self._average_length, 1.0
            )
            denominator = term_frequencies + self._k1 * (
                1.0 - self._b + self._b * length_ratio
            )
            scores.index_add_(
                0,
                ordinals,
                inverse_document_frequency
                * (term_frequencies * (self._k1 + 1.0) / denominator),
            )
        return self._hits_from_scores(
            list(enumerate(scores.cpu().tolist())), top_k=top_k
        )

    def _hits_from_scores(
        self, scored: Sequence[tuple[int, float]], *, top_k: int
    ) -> tuple[ReaderBM25Hit, ...]:
        ordered = sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]
        return tuple(
            ReaderBM25Hit(
                case_id=self._cases[index].id,
                context=self._cases[index].context,
                rank=rank,
                score=score,
            )
            for rank, (index, score) in enumerate(ordered, start=1)
        )

    def search(self, question: str, *, top_k: int) -> tuple[ReaderBM25Hit, ...]:
        """Search with question text only and stable source-order tie breaking."""

        if not question.strip():
            raise ValueError("Reader retrieval question must not be blank")
        if top_k < 1:
            raise ValueError("Reader retrieval top_k must be positive")
        if self.backend == "cuda":
            return self._search_cuda(question, top_k=top_k)
        return self._search_cpu(question, top_k=top_k)


__all__ = [
    "ReaderBM25Backend",
    "ReaderBM25CudaUnavailableError",
    "ReaderBM25Hit",
    "ReaderBM25Index",
]
