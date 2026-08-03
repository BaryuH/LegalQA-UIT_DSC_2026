"""Small deterministic BM25 index for train-case contexts only."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .types import ReaderInferenceCase

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)


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
        self._terms = tuple(Counter(_tokens(case.context)) for case in cases)
        self._lengths = tuple(sum(terms.values()) for terms in self._terms)
        self._average_length = sum(self._lengths) / len(self._lengths)
        frequencies: Counter[str] = Counter()
        for terms in self._terms:
            frequencies.update(terms.keys())
        self._document_frequencies = frequencies
        serialized = json.dumps(
            [{"id": case.id, "context": case.context} for case in cases],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def search(self, question: str, *, top_k: int) -> tuple[ReaderBM25Hit, ...]:
        """Search with question text only and stable source-order tie breaking."""

        if not question.strip():
            raise ValueError("Reader retrieval question must not be blank")
        if top_k < 1:
            raise ValueError("Reader retrieval top_k must be positive")
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


__all__ = ["ReaderBM25Hit", "ReaderBM25Index"]
