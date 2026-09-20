"""Deterministic text metrics used both as MBR utilities and as gold scorers.

Design rules:
- The pairwise METEOR matrix is the heart of the benchmark, so it must be fast.
  ``fast_meteor`` reproduces ``legal_rag.evaluation.meteor.compute_meteor`` bit
  for bit (validated: max abs diff 0.0 over real answer pairs) but runs in
  O(P + R) instead of O(P * R).
- Gold scoring reuses the repo adapters (``compute_meteor``/``compute_rouge_l``)
  and, when available, the official-style NLTK METEOR as a cross-check.
- METEOR is asymmetric.  Every function fixes the argument order as
  ``(reference, hypothesis)`` exactly like the official scorer
  ``meteor_score([reference], hypothesis)``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass

from .repo import compute_meteor, compute_rouge_l, normalize_text

Tokens = tuple[str, ...]

METRIC_VERSION = "benchmark-metrics-v1"


def tokenize(text: str) -> Tokens:
    """Repo tokenization policy (NFC, punctuation kept, legal codes preserved)."""

    return normalize_text(text).tokens


def fast_meteor(reference: Sequence[str], prediction: Sequence[str]) -> float:
    """Exact-token METEOR, identical semantics to ``compute_meteor``, O(P + R).

    ``reference`` and ``prediction`` are token sequences; direction matches the
    official scorer (reference first).
    """

    if not prediction or not reference:
        return 0.0
    positions: dict[str, deque[int]] = defaultdict(deque)
    for index, token in enumerate(reference):
        positions[token].append(index)  # ascending indices == repo's first-unused scan
    matched: list[int] = []
    for token in prediction:
        bucket = positions.get(token)
        if bucket:
            matched.append(bucket.popleft())
    matches = len(matched)
    if not matches:
        return 0.0
    precision = matches / len(prediction)
    recall = matches / len(reference)
    denominator = recall + 9.0 * precision
    f_mean = (10.0 * precision * recall / denominator) if denominator else 0.0
    chunks = 0
    previous: int | None = None
    for index in matched:
        if previous is None or index != previous + 1:
            chunks += 1
        previous = index
    fragmentation = 0.5 * (chunks / matches) ** 3 if chunks > 1 else 0.0
    return f_mean * (1.0 - fragmentation)


def meteor_exact(reference: str, prediction: str) -> float:
    """Repo exact-token METEOR on raw strings (gold scoring path)."""

    return compute_meteor(tokenize(reference), tokenize(prediction)).score


def rouge_l(reference: str, prediction: str) -> float:
    """Repo token-level ROUGE-L F1 on raw strings (gold scoring path)."""

    return compute_rouge_l(tokenize(reference), tokenize(prediction)).score


@dataclass(frozen=True, slots=True)
class GoldScore:
    """Official-style scores of one prediction against its gold reference."""

    meteor: float
    rouge_l: float
    meteor_official: float | None  # NLTK, when installed

    def as_dict(self) -> dict[str, float | None]:
        return {
            "meteor": self.meteor,
            "rouge_l": self.rouge_l,
            "meteor_official": self.meteor_official,
        }


def _nltk_meteor():
    try:
        from nltk.translate.meteor_score import meteor_score
    except Exception:  # noqa: BLE001 - optional cross-check only
        return None

    class _NoSynonym:
        @staticmethod
        def synsets(_word: str):
            return []

    class _NoStem:
        @staticmethod
        def stem(word: str):
            return word

    def scorer(reference: str, prediction: str) -> float:
        # str.split() token view, matching legal_rag.evaluation.source_scorer.
        return float(
            meteor_score(
                [reference.split()],
                prediction.split(),
                stemmer=_NoStem(),
                wordnet=_NoSynonym(),
            )
        )

    return scorer


_NLTK_METEOR = _nltk_meteor()


def score_against_gold(reference: str, prediction: str) -> GoldScore:
    """Score a prediction against gold with the repo metrics (eval-only)."""

    official = _NLTK_METEOR(reference, prediction) if _NLTK_METEOR else None
    return GoldScore(
        meteor=meteor_exact(reference, prediction),
        rouge_l=rouge_l(reference, prediction),
        meteor_official=official,
    )


__all__ = [
    "METRIC_VERSION",
    "GoldScore",
    "Tokens",
    "fast_meteor",
    "meteor_exact",
    "rouge_l",
    "score_against_gold",
    "tokenize",
]
