"""Deterministic text metrics for gold scoring.

Reuses the repo adapters (``compute_meteor`` / ``compute_rouge_l``) so the
benchmark never introduces a second scoring convention, plus an optional
official-style NLTK METEOR cross-check. METEOR is asymmetric: every function
fixes the argument order as ``(reference, hypothesis)`` exactly like the
official scorer ``meteor_score([reference], hypothesis)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .repo import compute_meteor, compute_rouge_l, normalize_text

Tokens = tuple[str, ...]

METRIC_VERSION = "benchmark-metrics-v1"


def tokenize(text: str) -> Tokens:
    """Repo tokenization policy (NFC, punctuation kept, legal codes preserved)."""

    return normalize_text(text).tokens


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
    "meteor_exact",
    "rouge_l",
    "score_against_gold",
    "tokenize",
]
