"""MBR utility functions: lexical, neural, ensemble, plus O(N) aggregates.

A utility answers three questions about a candidate pool:

- ``pairwise(cands)`` -> N x N matrix ``M`` with ``M[i][j] = u(ref=i, hyp=j)``.
  This is the exact object the user proposed ("ma trận METEOR").  Direction is
  fixed to reference-first so the correct MBR estimate is a *column* mean.
- ``aggregate(cands)`` -> length-N vector, an O(N) approximation of the column
  mean (reference aggregation / centroid), used for the fast decoding path.
- ``prior(question, cands)`` -> optional length-N quality-estimation vector used
  to weight the MBR sum (branch 4, QE-MBR).

Neural utilities lazily import ``sentence-transformers`` / ``torch`` so the
lexical path stays importable on any machine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from . import metrics

UTILITY_VERSION = "benchmark-utilities-v1"


class Utility(Protocol):
    name: str
    version: str

    def pairwise(self, candidates: Sequence[str]) -> np.ndarray: ...

    def aggregate(self, candidates: Sequence[str]) -> np.ndarray: ...

    def prior(self, question: str, candidates: Sequence[str]) -> np.ndarray | None: ...


# --------------------------------------------------------------------------- #
# Lexical utilities (METEOR / ROUGE-L) — the metric the leaderboard uses.
# --------------------------------------------------------------------------- #
class LexicalUtility:
    """METEOR or ROUGE-L pairwise utility over repo tokenization."""

    def __init__(self, metric: str = "meteor") -> None:
        if metric not in {"meteor", "rouge_l"}:
            raise ValueError(f"unknown lexical metric: {metric!r}")
        self.metric = metric
        self.name = f"lexical:{metric}"
        self.version = metrics.METRIC_VERSION

    def _fn(self, ref_tokens, hyp_tokens) -> float:
        if self.metric == "meteor":
            return metrics.fast_meteor(ref_tokens, hyp_tokens)
        from .repo import compute_rouge_l

        return compute_rouge_l(ref_tokens, hyp_tokens).score

    def pairwise(self, candidates: Sequence[str]) -> np.ndarray:
        toks = [metrics.tokenize(c) for c in candidates]
        n = len(toks)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = self._fn(toks[i], toks[j])
        return matrix

    def aggregate(self, candidates: Sequence[str]) -> np.ndarray:
        """Expected-reference aggregation (Vamvas & Sennrich, 2024).

        WARNING: this dilutes toward contaminated candidates; only safe after a
        grounding prune.  Kept for measured comparison against ``pairwise``.
        """

        from collections import Counter

        toks = [metrics.tokenize(c) for c in candidates]
        n = len(toks)
        expected: Counter[str] = Counter()
        for t in toks:
            expected.update(t)
        for key in expected:
            expected[key] /= n
        ref_len = sum(expected.values()) or 1.0
        scores = np.zeros(n, dtype=np.float64)
        for j, hyp in enumerate(toks):
            if not hyp:
                continue
            counts = Counter(hyp)
            matches = sum(min(counts[t], expected[t]) for t in counts)
            precision = matches / len(hyp)
            recall = matches / ref_len
            denom = recall + 9.0 * precision
            scores[j] = (10.0 * precision * recall / denom) if denom else 0.0
        return scores

    def prior(self, question: str, candidates: Sequence[str]) -> np.ndarray | None:
        return None


# --------------------------------------------------------------------------- #
# Neural utility (embedding cosine) — "high quality rather than high metric".
# --------------------------------------------------------------------------- #
class EmbeddingUtility:
    """Cosine-similarity pairwise utility from a sentence embedding model.

    Encoding is O(N); the cosine matrix is a cheap N x N dot product on unit
    vectors, so this is the natural home for centroid aggregation (CBMBR).
    """

    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        device: str = "cuda",
        batch_size: int = 16,
        max_seq_length: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.name = f"embedding:{model_name}"
        self.version = UTILITY_VERSION
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length:
                model.max_seq_length = self.max_seq_length
            self._model = model
        return self._model

    def _encode(self, candidates: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(candidates),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float64)

    def encode(self, candidates: Sequence[str]) -> np.ndarray:
        """Public unit-normalized embeddings (used by centroid/CBMBR)."""

        return self._encode(candidates)

    def pairwise(self, candidates: Sequence[str]) -> np.ndarray:
        vectors = self._encode(candidates)
        matrix = vectors @ vectors.T
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def aggregate(self, candidates: Sequence[str]) -> np.ndarray:
        """Cosine to the pool centroid — O(N) CBMBR-style score."""

        vectors = self._encode(candidates)
        centroid = vectors.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(centroid) or 1.0
        return (vectors @ centroid.T).ravel() / norm

    def prior(self, question: str, candidates: Sequence[str]) -> np.ndarray | None:
        model = self._load()
        q = model.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        )
        c = self._encode(candidates)
        return (c @ np.asarray(q, dtype=np.float64).T).ravel()


# --------------------------------------------------------------------------- #
# QE prior (cross-encoder relevance of question -> candidate).
# --------------------------------------------------------------------------- #
class RerankerPrior:
    """Per-candidate quality estimate from a Vietnamese cross-encoder.

    Not a pairwise utility; it produces the weight vector for QE-weighted MBR
    (branch 4).  ``pairwise``/``aggregate`` are intentionally unavailable.
    """

    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Reranker",
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 2304,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.name = f"reranker_prior:{model_name}"
        self.version = UTILITY_VERSION
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_length
            )
        return self._model

    def prior(self, question: str, candidates: Sequence[str]) -> np.ndarray:
        model = self._load()
        pairs = [(question, c) for c in candidates]
        scores = model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        scores = np.asarray(scores, dtype=np.float64)
        # squash raw logits to (0, 1) so weights are non-negative and comparable
        return 1.0 / (1.0 + np.exp(-scores))


# --------------------------------------------------------------------------- #
# Ensemble utility (branch 2) — blend to fight single-metric reward hacking.
# --------------------------------------------------------------------------- #
def _minmax_offdiag(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    if n < 2:
        return matrix
    mask = ~np.eye(n, dtype=bool)
    vals = matrix[mask]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return np.zeros_like(matrix)
    out = (matrix - lo) / (hi - lo)
    np.fill_diagonal(out, 0.0)
    return out


class EnsembleUtility:
    """Weighted, per-matrix min-max-normalized blend of sub-utilities."""

    def __init__(self, members: Sequence[tuple[Utility, float]]) -> None:
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members = list(members)
        self.name = "ensemble:" + "+".join(
            f"{u.name}*{w:g}" for u, w in self.members
        )
        self.version = UTILITY_VERSION

    def pairwise(self, candidates: Sequence[str]) -> np.ndarray:
        total = None
        weight_sum = 0.0
        for utility, weight in self.members:
            block = _minmax_offdiag(utility.pairwise(candidates))
            total = block * weight if total is None else total + block * weight
            weight_sum += weight
        assert total is not None
        return total / (weight_sum or 1.0)

    def aggregate(self, candidates: Sequence[str]) -> np.ndarray:
        total = None
        weight_sum = 0.0
        for utility, weight in self.members:
            vec = utility.aggregate(candidates)
            lo, hi = float(vec.min()), float(vec.max())
            vec = (vec - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(vec)
            total = vec * weight if total is None else total + vec * weight
            weight_sum += weight
        assert total is not None
        return total / (weight_sum or 1.0)

    def prior(self, question: str, candidates: Sequence[str]) -> np.ndarray | None:
        for utility, _ in self.members:
            vec = utility.prior(question, candidates)
            if vec is not None:
                return vec
        return None


__all__ = [
    "UTILITY_VERSION",
    "EmbeddingUtility",
    "EnsembleUtility",
    "LexicalUtility",
    "RerankerPrior",
    "Utility",
]
