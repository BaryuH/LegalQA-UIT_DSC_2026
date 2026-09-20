"""Candidate selection strategies operating on a utility matrix.

Every strategy returns a :class:`Selection` (chosen index + the score vector it
argmaxed) so the runner can persist an auditable trace.  Direction convention:
``matrix[i][j] = u(ref=i, hyp=j)`` and the MBR estimate of candidate ``j`` is
the mean over ``i != j`` of column ``j`` (validated as the correct direction;
the row mean collapses to picking short candidates).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Selection:
    strategy: str
    index: int
    scores: tuple[float, ...]
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "index": self.index,
            "scores": [round(float(s), 6) for s in self.scores],
            "note": self.note,
        }


def _argmax(scores: np.ndarray) -> int:
    # deterministic tie-break: smallest index wins
    return int(np.argmax(scores))


def column_mean(matrix: np.ndarray) -> np.ndarray:
    """MBR estimate: mean utility of each candidate as *hypothesis*."""

    n = matrix.shape[0]
    if n < 2:
        return np.zeros(n)
    return matrix.sum(axis=0) / (n - 1)


def mbr(matrix: np.ndarray) -> Selection:
    scores = column_mean(matrix)
    return Selection("mbr", _argmax(scores), tuple(scores))


def mbr_row(matrix: np.ndarray) -> Selection:
    """Wrong direction, kept only to demonstrate the failure empirically."""

    n = matrix.shape[0]
    scores = matrix.sum(axis=1) / max(n - 1, 1)
    return Selection("mbr_row", _argmax(scores), tuple(scores))


def mbr_weighted(matrix: np.ndarray, prior: np.ndarray) -> Selection:
    """QE-weighted MBR: pseudo-references weighted by their quality estimate."""

    weights = np.clip(np.asarray(prior, dtype=np.float64), 0.0, None)
    if weights.sum() <= 0:
        return Selection("mbr_weighted", _argmax(column_mean(matrix)), tuple(column_mean(matrix)), "flat_prior")
    n = matrix.shape[0]
    scores = np.zeros(n)
    for j in range(n):
        num = 0.0
        den = 0.0
        for i in range(n):
            if i != j:
                num += weights[i] * matrix[i, j]
                den += weights[i]
        scores[j] = num / den if den else 0.0
    return Selection("mbr_weighted", _argmax(scores), tuple(scores))


def mbr_pruned(
    matrix: np.ndarray, keep_mask: Sequence[bool], name: str = "mbr_pruned"
) -> Selection:
    """Drop gated-out candidates, then run MBR over the survivors."""

    keep = np.asarray(keep_mask, dtype=bool)
    n = matrix.shape[0]
    if keep.sum() == 0:
        # No silent fallback: report that the gate rejected everything.
        return Selection(name, -1, tuple(np.zeros(n)), "all_rejected")
    if keep.sum() == 1:
        idx = int(np.argmax(keep))
        return Selection(name, idx, tuple(keep.astype(float)), "single_survivor")
    sub = matrix[np.ix_(keep, keep)]
    sub_scores = column_mean(sub)
    survivors = np.where(keep)[0]
    scores = np.full(n, -np.inf)
    scores[survivors] = sub_scores
    return Selection(name, int(survivors[int(np.argmax(sub_scores))]), tuple(np.where(np.isfinite(scores), scores, 0.0)))


def aggregate(scores: np.ndarray, name: str = "aggregate") -> Selection:
    """O(N) reference-aggregation / centroid selection."""

    return Selection(name, _argmax(np.asarray(scores)), tuple(np.asarray(scores)))


def _kmeans(vectors: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    k = min(k, n)
    centers = vectors[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            members = vectors[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return labels


def cbmbr(vectors: np.ndarray, k: int = 3, seed: int = 0) -> Selection:
    """Centroid-based MBR: pick the medoid of the largest embedding cluster.

    Contamination-robust efficiency path (Deguchi et al., 2024): the majority
    cluster is the correct mode; its medoid is the consensus answer.
    """

    n = vectors.shape[0]
    if n < 2:
        return Selection("cbmbr", 0, tuple(np.zeros(n)))
    labels = _kmeans(np.asarray(vectors, dtype=np.float64), k, seed)
    largest = np.bincount(labels).argmax()
    members = np.where(labels == largest)[0]
    centroid = vectors[members].mean(axis=0, keepdims=True)
    sims = (vectors @ centroid.T).ravel()
    masked = np.full(n, -np.inf)
    masked[members] = sims[members]
    return Selection("cbmbr", int(np.argmax(masked)), tuple(np.where(np.isfinite(masked), masked, 0.0)))


# --------------------------------------------------------------------------- #
# Non-MBR baselines
# --------------------------------------------------------------------------- #
def baseline_first(candidates: Sequence[str]) -> Selection:
    return Selection("first", 0, tuple([0.0] * len(candidates)), "greedy/first candidate")


def baseline_longest(candidates: Sequence[str]) -> Selection:
    from . import metrics

    lengths = np.array([len(metrics.tokenize(c)) for c in candidates], dtype=np.float64)
    return Selection("longest", _argmax(lengths), tuple(lengths))


def baseline_random(candidates: Sequence[str], seed: int) -> Selection:
    idx = random.Random(seed).randrange(len(candidates))
    return Selection("random", idx, tuple([0.0] * len(candidates)))


def longest_grounded(
    candidates: Sequence[str],
    keep_mask: Sequence[bool],
    name: str = "longest_grounded",
) -> Selection:
    """Chosen production selector: longest candidate among gated survivors.

    Empirically ties the MBR field (ns vs ``longest``, len_resid ~ 0) at O(N)
    cost with no pairwise matrix. ``keep_mask`` should already combine the
    grounding gate and refusal filter. If the gate empties the pool, fall back
    to the global longest and flag it explicitly (never a silent fallback).
    """

    from . import metrics

    keep = np.asarray(keep_mask, dtype=bool)
    lengths = np.array([len(metrics.tokenize(c)) for c in candidates], dtype=np.float64)
    if keep.sum() == 0:
        return Selection(name, _argmax(lengths), tuple(lengths), "all_rejected_fallback_longest")
    masked = np.where(keep, lengths, -np.inf)
    return Selection(name, _argmax(masked), tuple(lengths))


__all__ = [
    "Selection",
    "aggregate",
    "baseline_first",
    "baseline_longest",
    "baseline_random",
    "cbmbr",
    "longest_grounded",
    "column_mean",
    "mbr",
    "mbr_pruned",
    "mbr_row",
    "mbr_weighted",
]
