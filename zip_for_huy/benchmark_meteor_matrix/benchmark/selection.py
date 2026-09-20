"""Answer-selection strategies.

The MBR / pairwise-METEOR-matrix line was evaluated and dropped (it never beat
``longest`` and had length_residual <= 0). What remains is the chosen production
selector plus a few cheap reference baselines used to keep it honest.

Direction/verdict history is preserved in git; this module only carries the
selectors still in use.
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


# --------------------------------------------------------------------------- #
# Reference baselines (context for the production selector, not shipped)
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


# --------------------------------------------------------------------------- #
# Chosen production selector
# --------------------------------------------------------------------------- #
def longest_grounded(
    candidates: Sequence[str],
    keep_mask: Sequence[bool],
    name: str = "longest_grounded",
) -> Selection:
    """Longest candidate among gated survivors (grounded AND not a refusal).

    O(N), no pairwise matrix. If the gate empties the pool, fall back to the
    global longest and flag it explicitly (never a silent fallback).
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
    "baseline_first",
    "baseline_longest",
    "baseline_random",
    "longest_grounded",
]
