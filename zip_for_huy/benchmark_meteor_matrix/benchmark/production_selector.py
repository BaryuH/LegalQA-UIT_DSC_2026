"""Chosen production answer selector: longest-among-grounded, refusal-pruned.

Decision record (validated on the grounded, length-fixed dev slice, Qwen3-8B):
- Oracle 0.4772, random 0.4126 -> selection headroom is only ~0.065.
- Every substantive strategy has length_residual <= 0: none picks better
  *content* than its length predicts. MBR ties ``longest`` (ns, p~0.27) and
  never beats it. The real levers were retrieval (reranker, +0.22 oracle) and
  generation length/completeness, not the selection algorithm.
- Therefore the pairwise METEOR matrix (MBR) is not worth its O(N^2) cost here.
  The cheap selector below ties the field at O(N): drop refusals and
  ungrounded candidates, then take the longest survivor (METEOR is
  recall-weighted, so length maximises coverage of the gold answer).

This module is self-contained (only ``grounding`` + ``metrics``) so the main RAG
pipeline can import it without pulling in the benchmark machinery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import grounding, metrics


@dataclass(frozen=True, slots=True)
class SelectionResult:
    index: int
    answer: str
    gate_mode: str
    kept: int
    total: int
    fallback: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "gate_mode": self.gate_mode,
            "kept": self.kept,
            "total": self.total,
            "fallback": self.fallback,
            "note": self.note,
        }


def select_final_answer(
    candidates: Sequence[str], evidence: str | None = None
) -> SelectionResult:
    """Pick the final answer from a candidate pool.

    Gate: keep candidates that are grounded (no legal id/date absent from the
    evidence) AND not a refusal. Among survivors, return the longest (token
    count). If the gate empties the pool, fall back to the global longest and
    flag it explicitly (contract: no silent fallback).
    """

    if not candidates:
        raise ValueError("no candidates to select from")

    keep, mode = grounding.substantive_keep_mask(candidates, evidence)
    lengths = [len(metrics.tokenize(c)) for c in candidates]
    survivors = [i for i, ok in enumerate(keep) if ok]

    fallback = not survivors
    pool = survivors if survivors else list(range(len(candidates)))
    index = max(pool, key=lambda i: lengths[i])
    note = "all_gated_out_fallback_longest" if fallback else ""
    return SelectionResult(
        index=index,
        answer=candidates[index],
        gate_mode=mode,
        kept=len(survivors),
        total=len(candidates),
        fallback=fallback,
        note=note,
    )


__all__ = ["SelectionResult", "select_final_answer"]
