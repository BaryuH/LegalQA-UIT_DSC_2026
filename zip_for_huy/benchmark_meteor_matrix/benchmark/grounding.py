"""Grounding gate used to prune candidates before MBR (branch 4).

The full ``legal_rag.sedar_sft.verifier`` needs a ``GroundedDraft`` and an
``EvidenceProfile``; wiring those into a standalone benchmark would couple the
harness to internals it does not own.  This module implements the *hard* subset
of that verifier — unsupported legal identifiers and dates — directly against the
evidence string, using the same identifier/date shapes.  The mode actually used
is recorded in the run manifest so nothing is a silent fallback.

Empirically (see README) MBR only breaks once contaminated candidates become the
majority; pruning obvious hallucinations keeps the effective contamination low
and also makes the O(N) aggregation path safe.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Same shapes as legal_rag.sedar_sft.verifier._LEGAL_ID_RE / _DATE_RE.
_LEGAL_ID_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐđ]{1,10}(?:-[A-ZĐđ]+)?\b")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")

GATE_VERSION = "benchmark-grounding-v1"

# Phrases the RAG prompt emits when evidence is insufficient; a refusal is a
# short non-answer that must be pruned before any answer selection.
REFUSAL_MARKERS = ("chưa đủ căn cứ", "không đủ căn cứ", "không có căn cứ")


def is_refusal(candidate: str) -> bool:
    lowered = candidate.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def substantive_keep_mask(
    candidates: "Sequence[str]", evidence: str | None
) -> tuple[list[bool], str]:
    """Combined gate: grounded (supported legal ids/dates) AND not a refusal.

    Returns (keep_mask, mode) where mode records whether grounding ran.
    """

    grounded, _reasons, mode = keep_mask(candidates, evidence)
    keep = [g and not is_refusal(c) for g, c in zip(grounded, candidates)]
    return keep, mode


@dataclass(frozen=True, slots=True)
class GateDecision:
    keep: bool
    reasons: tuple[str, ...]


def _unsupported(candidate: str, evidence: str, pattern: re.Pattern[str]) -> list[str]:
    ev = set(pattern.findall(evidence))
    return [tok for tok in pattern.findall(candidate) if tok not in ev]


def evaluate_candidate(candidate: str, evidence: str) -> GateDecision:
    """Reject candidates asserting legal IDs/dates absent from the evidence."""

    reasons: list[str] = []
    for name, pattern in (("legal_id", _LEGAL_ID_RE), ("date", _DATE_RE)):
        missing = _unsupported(candidate, evidence, pattern)
        reasons.extend(f"unsupported_{name}:{m}" for m in missing)
    return GateDecision(keep=not reasons, reasons=tuple(reasons))


def keep_mask(
    candidates: Sequence[str], evidence: str | None
) -> tuple[list[bool], list[tuple[str, ...]], str]:
    """Return (mask, per-candidate reasons, mode).

    ``mode`` is ``"grounded"`` when evidence was available and the gate ran, or
    ``"disabled_no_evidence"`` when closed-book (all kept).  Recorded verbatim in
    the manifest.
    """

    if not evidence:
        return [True] * len(candidates), [() for _ in candidates], "disabled_no_evidence"
    mask: list[bool] = []
    reasons: list[tuple[str, ...]] = []
    for cand in candidates:
        decision = evaluate_candidate(cand, evidence)
        mask.append(decision.keep)
        reasons.append(decision.reasons)
    return mask, reasons, "grounded"


__all__ = [
    "GATE_VERSION",
    "REFUSAL_MARKERS",
    "GateDecision",
    "evaluate_candidate",
    "is_refusal",
    "keep_mask",
    "substantive_keep_mask",
]
