"""Token budget of a built SFT dataset, measured with the model tokenizer.

SS-06 concluded the 512-token generation cap was not binding because the
measurement counted whitespace words. Re-measured with the model tokenizer the
cap was binding on 214 of 460 cases, and `ss03_feasibility.json` still carries
`tokenizer: whitespace_estimator_only`. Length claims must come from the
tokenizer the model actually uses.

`profile_sequence_length.py` cannot do this: its only implemented path is
`MockWhitespaceTokenizer`, and it rebuilds the dataset through the frozen-B2
builder rather than reading a prebuilt LTR dataset.

This module holds the pure reporting half, so the arithmetic is testable
without loading a tokenizer. Counts and IDs only; no prompt, question or
target text is retained or reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

TOKEN_BUDGET_SCHEMA_VERSION = "sedar_sft.reader_v2.token_budget.v1"

# Below this, a p99 fits 4096 with room for the generation cap; above it the
# sequence budget has to be raised before training, not after.
P99_HEADROOM_THRESHOLD = 3800


def percentile(values: Sequence[int], fraction: float) -> int:
    """Nearest-rank percentile, so every reported value is a real observation."""

    if not values:
        raise ValueError("no values to summarise")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    ordered = sorted(values)
    # ceil, not round: round() is banker's rounding in Python, so p99 of 100
    # values landed on the maximum instead of the 99th.
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarise(values: Sequence[int]) -> dict[str, int]:
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


@dataclass(frozen=True, slots=True)
class TokenBudgetReport:
    """What the dataset costs in tokens, and whether the budget holds."""

    prompt: dict[str, int]
    target: dict[str, int]
    total: dict[str, int]
    max_seq_length: int
    over_budget_count: int
    over_budget_ids: tuple[str, ...]
    recommended_max_seq_length: int
    verdict: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": TOKEN_BUDGET_SCHEMA_VERSION,
            "prompt_tokens": self.prompt,
            "target_tokens": self.target,
            "total_tokens": self.total,
            "max_seq_length": self.max_seq_length,
            "over_budget_count": self.over_budget_count,
            "over_budget_share": (
                self.over_budget_count / self.total["count"]
                if self.total["count"]
                else 0.0
            ),
            # Capped for readability; the count above is authoritative.
            "over_budget_ids_sample": list(self.over_budget_ids[:50]),
            "p99_headroom_threshold": P99_HEADROOM_THRESHOLD,
            "recommended_max_seq_length": self.recommended_max_seq_length,
            "verdict": self.verdict,
        }


def build_report(
    rows: Sequence[tuple[str, int, int]],
    *,
    max_seq_length: int,
) -> TokenBudgetReport:
    """Summarise (case_id, prompt_tokens, target_tokens) triples.

    Truncation is the failure this guards against: a target answer cut off at
    the sequence limit still produces a healthy-looking loss curve, so the run
    fails silently. Any example over budget is reported by ID.
    """

    if not rows:
        raise ValueError("no examples to profile")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")

    prompts = [prompt for _, prompt, _ in rows]
    targets = [target for _, _, target in rows]
    totals = [prompt + target for _, prompt, target in rows]
    over = tuple(
        case_id for case_id, prompt, target in rows if prompt + target > max_seq_length
    )
    total_summary = summarise(totals)

    if total_summary["p99"] > P99_HEADROOM_THRESHOLD or over:
        recommended = 8192
        verdict = "RAISE_MAX_SEQ_LENGTH"
    else:
        recommended = max_seq_length
        verdict = "BUDGET_HOLDS"

    return TokenBudgetReport(
        prompt=summarise(prompts),
        target=summarise(targets),
        total=total_summary,
        max_seq_length=max_seq_length,
        over_budget_count=len(over),
        over_budget_ids=over,
        recommended_max_seq_length=recommended,
        verdict=verdict,
    )


__all__ = [
    "P99_HEADROOM_THRESHOLD",
    "TOKEN_BUDGET_SCHEMA_VERSION",
    "TokenBudgetReport",
    "build_report",
    "percentile",
    "summarise",
]
