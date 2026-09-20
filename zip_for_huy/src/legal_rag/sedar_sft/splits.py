"""Deterministic dev/test split for the reader v2 comparison.

Clean-460 is spent: roughly twenty configurations were scored on it and the
champion was taken as the max, so selecting a new checkpoint on it would stack
another layer of selection bias on an exhausted set. Reader v2 therefore needs
two disjoint halves fixed *before* training: one for every training decision,
one touched exactly once at the end.

The split is stratified by the champion's own per-case score. An unstratified
draw can hand the dev half an easier slice, and the paired tests downstream
assume the two halves are comparable in difficulty.

This module holds the pure function so it can be tested without artifacts.
It handles IDs and numbers only; no question, prediction or gold text.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SPLIT_SCHEMA_VERSION = "sedar_sft.reader_v2.splits.v1"


class SplitError(ValueError):
    """Raised when a split cannot be produced deterministically."""


@dataclass(frozen=True, slots=True)
class StratumReport:
    """Per-stratum accounting, for the split artifact."""

    index: int
    size: int
    dev_size: int
    test_size: int
    score_min: float
    score_max: float

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "size": self.size,
            "dev_size": self.dev_size,
            "test_size": self.test_size,
            "score_min": self.score_min,
            "score_max": self.score_max,
        }


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Two disjoint, sorted ID lists plus the accounting behind them."""

    dev_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    strata: tuple[StratumReport, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "dev_count": len(self.dev_ids),
            "test_count": len(self.test_ids),
            "strata": [stratum.as_dict() for stratum in self.strata],
        }


def _assign_strata(
    ordered_ids: Sequence[str],
    *,
    strata: int,
) -> list[list[str]]:
    """Split a score-ordered ID list into equal-size rank buckets."""

    total = len(ordered_ids)
    buckets: list[list[str]] = [[] for _ in range(strata)]
    for position, case_id in enumerate(ordered_ids):
        # Rank-based, not value-based: quartiles of a metric with ties would
        # give uneven buckets, and equal sizes are what keeps the halves
        # comparable.
        index = min(strata - 1, position * strata // total)
        buckets[index].append(case_id)
    return buckets


def _dev_quota(bucket_sizes: Sequence[int], *, dev_size: int, seed: int) -> list[int]:
    """Largest-remainder apportionment, so the quotas sum to dev_size exactly.

    Ties in the remainder are broken by a seeded shuffle, not by index. With
    equal-size strata every remainder is identical, so an index tie-break hands
    every leftover dev case to the lowest-scoring strata: at 20 strata over 460
    cases that skewed dev 0.015 METEOR below test, twice the effect a finer
    stratification was supposed to remove. Seeded tie-breaking keeps the result
    reproducible without the systematic direction.
    """

    total = sum(bucket_sizes)
    exact = [size * dev_size / total for size in bucket_sizes]
    quota = [int(value) for value in exact]
    shortfall = dev_size - sum(quota)
    tie_break = list(range(len(bucket_sizes)))
    random.Random(f"{seed}:quota").shuffle(tie_break)
    rank = {index: position for position, index in enumerate(tie_break)}
    order = sorted(
        range(len(bucket_sizes)),
        key=lambda i: (-(exact[i] - quota[i]), rank[i]),
    )
    for i in order[:shortfall]:
        quota[i] += 1
    for index, size in enumerate(bucket_sizes):
        if quota[index] > size:
            raise SplitError(
                f"stratum {index} cannot supply {quota[index]} of {size} cases"
            )
    return quota


def make_stratified_split(
    scores: Mapping[str, float],
    *,
    dev_size: int,
    seed: int,
    strata: int = 10,
) -> SplitResult:
    """Split case IDs into dev/test, stratified by score, reproducibly.

    ``scores`` maps case_id to the champion's score for that case. Ordering is
    by (score, case_id) so equal scores never depend on dict insertion order,
    and the shuffle inside each stratum is seeded, so the same inputs always
    produce the same split.
    """

    if strata < 1:
        raise SplitError("strata must be >= 1")
    if not scores:
        raise SplitError("no scored cases supplied")
    if not 0 < dev_size < len(scores):
        raise SplitError(
            f"dev_size must be between 1 and {len(scores) - 1}, got {dev_size}"
        )

    ordered_ids = [
        case_id for case_id, _ in sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
    ]
    buckets = _assign_strata(ordered_ids, strata=strata)
    quota = _dev_quota(
        [len(bucket) for bucket in buckets], dev_size=dev_size, seed=seed
    )

    dev: list[str] = []
    test: list[str] = []
    reports: list[StratumReport] = []
    for index, bucket in enumerate(buckets):
        shuffled = sorted(bucket)
        random.Random(f"{seed}:{index}").shuffle(shuffled)
        take = quota[index]
        dev.extend(shuffled[:take])
        test.extend(shuffled[take:])
        bucket_scores = [scores[case_id] for case_id in bucket]
        reports.append(
            StratumReport(
                index=index,
                size=len(bucket),
                dev_size=take,
                test_size=len(bucket) - take,
                score_min=min(bucket_scores),
                score_max=max(bucket_scores),
            )
        )

    dev_ids = tuple(sorted(dev))
    test_ids = tuple(sorted(test))
    if set(dev_ids) & set(test_ids):
        raise SplitError("dev and test splits overlap")
    if len(dev_ids) + len(test_ids) != len(scores):
        raise SplitError("split does not cover every scored case exactly once")
    return SplitResult(dev_ids=dev_ids, test_ids=test_ids, strata=tuple(reports))


__all__ = [
    "SPLIT_SCHEMA_VERSION",
    "SplitError",
    "SplitResult",
    "StratumReport",
    "make_stratified_split",
]
