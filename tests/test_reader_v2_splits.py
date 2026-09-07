"""Acceptance tests for the frozen reader-v2 dev/test split."""

from __future__ import annotations

import random
from statistics import mean

import pytest

from legal_rag.sedar_sft.splits import SplitError, make_stratified_split


def _scores(n: int = 460, seed: int = 7) -> dict[str, float]:
    rng = random.Random(seed)
    return {str(100000 + i): round(rng.betavariate(2, 2), 4) for i in range(n)}


def test_split_is_exact_disjoint_and_covering() -> None:
    scores = _scores()
    result = make_stratified_split(scores, dev_size=230, seed=42)
    assert len(result.dev_ids) == 230
    assert len(result.test_ids) == 230
    assert not set(result.dev_ids) & set(result.test_ids)
    assert set(result.dev_ids) | set(result.test_ids) == set(scores)
    assert list(result.dev_ids) == sorted(result.dev_ids)
    assert list(result.test_ids) == sorted(result.test_ids)


def test_split_is_reproducible_and_seed_sensitive() -> None:
    scores = _scores()
    first = make_stratified_split(scores, dev_size=230, seed=42)
    assert first.dev_ids == make_stratified_split(scores, dev_size=230, seed=42).dev_ids
    assert first.dev_ids != make_stratified_split(scores, dev_size=230, seed=43).dev_ids


def test_strata_are_equal_sized_ordered_and_sum_to_dev_size() -> None:
    result = make_stratified_split(_scores(), dev_size=230, seed=42, strata=10)
    assert len(result.strata) == 10
    assert {stratum.size for stratum in result.strata} == {46}
    assert sum(stratum.dev_size for stratum in result.strata) == 230
    lows = [stratum.score_min for stratum in result.strata]
    assert lows == sorted(lows)


def test_stratification_keeps_the_halves_comparable() -> None:
    """The point of stratifying: dev and test must be equally hard.

    The gate reads confidence intervals against a +-0.008 noise floor, so a
    difficulty gap of that size between the halves would move a decision on
    its own.
    """

    scores = _scores()
    gaps = []
    for seed in range(1, 21):
        result = make_stratified_split(scores, dev_size=230, seed=seed)
        gaps.append(
            mean(scores[i] for i in result.dev_ids)
            - mean(scores[i] for i in result.test_ids)
        )
    assert max(abs(gap) for gap in gaps) < 0.008
    # No systematic direction: the tie-break in the apportionment must not
    # hand every leftover dev case to the low-scoring strata.
    assert abs(mean(gaps)) < 0.002


@pytest.mark.parametrize("dev_size", [0, 460, -1, 461])
def test_split_fails_closed_on_impossible_sizes(dev_size: int) -> None:
    with pytest.raises(SplitError):
        make_stratified_split(_scores(), dev_size=dev_size, seed=42)


def test_split_fails_closed_on_empty_input() -> None:
    with pytest.raises(SplitError):
        make_stratified_split({}, dev_size=1, seed=42)


def test_uneven_strata_still_apportion_exactly() -> None:
    scores = {str(i): i / 7 for i in range(7)}
    result = make_stratified_split(scores, dev_size=3, seed=1, strata=4)
    assert len(result.dev_ids) == 3
    assert len(result.test_ids) == 4
    assert sum(stratum.dev_size for stratum in result.strata) == 3
