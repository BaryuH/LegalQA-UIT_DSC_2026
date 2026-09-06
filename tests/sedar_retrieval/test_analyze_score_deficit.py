"""Tests for deficit-based prioritisation of failure groups."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "sedar_retrieval"
    / "analyze_score_deficit.py"
)
_spec = importlib.util.spec_from_file_location("_deficit", _PATH)
assert _spec and _spec.loader
_deficit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_deficit)


def _corpus(spec: dict[str, tuple[int, float]]):
    per_case, groups = {}, {}
    for name, (count, score) in spec.items():
        ids = [f"{name}{i}" for i in range(count)]
        groups[name] = ids
        per_case.update({i: {"meteor": score} for i in ids})
    return per_case, groups


def test_a_small_badly_failing_group_outranks_a_large_average_one() -> None:
    """Deficit share alone ranks these backwards; the ceiling does not."""

    per_case, groups = _corpus({"small_bad": (50, 0.20), "large_mid": (250, 0.55)})
    report = _deficit.deficit_by_group(per_case, groups)
    first = report["groups"][0]
    assert first["group"] == "small_bad"
    # The larger group holds most of the deficit yet offers nothing to win.
    large = next(g for g in report["groups"] if g["group"] == "large_mid")
    assert large["share_of_deficit"] > first["share_of_deficit"]
    assert large["ceiling_if_lifted_to_mean"] == pytest.approx(0.0)


def test_a_group_already_above_the_mean_has_no_ceiling() -> None:
    per_case, groups = _corpus({"good": (100, 0.90), "poor": (100, 0.30)})
    report = _deficit.deficit_by_group(per_case, groups)
    good = next(g for g in report["groups"] if g["group"] == "good")
    assert good["ceiling_if_lifted_to_mean"] == pytest.approx(0.0)


def test_deficit_shares_sum_to_one() -> None:
    per_case, groups = _corpus({"a": (30, 0.4), "b": (70, 0.6)})
    report = _deficit.deficit_by_group(per_case, groups)
    assert sum(g["share_of_deficit"] for g in report["groups"]) == pytest.approx(1.0)


def test_cases_outside_every_group_are_kept_as_a_remainder() -> None:
    per_case, groups = _corpus({"a": (10, 0.4)})
    per_case["orphan"] = {"meteor": 0.1}
    report = _deficit.deficit_by_group(per_case, groups)
    assert any(g["group"] == "(con lai)" for g in report["groups"])
    assert report["cases"] == 11


def test_without_groups_it_falls_back_to_quartiles() -> None:
    per_case = {str(i): {"meteor": i / 100} for i in range(100)}
    report = _deficit.deficit_by_group(per_case, None)
    assert len(report["groups"]) == 4
    worst = next(g for g in report["groups"] if "te nhat" in g["group"])
    assert worst["mean_metric"] < 0.2


def test_cases_missing_the_metric_are_excluded() -> None:
    per_case = {"a": {"meteor": 0.5}, "b": {"meteor": None}}
    report = _deficit.deficit_by_group(per_case, {"g": ["a", "b"]})
    assert report["cases"] == 1
    assert report["groups"][0]["n"] == 1


def test_a_group_above_the_corpus_mean_has_no_ceiling_however_large() -> None:
    """Summing below-mean cases rewards size; lifting the group mean does not.

    A group covering half the corpus accumulates a large within-group
    shortfall even when its own mean sits above the corpus mean. That is not
    a gain anyone can collect, and ranking on it puts the wrong group first.
    """

    per_case, groups = _corpus({"big_mixed": (236, 0.57), "small_bad": (38, 0.34)})
    report = _deficit.deficit_by_group(per_case, groups)
    big = next(g for g in report["groups"] if g["group"] == "big_mixed")
    small = next(g for g in report["groups"] if g["group"] == "small_bad")
    assert big["mean_metric"] > report["mean_metric"]
    assert big["ceiling_if_lifted_to_mean"] == pytest.approx(0.0)
    assert small["ceiling_if_lifted_to_mean"] > 0.0
    assert report["groups"][0]["group"] == "small_bad"


def test_within_group_shortfall_is_reported_and_never_smaller() -> None:
    per_case, groups = _corpus({"a": (60, 0.30), "b": (140, 0.70)})
    report = _deficit.deficit_by_group(per_case, groups)
    for g in report["groups"]:
        # Equal for a uniform group, larger for a mixed one. The tolerance is
        # for float accumulation, not for slack in the relationship.
        assert g["within_group_shortfall"] >= g["ceiling_if_lifted_to_mean"] - 1e-9
