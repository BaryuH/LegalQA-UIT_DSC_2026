"""Tests for the paired run comparison that gates every adoption decision."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "sedar_retrieval"
    / "compare_metrics_paired.py"
)
_spec = importlib.util.spec_from_file_location("_paired_module", _MODULE_PATH)
assert _spec and _spec.loader
_paired = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paired)


def _runs(deltas, metric="meteor", base=0.50):
    baseline = {str(i): {metric: base} for i in range(len(deltas))}
    candidate = {str(i): {metric: base + d} for i, d in enumerate(deltas)}
    return baseline, candidate


def test_a_constant_shift_is_recovered_with_a_tight_interval() -> None:
    baseline, candidate = _runs([0.02] * 200)
    m = _paired.compare(baseline, candidate, metrics=("meteor",),
                        resamples=1000)["metrics"]["meteor"]
    assert m["delta"] == pytest.approx(0.02)
    assert m["ci_low"] == pytest.approx(0.02)
    assert m["better"] == 200 and m["worse"] == 0


def test_no_difference_yields_a_large_p_value() -> None:
    baseline, candidate = _runs([0.0] * 200)
    m = _paired.compare(baseline, candidate, metrics=("meteor",),
                        resamples=1000)["metrics"]["meteor"]
    assert m["delta"] == pytest.approx(0.0)
    assert m["p_value"] > 0.5


def test_only_shared_cases_are_compared() -> None:
    baseline = {"a": {"meteor": 0.4}, "b": {"meteor": 0.4}}
    candidate = {"b": {"meteor": 0.6}, "c": {"meteor": 0.9}}
    r = _paired.compare(baseline, candidate, metrics=("meteor",), resamples=200)
    assert r["shared_cases"] == 1
    assert r["metrics"]["meteor"]["n"] == 1


def test_cases_missing_the_metric_are_dropped_not_zero_filled() -> None:
    baseline = {"a": {"meteor": 0.4}, "b": {"meteor": None}}
    candidate = {"a": {"meteor": 0.5}, "b": {"meteor": 0.9}}
    m = _paired.compare(baseline, candidate, metrics=("meteor",),
                        resamples=200)["metrics"]["meteor"]
    assert m["n"] == 1
    assert m["delta"] == pytest.approx(0.1)


def test_a_significant_but_tiny_effect_is_not_adoptable() -> None:
    """The decision rule reads the interval, not the p-value.

    Post-generation loop trimming measured +0.0025 ROUGE-L at p=0.002: real,
    and still inside the noise floor. The tool must say so rather than let a
    small p imply adoption.
    """

    baseline, candidate = _runs([0.0025] * 400)
    m = _paired.compare(baseline, candidate, metrics=("meteor",),
                        resamples=1000)["metrics"]["meteor"]
    assert m["p_value"] < 0.05
    assert not m["beats_noise_floor"]
    assert m["within_noise_floor"]


def test_a_real_regression_is_flagged_outside_the_noise_floor() -> None:
    baseline, candidate = _runs([-0.03] * 200)
    m = _paired.compare(baseline, candidate, metrics=("meteor",),
                        resamples=1000)["metrics"]["meteor"]
    assert not m["within_noise_floor"]


def test_duplicate_case_ids_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({"per_case": [{"id": "a", "meteor": 0.1},
                                 {"id": "a", "meteor": 0.2}]}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="Duplicate"):
        _paired.load_per_case(path)


def test_the_comparison_is_deterministic_for_a_given_seed() -> None:
    baseline, candidate = _runs([0.01, 0.03, -0.02, 0.05] * 40)
    a = _paired.compare(baseline, candidate, metrics=("meteor",), resamples=500)
    b = _paired.compare(baseline, candidate, metrics=("meteor",), resamples=500)
    assert a == b
