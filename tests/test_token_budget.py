"""Acceptance tests for the reader-v2 token budget report."""

from __future__ import annotations

import pytest

from legal_rag.sedar_sft.token_budget import (
    P99_HEADROOM_THRESHOLD,
    build_report,
    percentile,
    summarise,
)


def test_percentile_returns_real_observations() -> None:
    values = list(range(1, 101))
    assert percentile(values, 0.50) == 50
    assert percentile(values, 0.99) == 99
    assert percentile(values, 1.0) == 100
    assert percentile(values, 0.0) == 1


def test_summarise_reports_the_whole_shape() -> None:
    summary = summarise([10, 20, 30, 40])
    assert summary["count"] == 4
    assert summary["min"] == 10
    assert summary["max"] == 40
    assert summary["p50"] == 20


def test_budget_holds_when_everything_fits() -> None:
    rows = [(str(i), 1200, 400) for i in range(100)]
    report = build_report(rows, max_seq_length=4096)
    assert report.verdict == "BUDGET_HOLDS"
    assert report.over_budget_count == 0
    assert report.recommended_max_seq_length == 4096


def test_a_single_truncated_example_raises_the_budget() -> None:
    """One over-budget example is enough: truncation is silent, not averaged."""

    rows = [(str(i), 1200, 400) for i in range(99)]
    rows.append(("over", 3900, 800))
    report = build_report(rows, max_seq_length=4096)
    assert report.verdict == "RAISE_MAX_SEQ_LENGTH"
    assert report.over_budget_count == 1
    assert report.over_budget_ids == ("over",)
    assert report.recommended_max_seq_length == 8192


def test_p99_without_headroom_raises_the_budget_too() -> None:
    """Fitting today is not enough when the cap sits on top of the prompt."""

    # 3900 total: still inside 4096, but only 196 tokens of slack under a
    # generation cap of 768, so the budget has to move before training.
    rows = [(str(i), 3400, 500) for i in range(100)]
    report = build_report(rows, max_seq_length=4096)
    assert report.total["p99"] > P99_HEADROOM_THRESHOLD
    assert report.over_budget_count == 0
    assert report.verdict == "RAISE_MAX_SEQ_LENGTH"


def test_report_carries_ids_not_text() -> None:
    rows = [("c1", 4000, 200), ("c2", 100, 100)]
    payload = build_report(rows, max_seq_length=4096).as_dict()
    serialized = repr(payload)
    assert "c1" in payload["over_budget_ids_sample"]
    assert "rendered_text" not in serialized
    assert "target_answer" not in serialized


@pytest.mark.parametrize("bad", [[], None])
def test_build_report_fails_closed_on_no_rows(bad: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        build_report(bad or [], max_seq_length=4096)
