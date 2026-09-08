"""Acceptance tests for adaptive evidence-pack selection (TASK 25)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from legal_rag.sedar_retrieval.evidence.adaptive_pack import (
    AdaptivePackPolicy,
    PackCandidate,
    select_adaptive_pack,
)


@dataclass
class Unit:
    unit_id: str
    document_id: str
    char_count: int
    parent_unit_id: str | None = None
    level: str = "article"
    role: str = "substantive"
    packable: bool = True


def _units(*units: Unit) -> dict[str, Unit]:
    return {unit.unit_id: unit for unit in units}


def _cands(*pairs: tuple[str, float]) -> list[PackCandidate]:
    return [
        PackCandidate(unit_id=unit_id, rank=index, score=score)
        for index, (unit_id, score) in enumerate(pairs, start=1)
    ]


# --- the champion must be reproducible ------------------------------------


def test_defaults_reproduce_a_single_cap_fill() -> None:
    units = _units(*(Unit(f"a{i}", f"doc{i}", 1500) for i in range(6)))
    selection = select_adaptive_pack(
        _cands(*((f"a{i}", 5.0 - i) for i in range(6))),
        units,
        AdaptivePackPolicy(),
    )
    # target_chars == max_total_chars == 6000, so four 1500-char blocks fill it.
    assert selection.total_chars == 6000
    assert len(selection.blocks) == 4
    assert selection.stop_reason == "budget_exhausted"
    assert selection.starved is False
    assert selection.fill_ratio == pytest.approx(1.0)


def test_no_gating_by_default() -> None:
    policy = AdaptivePackPolicy()
    assert policy.gating_enabled is False
    units = _units(Unit("a0", "doc0", 100), Unit("a1", "doc1", 100))
    selection = select_adaptive_pack(_cands(("a0", 9.0), ("a1", -50.0)), units, policy)
    assert selection.ordered_ids == ("a0", "a1")


# --- the two stop reasons, which the old config could not distinguish -----


def test_budget_exhausted_means_evidence_existed_and_was_refused() -> None:
    units = _units(*(Unit(f"a{i}", f"doc{i}", 2000) for i in range(5)))
    selection = select_adaptive_pack(
        _cands(*((f"a{i}", 5.0 - i) for i in range(5))),
        units,
        AdaptivePackPolicy(target_chars=4000, max_total_chars=4000),
    )
    assert selection.stop_reason == "budget_exhausted"
    assert selection.starved is False


def test_content_exhausted_means_the_retriever_ran_out_not_the_budget() -> None:
    units = _units(Unit("a0", "doc0", 300), Unit("a1", "doc1", 200))
    selection = select_adaptive_pack(
        _cands(("a0", 5.0), ("a1", 4.0)),
        units,
        AdaptivePackPolicy(target_chars=6000, max_total_chars=6000),
    )
    assert selection.total_chars == 500
    assert selection.stop_reason == "content_exhausted"
    assert selection.starved is True
    assert selection.fill_ratio < 0.1


# --- starvation backfill --------------------------------------------------


def test_starvation_backfill_fills_an_under_target_pack_from_gated_out() -> None:
    units = _units(
        Unit("a0", "doc0", 800),
        Unit("a1", "doc1", 1200),
        Unit("a2", "doc2", 1500),
    )
    candidates = _cands(("a0", 9.0), ("a1", 1.0), ("a2", 0.5))
    policy = AdaptivePackPolicy(score_floor=5.0, target_chars=3000, max_total_chars=6000)
    with_backfill = select_adaptive_pack(candidates, units, policy)
    assert len(with_backfill.blocks) == 3
    assert with_backfill.backfilled_blocks == 2
    assert with_backfill.total_chars == 3500

    without = select_adaptive_pack(
        candidates,
        units,
        AdaptivePackPolicy(
            score_floor=5.0,
            target_chars=3000,
            max_total_chars=6000,
            starvation_backfill=False,
        ),
    )
    assert len(without.blocks) == 1
    assert without.starved is True
    assert without.dropped_reasons["score_floor"] == 2


def test_backfill_never_exceeds_the_hard_cap() -> None:
    units = _units(Unit("a0", "doc0", 900), Unit("a1", "doc1", 5000))
    selection = select_adaptive_pack(
        _cands(("a0", 9.0), ("a1", 0.1)),
        units,
        AdaptivePackPolicy(score_floor=5.0, target_chars=1000, max_total_chars=1000),
    )
    assert selection.total_chars == 900
    assert selection.dropped_reasons.get("max_total_chars") == 1


# --- distractor guard -----------------------------------------------------


def test_marginal_band_is_capped_so_backfill_cannot_flood_the_pack() -> None:
    units = _units(*(Unit(f"a{i}", f"doc{i}", 400) for i in range(6)))
    candidates = _cands(
        ("a0", 9.0), ("a1", 8.8), ("a2", 1.0), ("a3", 0.9), ("a4", 0.8), ("a5", 0.7)
    )
    policy = AdaptivePackPolicy(
        relative_margin=1.0,
        max_marginal_blocks=1,
        target_chars=6000,
        max_total_chars=6000,
        max_blocks=6,
    )
    selection = select_adaptive_pack(candidates, units, policy)
    assert selection.marginal_blocks <= 1
    assert selection.ordered_ids[:2] == ("a0", "a1")
    assert len(selection.blocks) == 3


# --- parent expansion (small to big) -------------------------------------


def test_child_hit_is_swapped_for_the_parent_article_when_it_fits() -> None:
    units = _units(
        Unit("art5", "doc0", 2200),
        Unit("art5::p0", "doc0", 700, parent_unit_id="art5", level="article_part"),
        Unit("art9", "doc1", 900),
    )
    selection = select_adaptive_pack(
        _cands(("art5::p0", 9.0), ("art9", 3.0)),
        units,
        AdaptivePackPolicy(target_chars=6000, max_total_chars=6000),
    )
    assert selection.ordered_ids[0] == "art5"
    assert selection.blocks[0].source == "parent_expansion"
    assert selection.blocks[0].replaced_unit_id == "art5::p0"
    assert selection.expanded_blocks == 1


def test_parent_is_not_used_when_it_does_not_fit_the_remaining_budget() -> None:
    units = _units(
        Unit("a0", "doc0", 900),
        Unit("art5", "doc1", 5000),
        Unit("art5::p0", "doc1", 400, parent_unit_id="art5", level="article_part"),
    )
    selection = select_adaptive_pack(
        _cands(("a0", 9.0), ("art5::p0", 8.0)),
        units,
        AdaptivePackPolicy(target_chars=2000, max_total_chars=2000),
    )
    assert selection.ordered_ids == ("a0", "art5::p0")
    assert selection.dropped_reasons.get("parent_did_not_fit") == 1


def test_expansion_can_be_disabled() -> None:
    units = _units(
        Unit("art5", "doc0", 2200),
        Unit("art5::p0", "doc0", 700, parent_unit_id="art5", level="article_part"),
    )
    selection = select_adaptive_pack(
        _cands(("art5::p0", 9.0)),
        units,
        AdaptivePackPolicy(expand_to_parent=False),
    )
    assert selection.ordered_ids == ("art5::p0",)
    assert selection.expanded_blocks == 0


# --- nested-span suppression ---------------------------------------------


def test_parent_and_its_own_child_never_share_a_pack() -> None:
    units = _units(
        Unit("art5", "doc0", 1000),
        Unit("art5::p0", "doc0", 400, parent_unit_id="art5", level="article_part"),
        Unit("art5::p1", "doc0", 400, parent_unit_id="art5", level="article_part"),
        Unit("art9", "doc1", 500),
    )
    selection = select_adaptive_pack(
        _cands(("art5::p0", 9.0), ("art5::p1", 8.0), ("art5", 7.0), ("art9", 6.0)),
        units,
        AdaptivePackPolicy(target_chars=6000, max_total_chars=6000, max_blocks=6),
    )
    assert selection.ordered_ids == ("art5", "art9")
    # p0 expanded to art5; p1 is then a nested span of an already-packed parent.
    # The third candidate (art5 itself) is a plain duplicate, not a suppression.
    assert selection.suppressed_nested == 1
    assert selection.dropped_reasons["nested_span"] == 1
    assert selection.expanded_blocks == 1


# --- caps and guards ------------------------------------------------------


def test_per_document_cap_holds() -> None:
    units = _units(
        Unit("a0", "doc0", 300),
        Unit("a1", "doc0", 300),
        Unit("a2", "doc0", 300),
        Unit("b0", "doc1", 300),
    )
    selection = select_adaptive_pack(
        _cands(("a0", 5.0), ("a1", 4.0), ("a2", 3.0), ("b0", 2.0)),
        units,
        AdaptivePackPolicy(max_per_document=2, target_chars=6000, max_total_chars=6000),
    )
    assert selection.ordered_ids == ("a0", "a1", "b0")
    assert selection.dropped_reasons["max_per_document"] == 1


def test_unpackable_units_are_excluded_but_never_leave_an_empty_pack() -> None:
    units = _units(
        Unit("a0", "doc0", 500, role="enforcement", packable=False),
        Unit("a1", "doc1", 500),
    )
    selection = select_adaptive_pack(_cands(("a0", 9.0), ("a1", 1.0)), units, AdaptivePackPolicy())
    assert selection.ordered_ids == ("a1",)
    assert selection.dropped_reasons["not_packable"] == 1

    only_unpackable = _units(Unit("a0", "doc0", 500, packable=False))
    fallback = select_adaptive_pack(_cands(("a0", 9.0)), only_unpackable, AdaptivePackPolicy())
    assert fallback.ordered_ids == ("a0",)
    assert fallback.stop_reason == "content_exhausted"


def test_min_block_chars_skips_fragments_while_longer_text_remains() -> None:
    units = _units(Unit("frag", "doc0", 40), Unit("a1", "doc1", 900))
    selection = select_adaptive_pack(
        _cands(("frag", 9.0), ("a1", 5.0)),
        units,
        AdaptivePackPolicy(min_block_chars=200, target_chars=6000, max_total_chars=6000),
    )
    assert selection.ordered_ids == ("a1",)
    assert selection.dropped_reasons["min_block_chars"] == 1


def test_max_blocks_binds_and_is_reported() -> None:
    units = _units(*(Unit(f"a{i}", f"doc{i}", 100) for i in range(10)))
    selection = select_adaptive_pack(
        _cands(*((f"a{i}", 10.0 - i) for i in range(10))),
        units,
        AdaptivePackPolicy(max_blocks=3, target_chars=6000, max_total_chars=6000),
    )
    assert len(selection.blocks) == 3
    assert selection.stop_reason == "max_blocks"


# --- ordering -------------------------------------------------------------


def test_best_last_puts_the_top_block_next_to_the_question() -> None:
    units = _units(*(Unit(f"a{i}", f"doc{i}", 300) for i in range(3)))
    candidates = _cands(("a0", 9.0), ("a1", 5.0), ("a2", 1.0))
    first = select_adaptive_pack(candidates, units, AdaptivePackPolicy())
    last = select_adaptive_pack(candidates, units, AdaptivePackPolicy(order="best_last"))
    assert first.ordered_ids == ("a0", "a1", "a2")
    assert last.ordered_ids == ("a2", "a1", "a0")
    assert first.total_chars == last.total_chars


# --- validation and edges -------------------------------------------------


def test_policy_validates_its_own_arguments() -> None:
    with pytest.raises(ValueError):
        AdaptivePackPolicy(min_blocks=0)
    with pytest.raises(ValueError):
        AdaptivePackPolicy(min_blocks=4, max_blocks=2)
    with pytest.raises(ValueError):
        AdaptivePackPolicy(target_chars=7000, max_total_chars=6000)
    with pytest.raises(ValueError):
        AdaptivePackPolicy(max_per_document=0)
    with pytest.raises(ValueError):
        AdaptivePackPolicy(relative_margin=-1.0)
    with pytest.raises(ValueError):
        AdaptivePackPolicy(order="random")  # type: ignore[arg-type]


def test_empty_candidates_and_unknown_units() -> None:
    empty = select_adaptive_pack([], {}, AdaptivePackPolicy())
    assert empty.stop_reason == "no_candidates"
    assert empty.starved is True
    with pytest.raises(KeyError):
        select_adaptive_pack(_cands(("ghost", 1.0)), {"a0": Unit("a0", "d", 10)}, AdaptivePackPolicy())


def test_unscored_tail_does_not_break_the_gate() -> None:
    units = _units(Unit("a0", "doc0", 500), Unit("a1", "doc1", 500))
    selection = select_adaptive_pack(
        [PackCandidate("a0", 1, 9.0), PackCandidate("a1", 2, float("-inf"))],
        units,
        AdaptivePackPolicy(score_floor=5.0, target_chars=1500, max_total_chars=6000),
    )
    assert selection.top_score == pytest.approx(9.0)
    assert selection.ordered_ids == ("a0", "a1")
    assert selection.backfilled_blocks == 1


def test_selection_serialises_for_the_run_artifact() -> None:
    units = _units(Unit("a0", "doc0", 500))
    payload = select_adaptive_pack(_cands(("a0", 3.0)), units, AdaptivePackPolicy()).as_dict()
    assert payload["block_count"] == 1
    assert payload["blocks"][0]["unit_id"] == "a0"
    assert set(payload) >= {
        "total_chars",
        "stop_reason",
        "starved",
        "fill_ratio",
        "dropped_reasons",
        "backfilled_blocks",
        "expanded_blocks",
    }


def test_starvation_uses_a_ratio_not_exact_equality() -> None:
    units = _units(Unit("a0", "doc0", 5900))
    nearly_full = select_adaptive_pack(
        _cands(("a0", 5.0)),
        units,
        AdaptivePackPolicy(target_chars=6000, max_total_chars=6000),
    )
    # 5,900 of 6,000 is not a starved pack, and calling it one would make the
    # diagnostic fire on almost every query.
    assert nearly_full.starved is False
    assert nearly_full.stop_reason == "candidates_exhausted"

    half = select_adaptive_pack(
        _cands(("a0", 5.0)),
        _units(Unit("a0", "doc0", 2000)),
        AdaptivePackPolicy(target_chars=6000, max_total_chars=6000),
    )
    assert half.starved is True
    assert half.stop_reason == "content_exhausted"


def test_starvation_ratio_is_validated() -> None:
    with pytest.raises(ValueError):
        AdaptivePackPolicy(starvation_ratio=0.0)
    with pytest.raises(ValueError):
        AdaptivePackPolicy(starvation_ratio=1.5)
