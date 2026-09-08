"""Acceptance tests for the corpus v4 -> v3 passage-view bridge (TASK 27)."""

from __future__ import annotations

import pytest

from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage
from legal_rag.sedar_retrieval.corpus_v4.export import (
    V4_TO_V3_LEVEL,
    ExportCounters,
    article_id_for,
    build_article_id_map,
    export_units,
    unit_to_passage_row,
)


def _unit(**kwargs) -> dict:
    base = {
        "unit_id": "d1::i0::art5",
        "document_id": "d1",
        "level": "article",
        "role": "substantive",
        "instrument_index": 0,
        "instrument_kind": "main",
        "article_number": "5",
        "article_title": "Phạm vi",
        "citation": "Nghị định 1/2020/NĐ-CP",
        "breadcrumb": "Nghị định 1/2020/NĐ-CP > Điều 5. Phạm vi",
        "reader_text": "Điều 5. Phạm vi\n1. Nội dung.",
        "retrieval_text": "Nghị định 1/2020/NĐ-CP > Điều 5. Phạm vi\nĐiều 5. Phạm vi\n1. Nội dung.",
        "char_count": 28,
        "indexed": True,
        "packable": True,
        "content_hash": "a" * 64,
        "source_path": "data/selected-contexts.zip",
        "source_member": "selected-contexts/context_d1.json",
        "parent_unit_id": None,
        "child_label": None,
        "link": None,
    }
    base.update(kwargs)
    return base


# --- the contract that unblocks everything ------------------------------


def test_exported_row_validates_as_a_canonical_passage() -> None:
    # This is the whole point of the bridge: load_passages_jsonl validates rows
    # with extra="forbid", so a v4 unit cannot be indexed without this mapping.
    row = unit_to_passage_row(_unit())
    assert row is not None
    passage = CanonicalPassage.model_validate(row)
    assert passage.passage_id == "d1::i0::art5"
    assert passage.retrieval_level == "article"
    assert passage.reader_text.startswith("Điều 5.")


def test_every_v4_level_maps_or_is_deliberately_dropped() -> None:
    assert V4_TO_V3_LEVEL == {
        "article": "article",
        "article_part": "clause",
        "block": "article",
    }
    # preamble is indexed=False in v4: provenance, not evidence.
    assert unit_to_passage_row(_unit(level="preamble")) is None
    assert unit_to_passage_row(_unit(level="something_new")) is None


def test_article_part_maps_to_clause_and_inherits_its_parent_article_id() -> None:
    part = _unit(
        unit_id="d1::i0::art5::p0",
        level="article_part",
        parent_unit_id="d1::i0::art5",
        child_label="Khoản 1 - Khoản 3",
    )
    row = unit_to_passage_row(part)
    assert row is not None
    passage = CanonicalPassage.model_validate(row)
    # clause level + the parent's article_id is what makes eval_retrieval.py's
    # article roll-up work with no change to the evaluator.
    assert passage.retrieval_level == "clause"
    assert passage.article_id == "d1::art::5"
    assert passage.clause_id == "d1::i0::art5::p0"
    assert passage.clause_number == "Khoản 1 - Khoản 3"
    assert passage.parent_id == "d1::i0::art5"


def test_block_units_are_their_own_scoring_unit() -> None:
    block = _unit(
        unit_id="d2::i0::blk0",
        document_id="d2",
        level="block",
        article_number=None,
        article_title=None,
        child_label="I. NHIỆM VỤ CHUNG",
    )
    row = unit_to_passage_row(block)
    assert row is not None
    passage = CanonicalPassage.model_validate(row)
    assert passage.retrieval_level == "article"
    # No Điều number, so no article_id to roll up to: it is the unit.
    assert passage.article_id is None


# --- label-compatible ids, scoped only where they must be ---------------


def test_main_text_keeps_the_id_spelling_the_existing_labels_use() -> None:
    assert article_id_for(_unit()) == "d1::art::5"


def test_annex_article_keeps_the_flat_id_when_the_number_is_unambiguous() -> None:
    units = [
        _unit(unit_id="d1::i0::art1", article_number="1"),
        _unit(unit_id="d1::i1::art25", article_number="25", instrument_index=1),
    ]
    id_map = build_article_id_map(units)
    # Điều 25 exists only in the annex, so scoping it would break the labels for
    # nothing. This was 16 of the 16 misses in the first measured export.
    assert article_id_for(units[1], article_id_map=id_map) == "d1::art::25"


def test_a_genuinely_colliding_number_is_scoped_by_instrument() -> None:
    units = [
        _unit(unit_id="d1::i0::art1", article_number="1"),
        _unit(unit_id="d1::i1::art1", article_number="1", instrument_index=1),
    ]
    id_map = build_article_id_map(units)
    assert article_id_for(units[0], article_id_map=id_map) == "d1::art::1"
    assert article_id_for(units[1], article_id_map=id_map) == "d1::i1::art::1"


def test_without_the_pre_pass_every_annex_article_is_scoped() -> None:
    annex = _unit(unit_id="d1::i1::art25", article_number="25", instrument_index=1)
    assert article_id_for(annex) == "d1::i1::art::25"


def test_article_id_needs_a_document_and_a_number() -> None:
    assert article_id_for(_unit(article_number=None)) is None
    assert article_id_for(_unit(document_id="")) is None


# --- streaming export and its counters ----------------------------------


def test_export_skips_unindexed_units_and_tallies_levels() -> None:
    units = [
        _unit(),
        _unit(unit_id="d1::i0::art5::p0", level="article_part", parent_unit_id="d1::i0::art5"),
        _unit(unit_id="d1::i0::preamble", level="preamble", indexed=False),
        _unit(unit_id="d3::i0::blk0", document_id="d3", level="block", article_number=None),
    ]
    counters = ExportCounters()
    rows = list(export_units(units, counters=counters))
    assert len(rows) == 3
    assert counters.skipped_not_indexed == 1
    assert counters.by_v3_level == {"article": 2, "clause": 1}
    assert counters.exported == 3
    for row in rows:
        CanonicalPassage.model_validate(row)


def test_duplicate_unit_ids_are_dropped_and_counted() -> None:
    counters = ExportCounters()
    rows = list(export_units([_unit(), _unit()], counters=counters))
    assert len(rows) == 1
    assert counters.duplicate_passage_ids == 1


def test_compact_units_reconstruct_retrieval_text() -> None:
    # --compact omits retrieval_text because breadcrumb + reader_text derives it.
    compact = _unit()
    compact.pop("retrieval_text")
    row = unit_to_passage_row(compact)
    assert row is not None
    assert row["retrieval_text"].startswith("Nghị định 1/2020/NĐ-CP > Điều 5.")
    assert compact["reader_text"] in row["retrieval_text"]
    CanonicalPassage.model_validate(row)


def test_rows_without_an_id_or_document_are_refused() -> None:
    assert unit_to_passage_row(_unit(unit_id="")) is None
    assert unit_to_passage_row(_unit(document_id="")) is None


def test_counters_serialise_for_the_report() -> None:
    counters = ExportCounters()
    list(export_units([_unit()], counters=counters))
    payload = counters.as_dict()
    assert payload["passages_exported"] == 1
    assert set(payload) >= {
        "units_read",
        "skipped_not_indexed",
        "exported_by_v3_level",
        "duplicate_passage_ids",
        "annex_scoped_article_ids",
    }


def test_content_hash_and_provenance_survive_the_mapping() -> None:
    row = unit_to_passage_row(_unit())
    assert row is not None
    passage = CanonicalPassage.model_validate(row)
    assert passage.source.content_hash == "a" * 64
    assert passage.source.source_member == "selected-contexts/context_d1.json"
    assert passage.source.document_id == "d1"


def test_citation_becomes_the_document_name() -> None:
    row = unit_to_passage_row(_unit())
    assert row is not None
    # include_document_name in the packer renders this; the zip member is what
    # it showed before, which was the wrong string.
    assert row["document_name"] == "Nghị định 1/2020/NĐ-CP"


@pytest.mark.parametrize("level", ["article", "article_part", "block"])
def test_all_exported_levels_validate(level: str) -> None:
    unit = _unit(level=level, unit_id=f"d9::i0::{level}")
    if level == "article_part":
        unit["parent_unit_id"] = "d9::i0::art5"
    row = unit_to_passage_row(unit)
    assert row is not None
    CanonicalPassage.model_validate(row)
