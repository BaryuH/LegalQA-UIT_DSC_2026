from types import SimpleNamespace

from scripts.sedar_retrieval.build_reranker_training_data import (
    _children_by_parent,
    _load_labels,
    _normalise_ranks,
)


def test_children_by_parent_indexes_containment_without_corpus_scan() -> None:
    units = {
        "parent": SimpleNamespace(parent_unit_id=None),
        "child-b": SimpleNamespace(parent_unit_id="parent"),
        "child-a": SimpleNamespace(parent_unit_id="parent"),
        "other": SimpleNamespace(parent_unit_id="elsewhere"),
    }

    assert _children_by_parent(units) == {
        "parent": ("child-a", "child-b"),
        "elsewhere": ("other",),
    }


def test_load_labels_accepts_v2_relevant_ids(tmp_path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        '{"query_id": "q1", "relevant_ids": ["doc::art::1"]}\n',
        encoding="utf-8",
    )

    assert _load_labels(labels_path) == {"q1": {"doc::art::1"}}


def test_normalise_ranks_maps_first_and_last_observed_rank() -> None:
    rows = [
        ("first", 0.9, 1),
        ("middle", 0.5, 3),
        ("last", 0.1, 5),
    ]

    assert _normalise_ranks(rows) == [1.0, 0.5, 0.0]
