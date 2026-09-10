from types import SimpleNamespace

from scripts.sedar_retrieval.build_reranker_training_data import (
    _children_by_parent,
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
