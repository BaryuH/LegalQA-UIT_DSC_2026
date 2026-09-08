"""Acceptance tests for the AITeamVN/Vietnamese_Reranker rerank path."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from legal_rag.sedar_retrieval.ranking.vietnamese_reranker import (
    DEFAULT_MAX_LENGTH,
    CutoffPolicy,
    RerankedCandidate,
    VietnameseRerankerConfig,
    VietnameseRerankerError,
    VietnameseRerankerScorer,
    apply_cutoff,
    merge_children_to_parent,
    rerank_query,
    unit_text,
)


@dataclass
class Unit:
    unit_id: str
    document_id: str
    reader_text: str
    breadcrumb: str = "Nghị định 1/2020/NĐ-CP > Điều 1. Phạm vi"
    parent_unit_id: str | None = None
    level: str = "article"
    role: str = "substantive"
    char_count: int = 0
    packable: bool = True

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.reader_text)


def _units(*units: Unit) -> dict[str, Unit]:
    return {unit.unit_id: unit for unit in units}


# --- text rendering -------------------------------------------------------


def test_reader_text_is_the_default_and_breadcrumb_is_opt_in() -> None:
    unit = Unit("u1", "d1", "Điều 76. Ký kết\n1. Nội dung.")
    assert unit_text(unit) == unit.reader_text
    with_path = unit_text(unit, text_field="breadcrumb_reader_text")
    assert with_path.startswith("Nghị định 1/2020/NĐ-CP")
    assert unit.reader_text in with_path


# --- child -> parent merging ---------------------------------------------


def test_child_hits_collapse_onto_the_parent_article_with_max_score() -> None:
    units = _units(
        Unit("a1", "d1", "Điều 5. " + "x" * 3000),
        Unit("a1::p0", "d1", "phần 1", parent_unit_id="a1", level="article_part"),
        Unit("a1::p1", "d1", "phần 2", parent_unit_id="a1", level="article_part"),
        Unit("a2", "d1", "Điều 6. ngắn"),
    )
    scored = [
        RerankedCandidate("a1::p0", "d1", 2.0, 0, 1),
        RerankedCandidate("a2", "d1", 3.0, 0, 2),
        RerankedCandidate("a1::p1", "d1", 5.0, 0, 3),
    ]
    merged = merge_children_to_parent(scored, units)
    assert [item.unit_id for item in merged] == ["a1", "a2"]
    # max, not sum: an article must not gain from having been split.
    assert merged[0].score == pytest.approx(5.0)
    assert merged[0].merged_from == ("a1::p0", "a1::p1")
    assert [item.rank for item in merged] == [1, 2]


def test_mean_aggregate_is_available_and_differs_from_max() -> None:
    units = _units(
        Unit("a1", "d1", "Điều 5. long"),
        Unit("a1::p0", "d1", "p0", parent_unit_id="a1", level="article_part"),
        Unit("a1::p1", "d1", "p1", parent_unit_id="a1", level="article_part"),
    )
    scored = [
        RerankedCandidate("a1::p0", "d1", 1.0, 0, 1),
        RerankedCandidate("a1::p1", "d1", 5.0, 0, 2),
    ]
    assert merge_children_to_parent(scored, units)[0].score == pytest.approx(5.0)
    mean = merge_children_to_parent(scored, units, aggregate="mean")
    assert mean[0].score == pytest.approx(3.0)


def test_orphan_child_keeps_itself_when_the_parent_is_not_in_the_view() -> None:
    units = _units(Unit("a1::p0", "d1", "phần", parent_unit_id="a1", level="article_part"))
    merged = merge_children_to_parent([RerankedCandidate("a1::p0", "d1", 1.0, 0, 1)], units)
    assert [item.unit_id for item in merged] == ["a1::p0"]


def test_ties_fall_back_to_retriever_rank_not_to_the_id() -> None:
    units = _units(Unit("zzz", "d1", "a"), Unit("aaa", "d2", "b"))
    scored = [
        RerankedCandidate("zzz", "d1", 1.0, 0, 1),
        RerankedCandidate("aaa", "d2", 1.0, 0, 2),
    ]
    merged = merge_children_to_parent(scored, units)
    # Sorting on the id would put "aaa" first and discard the retriever signal.
    assert [item.unit_id for item in merged] == ["zzz", "aaa"]


def test_unknown_unit_is_a_hard_error() -> None:
    with pytest.raises(VietnameseRerankerError):
        merge_children_to_parent([RerankedCandidate("ghost", "d1", 1.0, 0, 1)], {})


# --- variable-size cutoff -------------------------------------------------


def _ranked(*pairs: tuple[str, float]) -> tuple[RerankedCandidate, ...]:
    return tuple(
        RerankedCandidate(unit_id, "d1" if unit_id.startswith("a") else "d2", score, index, index)
        for index, (unit_id, score) in enumerate(pairs, start=1)
    )


def test_no_threshold_reproduces_fixed_k() -> None:
    units = _units(*(Unit(f"a{i}", f"doc{i}", "x" * 100) for i in range(5)))
    ranked = tuple(
        RerankedCandidate(f"a{i}", f"doc{i}", 5.0 - i, i + 1, i + 1) for i in range(5)
    )
    result = apply_cutoff(ranked, units, CutoffPolicy(max_keep=3))
    assert [item.unit_id for item in result.kept] == ["a0", "a1", "a2"]
    assert result.stop_reason == "max_keep"


def test_threshold_produces_a_variable_size_answer_set() -> None:
    units = _units(*(Unit(f"a{i}", f"doc{i}", "x" * 100) for i in range(5)))
    ranked = tuple(
        RerankedCandidate(f"a{i}", f"doc{i}", score, i + 1, i + 1)
        for i, score in enumerate([6.0, 5.5, 1.0, 0.5, 0.1])
    )
    policy = CutoffPolicy(score_threshold=2.0, min_keep=1, max_keep=8)
    result = apply_cutoff(ranked, units, policy)
    assert [item.unit_id for item in result.kept] == ["a0", "a1"]
    assert result.stop_reason == "score_threshold"


def test_min_keep_never_returns_an_empty_pack() -> None:
    units = _units(Unit("a0", "doc0", "x" * 100))
    ranked = _ranked(("a0", -9.0))
    result = apply_cutoff(ranked, units, CutoffPolicy(score_threshold=5.0))
    assert len(result.kept) == 1
    assert result.stop_reason in {"exhausted_candidates", "min_keep_override"}


def test_character_budget_binds_and_is_reported() -> None:
    units = _units(
        Unit("a0", "doc0", "x" * 500),
        Unit("a1", "doc1", "y" * 5000),
        Unit("a2", "doc2", "z" * 300),
    )
    ranked = _ranked(("a0", 5.0), ("a1", 4.0), ("a2", 3.0))
    result = apply_cutoff(ranked, units, CutoffPolicy(max_total_chars=1000, max_keep=8))
    assert [item.unit_id for item in result.kept] == ["a0", "a2"]
    assert result.dropped_reasons["max_total_chars"] == 1
    assert result.total_chars == 800


def test_per_document_cap_prevents_one_document_taking_the_pack() -> None:
    units = _units(
        Unit("a0", "doc0", "x" * 100),
        Unit("a1", "doc0", "x" * 100),
        Unit("a2", "doc0", "x" * 100),
        Unit("b0", "doc1", "y" * 100),
    )
    ranked = _ranked(("a0", 5.0), ("a1", 4.9), ("a2", 4.8), ("b0", 1.0))
    result = apply_cutoff(ranked, units, CutoffPolicy(max_per_document=2, max_keep=8))
    assert [item.unit_id for item in result.kept] == ["a0", "a1", "b0"]
    assert result.dropped_reasons["max_per_document"] == 1


def test_unpackable_units_are_ranked_but_do_not_consume_budget() -> None:
    units = _units(
        Unit("a0", "doc0", "x" * 100, role="enforcement", packable=False),
        Unit("a1", "doc1", "y" * 100),
    )
    ranked = _ranked(("a0", 9.0), ("a1", 1.0))
    result = apply_cutoff(ranked, units, CutoffPolicy(max_keep=8))
    assert [item.unit_id for item in result.kept] == ["a1"]
    assert result.dropped_reasons["not_packable"] == 1
    kept_anyway = apply_cutoff(ranked, units, CutoffPolicy(respect_packable=False))
    assert kept_anyway.kept[0].unit_id == "a0"


def test_relative_margin_adapts_to_the_top_score() -> None:
    units = _units(*(Unit(f"a{i}", f"doc{i}", "x" * 50) for i in range(4)))
    ranked = tuple(
        RerankedCandidate(f"a{i}", f"doc{i}", score, i + 1, i + 1)
        for i, score in enumerate([8.0, 7.6, 4.0, 3.0])
    )
    result = apply_cutoff(ranked, units, CutoffPolicy(relative_margin=1.0, max_keep=8))
    assert [item.unit_id for item in result.kept] == ["a0", "a1"]


def test_cutoff_policy_validates_its_own_arguments() -> None:
    with pytest.raises(ValueError):
        CutoffPolicy(min_keep=0)
    with pytest.raises(ValueError):
        CutoffPolicy(min_keep=4, max_keep=2)
    with pytest.raises(ValueError):
        CutoffPolicy(max_total_chars=0)
    with pytest.raises(ValueError):
        CutoffPolicy(relative_margin=-0.5)


# --- end-to-end reordering with a stub scorer ----------------------------


def test_rerank_query_keeps_the_unscored_tail_below_the_head() -> None:
    units = _units(*(Unit(f"a{i}", f"doc{i}", f"text {i}") for i in range(6)))
    order = ["a0", "a1", "a2", "a3", "a4", "a5"]

    def score_fn(query: str, texts) -> list[float]:
        assert query == "câu hỏi"
        # Reverse the head so the reordering is visible.
        return [float(len(texts) - index) for index in range(len(texts))]

    ranked = rerank_query("câu hỏi", order, units, score_fn=score_fn, top_k=3)
    assert [item.unit_id for item in ranked[:3]] == ["a0", "a1", "a2"]
    assert [item.unit_id for item in ranked[3:]] == ["a3", "a4", "a5"]
    assert all(item.score == float("-inf") for item in ranked[3:])


def test_rerank_query_rejects_a_scorer_that_returns_the_wrong_length() -> None:
    units = _units(Unit("a0", "doc0", "t"))
    with pytest.raises(VietnameseRerankerError):
        rerank_query("q", ["a0"], units, score_fn=lambda q, t: [1.0, 2.0])


def test_rerank_query_rejects_duplicate_candidates() -> None:
    units = _units(Unit("a0", "doc0", "t"))
    with pytest.raises(VietnameseRerankerError):
        rerank_query("q", ["a0", "a0"], units, score_fn=lambda q, t: [1.0, 1.0])


# --- configuration --------------------------------------------------------


def test_config_defaults_match_the_model_card() -> None:
    config = VietnameseRerankerConfig()
    assert config.model == "AITeamVN/Vietnamese_Reranker"
    assert (config.max_query_tokens, config.max_passage_tokens) == (256, 2048)
    assert config.max_length == DEFAULT_MAX_LENGTH == 2304
    # Raw logits by default: a sigmoid saturates and cannot carry a threshold.
    assert config.score_mode == "raw_logit"


def test_config_rejects_a_window_that_cannot_hold_both_sides() -> None:
    with pytest.raises(ValueError):
        VietnameseRerankerConfig(max_length=512)
    with pytest.raises(ValueError):
        VietnameseRerankerConfig(score_mode="softmax")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VietnameseRerankerConfig(batch_size=0)


def test_scorer_truncates_query_and_passage_against_separate_budgets() -> None:
    calls: list[dict] = []

    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if len(args) == 1 and kwargs.get("add_special_tokens") is False:
                return {"input_ids": list(range(min(len(args[0].split()), 256)))}
            batch = len(args[0])
            return _FakeBatch(batch)

        def decode(self, ids, skip_special_tokens=True):
            return "clipped query"

    class _FakeBatch(dict):
        def __init__(self, batch: int) -> None:
            super().__init__(input_ids=_FakeTensor(batch))

    class _FakeTensor:
        def __init__(self, batch: int) -> None:
            self.batch = batch

        def to(self, device):
            return self

    class FakeModel:
        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            raise AssertionError("not reached in this test")

    scorer = VietnameseRerankerScorer(
        VietnameseRerankerConfig(device="cpu", local_files_only=True),
        model_loader=lambda *a, **k: FakeModel(),
        tokenizer_loader=lambda *a, **k: FakeTokenizer(),
    )
    scorer._forward = lambda encoded: [0.5] * encoded["input_ids"].batch  # type: ignore[method-assign]
    scores = scorer("một câu hỏi rất dài " * 40, ["đoạn 1", "đoạn 2", "đoạn 3"])
    assert scores == [0.5, 0.5, 0.5]
    pair_call = next(c for c in calls if c["kwargs"].get("truncation") == "only_second")
    assert pair_call["kwargs"]["max_length"] == 2304
    assert pair_call["args"][0][0] == "clipped query"


def test_scorer_returns_empty_for_no_texts() -> None:
    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": [1, 2, 3]}

        def decode(self, ids, skip_special_tokens=True):
            return "q"

    class FakeModel:
        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    scorer = VietnameseRerankerScorer(
        VietnameseRerankerConfig(device="cpu"),
        model_loader=lambda *a, **k: FakeModel(),
        tokenizer_loader=lambda *a, **k: FakeTokenizer(),
    )
    assert scorer("q", []) == ()
