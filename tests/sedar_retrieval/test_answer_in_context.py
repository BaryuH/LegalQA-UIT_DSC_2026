"""Acceptance tests for the answer-in-context metric (TASK 28).

The metric matches citations by **resolved article identity**, so these tests
inject a resolver rather than a model. See the module docstring for why the
text-overlap-only version was recalibrated after the first oracle turned out to
be built from diluted silver labels.
"""

from __future__ import annotations

import pytest

from legal_rag.sedar_retrieval.corpus_v4.citations import Citation, extract_citations
from legal_rag.evaluation.answer_in_context import (
    ANSWER_IN_CONTEXT_SCHEMA_VERSION,
    answer_in_context,
    normalize_for_overlap,
    summarize_answer_in_context,
    tokenize,
)

_ANSWER = (
    "Căn cứ Điều 76 Bộ luật Lao động 2019 quy định về ký kết thỏa ước lao động "
    "tập thể như sau: Thỏa ước lao động tập thể được ký kết bởi đại diện hợp "
    "pháp của các bên thương lượng."
)
_GROUNDED_TEXT = (
    "Điều 76. Lấy ý kiến và ký kết thỏa ước lao động tập thể\n"
    "4. Thỏa ước lao động tập thể được ký kết bởi đại diện hợp pháp của các bên "
    "thương lượng."
)

#: The article the answer above actually cites.
_TARGET = "bldd2019::art::76"


def _resolver(mapping: dict[tuple[str, str], tuple[str, ...]]):
    """Resolve on (article number, a substring of the reference)."""

    def resolve(citation: Citation):
        for (number, needle), targets in mapping.items():
            if citation.article_number == number and needle.lower() in citation.reference.lower():
                return targets
        return ()

    return resolve


_RESOLVE = _resolver({("76", "Bộ luật Lao động"): (_TARGET,)})


# --- the headline metric --------------------------------------------------


def test_a_pack_containing_the_cited_article_scores_one() -> None:
    result = answer_in_context(
        "q1", _ANSWER, [_TARGET], resolve=_RESOLVE, pack_text=_GROUNDED_TEXT
    )
    assert result.citation_coverage == pytest.approx(1.0)
    assert result.citation_hit is True
    assert result.scorable is True
    assert result.articles_in_pack == result.resolved_articles


def test_a_pack_of_the_wrong_documents_scores_zero() -> None:
    # This is the case the first calibration got wrong: three documents that all
    # have an "Điều 12" are not the cited article.
    result = answer_in_context(
        "q2",
        _ANSWER,
        ["other::art::76", "another::art::76"],
        resolve=_RESOLVE,
        pack_text="Điều 76. Một điều khác hoàn toàn.",
    )
    assert result.citation_coverage == pytest.approx(0.0)
    assert result.citation_hit is False
    # And note the string "Điều 76" IS in the pack text: identity matters,
    # substring matching would have called this a hit.
    assert "Điều 76" in "Điều 76. Một điều khác hoàn toàn."


def test_partial_coverage_when_one_of_two_cited_articles_is_packed() -> None:
    answer = (
        "Theo Điều 76 Bộ luật Lao động 2019 và Điều 77 Bộ luật Lao động 2019 "
        "thì thủ tục như sau."
    )
    resolve = _resolver(
        {
            ("76", "Bộ luật Lao động"): (_TARGET,),
            ("77", "Bộ luật Lao động"): ("bldd2019::art::77",),
        }
    )
    result = answer_in_context("q3", answer, [_TARGET], resolve=resolve)
    assert len(result.resolved_articles) == 2
    assert result.citation_coverage == pytest.approx(0.5)
    assert result.citation_hit is False


# --- what must NOT count as a miss ---------------------------------------


def test_self_references_are_excluded_not_counted_as_misses() -> None:
    answer = "Theo Điều 5 Nghị định này thì áp dụng như sau."
    result = answer_in_context("q4", answer, [], resolve=_RESOLVE)
    assert result.self_reference_citations == 1
    assert result.resolved_articles == ()
    assert result.scorable is False
    # Unscorable, not zero: a self-reference is not a pack failure.
    assert result.citation_coverage == pytest.approx(1.0)


def test_a_citation_outside_the_corpus_is_unresolvable_not_a_miss() -> None:
    answer = "Căn cứ Điều 9 Thông tư 11/2022/TT-BTNMT quy định như sau."
    result = answer_in_context("q5", answer, [_TARGET], resolve=_RESOLVE)
    assert result.unresolvable_citations == 1
    assert result.resolved_articles == ()
    assert result.scorable is False


def test_an_answer_citing_nothing_is_unscorable() -> None:
    # 9% of gold answers cite no Điều at all.
    result = answer_in_context(
        "q6", "Bạn nên liên hệ cơ quan có thẩm quyền.", [_TARGET], resolve=_RESOLVE
    )
    assert result.cited_articles == ()
    assert result.scorable is False
    assert result.citation_hit is False


def test_an_empty_pack_is_a_miss_when_the_citation_resolved() -> None:
    result = answer_in_context("q7", _ANSWER, [], resolve=_RESOLVE)
    assert result.scorable is True
    assert result.citation_coverage == pytest.approx(0.0)


# --- the text-overlap secondary signal -----------------------------------


def test_text_overlap_separates_a_quoted_pack_from_an_unrelated_one() -> None:
    grounded = answer_in_context(
        "q8", _ANSWER, [_TARGET], resolve=_RESOLVE, pack_text=_GROUNDED_TEXT
    )
    unrelated = answer_in_context(
        "q9",
        _ANSWER,
        [_TARGET],
        resolve=_RESOLVE,
        pack_text="Điều 12. Tiêu chuẩn kỹ thuật của thiết bị đo lường nhóm 2.",
    )
    assert grounded.text_overlap > 0.3
    assert unrelated.text_overlap == pytest.approx(0.0)


def test_overlap_is_zero_without_pack_text_and_does_not_break_coverage() -> None:
    result = answer_in_context("q10", _ANSWER, [_TARGET], resolve=_RESOLVE)
    assert result.text_overlap == pytest.approx(0.0)
    assert result.pack_tokens == 0
    assert result.citation_coverage == pytest.approx(1.0)


def test_shingle_width_controls_how_strict_the_overlap_is() -> None:
    answer = "thỏa ước lao động tập thể được ký kết bởi đại diện hợp pháp"
    shuffled = "hợp pháp đại diện bởi kết ký được thể tập động lao ước thỏa"
    loose = answer_in_context(
        "q11", answer, [], resolve=_RESOLVE, pack_text=shuffled, overlap_shingle_size=1
    )
    strict = answer_in_context(
        "q12", answer, [], resolve=_RESOLVE, pack_text=shuffled, overlap_shingle_size=8
    )
    assert loose.text_overlap == pytest.approx(1.0)
    assert strict.text_overlap == pytest.approx(0.0)


def test_rejects_a_non_positive_shingle_size() -> None:
    with pytest.raises(ValueError):
        answer_in_context(
            "q13", _ANSWER, [], resolve=_RESOLVE, pack_text="x", overlap_shingle_size=0
        )


def test_normalisation_folds_case_and_whitespace_but_keeps_diacritics() -> None:
    assert normalize_for_overlap("  Thỏa   ƯỚC\nlao động ") == "thỏa ước lao động"
    assert tokenize("Điều 76: ký-kết") == ["điều", "76", "ký", "kết"]


def test_the_citation_extractor_and_the_metric_agree_on_what_a_citation_is() -> None:
    citations = extract_citations(_ANSWER)
    assert [item.article_number for item in citations] == ["76"]
    result = answer_in_context("q14", _ANSWER, [_TARGET], resolve=_RESOLVE)
    assert len(result.cited_articles) == 1


# --- aggregation ---------------------------------------------------------


def test_summary_scores_only_the_scorable_subset() -> None:
    rows = [
        answer_in_context("a", _ANSWER, [_TARGET], resolve=_RESOLVE, pack_text=_GROUNDED_TEXT),
        answer_in_context("b", _ANSWER, [], resolve=_RESOLVE, pack_text="x"),
        answer_in_context("c", "Không trích dẫn gì.", [], resolve=_RESOLVE),
    ]
    summary = summarize_answer_in_context(rows)
    assert summary.queries == 3
    assert summary.scorable_queries == 2
    assert summary.citation_hit_rate == pytest.approx(0.5)
    assert summary.mean_citation_coverage == pytest.approx(0.5)


def test_summary_reports_the_starved_quartile_separately() -> None:
    # The project's own analysis located the failure in the smallest packs, so
    # the aggregate must expose that slice rather than average it away.
    big = [
        answer_in_context(
            f"big{i}", _ANSWER, [_TARGET], resolve=_RESOLVE, pack_text=_GROUNDED_TEXT * 8
        )
        for i in range(3)
    ]
    small = answer_in_context("small", _ANSWER, [], resolve=_RESOLVE, pack_text="Điều 12.")
    summary = summarize_answer_in_context([*big, small])
    assert summary.starved_quartile_coverage == pytest.approx(0.0)
    assert summary.mean_citation_coverage > summary.starved_quartile_coverage


def test_empty_and_all_unscorable_input_summarise_to_zeros_not_an_error() -> None:
    assert summarize_answer_in_context([]).queries == 0
    unscorable = [answer_in_context("a", "Không có trích dẫn.", [], resolve=_RESOLVE)]
    summary = summarize_answer_in_context(unscorable)
    assert summary.queries == 1
    assert summary.scorable_queries == 0
    assert summary.citation_hit_rate == pytest.approx(0.0)


def test_summary_serialises_with_the_length_sensitivity_note() -> None:
    payload = summarize_answer_in_context(
        [answer_in_context("a", _ANSWER, [_TARGET], resolve=_RESOLVE, pack_text=_GROUNDED_TEXT)]
    ).as_dict()
    assert payload["schema_version"] == ANSWER_IN_CONTEXT_SCHEMA_VERSION
    assert set(payload) >= {
        "mean_citation_coverage",
        "citation_hit_rate",
        "starved_quartile_coverage",
        "unresolvable_citations",
        "self_reference_citations",
        "text_overlap",
    }
    assert "length-sensitive" in payload["text_overlap"]["note"]
