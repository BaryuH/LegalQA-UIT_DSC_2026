"""R3 cross-encoder reranking: reordering logic, tested without a model.

The measured gap this stage targets: on clean-460 the candidate set holds the
right article for 97.5% of labeled queries at rank 500, while the top-4 handed
to the reader holds it for 78%. Recall is near its ceiling; precision at the
pack cutoff has never been attacked.
"""

from __future__ import annotations

import pytest

from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage, SourceProvenance
from legal_rag.sedar_retrieval.ranking.cross_encoder import (
    CrossEncoderError,
    passage_text,
    rerank_rankings,
)


def _passage(passage_id: str, body: str) -> CanonicalPassage:
    return CanonicalPassage(
        passage_id=passage_id,
        document_id="doc-a",
        retrieval_level="article",
        article_number="76",
        article_title="Hợp đồng lao động",
        raw_text=body,
        reader_text=f"Điều 76. Hợp đồng lao động\n\n{body}",
        retrieval_text=f"[DOCUMENT CONTEXT]\nTên văn bản: X\n\n[HIERARCHY]\n{body}",
        source=SourceProvenance(
            source_path="ctx.zip", document_id="doc-a", content_hash="0" * 64
        ),
    )


@pytest.fixture
def passages() -> dict[str, CanonicalPassage]:
    return {f"p{i}": _passage(f"p{i}", f"noi dung {i}") for i in range(1, 8)}


@pytest.fixture
def questions() -> dict[str, str]:
    return {"q1": "Điều 76 quy định gì?"}


def test_reorders_head_by_score(passages, questions) -> None:
    rankings = {"q1": ["p1", "p2", "p3"]}

    def score_fn(query: str, texts):
        # Reverse the incoming order: last candidate scores highest.
        return [float(i) for i in range(len(texts))]

    (row,) = rerank_rankings(
        rankings, questions, passages, score_fn=score_fn, top_k=3
    )
    assert row.ranked_ids == ("p3", "p2", "p1")
    assert [s["ce_rank"] for s in row.scores] == [1, 2, 3]
    assert row.scores[0]["passage_id"] == "p3"


def test_tail_beyond_top_k_keeps_original_order(passages, questions) -> None:
    """Candidates past top_k are appended, not dropped: deep recall is unchanged."""

    rankings = {"q1": ["p1", "p2", "p3", "p4", "p5"]}

    def score_fn(query: str, texts):
        return [float(i) for i in range(len(texts))]

    (row,) = rerank_rankings(
        rankings, questions, passages, score_fn=score_fn, top_k=2
    )
    assert row.ranked_ids == ("p2", "p1", "p3", "p4", "p5")
    assert row.reranked_count == 2
    assert row.tail_count == 3


def test_ties_break_deterministically(passages, questions) -> None:
    rankings = {"q1": ["p3", "p1", "p2"]}
    (row,) = rerank_rankings(
        rankings,
        questions,
        passages,
        score_fn=lambda q, texts: [1.0] * len(texts),
        top_k=3,
    )
    assert row.ranked_ids == ("p1", "p2", "p3")


def test_scorer_receives_reader_text_not_retrieval_text(passages, questions) -> None:
    """retrieval_text carries the R2a wrapper and repeats the document name.

    Feeding it would spend the encoder's 512-token budget on boilerplate that is
    identical for every passage of the document.
    """

    seen: list[str] = []

    def score_fn(query: str, texts):
        seen.extend(texts)
        return [1.0] * len(texts)

    rerank_rankings(
        {"q1": ["p1"]}, questions, passages, score_fn=score_fn, top_k=1
    )
    assert seen[0].startswith("Điều 76. Hợp đồng lao động")
    assert "[DOCUMENT CONTEXT]" not in seen[0]


def test_raw_text_source(passages, questions) -> None:
    seen: list[str] = []
    rerank_rankings(
        {"q1": ["p1"]},
        questions,
        passages,
        score_fn=lambda q, texts: (seen.extend(texts), [1.0] * len(texts))[1],
        top_k=1,
        text_source="raw_text",
    )
    assert seen[0] == "noi dung 1"


def test_passage_text_helper(passages) -> None:
    p = passages["p1"]
    assert passage_text(p) == p.reader_text
    assert passage_text(p, text_source="raw_text") == p.raw_text


# ── Fail closed ───────────────────────────────────────────────────────────


def test_missing_question_fails(passages) -> None:
    with pytest.raises(CrossEncoderError, match="No question text"):
        rerank_rankings(
            {"q9": ["p1"]}, {}, passages,
            score_fn=lambda q, t: [1.0] * len(t), top_k=1,
        )


def test_missing_passage_fails(questions, passages) -> None:
    with pytest.raises(CrossEncoderError, match="absent from the corpus view"):
        rerank_rankings(
            {"q1": ["nope"]}, questions, passages,
            score_fn=lambda q, t: [1.0] * len(t), top_k=1,
        )


def test_duplicate_candidate_fails(questions, passages) -> None:
    with pytest.raises(CrossEncoderError, match="Duplicate passage_id"):
        rerank_rankings(
            {"q1": ["p1", "p1"]}, questions, passages,
            score_fn=lambda q, t: [1.0] * len(t), top_k=2,
        )


def test_score_count_mismatch_fails(questions, passages) -> None:
    with pytest.raises(CrossEncoderError, match="scores for"):
        rerank_rankings(
            {"q1": ["p1", "p2"]}, questions, passages,
            score_fn=lambda q, t: [1.0], top_k=2,
        )


def test_non_positive_top_k_fails(questions, passages) -> None:
    with pytest.raises(CrossEncoderError, match="top_k must be positive"):
        rerank_rankings(
            {"q1": ["p1"]}, questions, passages,
            score_fn=lambda q, t: [1.0], top_k=0,
        )
