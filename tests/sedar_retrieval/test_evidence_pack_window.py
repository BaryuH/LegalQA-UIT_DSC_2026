"""P1-P4 regression tests for the SEDAR evidence pack.

Each test below pins a defect that was silently costing the reader evidence:

P1  the candidate list was truncated to ``evidence_top_k`` *before*
    ``max_chunks_per_document`` and the character budget were applied, and
    whatever those dropped was never backfilled. ``pack_evidence`` only raises
    when *nothing* fits, so a four-slot pack quietly became a three- or
    two-block pack.
P2  ``pack_passage_retrieval_evidence`` never passed the document name down, so
    the header read ``Văn bản: <zip member>.txt`` even though every
    ``CanonicalPassage`` carries ``document_name``.
P3  ``reader_text`` — the field whose builder is documented as "Authoritative
    reader evidence" — never reached the reader; inference always used
    ``raw_text``.
P4  the retrieval views index ``article`` and ``clause`` levels together, so an
    article and its own clause competed for the same slots and burned the
    per-document cap.

The defaults must stay byte-identical to the historical pack, so the "before"
behaviour is asserted too: if a future change silently fixes P1 by default,
``test_default_config_still_loses_a_block`` fails and forces the decision to be
explicit.
"""

from __future__ import annotations

import pytest

from legal_rag.sedar_retrieval.corpus.schema import CanonicalPassage, SourceProvenance
from legal_rag.sedar_retrieval.evidence.passage_packer import (
    PassageEvidenceConfig,
    RankedPassageCandidate,
    dedup_candidates_by_article,
    pack_passage_retrieval_evidence,
)

_DOC_NAMES = {
    "A": "Nghị định 153/2020/NĐ-CP",
    "B": "Luật Doanh nghiệp 2020",
    "C": "Bộ luật Dân sự 2015",
}
_BODY = "Nội dung quy định chi tiết. " * 4
_HUGE_BUDGET = 100_000


def _passage(
    passage_id: str,
    document_id: str,
    level: str,
    article: int,
    clause: int | None = None,
    *,
    document_name: str | None = None,
) -> CanonicalPassage:
    name = document_name or _DOC_NAMES[document_id]
    return CanonicalPassage(
        passage_id=passage_id,
        document_id=document_id,
        article_id=f"{document_id}::art::{article}",
        clause_id=(f"{document_id}::art::{article}::cl::{clause}" if clause else None),
        retrieval_level=level,  # type: ignore[arg-type]
        document_name=name,
        article_number=str(article),
        article_title="Hợp đồng lao động",
        clause_number=str(clause) if clause else None,
        raw_text=_BODY,
        reader_text=f"Điều {article}. Hợp đồng lao động\n\n{_BODY}",
        retrieval_text=f"[DOCUMENT] {name}\n[ARTICLE] Điều {article}\n\n{_BODY}",
        source=SourceProvenance(
            source_path="selected-contexts.zip",
            source_member=f"{document_id}_12345.txt",
            document_id=document_id,
            content_hash="0" * 64,
        ),
    )


@pytest.fixture
def passages() -> dict[str, CanonicalPassage]:
    rows = [
        _passage("A::art76", "A", "article", 76),
        _passage("A::art76::cl1", "A", "clause", 76, 1),
        _passage("A::art76::cl2", "A", "clause", 76, 2),
        _passage("B::art10", "B", "article", 10),
        _passage("C::art20", "C", "article", 20),
    ]
    return {row.passage_id: row for row in rows}


@pytest.fixture
def candidates() -> list[RankedPassageCandidate]:
    """Three passages of one article ranked above two other documents."""

    order = ["A::art76", "A::art76::cl1", "A::art76::cl2", "B::art10", "C::art20"]
    return [
        RankedPassageCandidate(passage_id=pid, rank=rank, score=1.0 / rank)
        for rank, pid in enumerate(order, start=1)
    ]


# ── P1 ─────────────────────────────────────────────────────────────────────


def test_default_config_still_loses_a_block(passages, candidates) -> None:
    """The historical behaviour, pinned on purpose.

    Asking for four blocks yields three: the third same-document candidate is
    dropped by the per-document cap and nothing takes its place.
    """

    packed = pack_passage_retrieval_evidence(
        candidates,
        passages,
        config=PassageEvidenceConfig(evidence_top_k=4, max_total_chars=_HUGE_BUDGET),
    )
    assert len(packed.included_ids) == 3
    assert packed.dropped_reasons["A::art76::cl2"] == "max_chunks_per_document"
    assert packed.metadata["target_blocks"] == -1


def test_candidate_window_backfills_to_target(passages, candidates) -> None:
    packed = pack_passage_retrieval_evidence(
        candidates,
        passages,
        config=PassageEvidenceConfig(
            evidence_top_k=4, candidate_window=16, max_total_chars=_HUGE_BUDGET
        ),
    )
    assert len(packed.included_ids) == 4
    # Rank 5 is only reachable because the window is wider than the target.
    assert "C::art20" in packed.included_ids
    assert packed.metadata["target_blocks"] == 4
    assert packed.metadata["target_blocks_met"] is True


def test_candidate_window_below_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="candidate_window"):
        PassageEvidenceConfig(evidence_top_k=4, candidate_window=2)


# ── P2 ─────────────────────────────────────────────────────────────────────


def test_default_header_still_shows_the_zip_member(passages, candidates) -> None:
    """Pinned on purpose: P2 stays off by default.

    The frozen reader was fine-tuned on evidence rendered with the old header,
    so switching this on underneath the Phase 2 control run would make the
    control stop reproducing the champion. It has to be its own ablation arm.
    """

    packed = pack_passage_retrieval_evidence(
        candidates,
        passages,
        config=PassageEvidenceConfig(
            evidence_top_k=4, candidate_window=16, max_total_chars=_HUGE_BUDGET
        ),
    )
    name_lines = [
        line
        for line in packed.rendered_text.splitlines()
        if line.startswith("Văn bản:")
    ]
    assert name_lines
    assert all("Nghị định 153/2020/NĐ-CP" not in line for line in name_lines)
    assert any("A_12345.txt" in line for line in name_lines)
    assert packed.metadata["document_names_supplied"] is False


def test_header_shows_document_name_when_enabled(passages, candidates) -> None:
    packed = pack_passage_retrieval_evidence(
        candidates,
        passages,
        config=PassageEvidenceConfig(
            evidence_top_k=4,
            candidate_window=16,
            include_document_name=True,
            max_total_chars=_HUGE_BUDGET,
        ),
    )
    name_lines = [
        line
        for line in packed.rendered_text.splitlines()
        if line.startswith("Văn bản:")
    ]
    assert name_lines
    assert any("Nghị định 153/2020/NĐ-CP" in line for line in name_lines)
    assert all("12345.txt" not in line for line in name_lines)
    # The zip member is provenance and must stay on the Nguồn line.
    assert any(
        line.startswith("Nguồn:") and "A_12345.txt" in line
        for line in packed.rendered_text.splitlines()
    )
    assert packed.metadata["document_names_supplied"] is True


# ── P3 ─────────────────────────────────────────────────────────────────────


def test_body_source_reader_text_carries_article_heading(passages, candidates) -> None:
    heading = "Điều 76. Hợp đồng lao động"
    raw = pack_passage_retrieval_evidence(
        candidates,
        passages,
        config=PassageEvidenceConfig(evidence_top_k=1, max_total_chars=_HUGE_BUDGET),
    )
    reader = pack_passage_retrieval_evidence(
        candidates,
        passages,
        config=PassageEvidenceConfig(
            evidence_top_k=1,
            body_source="reader_text",
            max_total_chars=_HUGE_BUDGET,
        ),
    )
    assert heading not in raw.rendered_text
    assert heading in reader.rendered_text


def test_unknown_body_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="body_source"):
        PassageEvidenceConfig(body_source="summary")  # type: ignore[arg-type]


# ── P4 ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("article", ["A::art76", "B::art10", "C::art20"]),
        ("clause", ["A::art76::cl1", "B::art10", "C::art20"]),
        ("first", ["A::art76", "B::art10", "C::art20"]),
    ],
)
def test_dedup_by_article_modes(passages, candidates, mode, expected) -> None:
    kept = dedup_candidates_by_article(candidates, passages, mode=mode)
    assert [item.passage_id for item in kept] == expected


def test_dedup_off_is_identity(passages, candidates) -> None:
    assert dedup_candidates_by_article(candidates, passages, mode="off") == tuple(
        candidates
    )


def test_dedup_frees_slots_for_other_documents(passages, candidates) -> None:
    packed = pack_passage_retrieval_evidence(
        candidates,
        passages,
        config=PassageEvidenceConfig(
            evidence_top_k=4,
            candidate_window=16,
            dedup_article_mode="article",
            max_total_chars=_HUGE_BUDGET,
        ),
    )
    documents = {passages[pid].document_id for pid in packed.included_ids}
    assert len(packed.included_ids) == 3
    assert documents == {"A", "B", "C"}


def test_dedup_is_document_scoped(passages) -> None:
    """Two documents sharing 'Điều 76' must never collapse into one.

    This is the same trap that the legacy silver-label builder fell into by
    mapping article numbers globally.
    """

    corpus = dict(passages)
    corpus["Z::art76"] = _passage(
        "Z::art76", "Z", "article", 76, document_name="Thông tư 01/2021/TT-BTC"
    )
    trap = [
        RankedPassageCandidate(passage_id="A::art76", rank=1, score=1.0),
        RankedPassageCandidate(passage_id="Z::art76", rank=2, score=0.5),
    ]
    kept = dedup_candidates_by_article(trap, corpus, mode="article")
    assert [item.passage_id for item in kept] == ["A::art76", "Z::art76"]
