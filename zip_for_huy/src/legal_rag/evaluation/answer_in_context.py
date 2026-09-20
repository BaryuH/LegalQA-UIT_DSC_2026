"""Answer-in-context: did the evidence the reader received contain the answer?

## Why this metric, and why it is not retrieval recall

Retrieval recall answers "was the right article in the ranked list". It does not
answer "did the text needed to write the answer survive into the prompt", and
the two come apart. The one study that measures both reports **answer-in-context
correlating with F1 at +0.50 against retrieval recall's +0.31**, adding
**ΔR² = +0.17** over recall alone, and separating a **4.6x EM gap even on
questions where every gold document was retrieved**
([arXiv:2607.00725](https://arxiv.org/html/2607.00725v1)).

For this project that gap is not hypothetical. ``memory-bank/progress.md``
records article@4 = 0.7920 alongside a *"bottom quartile of starved packs"* and
*"65 UNDER_SPECIFIED cases ... actually out of content, not out of budget"*.
Those are answer-in-context failures that recall cannot see.

## The adaptation, and what calibrating it uncovered

The published metric is extractive: does the gold answer appear as a contiguous
span in the packed context. Task 2 answers are **generated prose** averaging
1,556 characters, so no contiguous match exists and the metric has to be adapted.
Two candidate adaptations were implemented and calibrated against an *oracle*
pack (the articles the answer actually cites) and a *random* pack, over the real
warmup queries.

**The first calibration said both metrics were dead.** Oracle 0.0005 against
random 0.0004 on 8-gram overlap; flat at every n from 2 to 8, and at n=2 the
random arm scored *higher*. Chasing that produced the finding that matters most
in this module:

**The v3 silver labels are diluted.** The first oracle was built from
``artifacts/sedar_retrieval/eval/warmup_silver_labels.jsonl``, and for query
101515 those labels list **six articles across three documents** -
``100125::art::12``, ``100139::art::12``, ``100325::art::12`` and the same three
at ``::art::9`` - which is the shape of a citation whose *document* was never
resolved, fanned out over every document in the subset that happens to have an
``Điều 12``. The citation resolver in ``corpus_v4.citations`` identifies exactly
one: ``236791::i0::art12`` (Thông tư 55/2021/TT-BCA). So the oracle was not an
oracle; it was mostly wrong documents. That dilution also inflates any recall
computed against those labels, because six ids count as a hit where one should.

**Rebuilt against a resolver oracle, both metrics separate completely** (379
scorable queries):

| | oracle | random | separation |
| --- | --- | --- | --- |
| ``citation_coverage`` | **0.9979** | 0.0000 | total |
| ``citation_hit`` rate | 0.9974 | 0.0000 | total |
| ``text_overlap`` (n=8) | **0.6035** | 0.0008 | 754x |

The random pack was the *longer* one (1,713 tokens against the oracle's) and
still scored far lower, so the length sensitivity that made the first
calibration unreadable does not dominate once the oracle is right.

Metrics reported per query:

``citation_coverage``
    of the articles the gold answer cites, what share are in the pack - matched
    by **resolved article identity**, not by the string ``Điều 76`` appearing
    somewhere. A pack quoting ``Điều 76`` of a different law is not grounding
    for this answer, and string matching cannot tell the difference. This is the
    headline.
``citation_hit``
    all of them, not just some. The binary an arm has to move.
``text_overlap``
    share of the answer's word n-grams recoverable from the pack. A real
    secondary signal after the recalibration above, but still length-sensitive:
    compare it only between packs of similar size, and never gate on it alone.

Unresolvable citations (a document outside the selected corpus) and
self-references (``Nghị định này``) are excluded from the denominator rather than
counted as misses - neither is a pack failure.

This is a **gold-reading, evaluation-only** module. Nothing it computes may enter
a retrieval query, an index, a prompt, or a submission artifact.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from legal_rag.sedar_retrieval.corpus_v4.citations import Citation

#: Maps one prose citation onto the article ids it may refer to.
CitationResolver = Callable[["Citation"], Sequence[str]]

ANSWER_IN_CONTEXT_SCHEMA_VERSION = "sedar-answer-in-context-v1"

#: n-gram width for ``text_overlap``. Calibrated: n=8 gives the cleanest
#: oracle/random separation (754x); smaller n lets common Vietnamese phrasing
#: leak in and made the first, mis-specified calibration unreadable.
DEFAULT_SHINGLE_SIZE = 8

_WORD_RE = re.compile(r"[0-9\w]+", re.UNICODE)

__all__ = [
    "ANSWER_IN_CONTEXT_SCHEMA_VERSION",
    "AnswerInContextResult",
    "AnswerInContextSummary",
    "CitationResolver",
    "answer_in_context",
    "normalize_for_overlap",
    "summarize_answer_in_context",
    "tokenize",
]


def normalize_for_overlap(text: str) -> str:
    """Fold case, accents-preserving NFC, and whitespace for comparison."""

    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()


def tokenize(text: str) -> list[str]:
    """Word tokens for shingling. No stemming: legal wording is quoted, not paraphrased."""

    return _WORD_RE.findall(normalize_for_overlap(text))


def _shingles(tokens: Sequence[str], size: int) -> set[tuple[str, ...]]:
    if size <= 0:
        raise ValueError("shingle size must be positive")
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


@dataclass(frozen=True, slots=True)
class AnswerInContextResult:
    """One query's answer-in-context measurement."""

    query_id: str
    cited_articles: tuple[str, ...]
    resolved_articles: tuple[str, ...]
    articles_in_pack: tuple[str, ...]
    unresolvable_citations: int
    self_reference_citations: int
    pack_units: int
    pack_tokens: int
    #: Secondary signal; length-sensitive. See the module docstring.
    text_overlap: float

    @property
    def citation_coverage(self) -> float:
        """Share of *resolvable* cited articles present in the pack.

        Unresolvable citations are excluded rather than counted as misses: a
        reference to a document outside the selected corpus, or a self-reference
        like "Nghị định này", is not a pack failure.
        """

        if not self.resolved_articles:
            return 1.0
        return len(self.articles_in_pack) / len(self.resolved_articles)

    @property
    def citation_hit(self) -> bool:
        return bool(self.resolved_articles) and len(self.articles_in_pack) == len(
            self.resolved_articles
        )

    @property
    def scorable(self) -> bool:
        """False when the answer cites nothing this corpus can resolve."""

        return bool(self.resolved_articles)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "cited_articles": list(self.cited_articles),
            "resolved_articles": list(self.resolved_articles),
            "articles_in_pack": list(self.articles_in_pack),
            "unresolvable_citations": self.unresolvable_citations,
            "self_reference_citations": self.self_reference_citations,
            "citation_coverage": round(self.citation_coverage, 4),
            "citation_hit": self.citation_hit,
            "scorable": self.scorable,
            "pack_units": self.pack_units,
            "pack_tokens": self.pack_tokens,
            "text_overlap": round(self.text_overlap, 4),
        }


def answer_in_context(
    query_id: str,
    answer: str,
    pack_article_ids: Iterable[str],
    *,
    resolve: "CitationResolver",
    pack_text: str = "",
    overlap_shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> AnswerInContextResult:
    """Did the pack contain the articles this answer cites?

    ``resolve`` maps one prose citation onto the article ids it could refer to
    (``corpus_v4.citations.CitationIndex.resolve`` composed with the exporter's
    article-id spelling). ``pack_article_ids`` is the pack's units mapped to
    their *article* identity, so an ``article_part`` counts for its parent.
    """

    from legal_rag.sedar_retrieval.corpus_v4.citations import extract_citations

    pack = {str(item) for item in pack_article_ids if item}
    citations = extract_citations(answer)
    self_refs = sum(1 for citation in citations if citation.self_reference)

    cited: list[str] = []
    resolved: list[str] = []
    in_pack: list[str] = []
    unresolvable = 0
    for citation in citations:
        if citation.self_reference:
            continue
        label = f"Điều {citation.article_number} — {citation.reference}"
        cited.append(label)
        targets = tuple(resolve(citation))
        if not targets:
            unresolvable += 1
            continue
        resolved.append(label)
        if pack & set(targets):
            in_pack.append(label)

    overlap = 0.0
    if pack_text:
        answer_shingles = _shingles(tokenize(answer), overlap_shingle_size)
        pack_shingles = _shingles(tokenize(pack_text), overlap_shingle_size)
        if answer_shingles:
            overlap = len(answer_shingles & pack_shingles) / len(answer_shingles)

    return AnswerInContextResult(
        query_id=query_id,
        cited_articles=tuple(cited),
        resolved_articles=tuple(resolved),
        articles_in_pack=tuple(in_pack),
        unresolvable_citations=unresolvable,
        self_reference_citations=self_refs,
        pack_units=len(pack),
        pack_tokens=len(tokenize(pack_text)) if pack_text else 0,
        text_overlap=overlap,
    )


@dataclass(frozen=True, slots=True)
class AnswerInContextSummary:
    """Corpus-level aggregate, shaped for the run artifact."""

    queries: int
    scorable_queries: int
    mean_citation_coverage: float
    citation_hit_rate: float
    starved_quartile_coverage: float
    mean_pack_units: float
    mean_pack_tokens: float
    unresolvable_citations: int
    self_reference_citations: int
    mean_text_overlap_diagnostic: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ANSWER_IN_CONTEXT_SCHEMA_VERSION,
            "queries": self.queries,
            "scorable_queries": self.scorable_queries,
            "mean_citation_coverage": round(self.mean_citation_coverage, 4),
            "citation_hit_rate": round(self.citation_hit_rate, 4),
            "starved_quartile_coverage": round(self.starved_quartile_coverage, 4),
            "mean_pack_units": round(self.mean_pack_units, 2),
            "mean_pack_tokens": round(self.mean_pack_tokens, 1),
            "unresolvable_citations": self.unresolvable_citations,
            "self_reference_citations": self.self_reference_citations,
            "text_overlap": {
                "mean": round(self.mean_text_overlap_diagnostic, 4),
                "note": (
                    "calibrated oracle 0.6035 vs random 0.0008 (754x), so this "
                    "is a real signal - but it is length-sensitive: compare it "
                    "only across packs of similar size, and do not gate on it "
                    "alone."
                ),
            },
        }


def summarize_answer_in_context(
    results: Iterable[AnswerInContextResult],
) -> AnswerInContextSummary:
    """Aggregate per-query results over the *scorable* subset.

    ``starved_quartile_coverage`` reports coverage restricted to the smallest
    quartile of packs by token count. That slice is where this project's own
    error analysis located the failure (*"bottom quartile is starved packs"*), so
    it is the number an adaptive-pack arm has to move.
    """

    rows = list(results)
    scorable = [row for row in rows if row.scorable]
    if not rows or not scorable:
        return AnswerInContextSummary(
            queries=len(rows),
            scorable_queries=0,
            mean_citation_coverage=0.0,
            citation_hit_rate=0.0,
            starved_quartile_coverage=0.0,
            mean_pack_units=mean(row.pack_units for row in rows) if rows else 0.0,
            mean_pack_tokens=mean(row.pack_tokens for row in rows) if rows else 0.0,
            unresolvable_citations=sum(row.unresolvable_citations for row in rows),
            self_reference_citations=sum(row.self_reference_citations for row in rows),
            mean_text_overlap_diagnostic=0.0,
        )

    by_pack_size = sorted(scorable, key=lambda row: row.pack_tokens)
    quartile = by_pack_size[: max(1, len(by_pack_size) // 4)]
    return AnswerInContextSummary(
        queries=len(rows),
        scorable_queries=len(scorable),
        mean_citation_coverage=mean(row.citation_coverage for row in scorable),
        citation_hit_rate=sum(1 for row in scorable if row.citation_hit) / len(scorable),
        starved_quartile_coverage=mean(row.citation_coverage for row in quartile),
        mean_pack_units=mean(row.pack_units for row in rows),
        mean_pack_tokens=mean(row.pack_tokens for row in rows),
        unresolvable_citations=sum(row.unresolvable_citations for row in rows),
        self_reference_citations=sum(row.self_reference_citations for row in rows),
        mean_text_overlap_diagnostic=mean(row.text_overlap for row in scorable),
    )
