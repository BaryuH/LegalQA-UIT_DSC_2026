"""Export corpus v4 units into the v3 ``CanonicalPassage`` contract (TASK 27).

## Why this exists

Corpus v4 writes ``units.jsonl``. Every index builder, retrieval runner and
evaluator in the repo reads a passage view through
``retrieval.passage_adapter.load_passages_jsonl``, which validates each row as a
``CanonicalPassage`` - a pydantic model with ``extra="forbid"``. A v4 unit
carries ``unit_id``, ``breadcrumb``, ``role`` and ``parent_unit_id``, none of
which that model knows, so a v4 build cannot be indexed at all without this
translation. It was the one thing blocking every downstream stage.

## The two decisions that make v4 immediately scoreable

**Level mapping.** ``CanonicalPassage.retrieval_level`` is the v3 vocabulary
(``document|chapter|section|article|clause|point``). v4 levels map on:

| v4 level | v3 level | why |
| --- | --- | --- |
| ``article`` | ``article`` | same thing |
| ``article_part`` | ``clause`` | it *is* a merged clause run, and mapping it to ``clause`` is what makes ``eval_retrieval.py`` treat it as a sub-article unit that rolls up to its parent |
| ``block`` | ``article`` | a top-level retrievable unit in a document that has no ``Điều``; it has no parent to roll up to, so it must be its own scoring unit |
| ``preamble`` | dropped | ``indexed=False`` in v4; it is provenance, not evidence |

**Label-compatible article ids, scoped only where they must be.** The existing
silver labels use ``{document_id}::art::{article_number}`` (verified against
``artifacts/sedar_retrieval/eval/warmup_silver_labels.jsonl``). v4 scopes article
numbers by instrument, because 2,004 documents of this corpus carry the same
``Điều`` number twice - an annexed ``Quy chế`` restarts numbering - and a shared
id would silently merge two different articles.

Scoping *every* annex article would break the labels for no gain. Measured: a
naive "annexes always get ``::i{n}::``" rule scored 85.84% label coverage, and
every one of the 16 misses was a document like 100686 (Nghị định 99/2003 with the
Quy chế Khu công nghệ cao), whose ``Điều 25``-``Điều 38`` exist **only** in the
annex and are therefore not ambiguous at all.

So the rule here is: **an article keeps the flat ``{doc}::art::{n}`` id unless
that number actually occurs more than once in the document.** Only genuinely
colliding numbers get ``{doc}::i{instrument}::art::{n}``. That needs one cheap
pre-pass over the units to count numbers per document, which
:func:`build_article_id_map` does.

``article_id`` is also what ``eval_retrieval.py``'s ``_load_level_maps`` reads to
build its article roll-up, so an ``article_part`` inherits its parent article's id
and article-level recall works with **no change to the evaluator**.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

V4_TO_V3_LEVEL: dict[str, str] = {
    "article": "article",
    "article_part": "clause",
    "block": "article",
}

#: Units at these v4 levels are never exported: v4 marks them ``indexed=False``.
SKIPPED_LEVELS = frozenset({"preamble"})

__all__ = [
    "SKIPPED_LEVELS",
    "V4_TO_V3_LEVEL",
    "ExportCounters",
    "article_id_for",
    "build_article_id_map",
    "export_units",
    "unit_to_passage_row",
]


class ExportCounters:
    """Mutable tally so the CLI can report what the mapping did."""

    def __init__(self) -> None:
        self.read = 0
        self.exported = 0
        self.skipped_level: dict[str, int] = {}
        self.skipped_not_indexed = 0
        self.by_v3_level: dict[str, int] = {}
        self.unmapped_level: dict[str, int] = {}
        self.duplicate_passage_ids = 0
        self.annex_articles = 0
        self.missing_article_number = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "units_read": self.read,
            "passages_exported": self.exported,
            "skipped_by_level": dict(self.skipped_level),
            "skipped_not_indexed": self.skipped_not_indexed,
            "exported_by_v3_level": dict(self.by_v3_level),
            "unmapped_v4_levels": dict(self.unmapped_level),
            "duplicate_passage_ids": self.duplicate_passage_ids,
            "annex_scoped_article_ids": self.annex_articles,
            "article_units_without_number": self.missing_article_number,
        }


def build_article_id_map(units: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count how many instruments use each ``(document, article number)``.

    One cheap streaming pre-pass. The result tells :func:`article_id_for`
    whether a number is ambiguous inside its document, so only real collisions
    pay the instrument-scoped id that the existing labels do not know about.
    """

    counts: Counter[str] = Counter()
    seen: set[tuple[str, int, str]] = set()
    for unit in units:
        if str(unit.get("level") or "") != "article":
            continue
        document_id = str(unit.get("document_id") or "").strip()
        number = unit.get("article_number")
        if not document_id or not number:
            continue
        instrument = int(unit.get("instrument_index") or 0)
        key = (document_id, instrument, str(number))
        if key in seen:
            continue
        seen.add(key)
        counts[f"{document_id}::{number}"] += 1
    return dict(counts)


def article_id_for(
    unit: Mapping[str, Any],
    *,
    article_id_map: Mapping[str, int] | None = None,
) -> str | None:
    """Article identity used for the evaluator's article-level roll-up.

    Flat ``{doc}::art::{n}`` unless the number genuinely collides inside the
    document, in which case ``{doc}::i{instrument}::art::{n}``. Without an
    ``article_id_map`` the caller has not done the pre-pass, so every annex
    article is scoped - correct but label-hostile; the CLI always passes one.
    """

    document_id = str(unit.get("document_id") or "").strip()
    number = unit.get("article_number")
    if not document_id or not number:
        return None
    instrument = int(unit.get("instrument_index") or 0)
    flat = f"{document_id}::art::{number}"
    if instrument == 0:
        return flat
    if article_id_map is not None:
        if article_id_map.get(f"{document_id}::{number}", 0) <= 1:
            # The number exists only in this annex: not ambiguous, so keep the
            # spelling the labels already use.
            return flat
    return f"{document_id}::i{instrument}::art::{number}"


def unit_to_passage_row(
    unit: Mapping[str, Any],
    *,
    article_id_map: Mapping[str, int] | None = None,
) -> dict[str, Any] | None:
    """Translate one v4 unit into a ``CanonicalPassage``-shaped dict.

    Returns ``None`` when the unit must not be exported (a ``preamble``, or
    anything v4 marked ``indexed=False``). Raises nothing: an unmapped level is
    reported by the caller rather than aborting a 300k-row export.
    """

    level = str(unit.get("level") or "")
    if level in SKIPPED_LEVELS:
        return None
    v3_level = V4_TO_V3_LEVEL.get(level)
    if v3_level is None:
        return None

    unit_id = str(unit.get("unit_id") or "").strip()
    document_id = str(unit.get("document_id") or "").strip()
    if not unit_id or not document_id:
        return None

    reader_text = str(unit.get("reader_text") or "")
    breadcrumb = str(unit.get("breadcrumb") or "")
    retrieval_text = str(unit.get("retrieval_text") or "")
    if not retrieval_text:
        # The compact writer omits retrieval_text because it is derivable.
        retrieval_text = f"{breadcrumb}\n{reader_text}".strip() or reader_text

    article_id = article_id_for(unit, article_id_map=article_id_map)
    parent_unit_id = unit.get("parent_unit_id")
    clause_id: str | None = None
    if v3_level == "clause":
        # A merged clause run's own identity. Its *article* identity comes from
        # article_id above, which is what the roll-up uses.
        clause_id = unit_id

    return {
        "passage_id": unit_id,
        "document_id": document_id,
        "article_id": article_id,
        "clause_id": clause_id,
        "point_id": None,
        "retrieval_level": v3_level,
        "document_name": unit.get("citation") or None,
        "article_number": unit.get("article_number"),
        "article_title": unit.get("article_title"),
        "clause_number": unit.get("child_label"),
        "point_label": None,
        "status": "unknown",
        "raw_text": reader_text,
        "reader_text": reader_text,
        "retrieval_text": retrieval_text,
        "summary": None,
        "references": [],
        "source": {
            "source_path": str(unit.get("source_path") or ""),
            "source_member": unit.get("source_member"),
            "document_id": document_id,
            "content_hash": str(unit.get("content_hash") or ""),
            "link": unit.get("link"),
        },
        "parent_id": str(parent_unit_id) if parent_unit_id else None,
        "parse_status": "ok",
    }


def export_units(
    units: Iterable[Mapping[str, Any]],
    *,
    counters: ExportCounters | None = None,
    article_id_map: Mapping[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream v4 units into passage rows, tallying what happened.

    Streaming matters: a full v4 build is ~300k units and ~1.5 GB of JSONL, so
    nothing here materialises the corpus in memory.
    """

    tally = counters if counters is not None else ExportCounters()
    seen: set[str] = set()
    for unit in units:
        tally.read += 1
        level = str(unit.get("level") or "")
        if not unit.get("indexed", True):
            tally.skipped_not_indexed += 1
            continue
        if level in SKIPPED_LEVELS:
            tally.skipped_level[level] = tally.skipped_level.get(level, 0) + 1
            continue
        if level not in V4_TO_V3_LEVEL:
            tally.unmapped_level[level] = tally.unmapped_level.get(level, 0) + 1
            continue
        row = unit_to_passage_row(unit, article_id_map=article_id_map)
        if row is None:
            tally.skipped_level[level] = tally.skipped_level.get(level, 0) + 1
            continue
        passage_id = row["passage_id"]
        if passage_id in seen:
            tally.duplicate_passage_ids += 1
            continue
        seen.add(passage_id)
        if row["retrieval_level"] == "article" and not row["article_number"]:
            tally.missing_article_number += 1
        if row["article_id"] and "::i" in row["article_id"]:
            tally.annex_articles += 1
        tally.exported += 1
        tally.by_v3_level[row["retrieval_level"]] = (
            tally.by_v3_level.get(row["retrieval_level"], 0) + 1
        )
        yield row
