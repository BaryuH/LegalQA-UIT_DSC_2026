"""Retrieval-unit construction for SEDAR corpus v4.

Design rules, each tied to a measurement on this corpus or to published
evidence:

* **One indexed unit per ``Điều``.** Article-level retrieval is the unit used by
  every published Vietnamese statute-retrieval system, and the ``Điều`` is the
  smallest span that is self-contained: a ``Khoản`` inherits its subject and
  scope from the article heading.
* **Children only for long articles.** 14.0% of articles exceed 512 words, which
  is where a single dense vector loses fidelity. Those are split at ``Khoản``
  boundaries into *merged* runs; every child keeps ``parent_unit_id`` so ranking
  can collapse children back onto one article.
* **No unit below the micro floor.** 33.1% of v3 clause units are under 120
  characters. Such fragments outrank full articles in both BM25 and dense
  retrieval and then starve the evidence pack. v4 never emits one, except when a
  whole article is genuinely that short, in which case it is complete.
* **Deterministic contextual header.** Every unit's ``retrieval_text`` opens with
  its citation and ancestry. This is the cheap, LLM-free form of contextual
  retrieval, and it also lengthens short units.
* **Role tagging, not deletion.** Enforcement and effect articles are 6.7% of
  articles but 21.4% of article characters. They are kept for recall and tagged
  so the pack builder can deprioritise them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from legal_rag.schemas import LegalDocument

from .metadata import DocumentMeta, citation_aliases, extract_document_meta
from .normalize import normalize_source_text, strip_distribution_block
from .schema import DocumentRecord, RetrievalUnit, UnitRole
from .segment import (
    Article,
    Instrument,
    parse_articles,
    segment_blocks,
    split_article_children,
    split_instruments,
)

__all__ = ["BuildOptions", "build_document"]

_EFFECT_RE = re.compile(
    r"(hiệu lực (thi hành|kể từ)|có hiệu lực|hết hiệu lực|điều khoản chuyển tiếp)",
    re.IGNORECASE,
)
_ENFORCE_RE = re.compile(
    r"(chịu trách nhiệm thi hành|tổ chức thực hiện|trách nhiệm thi hành)",
    re.IGNORECASE,
)
_PROMULGATE_RE = re.compile(
    r"(ban hành kèm theo|phê duyệt (đề án|chương trình|chiến lược|kế hoạch|quy hoạch))",
    re.IGNORECASE,
)
_FORM_RE = re.compile(r"^(mẫu số|biểu mẫu|phụ lục|biểu số)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """Tunable thresholds for the v4 unit builder."""

    #: An article longer than this is also emitted as merged clause children.
    long_article_chars: int = 2400
    #: No child unit may be shorter than this.
    min_child_chars: int = 500
    #: Hard cap on one child unit.
    max_child_chars: int = 2400
    #: Standalone units below this are folded into a neighbour where possible.
    micro_unit_chars: int = 120
    #: An article longer than this is never packed whole; its children are used.
    pack_max_chars: int = 6000
    #: Emit article children at all.
    emit_children: bool = True
    #: Emit the ``Căn cứ`` preamble as a non-indexed unit.
    keep_preamble: bool = True
    #: Article-less instruments are segmented into blocks of this target size.
    block_target_chars: int = 1200
    block_max_chars: int = 2600


def _hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _classify_article(article: Article) -> UnitRole:
    head = f"{article.title or ''} {article.body[:400]}"
    if _PROMULGATE_RE.search(head):
        return "annex_container" if len(article.body) > 4000 else "promulgation"
    if _ENFORCE_RE.search(head):
        return "enforcement"
    if _EFFECT_RE.search(head):
        return "effect"
    return "substantive"


def _breadcrumb(
    citation: str,
    instrument: Instrument,
    article: Article | None,
    child_label: str | None,
    block_label: str | None,
) -> str:
    parts = [citation]
    if instrument.kind == "annex" and instrument.title:
        parts.append(instrument.title)
    if article is not None:
        if article.part:
            parts.append(article.part)
        if article.chapter_number:
            chapter = f"Chương {article.chapter_number}"
            if article.chapter_title:
                chapter = f"{chapter}. {article.chapter_title}"
            parts.append(chapter)
        if article.section_number:
            section = f"Mục {article.section_number}"
            if article.section_title:
                section = f"{section}. {article.section_title}"
            parts.append(section)
        if article.subsection_title:
            parts.append(article.subsection_title)
        parts.append(article.heading)
    if block_label:
        parts.append(block_label)
    if child_label:
        parts.append(child_label)
    return " > ".join(part.strip() for part in parts if part and part.strip())


def _retrieval_text(breadcrumb: str, body: str) -> str:
    return f"{breadcrumb}\n{body}".strip()


def _unit_id(document_id: str, instrument: int, *parts: str) -> str:
    tail = "::".join(str(part) for part in parts if str(part))
    return f"{document_id}::i{instrument}::{tail}"


def build_document(
    document: LegalDocument,
    options: BuildOptions | None = None,
) -> tuple[DocumentRecord, tuple[RetrievalUnit, ...]]:
    """Normalise, segment and render one corpus document into v4 units."""

    opts = options or BuildOptions()
    normalized = normalize_source_text(document.passage)
    normalized = strip_distribution_block(normalized)
    lines = normalized.split("\n")
    meta = extract_document_meta(document.id, document.name, document.link, normalized)
    instruments = split_instruments(lines)

    units: list[RetrievalUnit] = []
    covered = 0
    profiles: set[str] = set()

    for instrument in instruments:
        articles = parse_articles(instrument)
        if articles:
            profiles.add("article")
            covered += _emit_articles(
                document, meta, instrument, articles, units, opts
            )
        else:
            profiles.add("block")
            covered += _emit_blocks(document, meta, instrument, units, opts)

    if opts.keep_preamble:
        preamble = _preamble_text(lines)
        if preamble:
            units.append(
                _make_unit(
                    document,
                    meta,
                    unit_id=_unit_id(document.id, 0, "preamble"),
                    level="preamble",
                    role="promulgation",
                    instrument=instruments[0],
                    breadcrumb=f"{meta.citation} > phần căn cứ",
                    reader_text=preamble,
                    indexed=False,
                    packable=False,
                )
            )

    units = _ensure_unique_ids(units)

    record = DocumentRecord(
        document_id=document.id,
        source_path=document.source_path,
        source_member=document.source_member,
        content_hash=document.content_hash,
        link=document.link,
        raw_name=document.name,
        doc_type=meta.doc_type,
        doc_number=meta.doc_number,
        title=meta.title,
        issuer=meta.issuer,
        issued_date=meta.issued_date,
        year=meta.year,
        citation=meta.citation,
        citation_aliases=citation_aliases(meta),
        metadata_source=meta.metadata_source,
        instrument_count=len(instruments),
        source_chars=len(normalized),
        covered_chars=covered,
        unit_count=len(units),
        parse_profile="+".join(sorted(profiles)) if profiles else "empty",
    )
    return record, tuple(units)


def _ensure_unique_ids(units: list[RetrievalUnit]) -> list[RetrievalUnit]:
    """Disambiguate repeated ids and record why they repeated.

    Amending instruments quote the article they replace ("Điều 5 được sửa đổi
    như sau: Điều 5. ..."), so the same article number can legitimately occur
    twice inside one instrument. The occurrence suffix keeps ids unique without
    silently dropping either copy, and the warning makes the collision visible
    to the citation resolver.
    """

    seen: dict[str, int] = {}
    resolved: list[RetrievalUnit] = []
    remap: dict[str, str] = {}
    for unit in units:
        count = seen.get(unit.unit_id, 0)
        seen[unit.unit_id] = count + 1
        if count == 0:
            resolved.append(unit)
            continue
        new_id = f"{unit.unit_id}~{count + 1}"
        remap[unit.unit_id] = new_id
        resolved.append(
            unit.model_copy(
                update={
                    "unit_id": new_id,
                    "warnings": (*unit.warnings, "duplicate_article_number"),
                }
            )
        )
    if remap:
        resolved = [
            unit.model_copy(update={"parent_unit_id": remap[unit.parent_unit_id]})
            if unit.parent_unit_id in remap
            else unit
            for unit in resolved
        ]
    return resolved


def _emit_articles(
    document: LegalDocument,
    meta: DocumentMeta,
    instrument: Instrument,
    articles: list[Article],
    units: list[RetrievalUnit],
    opts: BuildOptions,
) -> int:
    covered = 0
    for article in articles:
        role = _classify_article(article)
        text = article.text
        covered += len(text)
        breadcrumb = _breadcrumb(meta.citation, instrument, article, None, None)
        article_id = _unit_id(
            document.id, instrument.index, f"art{_slug(article.number)}"
        )
        children: list = []
        if opts.emit_children and len(text) > opts.long_article_chars:
            children = split_article_children(
                article,
                min_child_chars=opts.min_child_chars,
                max_child_chars=opts.max_child_chars,
            )
            if len(children) < 2:
                children = []
        # Exactly one granularity per article reaches the index: a short article
        # is indexed whole, a long one is indexed through its merged children and
        # keeps the article as the scoring/packing parent. Nothing is nested.
        units.append(
            _make_unit(
                document,
                meta,
                unit_id=article_id,
                level="article",
                role=role,
                instrument=instrument,
                breadcrumb=breadcrumb,
                reader_text=text,
                article=article,
                indexed=not children,
                packable=role != "enforcement" and len(text) <= opts.pack_max_chars,
            )
        )
        for position, child in enumerate(children):
            child_breadcrumb = _breadcrumb(
                meta.citation, instrument, article, child.label, None
            )
            units.append(
                _make_unit(
                    document,
                    meta,
                    unit_id=f"{article_id}::p{position}",
                    level="article_part",
                    role=role,
                    instrument=instrument,
                    breadcrumb=child_breadcrumb,
                    reader_text=f"{article.heading}\n{child.body}".strip(),
                    article=article,
                    parent_unit_id=article_id,
                    child_label=child.label,
                    child_index=position,
                    child_count=len(children),
                    indexed=True,
                    packable=len(text) > opts.pack_max_chars,
                )
            )
    return covered


def _emit_blocks(
    document: LegalDocument,
    meta: DocumentMeta,
    instrument: Instrument,
    units: list[RetrievalUnit],
    opts: BuildOptions,
) -> int:
    blocks = segment_blocks(
        instrument.lines,
        target_chars=opts.block_target_chars,
        max_chars=opts.block_max_chars,
        min_chars=max(opts.micro_unit_chars * 2, 320),
    )
    covered = 0
    for position, block in enumerate(blocks):
        covered += len(block.body)
        role: UnitRole = "form" if _FORM_RE.match(block.label or "") else "substantive"
        breadcrumb = _breadcrumb(meta.citation, instrument, None, None, block.label)
        units.append(
            _make_unit(
                document,
                meta,
                unit_id=_unit_id(document.id, instrument.index, f"blk{position}"),
                level="block",
                role=role,
                instrument=instrument,
                breadcrumb=breadcrumb,
                reader_text=block.body,
                child_label=block.label,
                child_index=position,
                child_count=len(blocks),
            )
        )
    return covered


def _preamble_text(lines: list[str], *, limit: int = 12) -> str | None:
    collected = [line.strip() for line in lines if line.strip().lower().startswith("căn cứ")]
    if not collected:
        return None
    return "\n".join(collected[:limit])


def _slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-zĐđ]+", "", value) or "0"


def _make_unit(
    document: LegalDocument,
    meta: DocumentMeta,
    *,
    unit_id: str,
    level: str,
    role: UnitRole,
    instrument: Instrument,
    breadcrumb: str,
    reader_text: str,
    article: Article | None = None,
    parent_unit_id: str | None = None,
    child_label: str | None = None,
    child_index: int | None = None,
    child_count: int | None = None,
    indexed: bool = True,
    packable: bool = True,
) -> RetrievalUnit:
    retrieval_text = _retrieval_text(breadcrumb, reader_text)
    return RetrievalUnit(
        unit_id=unit_id,
        document_id=document.id,
        parent_unit_id=parent_unit_id,
        level=level,  # type: ignore[arg-type]
        role=role,
        instrument_index=instrument.index,
        instrument_kind=instrument.kind,  # type: ignore[arg-type]
        instrument_title=instrument.title,
        part_label=article.part if article else None,
        chapter_number=article.chapter_number if article else None,
        chapter_title=article.chapter_title if article else None,
        section_number=article.section_number if article else None,
        section_title=article.section_title if article else None,
        subsection_title=article.subsection_title if article else None,
        article_number=article.number if article else None,
        article_title=article.title if article else None,
        child_label=child_label,
        child_index=child_index,
        child_count=child_count,
        citation=meta.citation,
        breadcrumb=breadcrumb,
        reader_text=reader_text,
        retrieval_text=retrieval_text,
        char_count=len(reader_text),
        word_count=len(reader_text.split()),
        indexed=indexed,
        packable=packable,
        content_hash=_hash(reader_text),
        source_path=document.source_path,
        source_member=document.source_member,
        link=document.link,
    )
