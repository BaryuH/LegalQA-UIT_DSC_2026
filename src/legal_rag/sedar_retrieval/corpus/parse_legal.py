"""Deterministic Vietnamese legal hierarchy parser.

Preserves original legal text. Unknown structures are retained with
``parse_status`` rather than dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from legal_rag.schemas import LegalDocument

from .schema import CanonicalNode, ParseStatus, SourceProvenance

_CHAPTER_RE = re.compile(
    r"^\s*chương\s+(?P<label>[IVXLC\d]+)(?:\s*[-–.:]\s*(?P<title>.*))?\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_SECTION_RE = re.compile(
    r"^\s*mục\s+(?P<label>\d+[A-Za-z]?)(?:\s*[-–.:]\s*(?P<title>.*))?\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_ARTICLE_RE = re.compile(
    r"^\s*điều\s+(?P<number>\d+[A-Za-z]?)"
    r"(?:\s*[.:]\s*(?P<title>.*))?\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_ARTICLE_ONLY_RE = re.compile(r"^\s*điều\s*$", flags=re.IGNORECASE | re.UNICODE)
_ARTICLE_NUMBER_RE = re.compile(
    r"^\s*(?P<number>\d+[A-Za-z]?)(?:\s*[.:]\s*(?P<title>.*))?\s*$",
    flags=re.UNICODE,
)
_CLAUSE_RE = re.compile(
    r"^\s*(?P<number>\d+)\s*[.)](?P<body>.*)$",
    flags=re.UNICODE,
)
_POINT_RE = re.compile(
    r"^\s*(?P<label>[A-Za-zĐđ])\s*[.)](?P<body>.*)$",
    flags=re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class _Line:
    start: int
    end: int
    text: str


@dataclass(slots=True)
class _ParseState:
    chapter_id: str | None = None
    chapter_title: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    article_id: str | None = None
    article_number: str | None = None
    article_title: str | None = None


def _iter_lines(text: str) -> tuple[_Line, ...]:
    lines: list[_Line] = []
    for match in re.finditer(
        r"[^\r\n\u2028\u2029]*(?:\r\n|\r|\n|\u2028|\u2029|$)",
        text,
    ):
        if match.start() == len(text) and not match.group():
            break
        raw = match.group()
        lines.append(
            _Line(
                start=match.start(),
                end=match.end(),
                text=raw.rstrip("\r\n\u2028\u2029"),
            )
        )
    return tuple(lines)


def _source_for(document: LegalDocument) -> SourceProvenance:
    return SourceProvenance(
        source_path=document.source_path,
        source_member=document.source_member,
        document_id=document.id,
        content_hash=document.content_hash,
        link=document.link,
    )


def _node_id(document_id: str, *parts: str) -> str:
    safe = [document_id]
    for part in parts:
        cleaned = part.strip().replace(" ", "_")
        if cleaned:
            safe.append(cleaned)
    return "::".join(safe)


def _heading_kind(text: str) -> str | None:
    if _CHAPTER_RE.fullmatch(text):
        return "chapter"
    if _SECTION_RE.fullmatch(text):
        return "section"
    if _ARTICLE_RE.fullmatch(text) or _ARTICLE_ONLY_RE.fullmatch(text):
        return "article"
    if _CLAUSE_RE.fullmatch(text):
        return "clause"
    if _POINT_RE.fullmatch(text):
        return "point"
    return None


def _span_until(
    lines: tuple[_Line, ...],
    start_index: int,
    stop_kinds: tuple[str, ...],
) -> tuple[int, int]:
    start = lines[start_index].start
    end = lines[-1].end if lines else start
    for index in range(start_index + 1, len(lines)):
        kind = _heading_kind(lines[index].text)
        if kind in stop_kinds:
            return start, lines[index].start
    return start, end


def _unique_id(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    index = 2
    while f"{base}__{index}" in used:
        index += 1
    value = f"{base}__{index}"
    used.add(value)
    return value


def parse_legal_document(document: LegalDocument) -> tuple[CanonicalNode, ...]:
    """Parse one document into hierarchical nodes without inventing structure."""

    text = document.passage
    lines = _iter_lines(text)
    source = _source_for(document)
    nodes: list[CanonicalNode] = []
    state = _ParseState()
    warnings: list[str] = []
    covered = bytearray(len(text))
    used_ids: set[str] = set()

    nodes.append(
        CanonicalNode(
            node_id=_unique_id(_node_id(document.id, "doc"), used_ids),
            parent_id=None,
            level="document",
            document_id=document.id,
            document_name=document.name,
            raw_text=text if text.strip() else document.name,
            parse_status="ok" if text.strip() else "partial",
            source=source,
        )
    )

    index = 0
    while index < len(lines):
        line = lines[index]

        chapter = _CHAPTER_RE.fullmatch(line.text)
        if chapter is not None:
            label = chapter.group("label")
            chapter_title = (chapter.group("title") or "").strip() or None
            node_id = _unique_id(_node_id(document.id, "chuong", label), used_ids)
            state.chapter_id = node_id
            state.chapter_title = chapter_title
            state.section_id = None
            state.section_title = None
            state.article_id = None
            state.article_number = None
            state.article_title = None
            # Chapter node stores heading line only; body is covered by children.
            heading_end = line.end
            nodes.append(
                CanonicalNode(
                    node_id=node_id,
                    parent_id=nodes[0].node_id,
                    level="chapter",
                    document_id=document.id,
                    document_name=document.name,
                    chapter_id=node_id,
                    chapter_title=chapter_title,
                    raw_text=text[line.start : heading_end] or line.text,
                    parse_status="ok",
                    source=source,
                )
            )
            covered[line.start : heading_end] = b"\x01" * (heading_end - line.start)
            index += 1
            continue

        section = _SECTION_RE.fullmatch(line.text)
        if section is not None:
            label = section.group("label")
            section_title = (section.group("title") or "").strip() or None
            node_id = _unique_id(_node_id(document.id, "muc", label), used_ids)
            parent = state.chapter_id or nodes[0].node_id
            state.section_id = node_id
            state.section_title = section_title
            state.article_id = None
            state.article_number = None
            state.article_title = None
            heading_end = line.end
            nodes.append(
                CanonicalNode(
                    node_id=node_id,
                    parent_id=parent,
                    level="section",
                    document_id=document.id,
                    document_name=document.name,
                    chapter_id=state.chapter_id,
                    chapter_title=state.chapter_title,
                    section_id=node_id,
                    section_title=section_title,
                    raw_text=text[line.start : heading_end] or line.text,
                    parse_status="ok",
                    source=source,
                )
            )
            covered[line.start : heading_end] = b"\x01" * (heading_end - line.start)
            index += 1
            continue

        number: str | None = None
        article_title: str | None = None
        heading_end = line.end
        advance = 1
        article_match = _ARTICLE_RE.fullmatch(line.text)
        if article_match is not None:
            number = article_match.group("number")
            article_title = (article_match.group("title") or "").strip() or None
        elif _ARTICLE_ONLY_RE.fullmatch(line.text) and index + 1 < len(lines):
            number_match = _ARTICLE_NUMBER_RE.fullmatch(lines[index + 1].text)
            if number_match is not None:
                number = number_match.group("number")
                article_title = (number_match.group("title") or "").strip() or None
                heading_end = lines[index + 1].end
                advance = 2

        if number is not None:
            article_id = _unique_id(_node_id(document.id, "art", number), used_ids)
            parent = state.section_id or state.chapter_id or nodes[0].node_id
            state.article_id = article_id
            state.article_number = number
            state.article_title = article_title
            span = _span_until(lines, index, ("chapter", "section", "article"))
            raw = text[span[0] : span[1]]
            nodes.append(
                CanonicalNode(
                    node_id=article_id,
                    parent_id=parent,
                    level="article",
                    document_id=document.id,
                    document_name=document.name,
                    chapter_id=state.chapter_id,
                    chapter_title=state.chapter_title,
                    section_id=state.section_id,
                    section_title=state.section_title,
                    article_id=article_id,
                    article_number=number,
                    article_title=article_title,
                    raw_text=raw if raw.strip() else line.text,
                    parse_status="ok",
                    source=source,
                )
            )
            covered[span[0] : span[1]] = b"\x01" * (span[1] - span[0])
            child_nodes = _parse_clauses_and_points(
                document=document,
                text=text,
                lines=lines,
                article_span=span,
                state=state,
                source=source,
                used_ids=used_ids,
            )
            nodes.extend(child_nodes)
            # Skip lines covered by this article.
            next_index = index + advance
            while next_index < len(lines) and lines[next_index].start < span[1]:
                next_index += 1
            index = next_index
            continue

        index += 1

    # Retain uncovered non-whitespace spans as explicit retained nodes.
    cursor = 0
    while cursor < len(text):
        if covered[cursor]:
            cursor += 1
            continue
        start = cursor
        while cursor < len(text) and not covered[cursor]:
            cursor += 1
        raw = text[start:cursor]
        if not raw.strip():
            continue
        retained_id = _unique_id(
            _node_id(document.id, "retained", str(start)), used_ids
        )
        parent = (
            state.article_id or state.section_id or state.chapter_id or nodes[0].node_id
        )
        nodes.append(
            CanonicalNode(
                node_id=retained_id,
                parent_id=parent,
                level="document",
                document_id=document.id,
                document_name=document.name,
                chapter_id=state.chapter_id,
                chapter_title=state.chapter_title,
                section_id=state.section_id,
                section_title=state.section_title,
                article_id=state.article_id,
                article_number=state.article_number,
                article_title=state.article_title,
                raw_text=raw,
                parse_status="retained",
                source=source,
                warnings=(f"unparsed_span@{start}",),
            )
        )
        warnings.append(f"{document.id}: retained unparsed span at {start}")

    if warnings:
        doc_node = nodes[0]
        nodes[0] = CanonicalNode(
            **{
                **doc_node.model_dump(),
                "warnings": tuple(warnings),
                "parse_status": "partial",
            }
        )
    return tuple(nodes)


def _parse_clauses_and_points(
    *,
    document: LegalDocument,
    text: str,
    lines: tuple[_Line, ...],
    article_span: tuple[int, int],
    state: _ParseState,
    source: SourceProvenance,
    used_ids: set[str],
) -> list[CanonicalNode]:
    nodes: list[CanonicalNode] = []
    article_id = state.article_id
    if article_id is None:
        return nodes

    local_lines = [
        line for line in lines if article_span[0] <= line.start < article_span[1]
    ]
    clause_indices = [
        index
        for index, line in enumerate(local_lines)
        if _CLAUSE_RE.fullmatch(line.text) is not None
    ]
    if not clause_indices:
        return nodes

    for position, clause_index in enumerate(clause_indices):
        line = local_lines[clause_index]
        match = _CLAUSE_RE.fullmatch(line.text)
        assert match is not None
        number = match.group("number")
        clause_id = _unique_id(
            _node_id(document.id, "art", state.article_number or "?", "cl", number),
            used_ids,
        )
        clause_end = (
            local_lines[clause_indices[position + 1]].start
            if position + 1 < len(clause_indices)
            else article_span[1]
        )
        raw = text[line.start : clause_end]
        nodes.append(
            CanonicalNode(
                node_id=clause_id,
                parent_id=article_id,
                level="clause",
                document_id=document.id,
                document_name=document.name,
                chapter_id=state.chapter_id,
                chapter_title=state.chapter_title,
                section_id=state.section_id,
                section_title=state.section_title,
                article_id=article_id,
                article_number=state.article_number,
                article_title=state.article_title,
                clause_id=clause_id,
                clause_number=number,
                raw_text=raw if raw.strip() else line.text,
                parse_status="ok",
                source=source,
            )
        )
        point_lines = [
            candidate
            for candidate in local_lines
            if line.start < candidate.start < clause_end
            and _POINT_RE.fullmatch(candidate.text) is not None
        ]
        for point_position, point_line in enumerate(point_lines):
            point_match = _POINT_RE.fullmatch(point_line.text)
            assert point_match is not None
            label = point_match.group("label")
            point_id = _unique_id(
                _node_id(
                    document.id,
                    "art",
                    state.article_number or "?",
                    "cl",
                    number,
                    "pt",
                    label.casefold(),
                ),
                used_ids,
            )
            point_end = (
                point_lines[point_position + 1].start
                if point_position + 1 < len(point_lines)
                else clause_end
            )
            point_raw = text[point_line.start : point_end]
            nodes.append(
                CanonicalNode(
                    node_id=point_id,
                    parent_id=clause_id,
                    level="point",
                    document_id=document.id,
                    document_name=document.name,
                    chapter_id=state.chapter_id,
                    chapter_title=state.chapter_title,
                    section_id=state.section_id,
                    section_title=state.section_title,
                    article_id=article_id,
                    article_number=state.article_number,
                    article_title=state.article_title,
                    clause_id=clause_id,
                    clause_number=number,
                    point_id=point_id,
                    point_label=label,
                    raw_text=point_raw if point_raw.strip() else point_line.text,
                    parse_status="ok",
                    source=source,
                )
            )
    return nodes


def content_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def infer_parse_status(nodes: tuple[CanonicalNode, ...]) -> ParseStatus:
    statuses = {node.parse_status for node in nodes}
    if statuses <= {"ok"}:
        return "ok"
    if "unparsed" in statuses:
        return "unparsed"
    return "partial"
