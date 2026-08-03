"""Deterministic legal-aware chunking from documents to retrieval chunks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

from ..schemas import LegalChunk, LegalDocument
from .normalize import normalize_retrieval_text


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Validated limits for legal-unit chunking and window fallback."""

    max_chars: int = 1200
    overlap_chars: int = 0
    min_chars: int = 100
    version: str = "legal-chunker-v1"

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        if self.min_chars <= 0:
            raise ValueError("min_chars must be greater than zero")
        if self.min_chars > self.max_chars:
            raise ValueError("min_chars must not exceed max_chars")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        if not self.version.strip():
            raise ValueError("version must not be blank")


@dataclass(frozen=True, slots=True)
class _Line:
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class _Heading:
    kind: str
    start: int
    end: int
    label: str
    value: str
    has_inline_content: bool = False


@dataclass(frozen=True, slots=True)
class _Unit:
    start: int
    end: int
    label: str


_ARTICLE_RE = re.compile(
    r"^\s*điều\s+(?P<number>\d+[A-Za-z]?)(?:\s*[.:]\s*.*)?\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_ARTICLE_ONLY_RE = re.compile(r"^\s*điều\s*$", flags=re.IGNORECASE | re.UNICODE)
_ARTICLE_NUMBER_RE = re.compile(
    r"^\s*(?P<number>\d+[A-Za-z]?)(?:\s*[.:]\s*.*)?\s*$",
    flags=re.UNICODE,
)
_CLAUSE_RE = re.compile(r"^\s*(?P<number>\d+)\s*[.)](?P<body>.*)$", flags=re.UNICODE)
_POINT_RE = re.compile(
    r"^\s*(?P<value>[A-Za-zĐđ])\s*[.)](?:\s*.*)?$",
    flags=re.IGNORECASE | re.UNICODE,
)


def _iter_lines(text: str) -> tuple[_Line, ...]:
    lines: list[_Line] = []
    for match in re.finditer(r"[^\r\n]*(?:\r\n|\r|\n|$)", text):
        if match.start() == len(text) and not match.group():
            break
        raw_line = match.group()
        lines.append(
            _Line(
                start=match.start(),
                end=match.end(),
                text=raw_line.rstrip("\r\n"),
            )
        )
    return tuple(lines)


def _find_headings(text: str) -> tuple[_Heading, ...]:
    """Find line-anchored headings without changing the source text."""

    lines = _iter_lines(text)
    headings: list[_Heading] = []
    consumed_lines: set[int] = set()

    for index, line in enumerate(lines):
        article_match = _ARTICLE_RE.fullmatch(line.text)
        if article_match is not None:
            number = article_match.group("number")
            headings.append(
                _Heading(
                    kind="article",
                    start=line.start,
                    end=line.end,
                    label=f"Điều {number}",
                    value=number,
                )
            )
            continue

        if not _ARTICLE_ONLY_RE.fullmatch(line.text) or index + 1 >= len(lines):
            continue
        number_match = _ARTICLE_NUMBER_RE.fullmatch(lines[index + 1].text)
        if number_match is None:
            continue
        number = number_match.group("number")
        headings.append(
            _Heading(
                kind="article",
                start=line.start,
                end=lines[index + 1].end,
                label=f"Điều {number}",
                value=number,
            )
        )
        consumed_lines.add(index + 1)

    for index, line in enumerate(lines):
        if index in consumed_lines:
            continue
        clause_match = _CLAUSE_RE.fullmatch(line.text)
        if clause_match is not None:
            number = clause_match.group("number")
            headings.append(
                _Heading(
                    kind="clause",
                    start=line.start,
                    end=line.end,
                    label=f"Khoản {number}",
                    value=number,
                    has_inline_content=bool(clause_match.group("body").strip()),
                )
            )
            continue

        point_match = _POINT_RE.fullmatch(line.text)
        if point_match is not None:
            value = point_match.group("value")
            headings.append(
                _Heading(
                    kind="point",
                    start=line.start,
                    end=line.end,
                    label=f"Điểm {value}",
                    value=value,
                )
            )

    return tuple(sorted(headings, key=lambda heading: (heading.start, heading.kind)))


def _window_ranges(
    start: int, end: int, config: ChunkingConfig
) -> tuple[tuple[int, int], ...]:
    """Return source ranges covering a unit without truncating its tail."""

    if end - start <= config.max_chars:
        return ((start, end),)

    step = config.max_chars - config.overlap_chars
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor + config.max_chars < end:
        ranges.append((cursor, cursor + config.max_chars))
        cursor += step

    tail_start = cursor
    if end - tail_start < config.min_chars and ranges:
        tail_start = max(start, end - config.max_chars)
    if not ranges or ranges[-1] != (tail_start, end):
        ranges.append((tail_start, end))
    return tuple(ranges)


def _has_content(text: str, start: int, end: int) -> bool:
    return bool(text[start:end].strip())


def _article_units(
    text: str,
    article: _Heading,
    article_end: int,
    headings: tuple[_Heading, ...],
) -> tuple[_Unit, ...]:
    clauses = [
        heading
        for heading in headings
        if heading.kind == "clause" and article.start < heading.start < article_end
    ]
    article_label = article.label
    if not clauses:
        return (_Unit(article.start, article_end, article_label),)

    units: list[_Unit] = []
    first_clause = clauses[0]
    if _has_content(text, article.start, first_clause.start):
        between_heading_and_clause = text[article.end : first_clause.start]
        if not between_heading_and_clause.strip():
            first_clause_start = article.start
        else:
            units.append(_Unit(article.start, first_clause.start, article_label))
            first_clause_start = first_clause.start
    else:  # pragma: no cover - article headings are non-blank by construction
        first_clause_start = first_clause.start

    for index, clause in enumerate(clauses):
        clause_start = first_clause_start if index == 0 else clause.start
        clause_end = (
            clauses[index + 1].start if index + 1 < len(clauses) else article_end
        )
        clause_label = f"{article_label} / {clause.label}"
        points = [
            heading
            for heading in headings
            if heading.kind == "point" and clause.start < heading.start < clause_end
        ]
        if not points:
            if _has_content(text, clause_start, clause_end):
                units.append(_Unit(clause_start, clause_end, clause_label))
            continue

        first_point = points[0]
        point_prefix_start = clause_start
        if _has_content(text, clause_start, first_point.start):
            between_clause_and_point = text[clause.end : first_point.start]
            if not clause.has_inline_content and not between_clause_and_point.strip():
                point_prefix_start = clause_start
            else:
                units.append(_Unit(clause_start, first_point.start, clause_label))
                point_prefix_start = first_point.start

        for point_index, point in enumerate(points):
            point_start = point_prefix_start if point_index == 0 else point.start
            point_end = (
                points[point_index + 1].start
                if point_index + 1 < len(points)
                else clause_end
            )
            if _has_content(text, point_start, point_end):
                units.append(
                    _Unit(
                        point_start,
                        point_end,
                        f"{clause_label} / {point.label}",
                    )
                )

    return tuple(units)


def _document_units(text: str) -> tuple[_Unit, ...]:
    headings = _find_headings(text)
    articles = [heading for heading in headings if heading.kind == "article"]
    if not articles:
        return (_Unit(0, len(text), "Document"),)

    units: list[_Unit] = []
    first_article = articles[0]
    if _has_content(text, 0, first_article.start):
        units.append(_Unit(0, first_article.start, "Document"))

    for index, article in enumerate(articles):
        article_end = (
            articles[index + 1].start if index + 1 < len(articles) else len(text)
        )
        units.extend(_article_units(text, article, article_end, headings))
    return tuple(units)


def chunk_document(
    document: LegalDocument, config: ChunkingConfig | None = None
) -> tuple[LegalChunk, ...]:
    """Chunk one legal document using the Document/Điều/Khoản/Điểm ladder."""

    selected_config = config or ChunkingConfig()
    units = _document_units(document.passage)
    chunks: list[LegalChunk] = []
    for unit in units:
        for range_start, range_end in _window_ranges(
            unit.start, unit.end, selected_config
        ):
            raw_text = document.passage[range_start:range_end]
            if not raw_text.strip():
                continue
            chunk_id = f"{document.id}:{len(chunks):04d}"
            chunks.append(
                LegalChunk(
                    chunk_id=chunk_id,
                    document_id=document.id,
                    source_path=document.source_path,
                    source_member=document.source_member,
                    raw_text=raw_text,
                    retrieval_text=normalize_retrieval_text(raw_text),
                    content_hash=sha256(raw_text.encode("utf-8")).hexdigest(),
                    chunker_version=selected_config.version,
                    section_label=unit.label,
                    start_offset=range_start,
                    end_offset=range_end,
                )
            )
    return tuple(chunks)


def chunk_documents(
    documents: Iterable[LegalDocument], config: ChunkingConfig | None = None
) -> tuple[LegalChunk, ...]:
    """Chunk documents in deterministic provenance order."""

    selected_config = config or ChunkingConfig()
    ordered_documents = sorted(
        documents,
        key=lambda document: (
            document.id,
            document.source_path,
            document.source_member or "",
        ),
    )
    return tuple(
        chunk
        for document in ordered_documents
        for chunk in chunk_document(document, selected_config)
    )
