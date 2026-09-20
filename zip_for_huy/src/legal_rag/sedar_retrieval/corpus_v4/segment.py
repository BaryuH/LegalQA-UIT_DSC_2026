"""Instrument-aware structural segmentation for SEDAR corpus v4.

Two structural facts about the competition corpus drive this module.

**Annexed instruments.** 3,709 of 8,512 documents (43.6%) contain the phrase
"ban hành kèm theo": a promulgating decision whose ``Điều 1`` attaches a whole
regulation that restarts its own ``Điều`` numbering. Parsing such a document as
one flat article list produces duplicate article numbers - measured at 10,995
collisions across 2,004 documents in the v3 parser - which makes both the node
ids and any ``Điều N`` citation ambiguous. v4 therefore splits a document into
*instruments* first and scopes every id and every citation to one instrument.

**Article-less documents.** 1,289 documents (15.1%) contain no ``Điều`` at all;
they are công văn, chỉ thị, thông báo, kế hoạch, hướng dẫn and QCVN/TCVN
standards whose native structure is roman/decimal headings. v3 emits nothing
retrievable for them. v4 segments them into ``block`` units so the corpus covers
every document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .normalize import is_structural_heading

__all__ = [
    "Article",
    "Block",
    "Instrument",
    "parse_articles",
    "segment_blocks",
    "split_article_children",
    "split_instruments",
]

_ANNEX_MARKER_RE = re.compile(r"ban\s+hành\s+kèm\s+theo", re.IGNORECASE)
_ANNEX_TITLE_RE = re.compile(
    r"""^\s*(
        phụ\s+lục
        | quy\s+chế | quy\s+định | quy\s+trình | quy\s+chuẩn | quy\s+hoạch
        | điều\s+lệ | danh\s+mục | biểu\s+mẫu | mẫu\s+số
        | chương\s+trình | đề\s+án | chiến\s+lược | kế\s+hoạch | phương\s+án
        | thể\s+lệ | nội\s+quy | tiêu\s+chuẩn | hướng\s+dẫn | định\s+mức
        | bảng\s+giá | khung\s+giá | lộ\s+trình | quy\s+tắc
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)
_PART_RE = re.compile(r"^\s*phần\s+(thứ\s+)?([\dIVXivx]+|[^\n]{0,60})\s*$", re.I)
_CHAPTER_RE = re.compile(r"^\s*chương\s+([\dIVXivx]+)\s*[\.:\-]?\s*(.*)$", re.I)
_SECTION_RE = re.compile(r"^\s*mục\s+([\dIVXivx]+)\s*[\.:\-]?\s*(.*)$", re.I)
_SUBSECTION_RE = re.compile(r"^\s*tiểu\s+mục\s+([\dIVXivx]+)\s*[\.:\-]?\s*(.*)$", re.I)
_ARTICLE_RE = re.compile(
    r"^\s*điều\s+(\d+[a-zđ]?)\s*[\.:\-–]?\s*(.*)$", re.IGNORECASE
)
_ARTICLE_BARE_RE = re.compile(r"^\s*điều\s*$", re.IGNORECASE)
_ARTICLE_NUMBER_ONLY_RE = re.compile(r"^\s*(\d+[a-zđ]?)\s*[\.:\-–]?\s*(.*)$")
_CLAUSE_RE = re.compile(r"^\s*(\d{1,3})[\.\)]\s*(?=\S)")
_POINT_RE = re.compile(r"^\s*([a-zđ])[\)\.]\s*(?=\S)", re.IGNORECASE)
_ROMAN_RE = re.compile(r"^\s*([IVX]{1,5})[\.\)]\s*(?=\S)")
_UPPER_HEADING_RE = re.compile(r"^[^a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị"
                               r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]{6,120}$")

#: A line-initial "Điều N" whose remainder opens with one of these words is a
#: cross-reference caught mid-sentence ("Điều 325. và Điều 326 của Bộ luật dân
#: sự thì ..."), not a heading. Measured on a 120-document integrity sample,
#: false-positive headings of this shape were the largest residual parse error.
_NOT_A_HEADING_TAIL = {
    "và", "của", "này", "thì", "hoặc", "đến", "trở", "được", "nêu", "tại",
    "sau", "trên", "cùng", "với", "là", "cho", "theo", "nhưng", "mà", "nếu",
    "bị", "do", "về",
}

_CITATION_HINT_RE = re.compile(
    r"(quy định tại|theo quy định|căn cứ|quy định ở|nêu tại)\s*$", re.IGNORECASE
)


@dataclass(slots=True)
class Instrument:
    """One self-contained numbering scope inside a corpus document."""

    index: int
    kind: str  # "main" | "annex"
    title: str | None
    lines: list[str]
    line_offset: int

    @property
    def label(self) -> str:
        if self.kind == "main":
            return "văn bản chính"
        return self.title or f"phụ lục {self.index}"


@dataclass(slots=True)
class Article:
    """One ``Điều`` with its ancestor headings resolved."""

    number: str
    title: str | None
    body: str
    part: str | None = None
    chapter_number: str | None = None
    chapter_title: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    subsection_title: str | None = None
    line_start: int = 0
    line_end: int = 0

    @property
    def heading(self) -> str:
        head = f"Điều {self.number}"
        return f"{head}. {self.title}" if self.title else head

    @property
    def text(self) -> str:
        return f"{self.heading}\n{self.body}".strip()


@dataclass(slots=True)
class Block:
    """A retrieval unit for documents that have no ``Điều`` structure."""

    label: str | None
    body: str
    line_start: int = 0
    line_end: int = 0
    path: tuple[str, ...] = field(default_factory=tuple)


def _is_upper_title(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 6:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) >= 0.85


def split_instruments(lines: list[str]) -> list[Instrument]:
    """Split a normalised document into main text plus annexed instruments.

    An annex starts at an ALL-CAPS instrument title (``QUY CHẾ``, ``PHỤ LỤC``,
    ``ĐỀ ÁN`` ...). The ``ban hành kèm theo`` line that usually follows is used
    as confirmation, and a title that is only *mentioned* inside a sentence -
    "quy định tại Quy chế ban hành kèm theo..." - is rejected because a genuine
    heading stands alone on its line.
    """

    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 160:
            continue
        if not _ANNEX_TITLE_RE.match(stripped):
            continue
        if not _is_upper_title(stripped) and not stripped.lower().startswith(
            ("phụ lục", "mẫu số", "biểu mẫu")
        ):
            continue
        window = " ".join(lines[index : index + 4])
        confirmed = bool(_ANNEX_MARKER_RE.search(window)) or stripped.lower().startswith(
            ("phụ lục", "mẫu số", "biểu mẫu")
        )
        if not confirmed:
            continue
        if index and _CITATION_HINT_RE.search(lines[index - 1].strip()):
            continue
        starts.append((index, stripped))

    # Merge starts that are adjacent (a two-line title) and drop very early ones
    # that belong to the masthead.
    cleaned: list[tuple[int, str]] = []
    for index, title in starts:
        if index < 3:
            continue
        if cleaned and index - cleaned[-1][0] <= 2:
            continue
        cleaned.append((index, title))

    if not cleaned:
        return [Instrument(index=0, kind="main", title=None, lines=lines, line_offset=0)]

    instruments = [
        Instrument(
            index=0,
            kind="main",
            title=None,
            lines=lines[: cleaned[0][0]],
            line_offset=0,
        )
    ]
    for position, (index, title) in enumerate(cleaned):
        end = cleaned[position + 1][0] if position + 1 < len(cleaned) else len(lines)
        instruments.append(
            Instrument(
                index=position + 1,
                kind="annex",
                title=title,
                lines=lines[index:end],
                line_offset=index,
            )
        )
    return [inst for inst in instruments if any(x.strip() for x in inst.lines)]


def parse_articles(instrument: Instrument) -> list[Article]:
    """Parse ``Điều`` units and their ancestor headings inside one instrument."""

    lines = instrument.lines
    articles: list[Article] = []
    part = chapter_no = chapter_title = section_no = section_title = subsection = None
    current: Article | None = None
    buffer: list[str] = []
    pending_title_for: Article | None = None

    def flush(end_index: int) -> None:
        nonlocal current, buffer
        if current is not None:
            current.body = "\n".join(buffer).strip()
            current.line_end = end_index
            articles.append(current)
        current = None
        buffer = []

    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            index += 1
            continue

        if pending_title_for is not None:
            # "Điều 2." on one line and its title on the next.
            if not is_structural_heading(line) and len(line) <= 200:
                pending_title_for.title = line
                pending_title_for = None
                index += 1
                continue
            pending_title_for = None

        part_match = _PART_RE.match(line)
        chapter_match = _CHAPTER_RE.match(line)
        section_match = _SECTION_RE.match(line)
        subsection_match = _SUBSECTION_RE.match(line)
        article_match = _ARTICLE_RE.match(line)

        if subsection_match:
            flush(index)
            subsection = (subsection_match.group(2) or "").strip() or None
            if not subsection and index + 1 < len(lines):
                subsection = lines[index + 1].strip() or None
            index += 1
            continue
        if section_match and not article_match:
            flush(index)
            section_no = section_match.group(1)
            section_title = (section_match.group(2) or "").strip() or None
            if not section_title and index + 1 < len(lines):
                nxt = lines[index + 1].strip()
                if nxt and not is_structural_heading(nxt):
                    section_title = nxt
                    index += 1
            subsection = None
            index += 1
            continue
        if chapter_match and not article_match:
            flush(index)
            chapter_no = chapter_match.group(1)
            chapter_title = (chapter_match.group(2) or "").strip() or None
            if not chapter_title and index + 1 < len(lines):
                nxt = lines[index + 1].strip()
                if nxt and not is_structural_heading(nxt):
                    chapter_title = nxt
                    index += 1
            section_no = section_title = subsection = None
            index += 1
            continue
        if part_match and not article_match and len(line) <= 80:
            flush(index)
            part = line
            chapter_no = chapter_title = section_no = section_title = None
            subsection = None
            index += 1
            continue

        if _ARTICLE_BARE_RE.match(line) and index + 1 < len(lines):
            follow = _ARTICLE_NUMBER_ONLY_RE.match(lines[index + 1].strip())
            if follow:
                article_match = _ARTICLE_RE.match(f"Điều {lines[index + 1].strip()}")
                index += 1

        if article_match:
            tail = (article_match.group(2) or "").strip()
            first_word = tail.split(" ", 1)[0].strip(".,;:)").lower() if tail else ""
            if first_word in _NOT_A_HEADING_TAIL:
                if current is not None:
                    buffer.append(line)
                index += 1
                continue
            flush(index)
            title = (article_match.group(2) or "").strip() or None
            current = Article(
                number=article_match.group(1),
                title=title,
                body="",
                part=part,
                chapter_number=chapter_no,
                chapter_title=chapter_title,
                section_number=section_no,
                section_title=section_title,
                subsection_title=subsection,
                line_start=index,
            )
            if title is None:
                pending_title_for = current
            index += 1
            continue

        if current is not None:
            buffer.append(line)
        index += 1

    flush(len(lines))
    return articles


def segment_blocks(
    lines: list[str],
    *,
    target_chars: int = 1200,
    max_chars: int = 2600,
    min_chars: int = 320,
) -> list[Block]:
    """Segment an article-less instrument on its own heading markers.

    Headings are roman numerals (``I.``), decimal outline numbers (``1.``,
    ``2.1.``) and ALL-CAPS lines. Consecutive segments below ``min_chars`` are
    merged so no block is ever indexed as a fragment.
    """

    segments: list[Block] = []
    label: str | None = None
    path: list[str] = []
    buffer: list[str] = []
    start = 0

    def flush(end: int) -> None:
        nonlocal buffer, label, start
        body = "\n".join(buffer).strip()
        if body:
            segments.append(
                Block(
                    label=label,
                    body=body,
                    line_start=start,
                    line_end=end,
                    path=tuple(path),
                )
            )
        buffer = []
        start = end

    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        roman = _ROMAN_RE.match(line)
        upper = _UPPER_HEADING_RE.match(line) and _is_upper_title(line)
        decimal = re.match(r"^\s*(\d+(?:\.\d+)*)\.\s+\S", line)
        is_heading = bool(roman or upper or (decimal and len(line) <= 160))
        if is_heading and buffer and len("\n".join(buffer)) >= target_chars:
            flush(index)
            label = line[:160]
            path = [label]
        elif is_heading and not buffer:
            label = line[:160]
            path = [label]
        buffer.append(line)
        if len("\n".join(buffer)) >= max_chars:
            flush(index + 1)
            label = None
    flush(len(lines))

    merged: list[Block] = []
    for block in segments:
        if merged and len(block.body) < min_chars:
            previous = merged[-1]
            if len(previous.body) + len(block.body) <= max_chars:
                previous.body = f"{previous.body}\n{block.body}"
                previous.line_end = block.line_end
                continue
        merged.append(block)
    return merged


def split_article_children(
    article: Article,
    *,
    min_child_chars: int = 500,
    max_child_chars: int = 2400,
) -> list[Block]:
    """Split one long ``Điều`` at ``Khoản`` (or roman) boundaries, merging runs.

    Children are only produced for articles that exceed the caller's length
    threshold. Adjacent clauses are merged until each child clears
    ``min_child_chars``, so a 40-character ``Khoản`` never becomes a unit of its
    own - that fragment is what currently wins retrieval and starves the
    evidence pack.
    """

    lines = article.body.split("\n")
    groups: list[tuple[str | None, list[str]]] = []
    label: str | None = None
    buffer: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        clause = _CLAUSE_RE.match(stripped)
        roman = _ROMAN_RE.match(stripped)
        if clause or roman:
            if buffer:
                groups.append((label, buffer))
            label = (
                f"Khoản {clause.group(1)}" if clause else f"Mục {roman.group(1)}"
            )
            buffer = [stripped]
        else:
            buffer.append(stripped)
    if buffer:
        groups.append((label, buffer))

    children: list[Block] = []
    pending_labels: list[str] = []
    pending: list[str] = []
    for group_label, group_lines in groups:
        pending.extend(group_lines)
        if group_label:
            pending_labels.append(group_label)
        body = "\n".join(pending)
        if len(body) >= min_child_chars:
            children.append(
                Block(
                    label=_range_label(pending_labels),
                    body=body,
                    path=tuple(pending_labels),
                )
            )
            pending = []
            pending_labels = []
        elif len(body) > max_child_chars:
            children.append(
                Block(
                    label=_range_label(pending_labels),
                    body=body,
                    path=tuple(pending_labels),
                )
            )
            pending = []
            pending_labels = []
    if pending:
        body = "\n".join(pending)
        if children and len(body) < min_child_chars:
            children[-1].body = f"{children[-1].body}\n{body}"
            children[-1].path = tuple(list(children[-1].path) + pending_labels)
            children[-1].label = _range_label(list(children[-1].path))
        else:
            children.append(
                Block(
                    label=_range_label(pending_labels),
                    body=body,
                    path=tuple(pending_labels),
                )
            )
    return children


def _range_label(labels: list[str]) -> str | None:
    if not labels:
        return None
    if len(labels) == 1:
        return labels[0]
    return f"{labels[0]} - {labels[-1]}"
