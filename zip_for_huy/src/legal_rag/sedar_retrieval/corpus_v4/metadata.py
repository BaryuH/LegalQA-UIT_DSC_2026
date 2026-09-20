"""Document-level metadata recovery for SEDAR corpus v4.

13.0% of the competition corpus (1,105 of 8,512 documents) has ``name`` equal to
its numeric id, so the document type, number and title have to be recovered from
the text header and the ``link`` slug. The recovered ``citation`` string is what
the generator quotes ("Căn cứ Điều 76 Bộ luật Lao động 2019"), so it is part of
the corpus contract rather than a nice-to-have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote

from .normalize import is_structural_heading as is_structural_marker

__all__ = [
    "DocumentMeta",
    "citation_aliases",
    "extract_document_meta",
    "normalize_doc_type",
]

_DOC_TYPES: tuple[tuple[str, str], ...] = (
    ("hien-phap", "Hiến pháp"),
    ("bo-luat", "Bộ luật"),
    ("van-ban-hop-nhat", "Văn bản hợp nhất"),
    ("phap-lenh", "Pháp lệnh"),
    ("nghi-quyet-lien-tich", "Nghị quyết liên tịch"),
    ("nghi-quyet", "Nghị quyết"),
    ("nghi-dinh", "Nghị định"),
    ("thong-tu-lien-tich", "Thông tư liên tịch"),
    ("thong-tu", "Thông tư"),
    ("quyet-dinh", "Quyết định"),
    ("chi-thi", "Chỉ thị"),
    ("cong-van", "Công văn"),
    ("thong-bao", "Thông báo"),
    ("huong-dan", "Hướng dẫn"),
    ("ke-hoach", "Kế hoạch"),
    ("quy-che", "Quy chế"),
    ("quy-chuan", "Quy chuẩn kỹ thuật quốc gia"),
    ("tieu-chuan", "Tiêu chuẩn quốc gia"),
    ("sac-lenh", "Sắc lệnh"),
    ("lenh", "Lệnh"),
    ("luat", "Luật"),
)

_NUMBER_RE = re.compile(
    r"số\s*[:.]?\s*([0-9]{1,6}\s*[/-]\s*[0-9A-ZĐ\-/\.]{2,40})",
    re.IGNORECASE,
)
_SLUG_NUMBER_RE = re.compile(
    r"^[-_]?(\d{1,5})[-_](\d{4})[-_]([A-Za-zĐđ]{2,8}(?:[-_][A-Za-zĐđ]{2,8}){0,3})(?=[-_]|$)"
)
_STANDARD_RE = re.compile(
    r"\b((?:QCVN|TCVN)\s*[\dA-Z\-:/\.]{3,40})", re.IGNORECASE
)
_TITLE_ANCHOR_RE = re.compile(
    r"(hà nội|thành phố hồ chí minh|tp\.?\s*hồ chí minh)?,?\s*ngày\s+\d{1,2}\s+tháng",
    re.IGNORECASE,
)
_TITLE_ONLY_TYPES = {"Luật", "Bộ luật", "Hiến pháp", "Pháp lệnh"}
_DATE_RE = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE
)
_QH_YEAR_RE = re.compile(r"/(\d{4})/QH", re.IGNORECASE)
_TITLE_STOP = re.compile(
    r"^(căn cứ|theo đề nghị|xét đề nghị|chính phủ|quốc hội|bộ trưởng|thủ tướng)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DocumentMeta:
    """Normalised identity of one corpus document."""

    document_id: str
    doc_type: str | None
    doc_number: str | None
    title: str | None
    issuer: str | None
    issued_date: str | None
    year: str | None
    citation: str
    metadata_source: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "document_id": self.document_id,
            "doc_type": self.doc_type,
            "doc_number": self.doc_number,
            "title": self.title,
            "issuer": self.issuer,
            "issued_date": self.issued_date,
            "year": self.year,
            "citation": self.citation,
            "metadata_source": self.metadata_source,
        }


def normalize_doc_type(raw: str | None) -> str | None:
    """Map a slug fragment or free text onto a canonical Vietnamese doc type."""

    if not raw:
        return None
    lowered = _deaccent(raw).lower()
    for slug, label in _DOC_TYPES:
        if slug in lowered:
            return label
    return None


def _deaccent(text: str) -> str:
    table = str.maketrans(
        "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ",
        "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd",
    )
    return text.lower().translate(table)


def _header_block(text: str, *, max_lines: int = 40) -> list[str]:
    lines = [line.strip() for line in text.split("\n")[:max_lines]]
    return [line for line in lines if line]


def _title_start_index(lines: list[str]) -> int:
    """First line after the masthead: the date line, else the ``Số:`` line."""

    anchor = 0
    for index, line in enumerate(lines[:30]):
        if _TITLE_ANCHOR_RE.search(line) or _NUMBER_RE.search(line):
            anchor = index + 1
    return anchor


def _extract_title(lines: list[str], doc_type: str | None) -> str | None:
    """Title block sitting between the masthead and the first ``Căn cứ`` line."""

    parts: list[str] = []
    for line in lines[_title_start_index(lines) :]:
        if _TITLE_STOP.match(line):
            break
        if is_structural_marker(line):
            break
        stripped = line.strip(" -–—*_.")
        if not stripped or len(stripped) < 3:
            if parts:
                break
            continue
        low = _deaccent(stripped)
        if low.startswith(("cong hoa", "doc lap", "so:", "so :")):
            continue
        if doc_type and low.strip(" :") == _deaccent(doc_type):
            continue
        if normalize_doc_type(stripped) and len(stripped) <= 24:
            continue
        parts.append(stripped)
        if len(" ".join(parts)) > 240:
            break
    if not parts:
        return None
    title = re.sub(r"\s+", " ", " ".join(parts)).strip(" -–—*:")
    if doc_type and _deaccent(title).startswith(_deaccent(doc_type)):
        title = title[len(doc_type) :].strip(" :-–—")
    return title or None


def _extract_issuer(lines: list[str]) -> str | None:
    for line in lines[:6]:
        letters = [c for c in line if c.isalpha()]
        if not letters:
            continue
        if sum(1 for c in letters if c.isupper()) / len(letters) < 0.8:
            continue
        low = _deaccent(line)
        if low.startswith(("cong hoa", "doc lap")):
            continue
        return re.sub(r"[-–—*]+$", "", line).strip()
    return None


def extract_document_meta(
    document_id: str, name: str | None, link: str | None, text: str
) -> DocumentMeta:
    """Recover identity from the header block, falling back to the link slug."""

    lines = _header_block(text)
    head = "\n".join(lines)
    sources: list[str] = []

    doc_number: str | None = None
    match = _NUMBER_RE.search(head)
    if match:
        doc_number = re.sub(r"\s+", "", match.group(1)).strip("/-")
        sources.append("header_number")

    slug = ""
    if link:
        slug = unquote(link).rsplit("/", 1)[-1]
    slug_source = slug or (name or "")
    slug_type = normalize_doc_type(slug_source)
    slug_tail = ""
    if slug_type:
        for slug_key, label in _DOC_TYPES:
            if label == slug_type and slug_key in _deaccent(slug_source):
                cut = _deaccent(slug_source).index(slug_key) + len(slug_key)
                slug_tail = slug_source[cut:]
                break
    if not doc_number and slug_tail:
        slug_match = _SLUG_NUMBER_RE.search(slug_tail)
        if slug_match:
            doc_number = (
                f"{slug_match.group(1)}/{slug_match.group(2)}/"
                f"{slug_match.group(3).replace('_', '-').upper()}"
            )
            sources.append("slug_number")

    doc_type = normalize_doc_type(slug_source)
    if doc_type:
        sources.append("slug_type")
    else:
        doc_type = normalize_doc_type(head[:400])
        if doc_type:
            sources.append("header_type")

    if not doc_number:
        standard = _STANDARD_RE.search(head)
        if standard:
            doc_number = re.sub(r"\s+", " ", standard.group(1)).strip()
            doc_type = doc_type or (
                "Quy chuẩn kỹ thuật quốc gia"
                if doc_number.upper().startswith("QCVN")
                else "Tiêu chuẩn quốc gia"
            )
            sources.append("standard_code")

    issued_date = None
    date_match = _DATE_RE.search(head)
    if date_match:
        day, month, year = date_match.groups()
        issued_date = f"{year}-{int(month):02d}-{int(day):02d}"
        sources.append("header_date")

    year = issued_date[:4] if issued_date else None
    if not year and doc_number:
        qh = _QH_YEAR_RE.search(doc_number)
        if qh:
            year = qh.group(1)
        else:
            parts = doc_number.split("/")
            if len(parts) > 1 and re.fullmatch(r"\d{4}", parts[1]):
                year = parts[1]

    title = _extract_title(lines, doc_type)
    issuer = _extract_issuer(lines)

    citation = _build_citation(doc_type, doc_number, title, year, name, document_id)
    return DocumentMeta(
        document_id=document_id,
        doc_type=doc_type,
        doc_number=doc_number,
        title=title,
        issuer=issuer,
        issued_date=issued_date,
        year=year,
        citation=citation,
        metadata_source="+".join(sources) if sources else "none",
    )


def _titlecase_vi(text: str) -> str:
    words = text.split()
    return " ".join(w.capitalize() if w.isupper() else w for w in words)


def _build_citation(
    doc_type: str | None,
    doc_number: str | None,
    title: str | None,
    year: str | None,
    name: str | None,
    document_id: str,
) -> str:
    """Render the short citation a legal answer is expected to quote.

    Codes and statutes are cited by name and year ("Bộ luật Lao động 2019"),
    which is how the gold answers in ``warmup.json`` cite them; every other
    instrument is cited by its number ("Nghị định 99/2003/NĐ-CP").
    """

    if doc_type in _TITLE_ONLY_TYPES and title and year:
        return f"{doc_type} {_titlecase_vi(title)} {year}".replace("  ", " ").strip()
    if doc_type and doc_number:
        return f"{doc_type} {doc_number}"
    if doc_number:
        return doc_number
    if doc_type and title:
        return f"{doc_type} {_titlecase_vi(title)}"[:160]
    if name and name.strip() != document_id:
        return name.replace("-", " ").strip()
    return f"Văn bản {document_id}"


def citation_aliases(meta: DocumentMeta) -> tuple[str, ...]:
    """Alternative surface forms used to resolve citations found in answers."""

    aliases: list[str] = [meta.citation]
    if meta.doc_type and meta.doc_number:
        aliases.append(f"{meta.doc_type} {meta.doc_number}")
        aliases.append(meta.doc_number)
    if meta.doc_type and meta.title:
        short = f"{meta.doc_type} {_titlecase_vi(meta.title)}"
        aliases.append(short[:160])
        if meta.year:
            aliases.append(f"{short[:150]} {meta.year}")
    seen: dict[str, None] = {}
    for alias in aliases:
        cleaned = re.sub(r"\s+", " ", alias).strip()
        if cleaned:
            seen.setdefault(cleaned, None)
    return tuple(seen)
