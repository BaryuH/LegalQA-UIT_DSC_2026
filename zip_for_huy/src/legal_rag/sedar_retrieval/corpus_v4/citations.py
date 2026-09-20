"""Citation resolution against a v4 corpus index.

Evaluation-only helper. Gold answers cite law in prose ("Căn cứ Điều 76 Bộ luật
Lao động 2019"), so projecting silver retrieval labels requires mapping a prose
reference onto ``(document_id, instrument_index, article_number)``. Two
properties of the corpus make this harder than a dictionary lookup:

* Statutes are cited by name and year, not by number, and the recovered title
  can be partial - so matching is done on a diacritic-folded, whitespace-folded
  key and accepts containment in either direction.
* ``Nghị định này`` / ``Luật này`` are self-references. They are reported as
  ``self_reference`` rather than counted as failures, following the filtering
  used when building Vietnamese legal retrieval training data.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["Citation", "CitationIndex", "extract_citations", "fold"]

_DOC_TYPES = (
    "bộ luật",
    "luật",
    "nghị định",
    "thông tư",
    "quyết định",
    "nghị quyết",
    "pháp lệnh",
    "hiến pháp",
    "văn bản hợp nhất",
    "chỉ thị",
    "công văn",
)
#: The reference must stop before a *following* citation. Without the negative
#: lookahead the greedy character class swallows "và Điều 77 Bộ luật ..." into
#: the first reference, and the second citation is never extracted at all -
#: which matters because 38% of the gold answers cite two or more articles.
_CITATION_RE = re.compile(
    r"Điều\s+(\d+[a-zđ]?)\s+"
    r"((?:của\s+)?(?:%s)(?:(?!\s+(?:và\s+)?Điều\s+\d)[^\n\.;:]){0,80})"
    % "|".join(_DOC_TYPES),
    re.IGNORECASE,
)
_SELF_REF_RE = re.compile(
    r"^(?:của\s+)?(?:%s)\s+(này|nêu trên|trên)\b" % "|".join(_DOC_TYPES),
    re.IGNORECASE,
)
_TRAILING_NOISE_RE = re.compile(
    r"\s+(quy định|quy đinh|nêu|về|có|thì|như|và|hướng dẫn|sửa đổi bởi|đề cập).*$",
    re.IGNORECASE,
)

_ACCENT_TABLE = str.maketrans(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd",
)


def fold(text: str) -> str:
    """Whitespace-, case- and diacritic-folded matching key."""

    normalized = unicodedata.normalize("NFC", text).lower()
    normalized = normalized.translate(_ACCENT_TABLE)
    normalized = re.sub(r"[^a-z0-9/\- ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


@dataclass(frozen=True, slots=True)
class Citation:
    """One prose citation found in an answer."""

    article_number: str
    reference: str
    self_reference: bool


def extract_citations(text: str) -> tuple[Citation, ...]:
    """Pull ``Điều N <document reference>`` pairs out of an answer."""

    found: list[Citation] = []
    for article, reference in _CITATION_RE.findall(text):
        cleaned = _TRAILING_NOISE_RE.sub("", reference).strip(" ,;:.")
        found.append(
            Citation(
                article_number=article,
                reference=cleaned,
                self_reference=bool(_SELF_REF_RE.match(cleaned)),
            )
        )
    return tuple(found)


class CitationIndex:
    """Resolve prose citations to v4 unit ids."""

    def __init__(self) -> None:
        self._alias_to_docs: dict[str, set[str]] = defaultdict(set)
        self._articles: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add_document(
        self, document_id: str, aliases: Iterable[str], *, link: str | None = None
    ) -> None:
        """Register a document under every surface form it can be cited by."""

        keys = {fold(alias) for alias in aliases if alias}
        if link:
            slug = link.rstrip("/").rsplit("/", 1)[-1]
            slug = re.sub(r"\.aspx$", "", slug)
            slug = re.sub(r"-\d{4,7}$", "", slug)
            keys.add(fold(slug.replace("-", " ").replace("_", " ")))
        for key in keys:
            if len(key) >= 6:
                self._alias_to_docs[key].add(document_id)

    def add_article_unit(
        self, document_id: str, article_number: str | None, unit_id: str
    ) -> None:
        if not article_number:
            return
        self._articles[document_id][article_number.lower()].append(unit_id)

    def resolve(self, citation: Citation) -> tuple[str, ...]:
        """Unit ids matching one citation; empty when it cannot be resolved."""

        if citation.self_reference:
            return ()
        key = fold(citation.reference)
        if not key:
            return ()
        candidates = set(self._alias_to_docs.get(key, ()))
        if not candidates:
            for alias, documents in self._alias_to_docs.items():
                if alias in key or key in alias:
                    candidates |= documents
                    if len(candidates) > 64:
                        break
        hits: list[str] = []
        wanted = citation.article_number.lower()
        for document_id in candidates:
            hits.extend(self._articles.get(document_id, {}).get(wanted, ()))
        return tuple(hits)
