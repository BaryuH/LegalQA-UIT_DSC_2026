"""Legal-text chunker with legal hierarchy parsing and provenance tracking.

Supports Clause-Level Chunking (Điều -> Khoản -> Điểm) and Small-to-Big
Parent Document Retrieval architecture.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import re
from dataclasses import dataclass
from hashlib import sha256

from .data_loader import LegalDocument

logger = logging.getLogger(__name__)

CHUNKER_VERSION = "nhan-chunker-v2"

# Regex for Article (Điều) headings: e.g. "Điều 37.", "Điều 1: Tiêu đề", "ĐIỀU 1"
_ARTICLE_RE = re.compile(
    r"^\s*điều\s+(?P<number>\d+[A-Za-z]?)(?:\s*[-–—.:]\s*(?P<title>.*))?\s*$",
    flags=re.IGNORECASE | re.UNICODE,
)
_ARTICLE_ONLY_RE = re.compile(r"^\s*điều\s*$", flags=re.IGNORECASE | re.UNICODE)
_ARTICLE_NUMBER_RE = re.compile(
    r"^\s*(?P<number>\d+[A-Za-z]?)(?:\s*[-–—.:]\s*(?P<title>.*))?\s*$",
    flags=re.UNICODE,
)

# Regex for Clause (Khoản) headings: e.g. "1.", "Khoản 1.", "1)", "Khoản 2:"
_CLAUSE_RE = re.compile(
    r"^\s*(?:khoản\s+)?(?P<number>\d+)\s*[.)]\s*(?P<body>.*)$",
    flags=re.IGNORECASE | re.UNICODE,
)

# Regex for Point (Điểm) headings: e.g. "a)", "đ)", "Điểm a.", "b."
_POINT_RE = re.compile(
    r"^\s*(?:điểm\s+)?(?P<label>[a-zđA-ZĐ])\s*[.)]\s*(?P<body>.*)$",
    flags=re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class ParentChunk:
    """Parent document chunk representing an entire legal Article (Điều) or section."""

    parent_id: str
    document_id: str
    source_path: str
    source_member: str | None
    header: str
    article_number: str | None
    raw_text: str
    start_offset: int
    end_offset: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class LegalChunk:
    """Child chunk representing a Clause (Khoản) or sub-unit for retrieval."""

    chunk_id: str
    document_id: str
    parent_id: str
    source_path: str
    source_member: str | None
    raw_text: str
    start_offset: int
    end_offset: int
    content_hash: str
    section_label: str = ""
    retrieval_text_override: str | None = None

    @property
    def retrieval_text(self) -> str:
        """Text used for retrieval indexing (with parent context if present)."""
        if self.retrieval_text_override is not None:
            return self.retrieval_text_override
        return self.raw_text


@dataclass(frozen=True, slots=True)
class DocumentChunkResult:
    """Result of chunking a single legal document."""

    chunks: list[LegalChunk]
    parents: list[ParentChunk]

    def __iter__(self):
        return iter((self.chunks, self.parents))


@dataclass(frozen=True, slots=True)
class ChunkCorpusResult:
    """Result of chunking all documents in the corpus."""

    chunks: list[LegalChunk]
    parents: list[ParentChunk]

    @property
    def chunks_by_id(self) -> dict[str, LegalChunk]:
        return {c.chunk_id: c for c in self.chunks}

    @property
    def parents_by_id(self) -> dict[str, ParentChunk]:
        return {p.parent_id: p for p in self.parents}

    def __iter__(self):
        return iter((self.chunks, self.parents))


@dataclass(frozen=True, slots=True)
class _Line:
    start: int
    end: int
    text: str


def _iter_lines(text: str) -> list[_Line]:
    """Split text into lines while capturing character boundaries."""
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
    return lines


def _window_split_text(
    text: str,
    base_offset: int,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[tuple[int, int, str]]:
    """Fallback sliding window splitter for texts exceeding max_chars."""
    if len(text) <= max_chars:
        return [(base_offset, base_offset + len(text), text)]

    results: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            last_break = text.rfind("\n", start, end)
            if last_break > start + min_chars:
                end = last_break + 1
            else:
                last_period = text.rfind(". ", start, end)
                if last_period > start + min_chars:
                    end = last_period + 2

        chunk_str = text[start:end]
        if chunk_str.strip():
            results.append((base_offset + start, base_offset + end, chunk_str))

        next_start = end - overlap_chars
        if next_start <= start:
            next_start = end
        start = next_start

    return results


def chunk_document(
    doc: LegalDocument,
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 100,
) -> DocumentChunkResult:
    """Chunk a legal document using legal hierarchy (Điều -> Khoản -> Điểm).

    Produces:
    - ParentChunks: Representing each Điều (or preamble/document if no Điều).
    - LegalChunks (Child): Representing each Khoản (or Điểm if Khoản is long),
      carrying parent_id and enriched retrieval_text.
    """
    passage = doc.passage
    if not passage.strip():
        return DocumentChunkResult(chunks=[], parents=[])

    lines = _iter_lines(passage)

    # 1. Identify all Article headings
    article_spans: list[tuple[int, str, str | None, str]] = []  # (line_idx, number, title, header)
    consumed_lines: set[int] = set()

    for idx, line in enumerate(lines):
        if idx in consumed_lines:
            continue
        m = _ARTICLE_RE.match(line.text)
        if m:
            num = m.group("number")
            title = m.group("title")
            # If title is missing on current line, check if next line is a short title
            if not title and idx + 1 < len(lines):
                next_line = lines[idx + 1].text.strip()
                if (
                    next_line
                    and len(next_line) < 150
                    and not _ARTICLE_RE.match(next_line)
                    and not _CLAUSE_RE.match(next_line)
                    and not _POINT_RE.match(next_line)
                ):
                    title = next_line
                    consumed_lines.add(idx + 1)
            header = f"Điều {num}" + (f": {title.strip()}" if title else "")
            article_spans.append((idx, num, title.strip() if title else None, header))
            continue

        # Check multi-line: Line 1 = "Điều", Line 2 = "37: Tiêu đề"
        if _ARTICLE_ONLY_RE.match(line.text) and idx + 1 < len(lines):
            next_line = lines[idx + 1]
            m_next = _ARTICLE_NUMBER_RE.match(next_line.text)
            if m_next:
                num = m_next.group("number")
                title = m_next.group("title")
                consumed_lines.add(idx + 1)
                header = f"Điều {num}" + (f": {title.strip()}" if title else "")
                article_spans.append((idx, num, title.strip() if title else None, header))
                continue

    parent_chunks: list[ParentChunk] = []
    child_chunks: list[LegalChunk] = []

    # Case 1: No articles found in the document -> whole document is parent
    if not article_spans:
        parent_id = f"{doc.id}_doc"
        p_hash = sha256(passage.encode("utf-8")).hexdigest()[:16]
        parent = ParentChunk(
            parent_id=parent_id,
            document_id=doc.id,
            source_path=doc.source_path,
            source_member=doc.source_member,
            header=doc.name or f"Văn bản {doc.id}",
            article_number=None,
            raw_text=passage,
            start_offset=0,
            end_offset=len(passage),
            content_hash=p_hash,
        )
        parent_chunks.append(parent)

        windows = _window_split_text(passage, 0, max_chars, overlap_chars, min_chars)
        for c_idx, (w_start, w_end, w_text) in enumerate(windows):
            c_hash = sha256(w_text.encode("utf-8")).hexdigest()[:16]
            child_chunks.append(
                LegalChunk(
                    chunk_id=f"{doc.id}_{c_idx}",
                    document_id=doc.id,
                    parent_id=parent_id,
                    source_path=doc.source_path,
                    source_member=doc.source_member,
                    raw_text=w_text,
                    start_offset=w_start,
                    end_offset=w_end,
                    content_hash=c_hash,
                    section_label=doc.name or "Văn bản",
                    retrieval_text_override=f"[{doc.name}]\n{w_text}",
                )
            )
        return DocumentChunkResult(chunks=child_chunks, parents=parent_chunks)

    # Case 2: Articles found
    first_art_line_idx = article_spans[0][0]
    first_art_start = lines[first_art_line_idx].start

    # Handle preamble if non-empty text exists before the first article
    if first_art_start > 0:
        preamble_text = passage[:first_art_start]
        if preamble_text.strip():
            preamble_id = f"{doc.id}_preamble"
            preamble_hash = sha256(preamble_text.encode("utf-8")).hexdigest()[:16]
            preamble_parent = ParentChunk(
                parent_id=preamble_id,
                document_id=doc.id,
                source_path=doc.source_path,
                source_member=doc.source_member,
                header=f"Phần mở đầu - {doc.name}",
                article_number=None,
                raw_text=preamble_text,
                start_offset=0,
                end_offset=first_art_start,
                content_hash=preamble_hash,
            )
            parent_chunks.append(preamble_parent)

            # Child chunk for preamble
            windows = _window_split_text(
                preamble_text, 0, max_chars, overlap_chars, min_chars
            )
            for c_idx, (w_start, w_end, w_text) in enumerate(windows):
                c_hash = sha256(w_text.encode("utf-8")).hexdigest()[:16]
                child_chunks.append(
                    LegalChunk(
                        chunk_id=f"{preamble_id}_{c_idx}",
                        document_id=doc.id,
                        parent_id=preamble_id,
                        source_path=doc.source_path,
                        source_member=doc.source_member,
                        raw_text=w_text,
                        start_offset=w_start,
                        end_offset=w_end,
                        content_hash=c_hash,
                        section_label=f"{doc.name} / Phần mở đầu",
                        retrieval_text_override=f"[{doc.name}] [Phần mở đầu]\n{w_text}",
                    )
                )

    # Process each Article
    for a_idx, (line_idx, art_num, art_title, art_header) in enumerate(article_spans):
        art_start = lines[line_idx].start
        art_end = (
            lines[article_spans[a_idx + 1][0]].start
            if a_idx + 1 < len(article_spans)
            else len(passage)
        )
        art_raw = passage[art_start:art_end]
        art_hash = sha256(art_raw.encode("utf-8")).hexdigest()[:16]

        parent_id = f"{doc.id}_dieu_{art_num}"
        parent = ParentChunk(
            parent_id=parent_id,
            document_id=doc.id,
            source_path=doc.source_path,
            source_member=doc.source_member,
            header=art_header,
            article_number=art_num,
            raw_text=art_raw,
            start_offset=art_start,
            end_offset=art_end,
            content_hash=art_hash,
        )
        parent_chunks.append(parent)

        # Find clauses inside this article
        next_art_line = (
            article_spans[a_idx + 1][0] if a_idx + 1 < len(article_spans) else len(lines)
        )
        article_lines = lines[line_idx:next_art_line]

        clause_matches: list[tuple[int, str]] = []  # (rel_line_idx, clause_num)
        for r_idx, l in enumerate(article_lines):
            # Skip heading line(s)
            if r_idx == 0 or (line_idx + r_idx in consumed_lines):
                continue
            m_cl = _CLAUSE_RE.match(l.text)
            if m_cl:
                clause_matches.append((r_idx, m_cl.group("number")))

        # Context prefix for retrieval: [Document Name] [Article Header]
        doc_header = f"[{doc.name}] " if (doc.name and doc.name != doc.id) else ""
        context_base = f"{doc_header}[{art_header}]"

        # Case 2A: Article has numbered clauses (Khoản)
        if clause_matches:
            # Introductory text before the first clause (e.g. "Người nộp thuế có các quyền:")
            first_clause_rel_idx = clause_matches[0][0]
            heading_end_offset = (
                lines[line_idx + 1].end
                if (line_idx + 1 in consumed_lines)
                else lines[line_idx].end
            )
            first_cl_start_offset = article_lines[first_clause_rel_idx].start
            intro_text = passage[heading_end_offset:first_cl_start_offset].strip()

            context_full = (
                f"{context_base}\n{intro_text}" if intro_text else context_base
            )

            for c_idx, (r_idx, cl_num) in enumerate(clause_matches):
                cl_start = article_lines[r_idx].start
                cl_end = (
                    article_lines[clause_matches[c_idx + 1][0]].start
                    if c_idx + 1 < len(clause_matches)
                    else art_end
                )
                cl_raw = passage[cl_start:cl_end]
                cl_label = f"Điều {art_num} / Khoản {cl_num}"

                # If clause fits in max_chars, create single child chunk
                if len(cl_raw) <= max_chars:
                    c_hash = sha256(cl_raw.encode("utf-8")).hexdigest()[:16]
                    retrieval_text = f"{context_full}\n{cl_raw.strip()}"
                    child_chunks.append(
                        LegalChunk(
                            chunk_id=f"{parent_id}_k{cl_num}",
                            document_id=doc.id,
                            parent_id=parent_id,
                            source_path=doc.source_path,
                            source_member=doc.source_member,
                            raw_text=cl_raw,
                            start_offset=cl_start,
                            end_offset=cl_end,
                            content_hash=c_hash,
                            section_label=cl_label,
                            retrieval_text_override=retrieval_text,
                        )
                    )
                else:
                    # Check for Point (Điểm) sub-divisions: e.g. a), b), c)
                    next_cl_rel = (
                        clause_matches[c_idx + 1][0]
                        if c_idx + 1 < len(clause_matches)
                        else len(article_lines)
                    )
                    sub_lines = article_lines[r_idx:next_cl_rel]
                    point_matches: list[tuple[int, str]] = []  # (sub_idx, point_label)

                    for s_idx, sl in enumerate(sub_lines):
                        if s_idx == 0:
                            continue  # The clause heading line itself
                        m_pt = _POINT_RE.match(sl.text)
                        if m_pt:
                            point_matches.append((s_idx, m_pt.group("label")))

                    if point_matches:
                        # Split by Points
                        for p_idx, (s_idx, pt_label) in enumerate(point_matches):
                            pt_start = sub_lines[s_idx].start
                            pt_end = (
                                sub_lines[point_matches[p_idx + 1][0]].start
                                if p_idx + 1 < len(point_matches)
                                else cl_end
                            )
                            pt_raw = passage[pt_start:pt_end]
                            pt_section = f"{cl_label} / Điểm {pt_label}"
                            pt_hash = sha256(pt_raw.encode("utf-8")).hexdigest()[:16]
                            retrieval_text = (
                                f"{context_full}\n[Khoản {cl_num}]\n{pt_raw.strip()}"
                            )
                            child_chunks.append(
                                LegalChunk(
                                    chunk_id=f"{parent_id}_k{cl_num}_d{pt_label}",
                                    document_id=doc.id,
                                    parent_id=parent_id,
                                    source_path=doc.source_path,
                                    source_member=doc.source_member,
                                    raw_text=pt_raw,
                                    start_offset=pt_start,
                                    end_offset=pt_end,
                                    content_hash=pt_hash,
                                    section_label=pt_section,
                                    retrieval_text_override=retrieval_text,
                                )
                            )
                    else:
                        # Fallback sliding window for long clause
                        windows = _window_split_text(
                            cl_raw, cl_start, max_chars, overlap_chars, min_chars
                        )
                        for w_idx, (w_start, w_end, w_text) in enumerate(windows):
                            c_hash = sha256(w_text.encode("utf-8")).hexdigest()[:16]
                            retrieval_text = f"{context_full}\n{w_text.strip()}"
                            child_chunks.append(
                                LegalChunk(
                                    chunk_id=f"{parent_id}_k{cl_num}_{w_idx}",
                                    document_id=doc.id,
                                    parent_id=parent_id,
                                    source_path=doc.source_path,
                                    source_member=doc.source_member,
                                    raw_text=w_text,
                                    start_offset=w_start,
                                    end_offset=w_end,
                                    content_hash=c_hash,
                                    section_label=cl_label,
                                    retrieval_text_override=retrieval_text,
                                )
                            )

        # Case 2B: Article without numbered clauses (entire article is single unit)
        else:
            if len(art_raw) <= max_chars:
                child_chunks.append(
                    LegalChunk(
                        chunk_id=f"{parent_id}_c0",
                        document_id=doc.id,
                        parent_id=parent_id,
                        source_path=doc.source_path,
                        source_member=doc.source_member,
                        raw_text=art_raw,
                        start_offset=art_start,
                        end_offset=art_end,
                        content_hash=art_hash,
                        section_label=f"Điều {art_num}",
                        retrieval_text_override=f"{context_base}\n{art_raw.strip()}",
                    )
                )
            else:
                windows = _window_split_text(
                    art_raw, art_start, max_chars, overlap_chars, min_chars
                )
                for w_idx, (w_start, w_end, w_text) in enumerate(windows):
                    c_hash = sha256(w_text.encode("utf-8")).hexdigest()[:16]
                    retrieval_text = f"{context_base}\n{w_text.strip()}"
                    child_chunks.append(
                        LegalChunk(
                            chunk_id=f"{parent_id}_{w_idx}",
                            document_id=doc.id,
                            parent_id=parent_id,
                            source_path=doc.source_path,
                            source_member=doc.source_member,
                            raw_text=w_text,
                            start_offset=w_start,
                            end_offset=w_end,
                            content_hash=c_hash,
                            section_label=f"Điều {art_num}",
                            retrieval_text_override=retrieval_text,
                        )
                    )

    return DocumentChunkResult(chunks=child_chunks, parents=parent_chunks)


def _chunk_doc_worker(
    args: tuple[LegalDocument, int, int, int],
) -> DocumentChunkResult:
    """Worker function for multiprocessing pool."""
    doc, max_chars, overlap_chars, min_chars = args
    return chunk_document(
        doc,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        min_chars=min_chars,
    )


def chunk_corpus(
    documents: list[LegalDocument],
    *,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 100,
    num_workers: int = 30,
) -> ChunkCorpusResult:
    """Chunk all documents in the corpus using legal hierarchy and multi-core CPU workers."""
    all_chunks: list[LegalChunk] = []
    all_parents: list[ParentChunk] = []

    workers = max(1, min(num_workers, mp.cpu_count() or 1))

    if workers > 1 and len(documents) > 20:
        logger.info(
            "Chunking %d documents across %d CPU workers...",
            len(documents),
            workers,
        )
        ctx = mp.get_context("fork")
        chunksize = max(10, len(documents) // (workers * 4))
        payload = [(doc, max_chars, overlap_chars, min_chars) for doc in documents]
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_chunk_doc_worker, payload, chunksize=chunksize)
        for res in results:
            all_chunks.extend(res.chunks)
            all_parents.extend(res.parents)
    else:
        for doc in documents:
            doc_result = chunk_document(
                doc,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
            )
            all_chunks.extend(doc_result.chunks)
            all_parents.extend(doc_result.parents)

    return ChunkCorpusResult(chunks=all_chunks, parents=all_parents)
