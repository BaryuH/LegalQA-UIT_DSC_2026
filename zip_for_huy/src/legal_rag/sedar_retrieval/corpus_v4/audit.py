"""Exit-gate audit for a SEDAR corpus v4 build."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence

from .schema import CorpusV4Audit, DocumentRecord, RetrievalUnit

__all__ = ["audit_corpus"]

MICRO_UNIT_CHARS = 120


def audit_corpus(
    records: Sequence[DocumentRecord],
    units: Sequence[RetrievalUnit],
    *,
    micro_unit_chars: int = MICRO_UNIT_CHARS,
) -> CorpusV4Audit:
    """Compute the counters the v4 build is allowed to be judged on."""

    ids = [unit.unit_id for unit in units]
    duplicates = len(ids) - len(set(ids))

    per_document_article: Counter[tuple[str, int, str]] = Counter()
    for unit in units:
        if unit.level == "article" and unit.article_number:
            key = (unit.document_id, unit.instrument_index, unit.article_number)
            per_document_article[key] += 1
    duplicate_articles = sum(v - 1 for v in per_document_article.values() if v > 1)

    indexed = [unit for unit in units if unit.indexed]
    micro = [unit for unit in indexed if unit.char_count < micro_unit_chars]
    sizes = sorted(unit.char_count for unit in indexed) or [0]

    source_chars = sum(record.source_chars for record in records) or 1
    covered_chars = sum(record.covered_chars for record in records)
    with_metadata = sum(
        1 for record in records if record.doc_type and record.doc_number
    )
    documents_with_units = {unit.document_id for unit in units}

    return CorpusV4Audit(
        document_count=len(records),
        instrument_count=sum(record.instrument_count for record in records),
        unit_count=len(units),
        indexed_unit_count=len(indexed),
        article_unit_count=sum(1 for u in units if u.level == "article"),
        article_part_count=sum(1 for u in units if u.level == "article_part"),
        block_unit_count=sum(1 for u in units if u.level == "block"),
        preamble_unit_count=sum(1 for u in units if u.level == "preamble"),
        documents_without_units=sum(
            1 for record in records if record.document_id not in documents_with_units
        ),
        duplicate_unit_id_count=duplicates,
        duplicate_article_number_count=duplicate_articles,
        micro_unit_count=len(micro),
        micro_unit_rate=len(micro) / max(1, len(indexed)),
        text_coverage_rate=min(1.0, covered_chars / source_chars),
        metadata_recovery_rate=with_metadata / max(1, len(records)),
        median_unit_chars=int(statistics.median(sizes)),
        p90_unit_chars=sizes[min(len(sizes) - 1, int(len(sizes) * 0.9))],
        role_counts=dict(Counter(unit.role for unit in units)),
    )
