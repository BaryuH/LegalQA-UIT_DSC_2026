"""Evidence-pack quality metrics for SEDAR Retrieval v3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidenceBlockView:
    query_id: str
    block_ids: tuple[str, ...]
    token_count: int
    contains_synthetic_summary: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceQualityBundle:
    n_queries: int
    avg_blocks: float
    avg_tokens: float
    duplicate_block_rate: float
    synthetic_summary_rate: float


def evaluate_evidence_packs(
    packs: Sequence[EvidenceBlockView],
) -> EvidenceQualityBundle:
    if not packs:
        return EvidenceQualityBundle(
            n_queries=0,
            avg_blocks=0.0,
            avg_tokens=0.0,
            duplicate_block_rate=0.0,
            synthetic_summary_rate=0.0,
        )

    block_counts: list[float] = []
    token_counts: list[float] = []
    dup_rates: list[float] = []
    synthetic: list[float] = []
    for pack in packs:
        block_counts.append(float(len(pack.block_ids)))
        token_counts.append(float(pack.token_count))
        if pack.block_ids:
            dup_rates.append(1.0 - (len(set(pack.block_ids)) / len(pack.block_ids)))
        else:
            dup_rates.append(0.0)
        synthetic.append(1.0 if pack.contains_synthetic_summary else 0.0)

    n = len(packs)
    return EvidenceQualityBundle(
        n_queries=n,
        avg_blocks=sum(block_counts) / n,
        avg_tokens=sum(token_counts) / n,
        duplicate_block_rate=sum(dup_rates) / n,
        synthetic_summary_rate=sum(synthetic) / n,
    )
