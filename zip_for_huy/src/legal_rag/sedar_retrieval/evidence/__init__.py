"""Evidence package for SEDAR Retrieval v3."""

from dataclasses import dataclass

from .adaptive_pack import (
    ADAPTIVE_PACK_SCHEMA_VERSION,
    AdaptivePackPolicy,
    AdaptivePackSelection,
    PackCandidate,
    SelectedBlock,
    select_adaptive_pack,
)
from .curation import (
    DEFAULT_BUDGETS,
    CandidateEvidence,
    EvidenceBlock,
    EvidencePack,
    curate_evidence,
)


@dataclass(frozen=True, slots=True)
class SufficiencyResult:
    sufficient: bool
    missing: tuple[str, ...]
    confidence: float
    followup_query: str | None = None
    reference_targets: tuple[str, ...] = ()


def fail_closed_sufficiency(*, error: str) -> SufficiencyResult:
    _ = error
    return SufficiencyResult(sufficient=True, missing=(), confidence=0.0)


__all__ = [
    "DEFAULT_BUDGETS",
    "CandidateEvidence",
    "EvidenceBlock",
    "EvidencePack",
    "SufficiencyResult",
    "curate_evidence",
    "fail_closed_sufficiency",
    "ADAPTIVE_PACK_SCHEMA_VERSION",
    "AdaptivePackPolicy",
    "AdaptivePackSelection",
    "PackCandidate",
    "SelectedBlock",
    "select_adaptive_pack",
]
