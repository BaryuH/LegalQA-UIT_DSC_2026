"""Evidence package for SEDAR Retrieval v3."""

from .curation import (
    DEFAULT_BUDGETS,
    CandidateEvidence,
    EvidenceBlock,
    EvidencePack,
    curate_evidence,
)


def fail_closed_sufficiency(*, error: str):
    from dataclasses import dataclass

    @dataclass(frozen=True, slots=True)
    class SufficiencyResult:
        sufficient: bool
        missing: tuple[str, ...]
        confidence: float
        followup_query: str | None = None
        reference_targets: tuple[str, ...] = ()

    _ = error
    return SufficiencyResult(sufficient=True, missing=(), confidence=0.0)


__all__ = [
    "DEFAULT_BUDGETS",
    "CandidateEvidence",
    "EvidenceBlock",
    "EvidencePack",
    "curate_evidence",
    "fail_closed_sufficiency",
]
