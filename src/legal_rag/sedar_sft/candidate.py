"""SS-18 adaptive second candidate (max 2; sequential on one model)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .analyzer import RequirementAnalysis
from .draft import GroundedDraft, attach_draft_attribution
from .evidence_profile import EvidenceProfile
from .verifier import VerifierResult, verify_draft


@dataclass(frozen=True, slots=True)
class CandidatePair:
    primary: GroundedDraft
    secondary: GroundedDraft | None
    selected: GroundedDraft
    disagreement: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.as_dict(),
            "secondary": None if self.secondary is None else self.secondary.as_dict(),
            "selected": self.selected.as_dict(),
            "disagreement": self.disagreement,
        }


def _token_jaccard(left: str, right: str) -> float:
    a = set(left.casefold().split())
    b = set(right.casefold().split())
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / max(len(a | b), 1))


def maybe_generate_second_candidate(
    *,
    question: str,
    evidence_text: str,
    analysis: RequirementAnalysis,
    evidence_profile: EvidenceProfile,
    primary: GroundedDraft,
    primary_verification: VerifierResult,
    generate_fn: Callable[[str], str] | None,
    enabled: bool,
) -> CandidatePair:
    """Generate candidate B only when routed; never parallel multi-model loads."""

    del question  # reserved for future diversity prompts
    if not enabled or generate_fn is None:
        return CandidatePair(primary, None, primary, 0.0)
    secondary_text = generate_fn("candidate_b")
    secondary = attach_draft_attribution(
        secondary_text,
        analysis=analysis,
        evidence_profile=evidence_profile,
        candidate_id="B",
    )
    secondary_verification = verify_draft(
        secondary,
        evidence_text=evidence_text,
        evidence_profile=evidence_profile,
    )
    disagreement = _token_jaccard(primary.answer_text, secondary.answer_text)
    # Prefer fewer high hard findings.
    primary_score = (
        int(primary_verification.hard_violation),
        sum(1 for item in primary_verification.claim_support if item.label == "unsupported"),
    )
    secondary_score = (
        int(secondary_verification.hard_violation),
        sum(
            1
            for item in secondary_verification.claim_support
            if item.label == "unsupported"
        ),
    )
    selected = secondary if secondary_score < primary_score else primary
    return CandidatePair(primary, secondary, selected, disagreement)


__all__ = ["CandidatePair", "maybe_generate_second_candidate"]
