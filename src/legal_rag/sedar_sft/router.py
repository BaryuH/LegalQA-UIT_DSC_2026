"""SS-16 RiskProfile heuristic router (no self-reported confidence primary)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .analyzer import RequirementAnalysis
from .draft import GroundedDraft
from .evidence_profile import EvidenceProfile
from .verifier import VerifierResult

Route = Literal["accept", "critic_patch", "second_candidate"]


@dataclass(frozen=True, slots=True)
class RiskProfile:
    hard_violation: bool
    retrieval_uncertainty: float
    requirement_gap: float
    unsupported_claim_risk: float
    citation_risk: float
    temporal_risk: float
    truncation_risk: float
    verbosity_risk: float
    candidate_disagreement: float
    route: Route

    def as_dict(self) -> dict[str, Any]:
        return {
            "hard_violation": self.hard_violation,
            "retrieval_uncertainty": self.retrieval_uncertainty,
            "requirement_gap": self.requirement_gap,
            "unsupported_claim_risk": self.unsupported_claim_risk,
            "citation_risk": self.citation_risk,
            "temporal_risk": self.temporal_risk,
            "truncation_risk": self.truncation_risk,
            "verbosity_risk": self.verbosity_risk,
            "candidate_disagreement": self.candidate_disagreement,
            "route": self.route,
        }


def route_risk(
    *,
    analysis: RequirementAnalysis,
    evidence_profile: EvidenceProfile,
    draft: GroundedDraft,
    verification: VerifierResult,
    candidate_disagreement: float = 0.0,
) -> RiskProfile:
    unsupported = sum(
        1 for item in verification.claim_support if item.label == "unsupported"
    )
    weak = sum(1 for item in verification.claim_support if item.label == "weak")
    total = max(len(verification.claim_support), 1)
    unsupported_risk = unsupported / total
    citation_risk = (
        1.0
        if any(
            item.code.startswith("UNSUPPORTED_") for item in verification.hard_findings
        )
        else 0.0
    )
    temporal_risk = (
        1.0
        if any(item.code == "TEMPORAL_CONFLICT" for item in verification.hard_findings)
        else (0.2 if evidence_profile.temporal_status == "unknown" else 0.0)
    )
    truncation_risk = (
        0.5
        if evidence_profile.dropped_evidence_ids
        or evidence_profile.truncated_evidence_ids
        else 0.0
    )
    retrieval_uncertainty = min(1.0, 0.1 * len(evidence_profile.dropped_evidence_ids))
    requirement_gap = 0.3 if analysis.uncertain_fields else 0.0
    verbosity_risk = 0.2 if len(draft.answer_text) > 2500 else 0.0

    if verification.hard_violation or citation_risk >= 1.0:
        route: Route = "critic_patch"
    elif unsupported_risk >= 0.5 or candidate_disagreement >= 0.4:
        route = "second_candidate"
    elif (
        unsupported_risk > 0 or weak > 0 or requirement_gap > 0 or temporal_risk >= 0.5
    ):
        route = "critic_patch"
    else:
        route = "accept"

    return RiskProfile(
        hard_violation=verification.hard_violation,
        retrieval_uncertainty=retrieval_uncertainty,
        requirement_gap=requirement_gap,
        unsupported_claim_risk=unsupported_risk,
        citation_risk=citation_risk,
        temporal_risk=temporal_risk,
        truncation_risk=truncation_risk,
        verbosity_risk=verbosity_risk,
        candidate_disagreement=candidate_disagreement,
        route=route,
    )


__all__ = ["RiskProfile", "Route", "route_risk"]
