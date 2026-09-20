"""SS-13 Grounded draft attribution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .analyzer import RequirementAnalysis
from .evidence_profile import EvidenceProfile


@dataclass(frozen=True, slots=True)
class DraftClaim:
    claim_id: str
    text: str
    claim_type: str
    requirement_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type,
            "requirement_ids": list(self.requirement_ids),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True, slots=True)
class GroundedDraft:
    answer_text: str
    claims: tuple[DraftClaim, ...]
    candidate_id: str = "A"

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "candidate_id": self.candidate_id,
            "claims": [claim.as_dict() for claim in self.claims],
        }


def attach_draft_attribution(
    answer_text: str,
    *,
    analysis: RequirementAnalysis,
    evidence_profile: EvidenceProfile,
    candidate_id: str = "A",
) -> GroundedDraft:
    """Attach minimal operational attribution (not chain-of-thought)."""

    if not answer_text.strip():
        raise ValueError("Draft answer_text must be non-blank")
    evidence_ids = evidence_profile.evidence_ids or ("E1",)
    claims = tuple(
        DraftClaim(
            claim_id=f"C{index + 1}",
            text=answer_text.strip() if index == 0 else requirement.type,
            claim_type=requirement.type,
            requirement_ids=(requirement.requirement_id,),
            evidence_ids=evidence_ids[:1],
        )
        for index, requirement in enumerate(analysis.requirements)
    )
    return GroundedDraft(
        answer_text=answer_text.strip(),
        claims=claims,
        candidate_id=candidate_id,
    )


__all__ = ["DraftClaim", "GroundedDraft", "attach_draft_attribution"]
