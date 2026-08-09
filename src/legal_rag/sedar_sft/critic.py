"""SS-17 structured one-pass critic (patch-only, max one call)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .draft import GroundedDraft
from .verifier import VerifierResult

ALLOWED_OPS = frozenset(
    {
        "REMOVE_UNSUPPORTED_CLAIM",
        "REMOVE_DUPLICATE",
        "REPAIR_CITATION_FROM_EVIDENCE",
        "REPAIR_NUMERIC_VALUE_FROM_EVIDENCE",
        "ADD_SUPPORTED_MISSING_REQUIREMENT",
        "REORDER_CURRENT_BEFORE_HISTORICAL",
        "SHORTEN_REDUNDANT_TEXT",
        "NO_CHANGE",
    }
)


@dataclass(frozen=True, slots=True)
class PatchOperation:
    type: str
    claim_id: str | None = None
    evidence_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.claim_id is not None:
            payload["claim_id"] = self.claim_id
        if self.evidence_ids:
            payload["evidence_ids"] = list(self.evidence_ids)
        return payload


@dataclass(frozen=True, slots=True)
class CriticPlan:
    action: Literal["PATCH", "NO_CHANGE"]
    operations: tuple[PatchOperation, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "operations": [op.as_dict() for op in self.operations],
        }


def plan_critic_patch(
    draft: GroundedDraft,
    verification: VerifierResult,
) -> CriticPlan:
    """Produce an allowlisted patch plan; never a second unconstrained solver."""

    operations: list[PatchOperation] = []
    for support in verification.claim_support:
        if support.label == "unsupported":
            operations.append(
                PatchOperation("REMOVE_UNSUPPORTED_CLAIM", claim_id=support.claim_id)
            )
    for finding in verification.hard_findings:
        if finding.code == "DUPLICATE_PARAGRAPH":
            operations.append(PatchOperation("REMOVE_DUPLICATE"))
        if finding.code.startswith("UNSUPPORTED_"):
            claim_id = draft.claims[0].claim_id if draft.claims else None
            evidence_ids = draft.claims[0].evidence_ids if draft.claims else ()
            operations.append(
                PatchOperation(
                    "REPAIR_CITATION_FROM_EVIDENCE",
                    claim_id=claim_id,
                    evidence_ids=evidence_ids,
                )
            )
    # Additive ops must carry evidence IDs; omit when none available.
    filtered = []
    for op in operations:
        if op.type not in ALLOWED_OPS:
            continue
        if op.type.startswith("ADD_") and not op.evidence_ids:
            continue
        filtered.append(op)
    if not filtered:
        return CriticPlan("NO_CHANGE", (PatchOperation("NO_CHANGE"),))
    return CriticPlan("PATCH", tuple(filtered))


def apply_critic_patch(draft: GroundedDraft, plan: CriticPlan) -> GroundedDraft:
    """Apply one-pass removals; does not invent unsupported claims."""

    if plan.action == "NO_CHANGE":
        return draft
    remove_ids = {
        op.claim_id
        for op in plan.operations
        if op.type == "REMOVE_UNSUPPORTED_CLAIM" and op.claim_id
    }
    claims = tuple(claim for claim in draft.claims if claim.claim_id not in remove_ids)
    answer = draft.answer_text
    if remove_ids and claims:
        answer = claims[0].text
    elif remove_ids and not claims:
        answer = "Không đủ căn cứ trong evidence để khẳng định nội dung bị loại."
    return GroundedDraft(
        answer_text=answer,
        claims=claims,
        candidate_id=draft.candidate_id,
    )


__all__ = [
    "ALLOWED_OPS",
    "CriticPlan",
    "PatchOperation",
    "apply_critic_patch",
    "plan_critic_patch",
]
