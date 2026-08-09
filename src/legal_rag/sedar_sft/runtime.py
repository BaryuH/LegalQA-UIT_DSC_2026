"""SS-19 finalizer + state machine (no critic cycle)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..schemas import PackedEvidence
from .analyzer import RequirementAnalysis, analyze_requirements
from .candidate import CandidatePair, maybe_generate_second_candidate
from .critic import CriticPlan, apply_critic_patch, plan_critic_patch
from .draft import GroundedDraft, attach_draft_attribution
from .evidence_profile import EvidenceProfile, build_evidence_profile
from .router import RiskProfile, route_risk
from .verifier import VerifierResult, verify_draft


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    answer_text: str
    accepted_claim_ids: tuple[str, ...]
    rejected_claim_ids: tuple[str, ...]
    route: str
    critic_calls: int
    candidate_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "accepted_claim_ids": list(self.accepted_claim_ids),
            "rejected_claim_ids": list(self.rejected_claim_ids),
            "route": self.route,
            "critic_calls": self.critic_calls,
            "candidate_count": self.candidate_count,
        }


@dataclass(frozen=True, slots=True)
class SedarRuntimeResult:
    evidence_profile: EvidenceProfile
    analysis: RequirementAnalysis
    draft: GroundedDraft
    verification: VerifierResult
    risk: RiskProfile
    critic_plan: CriticPlan | None
    candidates: CandidatePair
    final: FinalAnswer
    stages: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sedar_sft.ss19.runtime.v1",
            "stages": list(self.stages),
            "evidence_profile": self.evidence_profile.as_dict(),
            "analysis": self.analysis.as_dict(),
            "draft": self.draft.as_dict(),
            "verification": self.verification.as_dict(),
            "risk": self.risk.as_dict(),
            "critic_plan": None if self.critic_plan is None else self.critic_plan.as_dict(),
            "candidates": self.candidates.as_dict(),
            "final": self.final.as_dict(),
        }


def finalize_answer(
    draft: GroundedDraft,
    verification: VerifierResult,
    *,
    route: str,
    critic_calls: int,
    candidate_count: int,
) -> FinalAnswer:
    """Metric-aware evidence-preserving finalization; no unsupported inventions."""

    accepted = tuple(
        item.claim_id
        for item in verification.claim_support
        if item.label in {"supported", "weak", "unknown"}
    )
    rejected = tuple(
        item.claim_id
        for item in verification.claim_support
        if item.label == "unsupported"
    )
    answer = draft.answer_text
    if verification.hard_violation and not accepted:
        answer = (
            "Dựa trên evidence hiện có, chưa đủ căn cứ để khẳng định đầy đủ "
            "các nội dung bị phát hiện không được hỗ trợ."
        )
    return FinalAnswer(
        answer_text=answer,
        accepted_claim_ids=accepted,
        rejected_claim_ids=rejected,
        route=route,
        critic_calls=critic_calls,
        candidate_count=candidate_count,
    )


def run_sedar_runtime(
    *,
    question: str,
    evidence: PackedEvidence,
    generate_fn: Callable[[str], str],
    allow_second_candidate: bool = True,
) -> SedarRuntimeResult:
    """Execute the SEDAR state machine once; no cycle back to critic."""

    stages = [
        "VERIFY_MANIFEST_EXTERNAL",
        "RETRIEVE_B2_EXTERNAL",
        "PACK_EVIDENCE",
        "BUILD_EVIDENCE_PROFILE",
        "ANALYZE_REQUIREMENTS",
        "GENERATE_DRAFT",
        "VERIFY_DRAFT",
        "ROUTE",
    ]
    profile = build_evidence_profile(evidence)
    analysis = analyze_requirements(question)
    draft_text = generate_fn("draft")
    draft = attach_draft_attribution(
        draft_text, analysis=analysis, evidence_profile=profile
    )
    verification = verify_draft(
        draft, evidence_text=evidence.rendered_text, evidence_profile=profile
    )
    risk = route_risk(
        analysis=analysis,
        evidence_profile=profile,
        draft=draft,
        verification=verification,
    )
    critic_plan: CriticPlan | None = None
    critic_calls = 0
    working = draft
    working_verification = verification
    candidates = CandidatePair(draft, None, draft, 0.0)

    if risk.route == "critic_patch":
        stages.append("CRITIC_PATCH")
        critic_plan = plan_critic_patch(working, working_verification)
        working = apply_critic_patch(working, critic_plan)
        critic_calls = 1
        working_verification = verify_draft(
            working,
            evidence_text=evidence.rendered_text,
            evidence_profile=profile,
        )
        stages.append("REVERIFY")
    elif risk.route == "second_candidate" and allow_second_candidate:
        stages.append("SECOND_CANDIDATE")
        candidates = maybe_generate_second_candidate(
            question=question,
            evidence_text=evidence.rendered_text,
            analysis=analysis,
            evidence_profile=profile,
            primary=working,
            primary_verification=working_verification,
            generate_fn=generate_fn,
            enabled=True,
        )
        working = candidates.selected
        working_verification = verify_draft(
            working,
            evidence_text=evidence.rendered_text,
            evidence_profile=profile,
        )
        stages.append("VERIFY_B")
        # Optional one patch after B selection still counts toward max critic=1.
        if working_verification.hard_violation and critic_calls == 0:
            stages.append("CRITIC_PATCH")
            critic_plan = plan_critic_patch(working, working_verification)
            working = apply_critic_patch(working, critic_plan)
            critic_calls = 1
            working_verification = verify_draft(
                working,
                evidence_text=evidence.rendered_text,
                evidence_profile=profile,
            )
            stages.append("REVERIFY")

    stages.extend(["FINALIZE", "PREDICTION", "ARTIFACTS", "END"])
    final = finalize_answer(
        working,
        working_verification,
        route=risk.route,
        critic_calls=critic_calls,
        candidate_count=1 + (0 if candidates.secondary is None else 1),
    )
    return SedarRuntimeResult(
        evidence_profile=profile,
        analysis=analysis,
        draft=draft,
        verification=verification,
        risk=risk,
        critic_plan=critic_plan,
        candidates=candidates,
        final=final,
        stages=tuple(stages),
    )


__all__ = [
    "FinalAnswer",
    "SedarRuntimeResult",
    "finalize_answer",
    "run_sedar_runtime",
]
