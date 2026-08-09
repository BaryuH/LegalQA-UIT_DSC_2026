"""SS-14/SS-15 layered verifier (hard checks + lexical semantic support)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .draft import GroundedDraft
from .evidence_profile import EvidenceProfile

SupportLabel = Literal["supported", "weak", "unsupported", "unknown"]


@dataclass(frozen=True, slots=True)
class HardFinding:
    code: str
    message: str
    severity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class ClaimSupport:
    claim_id: str
    label: SupportLabel
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "label": self.label,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class VerifierResult:
    hard_findings: tuple[HardFinding, ...]
    claim_support: tuple[ClaimSupport, ...]
    hard_violation: bool
    semantic_mode: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "hard_findings": [item.as_dict() for item in self.hard_findings],
            "claim_support": [item.as_dict() for item in self.claim_support],
            "hard_violation": self.hard_violation,
            "semantic_mode": self.semantic_mode,
        }


_LEGAL_ID_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐđ]{1,10}(?:-[A-ZĐđ]+)?\b")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")


def verify_draft(
    draft: GroundedDraft,
    *,
    evidence_text: str,
    evidence_profile: EvidenceProfile,
    semantic_mode: str = "lexical_overlap",
) -> VerifierResult:
    """Hard checks always on; semantic support is lexical until model-backed gate."""

    findings: list[HardFinding] = []
    answer = draft.answer_text
    if evidence_profile.dropped_evidence_ids or evidence_profile.truncated_evidence_ids:
        findings.append(
            HardFinding(
                "EVIDENCE_DROP_OR_TRUNCATION",
                "Packed evidence reports dropped/truncated chunks",
                "medium",
            )
        )
    for legal_id in _LEGAL_ID_RE.findall(answer):
        if legal_id not in evidence_text:
            findings.append(
                HardFinding(
                    "UNSUPPORTED_LEGAL_IDENTIFIER",
                    f"Answer cites {legal_id} absent from evidence",
                    "high",
                )
            )
    for date in _DATE_RE.findall(answer):
        if date not in evidence_text:
            findings.append(
                HardFinding(
                    "UNSUPPORTED_DATE",
                    f"Answer date {date} absent from evidence",
                    "high",
                )
            )
    if (
        evidence_profile.temporal_status == "historical"
        and any(token in answer.casefold() for token in ("hiện nay", "hiện hành"))
    ):
        findings.append(
            HardFinding(
                "TEMPORAL_CONFLICT",
                "Answer asserts current status against historical evidence markers",
                "high",
            )
        )
    paragraphs = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(paragraphs) != len(set(paragraphs)):
        findings.append(
            HardFinding("DUPLICATE_PARAGRAPH", "Duplicate answer paragraphs", "low")
        )

    support: list[ClaimSupport] = []
    evidence_tokens = set(evidence_text.casefold().split())
    for claim in draft.claims:
        claim_tokens = set(claim.text.casefold().split())
        if not claim_tokens:
            label: SupportLabel = "unknown"
            note = "empty claim"
        else:
            overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
            if overlap >= 0.5:
                label = "supported"
                note = f"overlap={overlap:.2f}"
            elif overlap >= 0.2:
                label = "weak"
                note = f"overlap={overlap:.2f}"
            else:
                label = "unsupported"
                note = f"overlap={overlap:.2f}"
        support.append(ClaimSupport(claim.claim_id, label, note))

    hard_violation = any(item.severity == "high" for item in findings)
    return VerifierResult(
        hard_findings=tuple(findings),
        claim_support=tuple(support),
        hard_violation=hard_violation,
        semantic_mode=semantic_mode,
    )


__all__ = [
    "ClaimSupport",
    "HardFinding",
    "SupportLabel",
    "VerifierResult",
    "verify_draft",
]
