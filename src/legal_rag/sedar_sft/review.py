"""SS-23 final adversarial review checklist (server-specific)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Severity = Literal["Critical", "High", "Medium", "Low"]


CHECKS: tuple[tuple[str, Severity, str], ...] = (
    ("python_310_used", "High", "project accidentally using Python 3.10"),
    ("system_python_polluted", "High", "package installed into system Python"),
    ("hf_cache_on_root", "Critical", "model cache filling root disk"),
    ("checkpoint_on_root", "Critical", "checkpoint path on root disk"),
    ("unexpected_cpu_fallback", "Critical", "unexpected CPU fallback"),
    ("bf16_without_probe", "High", "BF16 config without verified support"),
    ("silent_attention_fallback", "High", "silent attention fallback"),
    ("flashattn_import_hidden", "High", "FlashAttention import failure hidden"),
    ("gpu_occupied", "Critical", "GPU already occupied"),
    ("silent_oom_recovery", "Critical", "OOM auto-recovery changed config silently"),
    ("multiple_4b_copies", "Critical", "multiple 4B copies loaded simultaneously"),
    ("reranker_resident_during_train", "High", "reranker left resident during training"),
    ("retrieval_during_training", "Critical", "on-the-fly retrieval during training"),
    ("dataloader_starving_gpu", "Medium", "DataLoader starving GPU"),
    ("non_nvme_cache", "High", "non-NVMe training cache despite available NVMe"),
    ("missing_vram_telemetry", "High", "telemetry missing peak VRAM/throughput"),
)


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    check_id: str
    severity: Severity
    description: str
    status: str
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "description": self.description,
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class AdversarialReview:
    status: str
    findings: tuple[ReviewFinding, ...]

    @property
    def unresolved_critical_or_high(self) -> tuple[ReviewFinding, ...]:
        return tuple(
            item
            for item in self.findings
            if item.status == "open" and item.severity in {"Critical", "High"}
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sedar_sft.ss23.adversarial_review.v1",
            "status": self.status,
            "findings": [item.as_dict() for item in self.findings],
            "unresolved_critical_or_high": [
                item.as_dict() for item in self.unresolved_critical_or_high
            ],
        }


def build_adversarial_review(*, local_dev: bool = True) -> AdversarialReview:
    findings: list[ReviewFinding] = []
    for check_id, severity, description in CHECKS:
        if local_dev:
            status = "deferred_server"
            evidence = "Local scaffolding; re-evaluate on Ubuntu RTX4090 before promotion"
        else:
            status = "open"
            evidence = "Needs server evidence"
        findings.append(
            ReviewFinding(check_id, severity, description, status, evidence)
        )
    status = (
        "pass_local_template"
        if local_dev
        else ("blocked" if any(f.status == "open" and f.severity in {"Critical", "High"} for f in findings) else "pass")
    )
    return AdversarialReview(status=status, findings=tuple(findings))


def write_adversarial_review(path: str | Path, review: AdversarialReview | None = None) -> Path:
    payload = review or build_adversarial_review()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


__all__ = [
    "AdversarialReview",
    "CHECKS",
    "ReviewFinding",
    "build_adversarial_review",
    "write_adversarial_review",
]
