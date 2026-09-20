"""SS-22 promotion freeze helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PromotionFreeze:
    status: str
    decision: str
    execution_profile: dict[str, Any]
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sedar_sft.ss22.promotion.v1",
            "status": self.status,
            "decision": self.decision,
            "execution_profile": self.execution_profile,
            "blockers": list(self.blockers),
            "rule": (
                "Promote only if SEDAR-SFT beats SFT-only under frozen B2 "
                "+ approved evaluator"
            ),
        }


def build_promotion_freeze(*, authorize_promote: bool = False) -> PromotionFreeze:
    profile = {
        "python_version": "UNRESOLVED_SERVER",
        "torch_version": "UNRESOLVED_SERVER",
        "cuda_build": "UNRESOLVED_SERVER",
        "transformers": "UNRESOLVED_SERVER",
        "peft": "UNRESOLVED_SERVER",
        "trl": "UNRESOLVED_SERVER",
        "bitsandbytes": "UNRESOLVED_SERVER",
        "attention_backend": "sdpa",
        "dtype": "bfloat16_if_supported",
        "quantization": "nf4_double_quant",
        "sequence_length": "UNRESOLVED_SS06",
        "micro_batch": "UNRESOLVED_SS04D",
        "gradient_accumulation": "AUTO",
        "dataloader_workers": "AUTO",
    }
    blockers = []
    if not authorize_promote:
        blockers.extend(
            [
                "canonical_training_not_completed",
                "sft_vs_sedar_ablation_missing",
                "server_execution_profile_unresolved",
                "adversarial_review_open",
            ]
        )
    decision = "PROMOTE" if authorize_promote and not blockers else "HOLD"
    return PromotionFreeze(
        status="frozen_template" if decision == "HOLD" else "promoted",
        decision=decision,
        execution_profile=profile,
        blockers=tuple(blockers),
    )


def write_promotion_freeze(
    path: str | Path, payload: PromotionFreeze | None = None
) -> Path:
    report = payload or build_promotion_freeze()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


__all__ = ["PromotionFreeze", "build_promotion_freeze", "write_promotion_freeze"]
