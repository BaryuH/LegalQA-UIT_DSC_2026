"""SS-21 controlled ablation matrix + efficiency metric schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_ABLATIONS = (
    "SFT_only",
    "SEDAR_SFT_full",
    "SEDAR_no_critic",
    "SEDAR_no_candidate_b",
    "QLoRA_SDPA",
    "QLoRA_FlashAttention2_if_promoted",
    "LoRA_BF16_if_ss04d_fits",
)


@dataclass(frozen=True, slots=True)
class AblationPlan:
    quality_metrics: tuple[str, ...]
    efficiency_metrics: tuple[str, ...]
    arms: tuple[str, ...]
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sedar_sft.ss21.ablation.v1",
            "status": self.status,
            "quality_metrics": list(self.quality_metrics),
            "efficiency_metrics": list(self.efficiency_metrics),
            "arms": list(self.arms),
            "rule": (
                "Do not mix kernel-performance comparisons with "
                "architecture-quality conclusions."
            ),
            "gpu_deferred": True,
        }


def build_ablation_plan() -> AblationPlan:
    return AblationPlan(
        quality_metrics=("meteor", "rouge_l"),
        efficiency_metrics=(
            "training_tokens_per_sec",
            "inference_generated_tokens_per_sec",
            "cases_per_sec",
            "peak_vram",
            "mean_latency",
            "p95_latency",
            "avg_model_calls_per_case",
            "gpu_time_per_1000_cases",
            "delta_meteor_per_additional_model_call",
            "delta_meteor_per_gpu_second",
        ),
        arms=REQUIRED_ABLATIONS,
        status="planned_pending_gpu_runs",
    )


def write_ablation_plan(path: str | Path, plan: AblationPlan | None = None) -> Path:
    payload = (plan or build_ablation_plan()).as_dict()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


__all__ = [
    "AblationPlan",
    "REQUIRED_ABLATIONS",
    "build_ablation_plan",
    "write_ablation_plan",
]
