#!/usr/bin/env python3
"""Emit SS-08..SS-23 local planning artifacts (no GPU execution)."""

from __future__ import annotations

from pathlib import Path

from legal_rag.sedar_sft.ablation import write_ablation_plan
from legal_rag.sedar_sft.observability import GpuObservability, write_observability
from legal_rag.sedar_sft.preflight import run_canonical_train_preflight, write_preflight
from legal_rag.sedar_sft.promotion import write_promotion_freeze
from legal_rag.sedar_sft.review import write_adversarial_review


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    preflight = run_canonical_train_preflight(
        root / "configs" / "sedar_sft_train.yaml",
        repo_root=root,
        authorize_gpu_execution=False,
    )
    write_preflight(root / "artifacts/sedar_sft/hardware/ss08_preflight.json", preflight)
    write_observability(
        root / "artifacts/sedar_sft/hardware/ss20_observability_template.json",
        GpuObservability(),
    )
    write_ablation_plan(root / "artifacts/sedar_sft/eval/ss21_ablation_plan.json")
    write_promotion_freeze(root / "artifacts/sedar_sft/eval/ss22_promotion_freeze.json")
    write_adversarial_review(root / "artifacts/sedar_sft/eval/ss23_adversarial_review.json")
    print("wrote SS-08..SS-23 local artifacts")
    print("preflight_status", preflight.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
