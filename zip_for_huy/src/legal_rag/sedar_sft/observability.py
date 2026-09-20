"""SS-20 GPU/run observability artifact helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuObservability:
    gpu_name: str | None = None
    gpu_compute_capability: tuple[int, int] | None = None
    driver_version: str | None = None
    torch_cuda_version: str | None = None
    attention_backend: str = "sdpa"
    quantization: str = "nf4"
    dtype: str = "bfloat16"
    peak_vram_allocated: float | None = None
    peak_vram_reserved: float | None = None
    avg_generation_latency_s: float | None = None
    median_generation_latency_s: float | None = None
    tokens_per_sec: float | None = None
    samples_per_sec: float | None = None
    critic_calls: int = 0
    candidate_b_calls: int = 0
    micro_batch: int | None = None
    gradient_accumulation: int | None = None
    effective_batch: int | None = None
    sequence_length: int | None = None
    gradient_checkpointing: bool | None = None
    dataloader_workers: str | int | None = "AUTO"
    pin_memory: bool | None = True
    persistent_workers: bool | None = True
    checkpoint_duration_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sedar_sft.ss20.observability.v1",
            "gpu_name": self.gpu_name,
            "gpu_compute_capability": (
                list(self.gpu_compute_capability)
                if self.gpu_compute_capability is not None
                else None
            ),
            "driver_version": self.driver_version,
            "torch_cuda_version": self.torch_cuda_version,
            "attention_backend": self.attention_backend,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "peak_vram_allocated": self.peak_vram_allocated,
            "peak_vram_reserved": self.peak_vram_reserved,
            "avg_generation_latency_s": self.avg_generation_latency_s,
            "median_generation_latency_s": self.median_generation_latency_s,
            "tokens_per_sec": self.tokens_per_sec,
            "samples_per_sec": self.samples_per_sec,
            "critic_calls": self.critic_calls,
            "candidate_b_calls": self.candidate_b_calls,
            "training": {
                "micro_batch": self.micro_batch,
                "gradient_accumulation": self.gradient_accumulation,
                "effective_batch": self.effective_batch,
                "sequence_length": self.sequence_length,
                "gradient_checkpointing": self.gradient_checkpointing,
                "dataloader_workers": self.dataloader_workers,
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
                "checkpoint_duration_s": self.checkpoint_duration_s,
            },
            "forbidden": ["secrets", "chain_of_thought", "gold_in_inference_logs"],
        }


def write_observability(path: str | Path, payload: GpuObservability) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


__all__ = ["GpuObservability", "write_observability"]
