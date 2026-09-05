"""SS-10 SFT-only inference baseline harness (GPU benchmarks deferred)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import PackedEvidence


@dataclass(frozen=True, slots=True)
class InferenceBenchmarkCase:
    case_id: str
    prompt_tokens_est: int
    generated_tokens_est: int
    latency_s: float
    answer: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prompt_tokens_est": self.prompt_tokens_est,
            "generated_tokens_est": self.generated_tokens_est,
            "latency_s": self.latency_s,
            "answer": self.answer,
        }


@dataclass(frozen=True, slots=True)
class SftOnlyBaselineReport:
    status: str
    cases: tuple[InferenceBenchmarkCase, ...]
    cases_per_sec: float | None
    tokens_per_sec: float | None
    peak_vram_mib: float | None
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sedar_sft.ss10.sft_only_baseline.v1",
            "status": self.status,
            "cases": [item.as_dict() for item in self.cases],
            "cases_per_sec": self.cases_per_sec,
            "tokens_per_sec": self.tokens_per_sec,
            "peak_vram_mib": self.peak_vram_mib,
            "notes": list(self.notes),
            "gpu_deferred": [
                "batched generation on RTX4090",
                "peak VRAM measurement",
                "unload reranker before generate",
            ],
        }


def run_sft_only_baseline(
    cases: Sequence[tuple[str, str, PackedEvidence]],
    *,
    generate_fn: Callable[[str, PackedEvidence], str],
    peak_vram_mib: float | None = None,
) -> SftOnlyBaselineReport:
    """Run SFT-only answers (no SEDAR repair) for comparison control."""

    results: list[InferenceBenchmarkCase] = []
    started = time.perf_counter()
    total_gen_tokens = 0
    for case_id, question, evidence in cases:
        t0 = time.perf_counter()
        answer = generate_fn(question, evidence)
        latency = time.perf_counter() - t0
        prompt_est = len(question.split()) + len(evidence.rendered_text.split())
        gen_est = max(len(answer.split()), 1)
        total_gen_tokens += gen_est
        results.append(
            InferenceBenchmarkCase(
                case_id=case_id,
                prompt_tokens_est=prompt_est,
                generated_tokens_est=gen_est,
                latency_s=latency,
                answer=answer,
            )
        )
    elapsed = max(time.perf_counter() - started, 1e-9)
    status = "pass_local_mock" if results else "empty"
    notes: tuple[str, ...] = (
        "SFT-only baseline uses one generator path; SEDAR runtime is disabled.",
        "Token counts are whitespace estimates until SS-04C tokenizer is locked.",
    )
    if peak_vram_mib is None:
        notes = (*notes, "peak_vram_mib unresolved until GPU run.")
    return SftOnlyBaselineReport(
        status=status,
        cases=tuple(results),
        cases_per_sec=len(results) / elapsed,
        tokens_per_sec=total_gen_tokens / elapsed,
        peak_vram_mib=peak_vram_mib,
        notes=notes,
    )


def write_baseline_report(path: str | Path, report: SftOnlyBaselineReport) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


__all__ = [
    "InferenceBenchmarkCase",
    "SftOnlyBaselineReport",
    "run_sft_only_baseline",
    "write_baseline_report",
]
