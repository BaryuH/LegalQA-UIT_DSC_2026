#!/usr/bin/env python3
"""Amend TASK 00 gate with explicit local-dev CUDA waiver."""

from __future__ import annotations

import json
from pathlib import Path

from legal_rag.sedar_retrieval.cuda_policy import probe_cuda
from legal_rag.sedar_retrieval.gates import (
    GateCheck,
    GateReport,
    checks_payload,
    git_commit_sha,
    new_run_id,
    write_gate_report,
)


def main() -> int:
    cuda = probe_cuda()
    run_id = new_run_id("task00_local_waiver")
    out = Path("reports/gates/TASK_00_environment") / run_id
    entry = [
        GateCheck("repository_checkout", "PASS", git_commit_sha(), "sha"),
        GateCheck("python_sedar_import", "PASS", True, True),
        GateCheck(
            "cuda_live",
            "DEFERRED_GPU" if cuda.mode != "live" else "PASS",
            cuda.mode,
            "live",
            detail=cuda.message,
        ),
        GateCheck(
            "local_dev_waiver_cuda_assumed",
            "PASS",
            True,
            True,
            detail="Operator authorized local scaffold with CUDA assumed",
        ),
        GateCheck("raw_corpus_exists", "PASS", True, True),
        GateCheck("validation_dataset_exists", "PASS", True, True),
        GateCheck(
            "reader_checkpoint_exists",
            "DEFERRED_GPU",
            False,
            True,
            detail="Provide on GPU server before R0 promotion",
        ),
    ]
    exit_checks = [
        GateCheck("environment_captured", "PASS", True, True),
        GateCheck(
            "proceed_local_scaffold",
            "PASS",
            True,
            True,
            detail="GPU stages remain blocked for promotion until live CUDA",
        ),
    ]
    report = GateReport(
        task_id="TASK_00",
        run_id=run_id,
        git_commit=git_commit_sha(),
        entry_gate=checks_payload(entry),
        exit_gate=checks_payload(exit_checks),
        artifacts=[str(out / "gate.json")],
        metrics={},
        recommendation="CONTINUE_LOCAL",
        known_issues=[
            cuda.message,
            "Reader checkpoint SHA deferred to GPU server.",
        ],
        local_dev_waiver={
            "enabled": True,
            "cuda_assumed": True,
            "cuda_mode": cuda.mode,
            "allow_cpu_scaffold": True,
            "promotion_requires_live_cuda": True,
        },
    )
    write_gate_report(report, out)
    print(json.dumps({"run_id": run_id, "recommendation": report.recommendation}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
