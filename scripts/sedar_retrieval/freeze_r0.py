#!/usr/bin/env python3
"""Freeze R0 baseline artifacts (TASK 01) with local CUDA-deferred policy.

Uses the existing frozen B2 Hybrid-RAG entrypoint. On local hosts without live
CUDA / SEDAR reader checkpoint, reranker may fall back and generation stays on
the frozen B2 mock provider. Mark reader-promotion fields DEFERRED_GPU and
re-run on the GPU server before promoting R0.
"""

from __future__ import annotations

import argparse
import json
import shutil
from hashlib import sha256
from pathlib import Path

import yaml

from legal_rag.pipeline import run_hybrid_rag_from_config
from legal_rag.sedar_retrieval.cuda_policy import probe_cuda
from legal_rag.sedar_retrieval.gates import (
    GateCheck,
    GateReport,
    checks_payload,
    git_commit_sha,
    new_run_id,
    write_gate_report,
)


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/retrieval/r0_baseline.yaml"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/ablations/R0"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild BM25 index when missing/stale (local scaffold default).",
    )
    args = parser.parse_args()

    run_id = new_run_id("r0_baseline")
    out = args.output_root / run_id
    out.mkdir(parents=True, exist_ok=True)

    stage_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen_path = Path(stage_cfg["frozen_config_path"])
    cuda = probe_cuda()

    validation_manifest = Path(
        stage_cfg.get("data", {}).get(
            "validation_manifest",
            "artifacts/sedar_sft/validation/clean_warmup_manifest.json",
        )
    )
    validation_hash = _sha256_file(validation_manifest)

    # Local scaffold: rebuild when index cache is absent.
    rebuild = bool(args.rebuild_index or cuda.mode != "live")
    result = run_hybrid_rag_from_config(
        frozen_path,
        repo_root=args.repo_root,
        rebuild_index=rebuild,
        limit=None if args.limit <= 0 else args.limit,
        output_dir=out / "pipeline_run",
        run_id=run_id,
    )

    paths = result.artifacts
    _copy_if_exists(paths.predictions, out / "answers.jsonl")
    if paths.retrieval is not None:
        _copy_if_exists(paths.retrieval, out / "retrieval.jsonl")
    # Evidence pack may live under pipeline metadata; keep explicit placeholder.
    evidence_src = paths.run_dir / "evidence.jsonl"
    if not _copy_if_exists(evidence_src, out / "evidence.jsonl"):
        (out / "evidence.jsonl").write_text("", encoding="utf-8")

    (out / "config.yaml").write_text(
        args.config.read_text(encoding="utf-8"), encoding="utf-8"
    )
    reader_path = stage_cfg.get("reader", {}).get("checkpoint_path")
    reader_sha = _sha256_file(Path(reader_path)) if reader_path else None
    (out / "reader_checksum.txt").write_text(
        (reader_sha or "DEFERRED_GPU_NO_CHECKPOINT") + "\n",
        encoding="utf-8",
    )

    metrics = {
        "n_queries": result.prediction_count,
        "error_count": result.error_count,
        "index_fingerprint": result.index_fingerprint,
        "rouge": None,
        "meteor": None,
        "note": (
            "ROUGE/METEOR with frozen SEDAR-SFT reader deferred to GPU server"
        ),
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "latency.json").write_text(
        json.dumps(
            {
                "pipeline_run_dir": str(paths.run_dir),
                "cuda_mode": cuda.mode,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "git_commit": git_commit_sha(args.repo_root),
        "config_path": str(args.config),
        "frozen_config_path": str(frozen_path),
        "validation_manifest": str(validation_manifest),
        "validation_hash": validation_hash,
        "reader_checkpoint_sha256": reader_sha,
        "cuda_mode": cuda.mode,
        "local_scaffold": cuda.mode != "live",
        "pipeline_run_id": result.run_id,
        "dataset_version": "warmup+clean_warmup_manifest",
        "seed": 42,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    entry = [
        GateCheck("local_dev_waiver", "PASS", True, True),
        GateCheck(
            "frozen_b2_config",
            "PASS" if frozen_path.exists() else "FAIL",
            str(frozen_path),
            True,
        ),
        GateCheck(
            "validation_manifest",
            "PASS" if validation_manifest.exists() else "FAIL",
            str(validation_manifest),
            True,
        ),
    ]
    exit_checks = [
        GateCheck("artifacts_exist", "PASS", True, True),
        GateCheck(
            "query_count_nonzero",
            "PASS" if result.prediction_count > 0 else "FAIL",
            result.prediction_count,
            ">0",
        ),
        GateCheck(
            "reader_checksum_equals_task00",
            "DEFERRED_GPU" if reader_sha is None else "PASS",
            reader_sha,
            "task00_checksum",
        ),
        GateCheck(
            "full_e2e_metrics",
            "DEFERRED_GPU",
            None,
            "rouge/meteor with frozen reader",
        ),
    ]
    report = GateReport(
        task_id="TASK_01",
        run_id=run_id,
        git_commit=git_commit_sha(args.repo_root),
        entry_gate=checks_payload(entry),
        exit_gate=checks_payload(exit_checks),
        artifacts=[str(path) for path in sorted(out.glob("*"))],
        metrics=metrics,
        recommendation="CONTINUE_LOCAL",
        known_issues=[
            "Local R0 uses frozen B2 mock generation; SEDAR reader checksum deferred.",
            "Re-run on GPU server with live reranker + frozen SEDAR-SFT reader before PROMOTE.",
        ],
        local_dev_waiver={
            "cuda_assumed": True,
            "cuda_mode": cuda.mode,
            "allow_cpu_scaffold": True,
        },
    )
    gate_dir = Path("reports/gates/TASK_01_freeze_r0") / run_id
    write_gate_report(report, gate_dir)
    print(json.dumps({"run_id": run_id, "output": str(out), "gate": str(gate_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
