#!/usr/bin/env python3
"""SS-07 CLI: inspect QLoRA training infra; refuse canonical train without GPU auth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.config import load_config
from legal_rag.sedar_sft.training_infra import inspect_sedar_training_infra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sedar_sft_train.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--authorize-gpu-execution",
        action="store_true",
        help="Only set on the GPU server after SS-04A..SS-04C gates pass.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/sedar_sft/hardware/training_infra.json"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    config = load_config(root / args.config)
    report = inspect_sedar_training_infra(
        config,
        repo_root=root,
        authorize_gpu_execution=args.authorize_gpu_execution,
    )
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.qlora_defaults_present else 1


if __name__ == "__main__":
    raise SystemExit(main())
