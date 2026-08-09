#!/usr/bin/env python3
"""SS-05 CLI: build SEDAR SFT dataset overlay over frozen B2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_sft.dataset import build_sedar_sft_dataset_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sedar_sft_train.yaml"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional smoke limit; omit for full effective-train build.",
    )
    args = parser.parse_args()
    result = build_sedar_sft_dataset_from_config(
        args.config,
        repo_root=args.repo_root,
        max_examples=args.max_examples,
    )
    print(
        json.dumps(
            {
                "status": result.underlying.status,
                "example_count": result.example_count,
                "output_dir": result.output_dir.as_posix(),
                "manifest_path": result.manifest_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.example_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
