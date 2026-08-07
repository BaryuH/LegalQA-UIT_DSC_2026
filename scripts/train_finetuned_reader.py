"""Run one explicit local Transformers + LoRA SFT job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    from legal_rag.config import load_config
    from legal_rag.finetuned_reader.trainer import RealTrainingError, run_real_sft
    from legal_rag.finetuned_reader.training import TrainingGateError

    parser = argparse.ArgumentParser(
        description=(
            "Train finetuned_reader with a local Transformers checkpoint and LoRA. "
            "The command never downloads a model."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/finetuned_reader_train.yaml"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional deterministic prefix for a server smoke run.",
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        config = load_config(config_path)
        result = run_real_sft(
            config,
            repo_root=root,
            run_id=args.run_id,
            max_examples=args.max_examples,
        )
    except (OSError, ValueError, RealTrainingError, TrainingGateError) as exc:
        print(f"TRAINING_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
