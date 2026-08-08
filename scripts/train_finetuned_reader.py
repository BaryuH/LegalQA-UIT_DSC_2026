"""Run one explicit local Transformers + LoRA SFT job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from legal_rag.config import ProjectConfig  # noqa: E402


def _apply_training_overrides(
    config: ProjectConfig,
    *,
    train_batch_size: int | None,
    gradient_accumulation_steps: int | None,
    bm25_backend: str | None = None,
) -> ProjectConfig:
    """Return an in-memory config with validated runtime-only train overrides."""

    if (
        train_batch_size is None
        and gradient_accumulation_steps is None
        and bm25_backend is None
    ):
        return config
    payload = config.model_dump(mode="json")
    settings = payload.get("finetuned_reader")
    if not isinstance(settings, dict):
        raise ValueError("finetuned_reader settings are required for overrides")
    dataset_build = settings.get("dataset_build")
    if not isinstance(dataset_build, dict):
        raise ValueError("finetuned_reader.dataset_build settings are required")
    training = settings.get("training")
    if not isinstance(training, dict):
        raise ValueError("finetuned_reader.training settings are required")
    if train_batch_size is not None:
        if train_batch_size <= 0:
            raise ValueError("--train-batch-size must be greater than zero")
        training["train_batch_size"] = train_batch_size
    if gradient_accumulation_steps is not None:
        if gradient_accumulation_steps <= 0:
            raise ValueError("--gradient-accumulation-steps must be greater than zero")
        training["gradient_accumulation_steps"] = gradient_accumulation_steps
    if bm25_backend is not None:
        if bm25_backend not in {"cpu", "cuda"}:
            raise ValueError("--bm25-backend must be 'cpu' or 'cuda'")
        dataset_build["bm25_backend"] = bm25_backend
    return ProjectConfig.model_validate(payload)


def main(argv: list[str] | None = None) -> int:
    from legal_rag.config import load_config
    from legal_rag.finetuned_reader.dataset import DatasetBuildError
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
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override train batch size without editing the config file.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override gradient accumulation without editing the config file.",
    )
    parser.add_argument(
        "--bm25-backend",
        choices=("cpu", "cuda"),
        default=None,
        help="Use the explicit CPU or CUDA BM25 dataset-build backend.",
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        config = load_config(config_path)
        config = _apply_training_overrides(
            config,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            bm25_backend=args.bm25_backend,
        )
        result = run_real_sft(
            config,
            repo_root=root,
            run_id=args.run_id,
            max_examples=args.max_examples,
        )
    except (
        OSError,
        ValueError,
        DatasetBuildError,
        RealTrainingError,
        TrainingGateError,
    ) as exc:
        print(f"TRAINING_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
