"""Training gates and a dependency-injected smoke path for generative SFT."""

from __future__ import annotations

import importlib.util
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import FineTunedReaderSection
from .contracts import SFTExample


class TrainingGateError(RuntimeError):
    """Raised before training when a required capability is unresolved."""


@dataclass(frozen=True, slots=True)
class TrainingStackReport:
    packages: dict[str, bool]
    torch_version: str | None
    cuda_available: bool | None
    device: str
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, object]:
        return {
            "blockers": list(self.blockers),
            "cuda_available": self.cuda_available,
            "device": self.device,
            "packages": dict(self.packages),
            "platform": platform.platform(),
            "torch_version": self.torch_version,
        }


def inspect_training_stack(
    config: FineTunedReaderSection,
    *,
    repo_root: str | Path | None = None,
) -> TrainingStackReport:
    """Probe optional training dependencies without importing model weights."""

    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "transformers", "peft", "accelerate")
    }
    torch_version: str | None = None
    cuda_available: bool | None = None
    if packages["torch"]:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    blockers: list[str] = []
    if not packages["torch"] or not packages["transformers"]:
        blockers.append("torch_and_transformers_required")
    if not packages["peft"]:
        blockers.append("peft_required_for_lora_or_qlora")
    if not packages["accelerate"]:
        blockers.append("accelerate_required_for_training_runtime")
    if config.model.revision == "UNRESOLVED":
        blockers.append("base_model_revision_unresolved")
    if config.model.base_model == "UNRESOLVED":
        blockers.append("base_model_unresolved")
    if not config.lora.target_modules:
        blockers.append("lora_target_modules_unresolved")
    if config.model.load_in_4bit and importlib.util.find_spec("bitsandbytes") is None:
        blockers.append("bitsandbytes_required_for_4bit")
    model_path = Path(config.model.base_model)
    if repo_root is not None and not model_path.is_absolute():
        model_path = Path(repo_root).resolve() / model_path
    if not model_path.exists() and config.model.local_files_only:
        blockers.append("local_base_model_path_missing")
    return TrainingStackReport(
        packages=packages,
        torch_version=torch_version,
        cuda_available=cuda_available,
        device=device,
        blockers=tuple(dict.fromkeys(blockers)),
    )


class SmokeTrainingBackend(Protocol):
    """Small backend contract used by offline forward/backward/save/reload tests."""

    def train_step(self, example: SFTExample) -> float: ...

    def save(self, output_dir: Path) -> None: ...

    def reload(self, output_dir: Path) -> None: ...

    def generate(self, example: SFTExample) -> str: ...


@dataclass(frozen=True, slots=True)
class SmokeTrainingResult:
    output_dir: Path
    steps: int
    losses: tuple[float, ...]
    generated_answer: str

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_answer": self.generated_answer,
            "losses": list(self.losses),
            "output_dir": self.output_dir.as_posix(),
            "steps": self.steps,
        }


def run_training_smoke(
    examples: Sequence[SFTExample],
    *,
    backend: SmokeTrainingBackend,
    output_dir: str | Path,
) -> SmokeTrainingResult:
    """Exercise one deterministic training/save/reload/generation cycle."""

    if not examples:
        raise TrainingGateError("Training smoke requires at least one SFT example")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise TrainingGateError(f"Refusing to overwrite smoke output: {root}")
    root.mkdir(parents=True, exist_ok=False)
    losses = tuple(float(backend.train_step(example)) for example in examples)
    if any(loss != loss or loss in {float("inf"), float("-inf")} for loss in losses):
        raise TrainingGateError("Training smoke produced non-finite loss")
    backend.save(root)
    backend.reload(root)
    generated = backend.generate(examples[0])
    if not isinstance(generated, str) or not generated.strip():
        raise TrainingGateError("Training smoke generation returned an empty answer")
    (root / "smoke_result.json").write_text(
        json.dumps(
            SmokeTrainingResult(root, len(losses), losses, generated).as_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return SmokeTrainingResult(root, len(losses), losses, generated)


def require_training_stack(
    config: FineTunedReaderSection,
    *,
    repo_root: str | Path | None = None,
) -> TrainingStackReport:
    """Fail closed instead of silently running a base model or mock trainer."""

    report = inspect_training_stack(config, repo_root=repo_root)
    if not report.ready:
        raise TrainingGateError(
            "finetuned_reader training is blocked: " + ", ".join(report.blockers)
        )
    return report


__all__ = [
    "SmokeTrainingBackend",
    "SmokeTrainingResult",
    "TrainingGateError",
    "TrainingStackReport",
    "inspect_training_stack",
    "require_training_stack",
    "run_training_smoke",
]
