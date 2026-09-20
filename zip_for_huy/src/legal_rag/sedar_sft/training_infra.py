"""SS-07 QLoRA training infrastructure inspection (GPU execution deferred)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import FineTunedReaderSection, ProjectConfig
from ..finetuned_reader.training import (
    TrainingGateError,
    TrainingStackReport,
    inspect_training_stack,
)

DEFAULT_RUNTIME_PROFILE = Path("configs/sedar_sft/runtime_profile.yaml")


@dataclass(frozen=True, slots=True)
class SedarTrainingInfraReport:
    stack: TrainingStackReport
    runtime_profile: dict[str, Any]
    qlora_defaults_present: bool
    gpu_execution_authorized: bool
    blockers: tuple[str, ...]

    @property
    def ready_for_code_smoke(self) -> bool:
        """True when injectable/offline smoke path can run without real weights."""

        return "runtime_profile_missing" not in self.blockers

    @property
    def ready_for_canonical_train(self) -> bool:
        return self.gpu_execution_authorized and not self.blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sedar_sft.ss07.training_infra.v1",
            "blockers": list(self.blockers),
            "gpu_execution_authorized": self.gpu_execution_authorized,
            "qlora_defaults_present": self.qlora_defaults_present,
            "ready_for_canonical_train": self.ready_for_canonical_train,
            "ready_for_code_smoke": self.ready_for_code_smoke,
            "runtime_profile": self.runtime_profile,
            "stack": self.stack.as_dict(),
        }


def load_runtime_profile(path: str | Path | None = None) -> dict[str, Any]:
    profile_path = Path(path) if path is not None else DEFAULT_RUNTIME_PROFILE
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TrainingGateError("SEDAR runtime profile must be a mapping")
    return raw


def inspect_sedar_training_infra(
    config: ProjectConfig | FineTunedReaderSection,
    *,
    repo_root: str | Path | None = None,
    runtime_profile_path: str | Path | None = None,
    authorize_gpu_execution: bool = False,
) -> SedarTrainingInfraReport:
    """Probe QLoRA infra without downloading models or inventing AUTO values."""

    settings = config.finetuned_reader if isinstance(config, ProjectConfig) else config
    if settings is None:
        raise TrainingGateError("SEDAR training requires finetuned_reader settings")

    root = Path(repo_root).resolve() if repo_root is not None else None
    stack = inspect_training_stack(settings, repo_root=root)
    blockers = list(stack.blockers)

    profile_path = (
        (root / DEFAULT_RUNTIME_PROFILE)
        if root is not None and runtime_profile_path is None
        else runtime_profile_path
    )
    try:
        runtime_profile = load_runtime_profile(
            profile_path if profile_path is not None else DEFAULT_RUNTIME_PROFILE
        )
    except (OSError, TrainingGateError, yaml.YAMLError) as exc:
        runtime_profile = {"error": str(exc)}
        blockers.append("runtime_profile_missing")

    qlora_ok = (
        settings.lora.adapter_type == "qlora"
        and settings.model.load_in_4bit
        and settings.training.packing is False
        and settings.training.gradient_checkpointing is True
    )
    if not qlora_ok:
        blockers.append("qlora_canonical_defaults_not_configured")

    packages = dict(stack.packages)
    for name in ("trl", "bitsandbytes", "datasets"):
        packages[name] = importlib.util.find_spec(name) is not None
    if settings.model.load_in_4bit and not packages.get("bitsandbytes"):
        if "bitsandbytes_required_for_4bit" not in blockers:
            blockers.append("bitsandbytes_required_for_4bit")

    if not authorize_gpu_execution:
        blockers.append("gpu_execution_deferred_by_operator")

    # Preserve package extras in a copy of the stack report.
    enriched = TrainingStackReport(
        packages=packages,
        torch_version=stack.torch_version,
        cuda_available=stack.cuda_available,
        device=stack.device,
        blockers=tuple(dict.fromkeys(blockers)),
    )
    return SedarTrainingInfraReport(
        stack=enriched,
        runtime_profile=runtime_profile,
        qlora_defaults_present=qlora_ok,
        gpu_execution_authorized=authorize_gpu_execution,
        blockers=enriched.blockers,
    )


def require_sedar_training_infra(
    config: ProjectConfig | FineTunedReaderSection,
    *,
    repo_root: str | Path | None = None,
    authorize_gpu_execution: bool = False,
) -> SedarTrainingInfraReport:
    report = inspect_sedar_training_infra(
        config,
        repo_root=repo_root,
        authorize_gpu_execution=authorize_gpu_execution,
    )
    if not report.ready_for_canonical_train:
        raise TrainingGateError(
            "SEDAR training infra not ready: " + ", ".join(report.blockers)
        )
    return report


__all__ = [
    "SedarTrainingInfraReport",
    "inspect_sedar_training_infra",
    "load_runtime_profile",
    "require_sedar_training_infra",
]
