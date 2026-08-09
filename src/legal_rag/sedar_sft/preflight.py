"""SS-08 canonical training preflight (fails closed without GPU authorization)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ProjectConfig, load_config
from ..finetuned_reader.b2_freeze import (
    load_b2_freeze_fingerprint,
    require_complete_b2_freeze,
)
from .training_infra import inspect_sedar_training_infra


class CanonicalTrainPreflightError(RuntimeError):
    """Raised when SS-08 entry gates fail."""


def _verify_source_manifest(repo_root: Path) -> None:
    from scripts.verify_data_manifest import (
        DEFAULT_MANIFEST_PATH,
        ManifestVerificationError,
        verify_manifest,
    )

    manifest_path = repo_root / DEFAULT_MANIFEST_PATH
    try:
        file_count = verify_manifest(repo_root, manifest_path)
    except (ManifestVerificationError, OSError, ValueError) as exc:
        raise CanonicalTrainPreflightError(str(exc)) from exc
    if file_count <= 0:
        raise CanonicalTrainPreflightError(
            "source data manifest verification returned no files"
        )


@dataclass(frozen=True, slots=True)
class CanonicalTrainPreflight:
    status: str
    checks: dict[str, Any]
    blockers: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status == "pass" and not self.blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sedar_sft.ss08.preflight.v1",
            "status": self.status,
            "blockers": list(self.blockers),
            "checks": self.checks,
            "gpu_deferred": [
                "exclusive GPU occupancy",
                "NVMe free-space probe on SEDAR_WORK_ROOT",
                "SS-04D/SS-07G batch-seq profile",
                "canonical training launch",
            ],
        }


def run_canonical_train_preflight(
    config: ProjectConfig | str | Path,
    *,
    repo_root: str | Path,
    authorize_gpu_execution: bool = False,
    dataset_manifest: str | Path | None = None,
) -> CanonicalTrainPreflight:
    """Validate SS-08 entry conditions without starting training."""

    root = Path(repo_root).resolve()
    project = (
        config
        if isinstance(config, ProjectConfig)
        else load_config(root / config if not Path(str(config)).is_absolute() else config)
    )
    blockers: list[str] = []
    checks: dict[str, Any] = {}

    try:
        _verify_source_manifest(root)
        checks["data_manifest"] = "pass"
    except Exception as exc:  # noqa: BLE001
        checks["data_manifest"] = f"fail:{exc}"
        blockers.append("data_manifest_failed")

    try:
        freeze = load_b2_freeze_fingerprint(root)
        require_complete_b2_freeze(freeze)
        checks["b2_freeze"] = {
            "status": freeze.status,
            "config_hash": freeze.config_hash,
            "index_fingerprint": freeze.index_fingerprint,
        }
    except Exception as exc:  # noqa: BLE001
        checks["b2_freeze"] = f"fail:{exc}"
        blockers.append("b2_freeze_invalid")

    infra = inspect_sedar_training_infra(
        project,
        repo_root=root,
        authorize_gpu_execution=authorize_gpu_execution,
    )
    checks["training_infra"] = infra.as_dict()
    if not authorize_gpu_execution:
        blockers.append("gpu_execution_deferred_by_operator")
    if not infra.qlora_defaults_present:
        blockers.append("qlora_defaults_missing")
    for item in infra.blockers:
        if item != "gpu_execution_deferred_by_operator":
            blockers.append(item)

    if dataset_manifest is not None:
        path = Path(dataset_manifest)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            checks["dataset_manifest"] = {
                "path": path.as_posix(),
                "present": True,
            }
        else:
            checks["dataset_manifest"] = {"path": path.as_posix(), "present": False}
            blockers.append("dataset_manifest_missing")
    else:
        checks["dataset_manifest"] = "not_provided"
        blockers.append("dataset_manifest_not_provided")

    settings = project.finetuned_reader
    if settings is None or settings.model.revision == "UNRESOLVED":
        blockers.append("model_revision_unresolved")
        checks["model_revision"] = "UNRESOLVED"
    else:
        checks["model_revision"] = settings.model.revision

    unique = tuple(dict.fromkeys(blockers))
    status = "pass" if not unique and authorize_gpu_execution else "blocked"
    return CanonicalTrainPreflight(status=status, checks=checks, blockers=unique)


def write_preflight(path: str | Path, report: CanonicalTrainPreflight) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


__all__ = [
    "CanonicalTrainPreflight",
    "CanonicalTrainPreflightError",
    "run_canonical_train_preflight",
    "write_preflight",
]
