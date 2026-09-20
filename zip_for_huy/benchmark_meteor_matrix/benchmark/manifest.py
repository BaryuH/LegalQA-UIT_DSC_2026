"""Run manifest: fingerprints and versions for reproducibility.

Every artifact directory carries one of these so a result can be traced back to
its exact model, sampling config, utility/metric versions, dataset, and code
state (AGENTS.md: cache/index must have version and fingerprint).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import grounding, metrics, utilities
from .generation import ModelConfig, SamplingConfig


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class RunManifest:
    dataset_name: str
    dataset_fingerprint: str
    case_count: int
    model: dict[str, Any]
    sampling: dict[str, Any]
    utilities: list[str]
    grounding_mode: str
    resolved_dtype: str | None
    versions: dict[str, str] = field(default_factory=dict)
    git_commit: str | None = None
    created_at: str = ""
    config_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "case_count": self.case_count,
            "model": self.model,
            "sampling": self.sampling,
            "utilities": self.utilities,
            "grounding_mode": self.grounding_mode,
            "resolved_dtype": self.resolved_dtype,
            "versions": self.versions,
            "git_commit": self.git_commit,
            "created_at": self.created_at,
            "config_sha256": self.config_sha256,
        }


def build_manifest(
    *,
    dataset_name: str,
    dataset_fingerprint: str,
    case_count: int,
    model: ModelConfig,
    sampling: SamplingConfig,
    utility_names: list[str],
    grounding_mode: str,
    resolved_dtype: str | None,
    raw_config: dict[str, Any],
) -> RunManifest:
    return RunManifest(
        dataset_name=dataset_name,
        dataset_fingerprint=dataset_fingerprint,
        case_count=case_count,
        model={
            "model_name": model.model_name,
            "device": model.device,
            "dtype": model.dtype,
            "load_in_4bit": model.load_in_4bit,
            "load_in_8bit": model.load_in_8bit,
        },
        sampling=sampling.as_dict(),
        utilities=utility_names,
        grounding_mode=grounding_mode,
        resolved_dtype=resolved_dtype,
        versions={
            "metrics": metrics.METRIC_VERSION,
            "utilities": utilities.UTILITY_VERSION,
            "grounding": grounding.GATE_VERSION,
            "python": platform.python_version(),
        },
        git_commit=_git_commit(),
        created_at=datetime.now(timezone.utc).isoformat(),
        config_sha256=config_hash(raw_config),
    )


def write_manifest(path: str | Path, manifest: RunManifest) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


__all__ = ["RunManifest", "build_manifest", "config_hash", "write_manifest"]
