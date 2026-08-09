"""SS-09 strict SEDAR checkpoint validator extensions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..finetuned_reader.checkpoint import (
    REQUIRED_MANIFEST_FIELDS,
    CheckpointValidationError,
    hash_directory,
)
from .contracts import SEDAR_METHOD, SEDAR_PROFILE

SEDAR_EXTRA_MANIFEST_FIELDS = frozenset(
    {
        "attention_backend",
        "compute_dtype",
        "quantization",
        "training_environment_profile",
        "cuda_driver_metadata",
        "training_resource_profile_hash",
        "method",
    }
)


@dataclass(frozen=True, slots=True)
class SedarCheckpointValidation:
    checkpoint_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_hash: str
    adapter_hash: str
    sedar_fields: dict[str, Any]
    environment_delta: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sedar_sft.ss09.checkpoint.v1",
            "checkpoint_dir": self.checkpoint_dir.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "manifest_hash": self.manifest_hash,
            "adapter_hash": self.adapter_hash,
            "sedar_fields": self.sedar_fields,
            "environment_delta": self.environment_delta,
        }


def _load_sedar_manifest(path: Path, *, require_sedar_extras: bool) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"Invalid checkpoint manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise CheckpointValidationError("Checkpoint manifest root must be an object")
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(payload))
    if missing:
        raise CheckpointValidationError(
            f"Checkpoint manifest is missing required fields: {', '.join(missing)}"
        )
    profile = payload.get("profile")
    method = payload.get("method", profile)
    if profile not in {SEDAR_PROFILE, "finetuned_reader"} and method != SEDAR_METHOD:
        raise CheckpointValidationError(
            f"Checkpoint profile/method must be sedar_sft-compatible, got {profile!r}"
        )
    for field in ("base_model", "base_revision", "tokenizer", "prompt_hash"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip() or value == "UNRESOLVED":
            raise CheckpointValidationError(
                f"Checkpoint manifest field {field!r} must be resolved"
            )
    if require_sedar_extras:
        missing_extra = sorted(SEDAR_EXTRA_MANIFEST_FIELDS - set(payload))
        if missing_extra:
            raise CheckpointValidationError(
                "SEDAR checkpoint missing fields: " + ", ".join(missing_extra)
            )
    return payload


def validate_sedar_checkpoint(
    checkpoint_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    expected: Mapping[str, object] | None = None,
    require_sedar_extras: bool = True,
    inference_environment: Mapping[str, object] | None = None,
) -> SedarCheckpointValidation:
    """Validate base + SEDAR GPU/runtime provenance fields. No silent fallback."""

    root = Path(checkpoint_dir).resolve()
    if not root.is_dir():
        raise CheckpointValidationError(f"Checkpoint directory not found: {root}")
    selected_manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / "checkpoint_manifest.json"
    )
    try:
        selected_manifest.relative_to(root)
    except ValueError as exc:
        raise CheckpointValidationError(
            "Checkpoint manifest must be inside checkpoint"
        ) from exc
    if not selected_manifest.is_file():
        raise CheckpointValidationError(
            f"Checkpoint manifest not found: {selected_manifest}"
        )
    manifest = _load_sedar_manifest(
        selected_manifest, require_sedar_extras=require_sedar_extras
    )
    if expected:
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise CheckpointValidationError(
                    f"Checkpoint field {key!r} mismatch: {manifest.get(key)!r} != {value!r}"
                )
    adapter_dir = root / "adapter"
    adapter_hash = hash_directory(adapter_dir) if adapter_dir.is_dir() else ""
    if not adapter_hash:
        raise CheckpointValidationError("Checkpoint adapter directory missing/empty")
    declared = str(manifest.get("adapter_hash", ""))
    if declared and declared != adapter_hash:
        raise CheckpointValidationError("adapter_hash does not match adapter bytes")
    sedar_fields = {
        field: manifest.get(field) for field in sorted(SEDAR_EXTRA_MANIFEST_FIELDS)
    }
    env_delta: dict[str, Any] = {
        "recorded": True,
        "exact_driver_match_required": False,
    }
    if inference_environment is not None:
        train_meta = manifest.get("cuda_driver_metadata")
        if train_meta != dict(inference_environment):
            env_delta["differences"] = {
                "training": train_meta,
                "inference": dict(inference_environment),
            }
            env_delta["note"] = (
                "Driver/environment differences recorded explicitly"
            )
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return SedarCheckpointValidation(
        checkpoint_dir=root,
        manifest_path=selected_manifest,
        manifest=manifest,
        manifest_hash=manifest_hash,
        adapter_hash=adapter_hash,
        sedar_fields=sedar_fields,
        environment_delta=env_delta,
    )


def build_sedar_manifest_template(
    *,
    base: Mapping[str, Any],
    attention_backend: str = "sdpa",
    compute_dtype: str = "bfloat16",
    quantization: Mapping[str, Any] | None = None,
    training_environment_profile: str = "linux_rtx4090_single_gpu",
    cuda_driver_metadata: Mapping[str, Any] | None = None,
    training_resource_profile_hash: str = "UNRESOLVED",
) -> dict[str, Any]:
    """Compose a SEDAR-extended manifest payload for later training writes."""

    payload = dict(base)
    payload.update(
        {
            "method": SEDAR_METHOD,
            "profile": SEDAR_PROFILE,
            "attention_backend": attention_backend,
            "compute_dtype": compute_dtype,
            "quantization": dict(
                quantization
                or {
                    "load_in_4bit": True,
                    "quant_type": "nf4",
                    "double_quant": True,
                }
            ),
            "training_environment_profile": training_environment_profile,
            "cuda_driver_metadata": dict(cuda_driver_metadata or {}),
            "training_resource_profile_hash": training_resource_profile_hash,
        }
    )
    return payload


__all__ = [
    "SEDAR_EXTRA_MANIFEST_FIELDS",
    "SedarCheckpointValidation",
    "build_sedar_manifest_template",
    "hash_directory",
    "validate_sedar_checkpoint",
]
