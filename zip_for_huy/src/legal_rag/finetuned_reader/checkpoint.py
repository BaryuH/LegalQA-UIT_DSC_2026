"""Strict checkpoint provenance and local-only validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import GENERATIVE_METHOD, GENERATIVE_TYPE, hash_file


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint cannot be proven compatible with the run."""


REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "profile",
        "type",
        "base_model",
        "base_revision",
        "tokenizer",
        "adapter_type",
        "target_modules",
        "adapter_hash",
        "dataset_manifest_hash",
        "retrieval_config_hash",
        "index_fingerprint",
        "prompt_hash",
        "seed",
        "best_checkpoint_criterion",
    }
)


def hash_directory(path: str | Path) -> str:
    """Hash relative file names and bytes in deterministic order."""

    root = Path(path)
    if not root.is_dir():
        raise CheckpointValidationError(f"Checkpoint directory not found: {root}")
    digest = __import__("hashlib").sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise CheckpointValidationError(f"Checkpoint directory is empty: {root}")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ValidatedCheckpoint:
    checkpoint_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_hash: str
    adapter_hash: str


def _load_manifest(path: Path) -> dict[str, Any]:
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
    if (
        payload.get("profile") != GENERATIVE_METHOD
        or payload.get("type") != GENERATIVE_TYPE
    ):
        raise CheckpointValidationError(
            "Checkpoint identity must be finetuned_reader/generative_sft_reader"
        )
    for field in ("base_model", "base_revision", "tokenizer", "prompt_hash"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip() or value == "UNRESOLVED":
            raise CheckpointValidationError(
                f"Checkpoint manifest field {field!r} must be resolved"
            )
    return payload


def validate_checkpoint(
    checkpoint_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    expected: Mapping[str, object] | None = None,
) -> ValidatedCheckpoint:
    """Validate identity, adapter bytes and declared comparison fingerprints."""

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
    manifest = _load_manifest(selected_manifest)

    adapter_relative = manifest.get("adapter_path", "adapter")
    if not isinstance(adapter_relative, str) or not adapter_relative.strip():
        raise CheckpointValidationError(
            "adapter_path must be a non-blank relative path"
        )
    adapter_path = (root / adapter_relative).resolve()
    try:
        adapter_path.relative_to(root)
    except ValueError as exc:
        raise CheckpointValidationError(
            "adapter_path must stay inside checkpoint"
        ) from exc
    adapter_hash = (
        hash_directory(adapter_path)
        if adapter_path.is_dir()
        else hash_file(str(adapter_path))
    )
    if adapter_hash != manifest.get("adapter_hash"):
        raise CheckpointValidationError(
            "Checkpoint adapter hash does not match manifest"
        )

    tokenizer_relative = manifest.get("tokenizer_path", "tokenizer")
    tokenizer_path = (root / str(tokenizer_relative)).resolve()
    if not tokenizer_path.exists():
        raise CheckpointValidationError(
            f"Checkpoint tokenizer path not found: {tokenizer_path}"
        )

    for key, value in (expected or {}).items():
        if manifest.get(key) != value:
            raise CheckpointValidationError(
                f"Checkpoint field {key!r} mismatch: expected {value!r}, "
                f"got {manifest.get(key)!r}"
            )

    return ValidatedCheckpoint(
        checkpoint_dir=root,
        manifest_path=selected_manifest,
        manifest=manifest,
        manifest_hash=hash_file(str(selected_manifest)),
        adapter_hash=adapter_hash,
    )


__all__ = [
    "CheckpointValidationError",
    "REQUIRED_MANIFEST_FIELDS",
    "ValidatedCheckpoint",
    "hash_directory",
    "validate_checkpoint",
]
