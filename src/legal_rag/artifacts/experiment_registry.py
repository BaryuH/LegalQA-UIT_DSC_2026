"""Small append-only JSONL registry for reproducible experiment runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class ExperimentRegistryError(RuntimeError):
    """Raised when an experiment registry cannot be read or extended safely."""


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


def _validate_rate(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return numeric


def _validate_metric(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _validate_rate(value, name)


def _validate_latency(value: float | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("latency_ms must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError("latency_ms must be finite and non-negative")
    return numeric


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """One complete, content-free registry row for an experiment run."""

    run_id: str
    git_commit: str
    dirty: bool
    command: str
    config_hash: str
    split: str
    data_manifest_hash: str
    chunk_fingerprint: str | None
    index_fingerprint: str | None
    prompt_hash: str
    model: str
    seed: int
    meteor: float | None
    rouge_l: float | None
    error_rate: float
    latency_ms: float | None
    reranker_fallback_rate: float
    output_hash: str
    notes: str
    schema_version: str = "i1.experiment.v1"

    def __post_init__(self) -> None:
        for text_value, name in (
            (self.run_id, "run_id"),
            (self.git_commit, "git_commit"),
            (self.command, "command"),
            (self.config_hash, "config_hash"),
            (self.split, "split"),
            (self.data_manifest_hash, "data_manifest_hash"),
            (self.prompt_hash, "prompt_hash"),
            (self.model, "model"),
            (self.output_hash, "output_hash"),
            (self.schema_version, "schema_version"),
        ):
            _require_text(text_value, name)
        if not isinstance(self.dirty, bool):
            raise TypeError("dirty must be a boolean")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        for fingerprint_value, name in (
            (self.chunk_fingerprint, "chunk_fingerprint"),
            (self.index_fingerprint, "index_fingerprint"),
        ):
            if fingerprint_value is not None:
                _require_text(fingerprint_value, name)
        object.__setattr__(self, "meteor", _validate_metric(self.meteor, "meteor"))
        object.__setattr__(self, "rouge_l", _validate_metric(self.rouge_l, "rouge_l"))
        object.__setattr__(
            self,
            "error_rate",
            _validate_rate(self.error_rate, "error_rate"),
        )
        object.__setattr__(
            self,
            "latency_ms",
            _validate_latency(self.latency_ms),
        )
        object.__setattr__(
            self,
            "reranker_fallback_rate",
            _validate_rate(
                self.reranker_fallback_rate,
                "reranker_fallback_rate",
            ),
        )
        _require_text(self.notes, "notes")

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON object written to the registry."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "dirty": self.dirty,
            "command": self.command,
            "config_hash": self.config_hash,
            "split": self.split,
            "data_manifest_hash": self.data_manifest_hash,
            "chunk_fingerprint": self.chunk_fingerprint,
            "index_fingerprint": self.index_fingerprint,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
            "seed": self.seed,
            "meteor": self.meteor,
            "rouge_l": self.rouge_l,
            "error_rate": self.error_rate,
            "latency_ms": self.latency_ms,
            "reranker_fallback_rate": self.reranker_fallback_rate,
            "output_hash": self.output_hash,
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExperimentRecord:
        """Validate a JSON object read from a registry."""

        required = {
            "run_id",
            "git_commit",
            "dirty",
            "command",
            "config_hash",
            "split",
            "data_manifest_hash",
            "chunk_fingerprint",
            "index_fingerprint",
            "prompt_hash",
            "model",
            "seed",
            "meteor",
            "rouge_l",
            "error_rate",
            "latency_ms",
            "reranker_fallback_rate",
            "output_hash",
            "notes",
        }
        if set(value) != required | {"schema_version"}:
            raise ExperimentRegistryError(
                "Experiment registry row has unexpected or missing fields"
            )
        try:
            return cls(
                run_id=value["run_id"],  # type: ignore[arg-type]
                git_commit=value["git_commit"],  # type: ignore[arg-type]
                dirty=value["dirty"],  # type: ignore[arg-type]
                command=value["command"],  # type: ignore[arg-type]
                config_hash=value["config_hash"],  # type: ignore[arg-type]
                split=value["split"],  # type: ignore[arg-type]
                data_manifest_hash=value["data_manifest_hash"],  # type: ignore[arg-type]
                chunk_fingerprint=value["chunk_fingerprint"],  # type: ignore[arg-type]
                index_fingerprint=value["index_fingerprint"],  # type: ignore[arg-type]
                prompt_hash=value["prompt_hash"],  # type: ignore[arg-type]
                model=value["model"],  # type: ignore[arg-type]
                seed=value["seed"],  # type: ignore[arg-type]
                meteor=value["meteor"],  # type: ignore[arg-type]
                rouge_l=value["rouge_l"],  # type: ignore[arg-type]
                error_rate=value["error_rate"],  # type: ignore[arg-type]
                latency_ms=value["latency_ms"],  # type: ignore[arg-type]
                reranker_fallback_rate=value["reranker_fallback_rate"],  # type: ignore[arg-type]
                output_hash=value["output_hash"],  # type: ignore[arg-type]
                notes=value["notes"],  # type: ignore[arg-type]
                schema_version=value["schema_version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ExperimentRegistryError("Invalid experiment registry row") from exc


def current_command(argv: Sequence[str] | None = None) -> str:
    """Return a shell-readable command with secret-shaped values redacted."""

    arguments = list(sys.argv if argv is None else argv)
    if not arguments:
        return "UNRESOLVED"
    redacted: list[str] = []
    redact_next = False
    secret_markers = (
        "apikey",
        "authorization",
        "credential",
        "password",
        "secret",
        "token",
    )
    for argument in arguments:
        normalized = "".join(
            character for character in argument.casefold() if character.isalnum()
        )
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
        elif any(marker in normalized for marker in secret_markers):
            if "=" in argument:
                redacted.append(argument.split("=", 1)[0] + "=<redacted>")
            else:
                redacted.append(argument)
                redact_next = True
        else:
            redacted.append(argument)
    return shlex.join(redacted)


def hash_run_outputs(run_dir: str | Path) -> str:
    """Hash all completed files in one run directory, excluding temporary files."""

    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ExperimentRegistryError(f"Run directory is missing: {root}")
    files = tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and not path.name.endswith(".tmp")
        )
    )
    if not files:
        raise ExperimentRegistryError(
            f"Run directory has no completed artifacts: {root}"
        )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class ExperimentRegistry:
    """Append-only JSONL registry that never overwrites a run row."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: ExperimentRecord) -> None:
        """Append one validated row and reject duplicate run IDs."""

        existing = self.read()
        if any(item.run_id == record.run_id for item in existing):
            raise FileExistsError(
                f"Experiment registry already contains run_id: {record.run_id}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            record.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ExperimentRegistryError(
                f"Unable to append experiment registry: {self.path}"
            ) from exc

    def read(self) -> tuple[ExperimentRecord, ...]:
        """Read and validate all rows in deterministic file order."""

        if not self.path.exists():
            return ()
        records: list[ExperimentRecord] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise ExperimentRegistryError(
                f"Unable to read experiment registry: {self.path}"
            ) from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExperimentRegistryError(
                    f"Invalid experiment registry JSON at line {line_number}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ExperimentRegistryError(
                    f"Experiment registry line {line_number} must be an object"
                )
            records.append(ExperimentRecord.from_mapping(payload))
        return tuple(records)


__all__ = [
    "ExperimentRecord",
    "ExperimentRegistry",
    "ExperimentRegistryError",
    "current_command",
    "hash_run_outputs",
]
