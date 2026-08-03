"""Run identity, environment capture, and atomic artifact writing.

The run manager owns only provenance and serialization.  It never receives a
gold answer, prompt text, or private reference, so inference callers cannot
accidentally place those values in reproducibility metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path

from ..config import redact_secrets

_RUN_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REDACTED = "<redacted>"


class ArtifactWriteError(RuntimeError):
    """Raised when an artifact cannot be written without losing provenance."""


@dataclass(frozen=True, slots=True)
class RunArtifactPaths:
    """Stable paths for the artifacts belonging to one run."""

    run_dir: Path
    config: Path
    environment: Path
    predictions: Path
    generation: Path
    errors: Path
    summary: Path
    metrics: Path
    submission: Path
    retrieval: Path | None = None
    reader: Path | None = None
    registry: Path | None = None

    @property
    def metadata(self) -> Path:
        """Backward-compatible alias for the G1 environment artifact."""

        return self.environment


def fingerprint_json(value: object) -> str:
    """Return the deterministic SHA256 fingerprint of a JSON-compatible value."""

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("Fingerprint value must be JSON serializable") from exc
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_component(value: str, name: str) -> str:
    if not isinstance(value, str) or _RUN_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{name} must contain only letters, numbers, '.', '_' or '-'")
    return value


def _timestamped_run_id(split: str, method: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{split}_{method}"


@lru_cache(maxsize=1)
def _package_versions() -> dict[str, str]:
    """Capture installed distribution versions in deterministic key order."""

    collected: dict[str, set[str]] = {}
    for distribution in importlib_metadata.distributions():
        name = distribution.name
        version = distribution.version
        if name and version:
            collected.setdefault(name, set()).add(version)
    return {
        name: ",".join(sorted(versions))
        for name, versions in sorted(
            collected.items(), key=lambda item: item[0].casefold()
        )
    }


def _git_metadata(repo_root: Path) -> dict[str, str | bool]:
    """Read commit and dirty state without including the status file list."""

    def run_git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ArtifactWriteError(
                f"Unable to capture git metadata under {repo_root}"
            ) from exc
        return completed.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    dirty_output = run_git("status", "--porcelain", "--untracked-files=all")
    if not commit:
        raise ArtifactWriteError("Git commit metadata is blank")
    return {"commit": commit, "dirty": bool(dirty_output)}


def _stable_record_key(record: Mapping[str, object]) -> tuple[str, str]:
    identifier = record.get("id", record.get("case_id"))
    if identifier is not None:
        return ("0", str(identifier))
    return (
        "1",
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


class RunManager:
    """Create one immutable run directory and write its artifacts atomically."""

    def __init__(
        self,
        paths: RunArtifactPaths,
        *,
        repo_root: str | Path,
    ) -> None:
        self.paths = paths
        self.repo_root = Path(repo_root).resolve()

    @classmethod
    def create(
        cls,
        output_dir: str | Path,
        *,
        split: str,
        method: str,
        repo_root: str | Path,
        run_id: str | None = None,
        registry_path: str | Path | None = None,
    ) -> RunManager:
        """Create a manager without overwriting an existing run directory."""

        selected_split = _validate_component(split, "split")
        selected_method = _validate_component(method, "method")
        selected_run_id = (
            _timestamped_run_id(selected_split, selected_method)
            if run_id is None
            else _validate_component(run_id, "run_id")
        )
        root = Path(output_dir)
        run_dir = root / selected_run_id
        if run_dir.exists():
            raise FileExistsError(
                f"Refusing to reuse existing run directory: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        return cls(
            RunArtifactPaths(
                run_dir=run_dir,
                config=run_dir / "config.json",
                environment=run_dir / "environment.json",
                predictions=run_dir / "predictions.jsonl",
                generation=run_dir / "generation.jsonl",
                errors=run_dir / "errors.jsonl",
                summary=run_dir / "run_summary.json",
                metrics=run_dir / "metrics.json",
                submission=run_dir / "submission.json",
                retrieval=(
                    run_dir / "retrieval.jsonl"
                    if selected_method
                    in {
                        "bm25_rag",
                        "bm25-rag",
                        "hybrid_rag",
                        "hybrid-rag",
                        "tuned_bm25_reader",
                        "tuned-bm25-reader",
                    }
                    else None
                ),
                reader=(
                    run_dir / "reader.jsonl"
                    if selected_method
                    in {
                        "finetuned_reader",
                        "finetuned-reader",
                        "tuned_bm25_reader",
                        "tuned-bm25-reader",
                    }
                    else None
                ),
                registry=None if registry_path is None else Path(registry_path),
            ),
            repo_root=repo_root,
        )

    @property
    def run_id(self) -> str:
        """Return the immutable directory name used as this run's identity."""

        return self.paths.run_dir.name

    def environment(self, *, seed: int) -> dict[str, object]:
        """Capture reproducibility-relevant interpreter, package, and Git state."""

        git = _git_metadata(self.repo_root)
        return {
            "schema_version": "g1.environment.v1",
            "python": platform.python_version(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "seed": seed,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "git": git,
            "package_versions": _package_versions(),
            "secrets": _REDACTED,
        }

    def write_json(self, path: Path, value: object) -> None:
        """Write a redacted, sorted JSON object without overwriting an artifact."""

        redacted = redact_secrets(value)
        self._write_atomic(
            path,
            json.dumps(redacted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def write_jsonl(
        self,
        path: Path,
        records: Sequence[Mapping[str, object]],
    ) -> None:
        """Write records in stable ID order with sorted object keys."""

        ordered = sorted(records, key=_stable_record_key)
        content = "".join(
            json.dumps(
                redact_secrets(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in ordered
        )
        self._write_atomic(path, content)

    def _write_atomic(self, path: Path, content: str) -> None:
        try:
            path.resolve().relative_to(self.paths.run_dir.resolve())
        except ValueError as exc:
            raise ArtifactWriteError(
                f"Artifact path must be inside the run directory: {path}"
            ) from exc
        if path.exists():
            try:
                if path.read_text(encoding="utf-8") == content:
                    return
            except OSError as exc:
                raise ArtifactWriteError(f"Unable to inspect artifact: {path}") from exc
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(path)
        except OSError as exc:
            raise ArtifactWriteError(
                f"Unable to atomically write artifact: {path}"
            ) from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError as exc:  # pragma: no cover - cleanup-only failure
                    raise ArtifactWriteError(
                        f"Unable to clean temporary artifact: {temporary_path}"
                    ) from exc


__all__ = [
    "ArtifactWriteError",
    "RunArtifactPaths",
    "RunManager",
    "fingerprint_json",
]
