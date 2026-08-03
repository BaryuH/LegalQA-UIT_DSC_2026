"""Generate and verify a deterministic manifest for immutable source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
SOURCE_DIRECTORIES = ("data",)
SOURCE_FILES = ("selected-contexts.zip",)
EXCLUDED_DIRECTORY_NAMES = {"cache", "output", "outputs"}
DEFAULT_MANIFEST_PATH = Path("artifacts/data-baseline/manifest.json")


class ManifestVerificationError(RuntimeError):
    """Raised when the current source files do not match the baseline."""


@dataclass(frozen=True)
class ManifestEntry:
    """One source-file entry in the manifest."""

    path: str
    size: int
    sha256: str


def _relative_path(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _is_excluded(path: Path, repo_root: Path) -> bool:
    relative_parts = path.relative_to(repo_root).parts
    return any(part.casefold() in EXCLUDED_DIRECTORY_NAMES for part in relative_parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths(repo_root: Path) -> list[Path]:
    paths: set[Path] = set()

    for directory_name in SOURCE_DIRECTORIES:
        source_directory = repo_root / directory_name
        if not source_directory.is_dir():
            raise ManifestVerificationError(
                f"Source directory missing: {directory_name}/"
            )
        for path in source_directory.rglob("*"):
            if path.is_file() and not _is_excluded(path, repo_root):
                paths.add(path)

    for file_name in SOURCE_FILES:
        source_file = repo_root / file_name
        if source_file.is_file() and not _is_excluded(source_file, repo_root):
            paths.add(source_file)

    return sorted(paths, key=lambda path: _relative_path(path, repo_root))


def build_manifest(repo_root: Path) -> dict[str, Any]:
    """Build deterministic manifest content without writing any file."""

    resolved_root = repo_root.resolve()
    entries = [
        ManifestEntry(
            path=_relative_path(path, resolved_root),
            size=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path in _source_paths(resolved_root)
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_paths": [*SOURCE_DIRECTORIES, *SOURCE_FILES],
        "files": [asdict(entry) for entry in entries],
    }


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise ManifestVerificationError(f"Manifest file missing: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestVerificationError(
            f"Manifest is not valid JSON: {manifest_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ManifestVerificationError("Manifest root must be a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestVerificationError(
            f"Unsupported manifest schema_version: {payload.get('schema_version')!r}"
        )
    files = payload.get("files")
    if not isinstance(files, list):
        raise ManifestVerificationError("Manifest field 'files' must be a list")
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ManifestVerificationError(
                f"Manifest entry {index} must be a JSON object"
            )
        required_fields = {"path", "size", "sha256"}
        if set(entry) != required_fields:
            raise ManifestVerificationError(
                f"Manifest entry {index} must contain exactly {sorted(required_fields)}"
            )
        if not isinstance(entry["path"], str):
            raise ManifestVerificationError(
                f"Manifest entry {index} path must be a string"
            )
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ManifestVerificationError(
                f"Manifest entry {index} size must be a non-negative integer"
            )
        if not isinstance(entry["sha256"], str):
            raise ManifestVerificationError(
                f"Manifest entry {index} sha256 must be a string"
            )
    return payload


def _entries_by_path(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = payload["files"]
    result = {entry["path"]: entry for entry in entries}
    if len(result) != len(entries):
        raise ManifestVerificationError("Manifest contains duplicate file paths")
    return result


def write_manifest(repo_root: Path, manifest_path: Path) -> int:
    """Write a newly computed manifest and return its source-file count."""

    payload = build_manifest(repo_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(payload["files"])


def verify_manifest(repo_root: Path, manifest_path: Path) -> int:
    """Verify missing, extra, size, and SHA256 mismatches against a baseline."""

    expected = _entries_by_path(_read_manifest(manifest_path))
    actual = _entries_by_path(build_manifest(repo_root))

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatches = []
    for path in sorted(set(expected) & set(actual)):
        expected_entry = expected[path]
        actual_entry = actual[path]
        if (
            expected_entry["size"] != actual_entry["size"]
            or expected_entry["sha256"].casefold() != actual_entry["sha256"].casefold()
        ):
            mismatches.append(
                {
                    "path": path,
                    "expected_size": expected_entry["size"],
                    "actual_size": actual_entry["size"],
                    "expected_sha256": expected_entry["sha256"],
                    "actual_sha256": actual_entry["sha256"],
                }
            )

    errors = []
    if missing:
        errors.append(f"missing files: {', '.join(missing)}")
    if extra:
        errors.append(f"extra files: {', '.join(extra)}")
    if mismatches:
        details = "; ".join(
            f"{item['path']} (expected size/hash {item['expected_size']}/"
            f"{item['expected_sha256']}, actual {item['actual_size']}/"
            f"{item['actual_sha256']})"
            for item in mismatches
        )
        errors.append(f"size/hash mismatches: {details}")

    if errors:
        raise ManifestVerificationError(
            "Data manifest verification failed:\n- " + "\n- ".join(errors)
        )
    return len(actual)


def _resolve_manifest_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a baseline with ``--write`` or verify it by default."""

    parser = argparse.ArgumentParser(
        description="Verify immutable source-data files against a SHA256 manifest."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root; defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest path relative to the repository root.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write/update the manifest instead of verifying it.",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = _resolve_manifest_path(repo_root, args.manifest)

    try:
        if args.write:
            count = write_manifest(repo_root, manifest_path)
            print(f"Wrote {manifest_path} for {count} source file(s).")
        else:
            count = verify_manifest(repo_root, manifest_path)
            print(f"Data manifest verified: {count} source file(s).")
    except (ManifestVerificationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
