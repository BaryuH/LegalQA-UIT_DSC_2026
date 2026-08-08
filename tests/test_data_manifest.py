from pathlib import Path

import pytest

from scripts.verify_data_manifest import (
    ManifestVerificationError,
    build_manifest,
    verify_manifest,
    write_manifest,
)


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "z-source.txt").write_bytes(b"z fixture\n")
    (data_dir / "a-source.txt").write_bytes(b"a fixture\n")
    (data_dir / "cache").mkdir()
    (data_dir / "cache" / "ignored.txt").write_bytes(b"ignore me")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "ignored.txt").write_bytes(b"ignore me too")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "ignored.txt").write_bytes(b"ignore me also")
    (tmp_path / "selected-contexts.zip").write_bytes(b"synthetic archive fixture")
    return tmp_path


def test_manifest_is_sorted_and_excludes_cache_and_outputs(
    fixture_root: Path,
) -> None:
    payload = build_manifest(fixture_root)

    paths = [entry["path"] for entry in payload["files"]]
    assert paths == [
        "data/a-source.txt",
        "data/z-source.txt",
        "selected-contexts.zip",
    ]
    assert all("cache" not in path and "outputs" not in path for path in paths)


def test_manifest_detects_missing_extra_and_hash_mismatch(
    fixture_root: Path,
) -> None:
    manifest_path = fixture_root / "artifacts" / "manifest.json"
    write_manifest(fixture_root, manifest_path)

    (fixture_root / "data" / "a-source.txt").unlink()
    with pytest.raises(ManifestVerificationError, match="missing files"):
        verify_manifest(fixture_root, manifest_path)

    (fixture_root / "data" / "a-source.txt").write_bytes(b"a fixture\n")
    (fixture_root / "data" / "extra.txt").write_bytes(b"extra fixture\n")
    with pytest.raises(ManifestVerificationError, match="extra files"):
        verify_manifest(fixture_root, manifest_path)

    (fixture_root / "data" / "extra.txt").unlink()
    (fixture_root / "data" / "a-source.txt").write_bytes(b"changed fixture\n")
    with pytest.raises(ManifestVerificationError, match="size/hash mismatches"):
        verify_manifest(fixture_root, manifest_path)


def test_manifest_normalizes_text_line_endings_without_rewriting_source(
    fixture_root: Path,
) -> None:
    manifest_path = fixture_root / "artifacts" / "manifest.json"
    source = fixture_root / "data" / "a-source.txt"
    source.write_bytes(b"first\r\nsecond\r\n")
    write_manifest(fixture_root, manifest_path)

    source.write_bytes(b"first\nsecond\n")
    assert verify_manifest(fixture_root, manifest_path) == 3

    source.write_bytes(b"first\nchanged\n")
    with pytest.raises(ManifestVerificationError, match="size/hash mismatches"):
        verify_manifest(fixture_root, manifest_path)


def test_manifest_keeps_binary_line_endings_byte_sensitive(
    fixture_root: Path,
) -> None:
    manifest_path = fixture_root / "artifacts" / "manifest.json"
    archive = fixture_root / "selected-contexts.zip"
    archive.write_bytes(b"archive\r\ncontents")
    write_manifest(fixture_root, manifest_path)

    archive.write_bytes(b"archive\ncontents")
    with pytest.raises(ManifestVerificationError, match="size/hash mismatches"):
        verify_manifest(fixture_root, manifest_path)
