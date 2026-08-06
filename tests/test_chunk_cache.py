"""Acceptance tests for the auditable chunk cache."""

import json
from pathlib import Path

import pytest

from legal_rag.schemas import LegalDocument
from legal_rag.text import (
    ChunkCacheError,
    ChunkCacheFingerprint,
    ChunkCacheStaleError,
    ChunkingConfig,
    build_chunk_cache,
    chunk_document,
    read_chunk_cache,
)
from legal_rag.text import cache as cache_module


def _document(
    document_id: str, passage: str = "Điều 1. Nội dung pháp lý."
) -> LegalDocument:
    return LegalDocument(
        id=document_id,
        name=f"Văn bản {document_id}",
        passage=passage,
        source_path=f"contexts/context_{document_id}.json",
        source_member=f"context_{document_id}.json",
        content_hash=f"source-{document_id}",
    )


def _config(**overrides: object) -> ChunkingConfig:
    values: dict[str, object] = {
        "max_chars": 120,
        "overlap_chars": 10,
        "min_chars": 10,
        "version": "legal-chunker-v1",
    }
    values.update(overrides)
    return ChunkingConfig(**values)


class _ShortWritingHandle:
    """Test double that accepts only a short prefix per write call."""

    def __init__(self, limit: int = 7) -> None:
        self.limit = limit
        self.value = ""

    def write(self, text: str) -> int:
        written = min(self.limit, len(text))
        self.value += text[:written]
        return written


def test_jsonl_writer_completes_partial_text_writes() -> None:
    config = _config()
    fingerprint = ChunkCacheFingerprint.from_config("manifest-a", config)
    chunks = chunk_document(_document("partial"), config)
    handle = _ShortWritingHandle()

    cache_module._write_cache_lines(handle, chunks, fingerprint)

    records = [json.loads(line) for line in handle.value.splitlines()]
    assert records[0]["record_type"] == "metadata"
    assert [record["record_type"] for record in records[1:]] == ["chunk"] * len(chunks)


def test_jsonl_cache_miss_hit_and_summary_are_auditable(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    document = _document("2", "Điều 2. Nội dung pháp lý.")
    config = _config()

    first = build_chunk_cache([document], cache_root, "manifest-a", config)
    second = build_chunk_cache([document], cache_root, "manifest-a", config)

    assert first.status == "miss"
    assert second.status == "hit"
    assert first.chunks == second.chunks
    assert first.summary.chunk_count == 1
    assert first.summary.document_count == 1
    assert first.summary.chunks_by_document == (("2", 1),)
    assert first.cache_path.name == "chunks.jsonl"
    assert first.cache_path.is_file()
    assert first.cache_path.is_relative_to(cache_root)
    assert not list(cache_root.rglob("*.pickle"))
    records = [
        json.loads(line)
        for line in first.cache_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["record_type"] == "metadata"
    assert records[0]["fingerprint"]["source_manifest_hash"] == "manifest-a"
    assert records[0]["fingerprint"]["chunker_version"] == config.version
    assert (
        records[0]["fingerprint"]["normalization_version"]
        == "retrieval-normalization-v1"
    )
    assert all(record["record_type"] == "chunk" for record in records[1:])


def test_source_config_chunker_and_normalization_changes_use_new_fingerprints(
    tmp_path: Path,
) -> None:
    document = _document("1", "Điều 1. Nội dung đủ dài để tạo cache.")
    root = tmp_path / "cache"
    base_config = _config()

    base = build_chunk_cache([document], root, "manifest-a", base_config)
    changed_source = build_chunk_cache([document], root, "manifest-b", base_config)
    changed_config = build_chunk_cache(
        [document], root, "manifest-a", _config(max_chars=80)
    )
    changed_chunker = build_chunk_cache(
        [document], root, "manifest-a", _config(version="legal-chunker-v2")
    )
    changed_normalization = build_chunk_cache(
        [document],
        root,
        "manifest-a",
        base_config,
        normalization_version="retrieval-normalization-v2",
    )

    paths = {
        base.cache_path,
        changed_source.cache_path,
        changed_config.cache_path,
        changed_chunker.cache_path,
        changed_normalization.cache_path,
    }
    assert len(paths) == 5
    assert all(
        result.status == "miss"
        for result in [
            changed_source,
            changed_config,
            changed_chunker,
            changed_normalization,
        ]
    )


def test_cache_serialization_is_deterministic_for_input_order_changes(
    tmp_path: Path,
) -> None:
    documents = [_document("2"), _document("1")]
    config = _config()

    first = build_chunk_cache(documents, tmp_path / "cache-a", "manifest-a", config)
    second = build_chunk_cache(
        list(reversed(documents)), tmp_path / "cache-b", "manifest-a", config
    )

    assert first.cache_path.read_bytes() == second.cache_path.read_bytes()
    assert [chunk.document_id for chunk in first.chunks] == ["1", "2"]


def test_reading_with_another_fingerprint_reports_stale_without_rewriting(
    tmp_path: Path,
) -> None:
    document = _document("3")
    config = _config()
    written = build_chunk_cache([document], tmp_path / "cache", "manifest-a", config)
    before = written.cache_path.read_bytes()

    stale_fingerprint = written.fingerprint.__class__.from_config("manifest-b", config)
    stale = read_chunk_cache(written.cache_path, stale_fingerprint)

    assert stale.status == "stale"
    assert stale.reason == "fingerprint_mismatch"
    assert written.cache_path.read_bytes() == before


def test_stale_cache_is_not_silently_overwritten(tmp_path: Path) -> None:
    document = _document("4")
    config = _config()
    written = build_chunk_cache([document], tmp_path / "cache", "manifest-a", config)
    payload = written.cache_path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(payload[0])
    metadata["fingerprint"]["chunker_version"] = "tampered"
    payload[0] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    written.cache_path.write_text("\n".join(payload) + "\n", encoding="utf-8")

    with pytest.raises(ChunkCacheStaleError, match="Stale chunk cache"):
        build_chunk_cache([document], tmp_path / "cache", "manifest-a", config)


def test_atomic_cache_does_not_overwrite_unrelated_files_or_source_data(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    unrelated = cache_root / "unrelated.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("keep", encoding="utf-8")
    document = _document("5")

    result = build_chunk_cache(
        [document],
        cache_root,
        "manifest-a",
        _config(),
        data_root=tmp_path / "data",
    )

    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert result.cache_path.is_file()
    assert not list(cache_root.rglob("*.tmp"))

    with pytest.raises(ChunkCacheError):
        build_chunk_cache(
            [document],
            tmp_path / "data" / "cache",
            "manifest-a",
            _config(),
            data_root=tmp_path / "data",
        )
