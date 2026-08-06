"""Auditable, fingerprinted JSONL cache for derived legal chunks."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from ..schemas import LegalChunk, LegalDocument
from .chunking import ChunkingConfig, chunk_documents
from .normalize import NORMALIZATION_VERSION

CACHE_SCHEMA_VERSION = "c3.chunk-cache.v1"
CacheStatus = Literal["hit", "miss", "stale"]


class ChunkCacheError(RuntimeError):
    """Base error for unsafe or invalid chunk-cache operations."""


class ChunkCacheConflictError(ChunkCacheError):
    """Raised when a cache path already contains different content."""


class ChunkCacheStaleError(ChunkCacheError):
    """Raised when an existing cache cannot satisfy its expected fingerprint."""


def _sha256_json(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _require_fingerprint_part(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be blank")
    if any(character in value for character in "/\\"):
        raise ValueError(f"{name} must not contain path separators")
    return value


@dataclass(frozen=True, slots=True)
class ChunkCacheFingerprint:
    """All identities that can change the derived chunk cache."""

    source_manifest_hash: str
    chunk_config_hash: str
    chunker_version: str
    normalization_version: str
    schema_version: str = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_fingerprint_part(self.source_manifest_hash, "source_manifest_hash")
        _require_fingerprint_part(self.chunk_config_hash, "chunk_config_hash")
        _require_fingerprint_part(self.chunker_version, "chunker_version")
        _require_fingerprint_part(self.normalization_version, "normalization_version")
        _require_fingerprint_part(self.schema_version, "schema_version")

    @classmethod
    def from_config(
        cls,
        source_manifest_hash: str,
        config: ChunkingConfig,
        normalization_version: str = NORMALIZATION_VERSION,
    ) -> ChunkCacheFingerprint:
        chunk_config = {
            "max_chars": config.max_chars,
            "min_chars": config.min_chars,
            "overlap_chars": config.overlap_chars,
        }
        return cls(
            source_manifest_hash=source_manifest_hash,
            chunk_config_hash=_sha256_json(chunk_config),
            chunker_version=config.version,
            normalization_version=normalization_version,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "source_manifest_hash": self.source_manifest_hash,
            "chunk_config_hash": self.chunk_config_hash,
            "chunker_version": self.chunker_version,
            "normalization_version": self.normalization_version,
        }

    @property
    def cache_fingerprint(self) -> str:
        """Return a stable directory-safe identity for the complete key."""

        return _sha256_json(self.as_dict())


@dataclass(frozen=True, slots=True)
class ChunkCacheSummary:
    """Deterministic statistics stored in the JSONL metadata record."""

    chunk_count: int
    document_count: int
    raw_char_count: int
    retrieval_char_count: int
    min_raw_chars: int | None
    max_raw_chars: int | None
    chunks_by_document: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_count": self.chunk_count,
            "document_count": self.document_count,
            "raw_char_count": self.raw_char_count,
            "retrieval_char_count": self.retrieval_char_count,
            "min_raw_chars": self.min_raw_chars,
            "max_raw_chars": self.max_raw_chars,
            "chunks_by_document": {
                document_id: count for document_id, count in self.chunks_by_document
            },
        }


@dataclass(frozen=True, slots=True)
class ChunkCacheResult:
    """Cache operation result with explicit hit/miss/stale status."""

    status: CacheStatus
    cache_path: Path
    fingerprint: ChunkCacheFingerprint
    chunks: tuple[LegalChunk, ...]
    summary: ChunkCacheSummary
    reason: str | None = None


def _summary(chunks: tuple[LegalChunk, ...]) -> ChunkCacheSummary:
    lengths = [len(chunk.raw_text) for chunk in chunks]
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
    return ChunkCacheSummary(
        chunk_count=len(chunks),
        document_count=len(counts),
        raw_char_count=sum(lengths),
        retrieval_char_count=sum(len(chunk.retrieval_text) for chunk in chunks),
        min_raw_chars=min(lengths) if lengths else None,
        max_raw_chars=max(lengths) if lengths else None,
        chunks_by_document=tuple(sorted(counts.items())),
    )


def _chunk_sort_key(chunk: LegalChunk) -> tuple[str, str, str, int, int, str]:
    return (
        chunk.document_id,
        chunk.source_path,
        chunk.source_member or "",
        chunk.start_offset if chunk.start_offset is not None else -1,
        chunk.end_offset if chunk.end_offset is not None else -1,
        chunk.chunk_id,
    )


def _ordered_chunks(chunks: Iterable[LegalChunk]) -> tuple[LegalChunk, ...]:
    ordered = tuple(sorted(chunks, key=_chunk_sort_key))
    chunk_ids = [chunk.chunk_id for chunk in ordered]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ChunkCacheError("Chunk cache cannot contain duplicate chunk IDs")
    return ordered


def _validate_cache_root(cache_root: str | Path, data_root: str | Path | None) -> Path:
    resolved_cache = Path(cache_root).resolve()
    if data_root is not None:
        resolved_data = Path(data_root).resolve()
        try:
            resolved_cache.relative_to(resolved_data)
        except ValueError:
            pass
        else:
            raise ChunkCacheError(
                f"Chunk cache must be outside source data directory: {resolved_cache}"
            )
    if any(part.casefold() == "data" for part in resolved_cache.parts):
        raise ChunkCacheError(
            f"Chunk cache path appears inside a data directory: {resolved_cache}"
        )
    return resolved_cache


def cache_path(
    cache_root: str | Path,
    fingerprint: ChunkCacheFingerprint,
    data_root: str | Path | None = None,
) -> Path:
    """Return the deterministic JSONL path outside the source-data tree."""

    root = _validate_cache_root(cache_root, data_root)
    return (
        root
        / "chunks"
        / fingerprint.source_manifest_hash
        / fingerprint.chunk_config_hash
        / fingerprint.cache_fingerprint
        / "chunks.jsonl"
    )


def _empty_result(
    status: CacheStatus,
    path: Path,
    fingerprint: ChunkCacheFingerprint,
    reason: str | None = None,
) -> ChunkCacheResult:
    empty: tuple[LegalChunk, ...] = ()
    return ChunkCacheResult(
        status=status,
        cache_path=path,
        fingerprint=fingerprint,
        chunks=empty,
        summary=_summary(empty),
        reason=reason,
    )


def read_chunk_cache(
    path: str | Path,
    fingerprint: ChunkCacheFingerprint,
    data_root: str | Path | None = None,
) -> ChunkCacheResult:
    """Read one cache file and explicitly classify misses and stale content."""

    resolved_path = Path(path).resolve()
    _validate_cache_root(resolved_path.parent, data_root)
    if not resolved_path.is_file():
        return _empty_result("miss", resolved_path, fingerprint, "cache_missing")

    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("cache is empty")
        metadata = json.loads(lines[0])
        if not isinstance(metadata, dict):
            raise ValueError("metadata record must be an object")
        if metadata.get("record_type") != "metadata":
            raise ValueError("first record must be metadata")
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported cache schema version")
        if metadata.get("fingerprint") != fingerprint.as_dict():
            return _empty_result(
                "stale", resolved_path, fingerprint, "fingerprint_mismatch"
            )
        if metadata.get("cache_fingerprint") != fingerprint.cache_fingerprint:
            return _empty_result(
                "stale", resolved_path, fingerprint, "cache_key_mismatch"
            )

        chunks: list[LegalChunk] = []
        for line_number, line in enumerate(lines[1:], start=2):
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("record_type") != "chunk":
                raise ValueError(f"line {line_number} is not a chunk record")
            chunks.append(LegalChunk.model_validate(record.get("chunk")))
        ordered = tuple(chunks)
        if ordered != _ordered_chunks(ordered):
            raise ValueError("chunk records are not in deterministic order")
        if any(
            chunk.chunker_version != fingerprint.chunker_version for chunk in ordered
        ):
            return _empty_result(
                "stale", resolved_path, fingerprint, "chunker_version_mismatch"
            )
        actual_summary = _summary(ordered)
        if metadata.get("summary") != actual_summary.as_dict():
            raise ValueError("summary does not match chunk records")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        return _empty_result("stale", resolved_path, fingerprint, str(exc))

    return ChunkCacheResult(
        status="hit",
        cache_path=resolved_path,
        fingerprint=fingerprint,
        chunks=ordered,
        summary=actual_summary,
    )


def load_chunk_cache(
    cache_root: str | Path,
    fingerprint: ChunkCacheFingerprint,
    data_root: str | Path | None = None,
) -> ChunkCacheResult:
    """Load the cache selected by a complete fingerprint."""

    return read_chunk_cache(
        cache_path(cache_root, fingerprint, data_root), fingerprint, data_root
    )


def _write_cache_lines(
    handle: Any,
    chunks: tuple[LegalChunk, ...],
    fingerprint: ChunkCacheFingerprint,
) -> None:
    """Stream JSONL cache records without materializing the full file in memory."""

    summary = _summary(chunks)
    metadata = {
        "record_type": "metadata",
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_fingerprint": fingerprint.cache_fingerprint,
        "fingerprint": fingerprint.as_dict(),
        "summary": summary.as_dict(),
    }
    handle.write(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    for chunk in chunks:
        record = {"record_type": "chunk", "chunk": chunk.model_dump(mode="json")}
        handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )


def _serialize_cache(
    chunks: tuple[LegalChunk, ...], fingerprint: ChunkCacheFingerprint
) -> str:
    summary = _summary(chunks)
    records: list[dict[str, Any]] = [
        {
            "record_type": "metadata",
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_fingerprint": fingerprint.cache_fingerprint,
            "fingerprint": fingerprint.as_dict(),
            "summary": summary.as_dict(),
        }
    ]
    records.extend(
        {"record_type": "chunk", "chunk": chunk.model_dump(mode="json")}
        for chunk in chunks
    )
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )


def write_chunk_cache(
    cache_root: str | Path,
    chunks: Iterable[LegalChunk],
    fingerprint: ChunkCacheFingerprint,
    data_root: str | Path | None = None,
) -> ChunkCacheResult:
    """Write a new cache atomically without replacing differing content."""

    path = cache_path(cache_root, fingerprint, data_root)
    ordered = _ordered_chunks(chunks)
    existing = read_chunk_cache(path, fingerprint, data_root)
    if existing.status == "hit":
        if existing.chunks != ordered:
            raise ChunkCacheConflictError(
                f"Cache already exists with different chunk content: {path}"
            )
        return existing
    if path.exists() and existing.status == "stale":
        raise ChunkCacheStaleError(
            f"Refusing to overwrite stale chunk cache {path}: {existing.reason}"
        )

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
            _write_cache_lines(handle, ordered, fingerprint)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise ChunkCacheConflictError(f"Cache appeared during write: {path}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    result = read_chunk_cache(path, fingerprint, data_root)
    if result.status != "hit":
        raise ChunkCacheError(
            f"Atomic chunk-cache write could not be verified: {result.reason}"
        )
    return ChunkCacheResult(
        status="miss",
        cache_path=result.cache_path,
        fingerprint=result.fingerprint,
        chunks=result.chunks,
        summary=result.summary,
        reason="cache_built",
    )


def build_chunk_cache(
    documents: Iterable[LegalDocument],
    cache_root: str | Path,
    source_manifest_hash: str,
    config: ChunkingConfig,
    normalization_version: str = NORMALIZATION_VERSION,
    data_root: str | Path | None = None,
) -> ChunkCacheResult:
    """Load a matching cache or build and atomically persist a cache miss."""

    fingerprint = ChunkCacheFingerprint.from_config(
        source_manifest_hash,
        config,
        normalization_version,
    )
    loaded = load_chunk_cache(cache_root, fingerprint, data_root)
    if loaded.status == "hit":
        return loaded
    if loaded.status == "stale":
        raise ChunkCacheStaleError(
            f"Stale chunk cache requires an explicit new fingerprint: "
            f"{loaded.cache_path} ({loaded.reason})"
        )
    chunks = chunk_documents(documents, config)
    return write_chunk_cache(cache_root, chunks, fingerprint, data_root)


build_or_load_chunk_cache = build_chunk_cache


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "ChunkCacheConflictError",
    "ChunkCacheError",
    "ChunkCacheFingerprint",
    "ChunkCacheResult",
    "ChunkCacheStaleError",
    "ChunkCacheSummary",
    "build_chunk_cache",
    "build_or_load_chunk_cache",
    "cache_path",
    "load_chunk_cache",
    "read_chunk_cache",
    "write_chunk_cache",
]
