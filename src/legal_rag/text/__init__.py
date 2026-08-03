"""Derived text views, legal-aware chunking, and auditable caches."""

from .cache import (
    CACHE_SCHEMA_VERSION,
    ChunkCacheConflictError,
    ChunkCacheError,
    ChunkCacheFingerprint,
    ChunkCacheResult,
    ChunkCacheStaleError,
    ChunkCacheSummary,
    build_chunk_cache,
    build_or_load_chunk_cache,
    cache_path,
    load_chunk_cache,
    read_chunk_cache,
    write_chunk_cache,
)
from .chunking import ChunkingConfig, chunk_document, chunk_documents
from .normalize import (
    NORMALIZATION_VERSION,
    normalize_retrieval_text,
    tokenize_legal_text,
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "ChunkingConfig",
    "ChunkCacheConflictError",
    "ChunkCacheError",
    "ChunkCacheFingerprint",
    "ChunkCacheResult",
    "ChunkCacheStaleError",
    "ChunkCacheSummary",
    "NORMALIZATION_VERSION",
    "build_chunk_cache",
    "build_or_load_chunk_cache",
    "cache_path",
    "chunk_document",
    "chunk_documents",
    "load_chunk_cache",
    "normalize_retrieval_text",
    "read_chunk_cache",
    "tokenize_legal_text",
    "write_chunk_cache",
]
