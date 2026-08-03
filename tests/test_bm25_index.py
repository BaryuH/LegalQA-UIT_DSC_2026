"""Acceptance tests for the auditable BM25 index."""

import json
from pathlib import Path

import pytest

from legal_rag.config import load_config
from legal_rag.retrieval import (
    BM25Config,
    BM25IndexError,
    BM25IndexMissingError,
    BM25IndexStaleError,
    build_bm25_index,
    load_bm25_index,
    load_or_build_bm25_index,
)
from legal_rag.schemas import LegalChunk


def _chunk(
    chunk_id: str,
    document_id: str,
    retrieval_text: str,
) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_path=f"contexts/context_{document_id}.json",
        source_member=f"context_{document_id}.json",
        raw_text=f"RAW-ONLY-{chunk_id}",
        retrieval_text=retrieval_text,
        content_hash=f"hash-{chunk_id}",
        chunker_version="legal-chunker-v1",
        section_label="Điều 1",
        start_offset=0,
        end_offset=len(retrieval_text),
    )


def _chunks() -> list[LegalChunk]:
    return [
        _chunk(
            "chunk-2",
            "2",
            "Điều 37 áp dụng nghị định 65/2022/NĐ-CP trong 30 ngày.",
        ),
        _chunk(
            "chunk-1",
            "1",
            "Điều 1 áp dụng nghị định 153/2020/NĐ-CP với mức 5.000 đồng.",
        ),
    ]


def test_build_reload_and_summary_use_only_retrieval_view(tmp_path: Path) -> None:
    config = BM25Config(k1=1.2, b=0.7)
    built = build_bm25_index(
        reversed(_chunks()), tmp_path / "cache", "chunk-cache-a", config
    )

    assert built.status == "miss"
    assert built.index is not None
    assert built.summary.corpus_size == 2
    assert built.summary.vocabulary_size > 0
    assert [document.chunk_id for document in built.index.documents] == [
        "chunk-1",
        "chunk-2",
    ]

    loaded = load_bm25_index(tmp_path / "cache", "chunk-cache-a", config)
    assert loaded.status == "hit"
    assert loaded.index == built.index

    payload = built.cache_path.read_text(encoding="utf-8")
    assert "RAW-ONLY" not in payload
    assert '"raw_text"' not in payload
    assert '"answer"' not in payload
    records = [json.loads(line) for line in payload.splitlines()]
    assert records[0]["summary"]["corpus_size"] == 2
    assert records[0]["summary"]["vocabulary_size"] == built.summary.vocabulary_size
    assert all(record["record_type"] == "document" for record in records[1:])


def test_serialization_is_deterministic_and_fingerprint_ties_to_chunk_cache(
    tmp_path: Path,
) -> None:
    config = BM25Config()
    first = build_bm25_index(_chunks(), tmp_path / "cache-a", "chunk-cache-a", config)
    second = build_bm25_index(
        list(reversed(_chunks())), tmp_path / "cache-b", "chunk-cache-a", config
    )
    changed_chunk_cache = build_bm25_index(
        _chunks(), tmp_path / "cache-a", "chunk-cache-b", config
    )
    changed_config = build_bm25_index(
        _chunks(), tmp_path / "cache-a", "chunk-cache-a", BM25Config(k1=2.0)
    )

    assert first.cache_path.read_bytes() == second.cache_path.read_bytes()
    assert first.fingerprint.chunk_cache_fingerprint == "chunk-cache-a"
    assert changed_chunk_cache.cache_path != first.cache_path
    assert changed_config.cache_path != first.cache_path
    assert first.cache_path.is_file()
    assert changed_chunk_cache.cache_path.is_file()
    assert changed_config.cache_path.is_file()


def test_strict_load_requires_explicit_auto_rebuild(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    config = BM25Config()

    with pytest.raises(BM25IndexMissingError, match="strict policy"):
        load_or_build_bm25_index(
            _chunks(), cache_root, "chunk-cache-a", config, policy="strict"
        )

    rebuilt = load_or_build_bm25_index(
        _chunks(), cache_root, "chunk-cache-a", config, policy="auto_rebuild"
    )
    assert rebuilt.status == "miss"
    assert rebuilt.reason == "index_built"
    assert (
        load_or_build_bm25_index(
            _chunks(), cache_root, "chunk-cache-a", config, policy="strict"
        ).status
        == "hit"
    )


def test_stale_metadata_fails_closed_and_auto_rebuild_is_explicit(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    config = BM25Config()
    built = build_bm25_index(_chunks(), cache_root, "chunk-cache-a", config)
    before = built.cache_path.read_bytes()
    records = built.cache_path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(records[0])
    metadata["config"]["k1"] = 2.0
    records[0] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    built.cache_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    stale = load_bm25_index(cache_root, "chunk-cache-a", config)
    assert stale.status == "stale"
    assert stale.reason == "config_mismatch"
    assert built.cache_path.read_bytes() != before

    with pytest.raises(BM25IndexStaleError, match="strict policy"):
        load_or_build_bm25_index(
            _chunks(), cache_root, "chunk-cache-a", config, policy="strict"
        )

    rebuilt = load_or_build_bm25_index(
        _chunks(), cache_root, "chunk-cache-a", config, policy="auto_rebuild"
    )
    assert rebuilt.status == "miss"
    assert load_bm25_index(cache_root, "chunk-cache-a", config).status == "hit"


def test_invalid_bm25_config_and_source_data_cache_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        BM25Config(k1=0)
    with pytest.raises(ValueError):
        BM25Config(b=1.1)
    with pytest.raises(BM25IndexError, match="source data"):
        build_bm25_index(
            _chunks(),
            tmp_path / "data" / "cache",
            "chunk-cache-a",
            BM25Config(),
            data_root=tmp_path / "data",
        )


def test_bm25_profile_exposes_k1_and_b(tmp_path: Path) -> None:
    config = load_config(Path("configs/bm25_rag.yaml"))
    assert config.retrieval.k1 == 1.5
    assert config.retrieval.b == 0.75
