"""Write B2 freeze fingerprints and optionally run representative Hybrid-RAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from legal_rag.finetuned_reader.b2_freeze import (
        build_b2_freeze_fingerprint,
        require_complete_b2_freeze,
        write_b2_freeze_fingerprint,
    )
    from legal_rag.pipeline import prepare_bm25_index_from_config, run_hybrid_rag_from_config

    parser = argparse.ArgumentParser(description="Refresh B2 freeze artifact (FTR-02).")
    parser.add_argument(
        "--config",
        default="configs/frozen/hybrid_rag_b2.yaml",
        help="Frozen Hybrid-RAG config path.",
    )
    parser.add_argument(
        "--run-id",
        default="ftr02_representative_b2_warmup",
        help="Representative warmup Hybrid-RAG run ID.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force BM25 index rebuild before the representative run.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only refresh fingerprints from an existing index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional question limit for the representative run.",
    )
    args = parser.parse_args(argv)

    config_path = root / args.config
    preparation = prepare_bm25_index_from_config(
        config_path,
        repo_root=root,
        rebuild_index=args.rebuild_index,
    )
    run_path: str | None = None
    if not args.skip_run:
        result = run_hybrid_rag_from_config(
            config_path,
            repo_root=root,
            rebuild_index=False,
            limit=args.limit,
            run_id=args.run_id,
        )
        run_path = result.artifacts.run_dir.relative_to(root).as_posix()

    fingerprint = build_b2_freeze_fingerprint(
        root,
        frozen_config_path=config_path.relative_to(root).as_posix(),
        chunk_cache_fingerprint=preparation.chunk_cache_fingerprint,
        index_fingerprint=preparation.index.index_fingerprint,
        representative_run_path=run_path,
    )
    written = write_b2_freeze_fingerprint(root, fingerprint)
    require_complete_b2_freeze(fingerprint)
    print(
        json.dumps(
            {
                "status": fingerprint.status,
                "written": written.as_posix(),
                "chunk_cache_fingerprint": fingerprint.chunk_cache_fingerprint,
                "index_fingerprint": fingerprint.index_fingerprint,
                "representative_run": fingerprint.representative_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
