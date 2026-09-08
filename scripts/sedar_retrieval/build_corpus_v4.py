#!/usr/bin/env python3
"""Build the SEDAR corpus v4 artifact from the read-only competition archive.

Outputs, all under ``--output-dir``:

``documents.jsonl``  one :class:`DocumentRecord` per corpus document
``units.jsonl``      one :class:`RetrievalUnit` per retrieval unit
``audit.json``       :class:`CorpusV4Audit` exit-gate counters
``manifest.json``    run id, git sha, options, corpus hash

The competition archive under ``data/`` is never modified, extracted or
rewritten; it is streamed straight out of the zip.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from legal_rag.contexts import load_selected_contexts
from legal_rag.sedar_retrieval.corpus_v4 import (
    BuildOptions,
    audit_corpus,
    build_document,
)


def _new_run_id(prefix: str) -> str:
    try:
        from legal_rag.sedar_retrieval.gates import new_run_id

        return new_run_id(prefix)
    except Exception:  # pragma: no cover - offline fallback
        from datetime import UTC, datetime

        return f"{datetime.now(UTC):%Y%m%d_%H%M%S}_{prefix}"


def _git_sha() -> str | None:
    try:
        from legal_rag.sedar_retrieval.gates import git_commit_sha

        return git_commit_sha()
    except Exception:  # pragma: no cover - offline fallback
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, default=Path("data/selected-contexts.zip"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--long-article-chars", type=int, default=2400)
    parser.add_argument("--min-child-chars", type=int, default=500)
    parser.add_argument("--max-child-chars", type=int, default=2400)
    parser.add_argument("--pack-max-chars", type=int, default=6000)
    parser.add_argument("--micro-unit-chars", type=int, default=120)
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Omit retrieval_text from units.jsonl. It is derivable as "
            "breadcrumb + newline + reader_text, and repeating it is why a full "
            "build is ~1.5 GB. export_corpus_v4_passages.py reconstructs it."
        ),
    )
    parser.add_argument("--no-children", action="store_true")
    parser.add_argument("--no-preamble", action="store_true")
    args = parser.parse_args()

    run_id = _new_run_id("corpus_v4")
    output_dir = args.output_dir or Path("artifacts/sedar_retrieval/corpus_v4") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    options = BuildOptions(
        long_article_chars=args.long_article_chars,
        min_child_chars=args.min_child_chars,
        max_child_chars=args.max_child_chars,
        pack_max_chars=args.pack_max_chars,
        micro_unit_chars=args.micro_unit_chars,
        emit_children=not args.no_children,
        keep_preamble=not args.no_preamble,
    )

    documents_path = output_dir / "documents.jsonl"
    units_path = output_dir / "units.jsonl"
    records = []
    units = []

    with (
        documents_path.open("w", encoding="utf-8") as documents_file,
        units_path.open("w", encoding="utf-8") as units_file,
    ):
        for position, document in enumerate(load_selected_contexts(args.contexts)):
            if args.max_documents and position >= args.max_documents:
                break
            record, document_units = build_document(document, options)
            records.append(record)
            units.extend(document_units)
            documents_file.write(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
            )
            for unit in document_units:
                payload = unit.model_dump(mode="json")
                if args.compact:
                    payload.pop("retrieval_text", None)
                units_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    report = audit_corpus(records, units, micro_unit_chars=args.micro_unit_chars)
    (output_dir / "audit.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "git_commit": _git_sha(),
        "contexts_path": str(args.contexts),
        "options": asdict(options),
        "compact": bool(args.compact),
        "documents_path": str(documents_path),
        "units_path": str(units_path),
        "corpus_sha256": sha256(args.contexts.read_bytes()).hexdigest()
        if args.contexts.is_file()
        else None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "audit": report.model_dump(mode="json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
