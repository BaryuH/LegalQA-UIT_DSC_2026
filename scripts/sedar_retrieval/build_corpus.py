#!/usr/bin/env python3
"""Build canonical legal hierarchy corpus (TASK 03/04 scaffold)."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from legal_rag.contexts import load_selected_contexts
from legal_rag.sedar_retrieval.corpus.hierarchy import (
    audit_canonical_nodes,
    nodes_to_passages,
)
from legal_rag.sedar_retrieval.corpus.parse_legal import parse_legal_document
from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.io.jsonl import count_jsonl_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("data/selected-contexts.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: artifacts/sedar_retrieval/canonical/<run_id>",
    )
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--levels", default="article,clause")
    parser.add_argument(
        "--compact-nodes",
        action="store_true",
        help=(
            "Skip point/retained nodes in nodes.jsonl to reduce disk "
            "(audit still counts them)."
        ),
    )
    args = parser.parse_args()

    run_id = new_run_id("canonical_corpus")
    output_dir = args.output_dir or Path("artifacts/sedar_retrieval/canonical") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = list(load_selected_contexts(args.contexts))
    if args.max_documents > 0:
        documents = documents[: args.max_documents]

    levels = tuple(item.strip() for item in args.levels.split(",") if item.strip())
    nodes_path = output_dir / "nodes.jsonl"
    passages_path = output_dir / "passages.jsonl"
    all_nodes = []
    source_texts = {doc.id: doc.passage for doc in documents}
    compact_skip = {"point", "retained"} if args.compact_nodes else set()

    with (
        nodes_path.open("w", encoding="utf-8") as nodes_f,
        passages_path.open("w", encoding="utf-8") as passages_f,
    ):
        for document in documents:
            nodes = parse_legal_document(document)
            all_nodes.extend(nodes)
            for node in nodes:
                if args.compact_nodes and (
                    node.level in compact_skip or node.parse_status == "retained"
                ):
                    continue
                nodes_f.write(
                    json.dumps(node.model_dump(mode="json"), ensure_ascii=False) + "\n"
                )
            for passage in nodes_to_passages(nodes, levels=levels):  # type: ignore[arg-type]
                passages_f.write(
                    json.dumps(passage.model_dump(mode="json"), ensure_ascii=False)
                    + "\n"
                )

    audit = audit_canonical_nodes(all_nodes, source_texts=source_texts)
    audit_payload = audit.model_dump(mode="json")
    audit_payload["passage_count"] = count_jsonl_records(passages_path)
    (output_dir / "audit.json").write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "git_commit": git_commit_sha(),
        "contexts_path": str(args.contexts),
        "document_count": len(documents),
        "levels": list(levels),
        "nodes_path": str(nodes_path),
        "passages_path": str(passages_path),
        "audit_path": str(output_dir / "audit.json"),
        "corpus_sha256": sha256(args.contexts.read_bytes()).hexdigest()
        if args.contexts.is_file()
        else None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
