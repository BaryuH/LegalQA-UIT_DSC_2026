#!/usr/bin/env python3
"""Build R1/R2a passage views from canonical nodes (TASK 04/05)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.corpus.context_augment import (
    assert_no_summary_in_reader,
    augment_retrieval_text_r2a,
)
from legal_rag.sedar_retrieval.corpus.hierarchy import nodes_to_passages
from legal_rag.sedar_retrieval.corpus.schema import CanonicalNode
from legal_rag.sedar_retrieval.gates import git_commit_sha, new_run_id
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines


def _load_nodes(path: Path) -> tuple[CanonicalNode, ...]:
    rows: list[CanonicalNode] = []
    for line in iter_jsonl_lines(path):
        rows.append(CanonicalNode.model_validate(json.loads(line)))
    return tuple(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", default="article,clause")
    args = parser.parse_args()

    run_id = new_run_id("r1_r2a_passages")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    nodes = _load_nodes(args.nodes)
    levels = tuple(x.strip() for x in args.levels.split(",") if x.strip())
    r1 = nodes_to_passages(nodes, levels=levels)  # type: ignore[arg-type]
    r2a = tuple(augment_retrieval_text_r2a(p) for p in r1)
    for p in r2a:
        assert_no_summary_in_reader(p)

    def write(path: Path, rows: tuple) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n"
                )

    write(out / "passages_r1.jsonl", r1)
    write(out / "passages_r2a.jsonl", r2a)
    manifest = {
        "run_id": run_id,
        "git_commit": git_commit_sha(),
        "nodes": str(args.nodes),
        "r1_count": len(r1),
        "r2a_count": len(r2a),
        "levels": list(levels),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
