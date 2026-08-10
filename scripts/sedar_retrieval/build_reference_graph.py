#!/usr/bin/env python3
"""Build reference graph from canonical nodes (TASK 17 local)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.sedar_retrieval.corpus.reference_graph import parse_reference_edges
from legal_rag.sedar_retrieval.corpus.schema import CanonicalNode
from legal_rag.sedar_retrieval.gates import new_run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nodes = tuple(
        CanonicalNode.model_validate(json.loads(line))
        for line in args.nodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    edges = parse_reference_edges(nodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(
                json.dumps(
                    {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "relation_type": edge.relation_type,
                        "original_reference_text": edge.original_reference_text,
                        "resolution_status": edge.resolution_status,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    resolved = sum(1 for e in edges if e.resolution_status == "resolved")
    print(
        json.dumps(
            {
                "run_id": new_run_id("reference_graph"),
                "edges": len(edges),
                "resolved": resolved,
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
