#!/usr/bin/env python3
"""Verify a SEDAR corpus v4 build: text integrity and citation coverage.

Two checks, both offline and deterministic:

**Integrity.** For a random sample of documents, every unit's ``reader_text``
must appear in the normalised source. This is what catches a normalisation or
segmentation bug that silently rewrites legal text - the failure mode that
matters most in a corpus whose whole job is to be quoted verbatim.

**Citation coverage.** Gold answers cite law in prose. Every ``Điều N <document>``
pair found in an answer is resolved against the built corpus. Self-references
("Nghị định này") are reported separately because they are unresolvable by
construction, not a corpus defect.

Reading gold answers here is an *evaluation-only* use: nothing this script
produces is allowed into a retrieval query, an index, a prompt or an inference
artifact.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from legal_rag.contexts import load_selected_contexts
from legal_rag.sedar_retrieval.corpus_v4.citations import (
    CitationIndex,
    extract_citations,
)
from legal_rag.sedar_retrieval.corpus_v4.normalize import (
    normalize_source_text,
    strip_distribution_block,
)


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, default=Path("data/selected-contexts.zip"))
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Evaluation-only answers file (e.g. data/warmup.json).",
    )
    parser.add_argument("--sample-documents", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-chars", type=int, default=160)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    documents_path = args.build_dir / "documents.jsonl"
    units_path = args.build_dir / "units.jsonl"

    records = {}
    for line in documents_path.open(encoding="utf-8"):
        record = json.loads(line)
        records[record["document_id"]] = record

    random.seed(args.seed)
    sample = set(
        random.sample(sorted(records), min(args.sample_documents, len(records)))
    )

    index = CitationIndex()
    for record in records.values():
        index.add_document(
            record["document_id"],
            record.get("citation_aliases") or [record["citation"]],
            link=record.get("link"),
        )

    sampled_units = defaultdict(list)
    for line in units_path.open(encoding="utf-8"):
        unit = json.loads(line)
        if unit["level"] == "article":
            index.add_article_unit(
                unit["document_id"], unit.get("article_number"), unit["unit_id"]
            )
        if unit["document_id"] in sample:
            sampled_units[unit["document_id"]].append(unit)

    sources = {}
    for document in load_selected_contexts(args.contexts):
        if document.id in sample:
            sources[document.id] = _fold(
                strip_distribution_block(normalize_source_text(document.passage))
            )
        if len(sources) == len(sample):
            break

    found = missing = 0
    failures = []
    for document_id, units in sampled_units.items():
        source = sources.get(document_id, "")
        for unit in units:
            if unit["level"] == "preamble":
                continue
            body = unit["reader_text"]
            if unit["level"] == "article_part" and "\n" in body:
                # Children carry the parent heading, which is not contiguous
                # with the child body in the source.
                body = body.split("\n", 1)[1]
            probe = _fold(body)[: args.probe_chars]
            if probe and probe in source:
                found += 1
            else:
                missing += 1
                if len(failures) < 20:
                    failures.append(
                        {
                            "unit_id": unit["unit_id"],
                            "level": unit["level"],
                            "probe": probe[:120],
                        }
                    )

    report = {
        "build_dir": str(args.build_dir),
        "integrity": {
            "sampled_documents": len(sample),
            "units_checked": found + missing,
            "units_found_in_source": found,
            "units_missing": missing,
            "integrity_rate": round(found / max(1, found + missing), 4),
            "failures": failures,
        },
    }

    if args.gold and args.gold.is_file():
        gold = json.loads(args.gold.read_text(encoding="utf-8"))
        total = resolved = self_reference = 0
        per_query = []
        unresolved = []
        for item in gold.values():
            answer = item.get("answer") or ""
            citations = extract_citations(answer)
            hits = need = 0
            for citation in citations:
                if citation.self_reference:
                    self_reference += 1
                    continue
                total += 1
                need += 1
                if index.resolve(citation):
                    resolved += 1
                    hits += 1
                elif len(unresolved) < 20:
                    unresolved.append(
                        {
                            "article": citation.article_number,
                            "reference": citation.reference[:80],
                        }
                    )
            per_query.append((hits, need) if need else None)
        scored = [x for x in per_query if x]
        report["citation_coverage"] = {
            "gold_file": str(args.gold),
            "resolvable_citations": total,
            "resolved": resolved,
            "resolution_rate": round(resolved / max(1, total), 4),
            "self_reference_citations": self_reference,
            "queries_with_resolvable_citation": len(scored),
            "queries_fully_resolved": sum(1 for h, n in scored if h == n),
            "queries_partially_resolved": sum(1 for h, n in scored if 0 < h < n),
            "queries_unresolved": sum(1 for h, n in scored if h == 0),
            "unresolved_examples": unresolved,
        }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
