#!/usr/bin/env python3
"""Score answer-in-context over a pack artifact (TASK 28).

Retrieval recall says whether the right article was in the ranked list. It does
not say whether the text needed to write the answer survived into the prompt,
and the two come apart: answer-in-context correlates with F1 at **+0.50** against
recall's **+0.31**, and separates a **4.6x EM gap even where every gold document
was retrieved** (arXiv:2607.00725). This project's own analysis found the same
shape from the other side - a starved bottom quartile that article@4 cannot see.

Input is either:

* ``--packs`` : the output of ``build_adaptive_pack.py`` (has ``pack_ids``), or
* ``--ranking`` + ``--top-k`` : any ranking JSONL, scoring the first k as if
  they were the pack. Use this to get an AIC baseline before the adaptive pack
  exists.

Reads gold answers. **Evaluation-only**: nothing it writes may enter a query, an
index, a prompt, or a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from legal_rag.evaluation.answer_in_context import (
    DEFAULT_SHINGLE_SIZE,
    answer_in_context,
    summarize_answer_in_context,
)
from legal_rag.sedar_retrieval.corpus_v4.citations import CitationIndex
from legal_rag.sedar_retrieval.corpus_v4.export import (
    article_id_for,
    build_article_id_map,
)
from legal_rag.sedar_retrieval.ranking.vietnamese_reranker import (
    load_rerank_units,
    unit_text,
)


def _require_file(path: Path, flag: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"{flag} does not exist or is not a file: {path}")
    return path


def _load_pack_ids(
    packs: Path | None, ranking: Path | None, top_k: int
) -> dict[str, list[str]]:
    source = packs or ranking
    if source is None:
        raise SystemExit("Pass --packs or --ranking")
    per_query: dict[str, list[str]] = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", "")).strip()
            if not query_id:
                continue
            ids = row.get("pack_ids")
            if not isinstance(ids, list) or not ids:
                ranked = row.get("ranked_ids")
                if not isinstance(ranked, list):
                    continue
                ids = ranked[:top_k]
            per_query[query_id] = [str(item) for item in ids]
    if not per_query:
        raise SystemExit(f"No pack rows found in {source}")
    return per_query


def _load_raw_units(path: Path) -> list[dict[str, Any]]:
    """Load corpus rows with the common id expected by the AIC mapper.

    ``load_rerank_units`` accepts both corpus v4 ``unit_id`` rows and v3
    ``passage_id`` rows.  The article-mapping pass must apply the same
    compatibility rule; otherwise a valid v3 AIC run fails after loading with
    ``KeyError: 'unit_id'``.
    """

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = dict(json.loads(line))
            unit_id = row.get("unit_id") or row.get("passage_id")
            if not unit_id:
                raise SystemExit(
                    f"Corpus row {line_number} in {path} has neither "
                    "'unit_id' nor 'passage_id'"
                )
            row["unit_id"] = str(unit_id)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No corpus rows found in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, default=None)
    parser.add_argument("--ranking", type=Path, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="With --ranking, treat the first k as the pack (champion is 6).",
    )
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Evaluation-only answers file, e.g. data/warmup.json.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--documents",
        type=Path,
        required=True,
        help="corpus_v4 documents.jsonl: supplies the citation aliases.",
    )
    parser.add_argument(
        "--overlap-shingle-size",
        type=int,
        default=DEFAULT_SHINGLE_SIZE,
        help=(
            "n-gram width for the secondary text-overlap signal. n=8 is "
            "calibrated (oracle 0.6035 vs random 0.0008); smaller n lets common "
            "Vietnamese phrasing leak in."
        ),
    )
    parser.add_argument(
        "--body-source",
        choices=("reader_text", "raw_text", "breadcrumb_reader_text"),
        default="reader_text",
        help="Must match what the reader actually receives, or the number lies.",
    )
    parser.add_argument("--per-query-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.overlap_shingle_size <= 0:
        raise SystemExit("--overlap-shingle-size must be positive")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    _require_file(args.units, "--units")
    _require_file(args.gold, "--gold")
    _require_file(args.documents, "--documents")
    if args.packs:
        _require_file(args.packs, "--packs")
    if args.ranking:
        _require_file(args.ranking, "--ranking")

    packs = _load_pack_ids(args.packs, args.ranking, args.top_k)
    units = load_rerank_units(args.units)
    gold = json.loads(args.gold.read_text(encoding="utf-8"))

    # Citations resolve to ARTICLE identity, so a pack unit counts for its
    # parent article and an article_part is not a miss.
    raw_units = _load_raw_units(args.units)
    article_id_map = build_article_id_map(raw_units)
    # Index by id first: resolving each unit's parent by scanning the list is
    # O(n^2) and hangs on a 300k-unit build.
    by_id = {str(row["unit_id"]): row for row in raw_units if row.get("unit_id")}
    unit_to_article: dict[str, str] = {}
    articles_seen: set[str] = set()
    for row in raw_units:
        parent = row.get("parent_unit_id")
        holder = by_id.get(str(parent), row) if parent else row
        article = article_id_for(holder, article_id_map=article_id_map)
        if article:
            unit_to_article[str(row["unit_id"])] = article
            articles_seen.add(article)

    index = CitationIndex()
    with args.documents.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index.add_document(
                record["document_id"],
                record.get("citation_aliases") or [record["citation"]],
                link=record.get("link"),
            )
    for article in articles_seen:
        document_id = article.split("::", 1)[0]
        number = article.rsplit("::", 1)[-1]
        # resolve() must return ARTICLE ids, which is what the pack is mapped to.
        index.add_article_unit(document_id, number, article)

    def resolve(citation):
        return index.resolve(citation)

    scope: set[str] | None = None
    if args.manifest is not None:
        _require_file(args.manifest, "--manifest")
        from legal_rag.finetuned_reader.warmup_eval import load_clean_warmup_manifest

        scope = set(load_clean_warmup_manifest(args.manifest).included_ids)

    results = []
    skipped_no_answer = 0
    skipped_no_pack = 0
    missing_units = 0
    for query_id, item in gold.items():
        if scope is not None and query_id not in scope:
            continue
        answer = (item or {}).get("answer")
        if not answer:
            skipped_no_answer += 1
            continue
        pack_ids = packs.get(query_id)
        if not pack_ids:
            skipped_no_pack += 1
            continue
        blocks = []
        pack_articles: set[str] = set()
        for unit_id in pack_ids:
            unit = units.get(unit_id)
            if unit is None:
                missing_units += 1
                continue
            blocks.append(unit_text(unit, text_field=args.body_source))
            article = unit_to_article.get(unit_id)
            if article:
                pack_articles.add(article)
        results.append(
            answer_in_context(
                query_id,
                answer,
                pack_articles,
                resolve=resolve,
                pack_text="\n\n".join(blocks),
                overlap_shingle_size=args.overlap_shingle_size,
            )
        )

    summary = summarize_answer_in_context(results)
    report: dict[str, Any] = {
        "source": str(args.packs or args.ranking),
        "units": str(args.units),
        "gold": str(args.gold),
        "body_source": args.body_source,
        "overlap_shingle_size": args.overlap_shingle_size,
        "top_k_if_ranking": args.top_k if args.ranking else None,
        "scored_queries": len(results),
        "skipped_without_answer": skipped_no_answer,
        "skipped_without_pack": skipped_no_pack,
        "pack_ids_missing_from_units": missing_units,
        "summary": summary.as_dict(),
    }

    if args.per_query_output:
        args.per_query_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_query_output.open("w", encoding="utf-8") as handle:
            for row in sorted(results, key=lambda item: item.citation_coverage):
                handle.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
