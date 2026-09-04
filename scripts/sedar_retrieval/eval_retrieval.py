#!/usr/bin/env python3
"""Evaluate retrieval predictions with SEDAR Retrieval v3 metrics (TASK 02).

Level-aware since the M1 fix: ``--passages`` is required so that
``document_recall_at`` / ``article_recall_at`` / ``clause_recall_at`` and
``wrong_document_rate`` are actually computed.  Before the fix these mappings
were never passed, so every level recall reported ``0.0`` and ``recall_at`` /
``mrr_at_10`` / ``ndcg_at_10`` only credited an exact ``passage_id`` match.
Because silver labels prefer article-level passage IDs while the retrieval
views index both ``article`` and ``clause`` levels, a system that returned the
correct clause of the correct article scored as a total miss.

Article identity is always document-scoped: the fallback key is
``{document_id}::art::{article_number}``, never a bare article number, so two
different documents that share ``Điều 76`` are never merged.

``mrr_at_10`` / ``ndcg_at_10`` inside ``evaluate_retrieval`` only ever consider
``relevant_ids``, so they stay exact-passage even with the level maps wired.
This script therefore reports two bundles:

``exact``
    the historical semantics — one specific passage_id must be retrieved.
``article_expanded``
    ``relevant_ids`` widened to every passage sharing an article with a labeled
    passage, so ``recall_at`` / ``mrr_at_10`` / ``ndcg_at_10`` become
    article-level. Use this bundle for ranking gates; ``article_recall_at`` in
    either bundle answers the recall question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_rag.finetuned_reader.warmup_eval import load_clean_warmup_manifest
from legal_rag.sedar_retrieval.eval.retrieval_metrics import (
    QueryRelevance,
    RankedList,
    evaluate_retrieval,
    metrics_to_dict,
)
from legal_rag.sedar_retrieval.io.jsonl import iter_jsonl_lines, load_jsonl_records

DEFAULT_CUTOFFS = "4,10,20,50,100"


def _require_input_file(path: Path, flag: str) -> Path:
    """Fail loudly on an unset shell variable instead of reading a directory.

    ``--labels "$SILVER_LABELS"`` with the variable unset hands argparse an
    empty string, and ``Path("")`` is ``Path(".")`` — so the read reaches the
    working directory and dies with ``IsADirectoryError: '.'`` several frames
    deep, pointing at the JSONL reader rather than at the missing export.
    """

    if str(path) in {"", "."}:
        raise SystemExit(
            f"{flag} resolved to an empty path. An unset shell variable "
            f"expands to '' and Path('') is '.', so the read would hit the "
            f"working directory. Export the variable you passed to {flag} and "
            f"verify it with 'ls -l' before re-running."
        )
    if path.is_dir():
        raise SystemExit(f"{flag} is a directory, expected a file: {path}")
    if not path.is_file():
        raise SystemExit(f"{flag} does not exist: {path}")
    return path


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return load_jsonl_records(path)


def _parse_cutoffs(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise SystemExit(f"Invalid --cutoffs entry: {item!r}") from exc
        if value <= 0:
            raise SystemExit("--cutoffs entries must be positive")
        values.append(value)
    if not values:
        raise SystemExit("--cutoffs must contain at least one positive integer")
    return tuple(sorted(set(values)))


def _load_level_maps(
    passages_path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Stream the passage view and build passage_id -> level identity maps.

    Read as raw JSON rather than through the pydantic model: this only needs
    five fields and the view can hold hundreds of thousands of rows.
    """

    to_document: dict[str, str] = {}
    to_article: dict[str, str] = {}
    to_clause: dict[str, str] = {}

    for line in iter_jsonl_lines(passages_path):
        row = json.loads(line)
        passage_id = str(row.get("passage_id", "")).strip()
        if not passage_id:
            raise SystemExit(f"Passage row without passage_id in {passages_path}")
        document_id = str(row.get("document_id", "")).strip()
        if not document_id:
            raise SystemExit(f"Passage {passage_id!r} has no document_id")
        if passage_id in to_document:
            raise SystemExit(f"Duplicate passage_id in view: {passage_id!r}")
        to_document[passage_id] = document_id

        article_id = row.get("article_id")
        article_number = row.get("article_number")
        if article_id:
            # Document-scoped by construction (canonical IDs embed the document).
            to_article[passage_id] = str(article_id)
        elif article_number:
            # Fallback stays document-scoped: never a bare article number.
            to_article[passage_id] = f"{document_id}::art::{article_number}"

        clause_id = row.get("clause_id")
        if clause_id:
            to_clause[passage_id] = str(clause_id)

    if not to_document:
        raise SystemExit(f"Passage view is empty: {passages_path}")
    return to_document, to_article, to_clause


def _article_members(
    passage_to_article: dict[str, str],
) -> dict[str, frozenset[str]]:
    """Return article key -> every passage_id belonging to that article."""

    grouped: dict[str, set[str]] = {}
    for passage_id, article_key in passage_to_article.items():
        grouped.setdefault(article_key, set()).add(passage_id)
    return {key: frozenset(values) for key, values in grouped.items()}


def _expand_labels_to_article(
    labels: list[QueryRelevance],
    *,
    passage_to_article: dict[str, str],
    article_members: dict[str, frozenset[str]],
) -> list[QueryRelevance]:
    """Widen relevant_ids to every sibling passage of each labeled article.

    A labeled article-level passage and its own clauses are the same legal
    provision, so crediting either is correct. Passages without an article
    mapping are kept as-is.
    """

    expanded: list[QueryRelevance] = []
    for item in labels:
        widened: set[str] = set(item.relevant_ids)
        for passage_id in item.relevant_ids:
            article_key = passage_to_article.get(passage_id)
            if article_key is None:
                continue
            widened |= article_members.get(article_key, frozenset())
        expanded.append(
            QueryRelevance(
                query_id=item.query_id,
                relevant_ids=frozenset(widened),
                graded=item.graded,
                provenance=item.provenance,
                relevant_document_ids=item.relevant_document_ids,
            )
        )
    return expanded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--passages",
        type=Path,
        required=True,
        help=(
            "Canonical passage view (passages_r2a.jsonl) used to build the "
            "document/article/clause level mappings. Required: without it the "
            "level recalls silently report 0.0 and only exact passage_id "
            "matches are credited."
        ),
    )
    parser.add_argument(
        "--cutoffs",
        default=DEFAULT_CUTOFFS,
        help=(
            "Comma-separated cutoffs. Keep 4 in the list: it is the evidence "
            f"pack cutoff and the reader's upper bound. Default: {DEFAULT_CUTOFFS}"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "VAL-00 clean-warmup manifest. Without it the scope is every query "
            "in --pred (all 500 warmup IDs), which is NOT the clean-460 scope "
            "every champion and e2e number uses — the two are not comparable. "
            "Pass the manifest whenever the result will be read next to an e2e "
            "METEOR/ROUGE-L figure."
        ),
    )
    parser.add_argument("--mrr-cutoff", type=int, default=10)
    parser.add_argument("--ndcg-cutoff", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing metrics artifact.",
    )
    args = parser.parse_args()

    if args.mrr_cutoff <= 0 or args.ndcg_cutoff <= 0:
        raise SystemExit("--mrr-cutoff and --ndcg-cutoff must be positive")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite artifact: {args.output}")

    cutoffs = _parse_cutoffs(args.cutoffs)

    # Validate every input up front: one clear message beats three separate
    # failures discovered one run at a time.
    _require_input_file(args.pred, "--pred")
    _require_input_file(args.labels, "--labels")
    _require_input_file(args.passages, "--passages")

    preds_raw = _load_jsonl(args.pred)
    labels_raw = _load_jsonl(args.labels)
    passage_to_document, passage_to_article, passage_to_clause = _load_level_maps(
        args.passages
    )

    predictions = [
        RankedList(
            query_id=str(row["query_id"]),
            ranked_ids=tuple(str(x) for x in row["ranked_ids"]),  # type: ignore[index]
        )
        for row in preds_raw
    ]
    labels = []
    for row in labels_raw:
        provenance = str(row.get("provenance", "unlabeled"))
        if provenance not in {"gold", "silver", "unlabeled"}:
            raise SystemExit(f"Invalid provenance: {provenance}")
        relevant = frozenset(str(x) for x in row.get("relevant_ids", []))  # type: ignore[arg-type]
        labels.append(
            QueryRelevance(
                query_id=str(row["query_id"]),
                relevant_ids=relevant,
                provenance=provenance,  # type: ignore[arg-type]
            )
        )

    id_scope = "all_predictions"
    manifest_path: str | None = None
    n_predictions_before_scope = len(predictions)
    if args.manifest is not None:
        _require_input_file(args.manifest, "--manifest")
        manifest = load_clean_warmup_manifest(args.manifest)
        included = set(manifest.included_ids)
        predictions = [item for item in predictions if item.query_id in included]
        if not predictions:
            raise SystemExit(
                f"No prediction query_id is in the manifest: {args.manifest}"
            )
        missing = sorted(included - {item.query_id for item in predictions})
        if missing:
            raise SystemExit(
                f"{len(missing)} manifest IDs are absent from --pred "
                f"(e.g. {missing[:5]}); the ranking does not cover the clean scope."
            )
        id_scope = "clean_manifest"
        manifest_path = str(args.manifest)

    pred_ids = {item.query_id for item in predictions}
    labels = [item for item in labels if item.query_id in pred_ids]
    if len(labels) != len(predictions):
        raise SystemExit(
            "Label/prediction query_id mismatch after intersection "
            f"({len(labels)} labels vs {len(predictions)} preds)"
        )

    # Fail closed on labels whose relevant IDs are absent from the view: that
    # means the labels and the ranking were built against different corpora.
    unknown = sorted(
        {
            passage_id
            for item in labels
            for passage_id in item.relevant_ids
            if passage_id not in passage_to_document
        }
    )
    if unknown:
        raise SystemExit(
            "Label relevant_ids missing from the passage view "
            f"({len(unknown)} IDs, e.g. {unknown[:5]}). "
            "Labels, rankings and --passages must share one corpus fingerprint."
        )

    def _evaluate(label_rows: list[QueryRelevance]) -> dict[str, object]:
        return metrics_to_dict(
            evaluate_retrieval(
                predictions,
                label_rows,
                cutoffs=cutoffs,
                passage_to_document=passage_to_document,
                passage_to_article=passage_to_article,
                passage_to_clause=passage_to_clause,
                mrr_cutoff=args.mrr_cutoff,
                ndcg_cutoff=args.ndcg_cutoff,
            )
        )

    article_members = _article_members(passage_to_article)
    labels_expanded = _expand_labels_to_article(
        labels,
        passage_to_article=passage_to_article,
        article_members=article_members,
    )

    metrics = _evaluate(labels)
    metrics["article_expanded"] = _evaluate(labels_expanded)
    metrics["evaluator"] = {
        "level_aware": True,
        "cutoffs": list(cutoffs),
        "mrr_cutoff": args.mrr_cutoff,
        "ndcg_cutoff": args.ndcg_cutoff,
        "id_scope": id_scope,
        "manifest_path": manifest_path,
        "n_predictions_before_scope": n_predictions_before_scope,
        "n_predictions_scored": len(predictions),
        "pred_path": str(args.pred),
        "labels_path": str(args.labels),
        "passages_path": str(args.passages),
        "article_key_policy": "article_id_or_document_scoped_article_number",
        "passage_count": len(passage_to_document),
        "article_mapped_passages": len(passage_to_article),
        "clause_mapped_passages": len(passage_to_clause),
        "bundles": {
            "top_level": "exact passage_id relevance (historical semantics)",
            "article_expanded": (
                "relevant_ids widened to article siblings; use for ranking gates"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path = args.output.with_suffix(".md")
    lines = ["# Retrieval metrics", ""]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(args.output), "markdown": str(md_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
