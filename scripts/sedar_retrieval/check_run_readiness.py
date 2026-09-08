#!/usr/bin/env python3
"""Pre-run consistency check for a SEDAR retrieval run (TASK 29).

Answers one question before any GPU time: **can this run start, and will the
numbers it produces mean what we think they mean?**

Every check here exists because of a defect this project has already hit, or a
documented trap:

``passages``
    row count of the view, and whether ids are unique. A view and an index that
    disagree on row count produce silently wrong metadata joins.
``bm25_index``
    that the manifest exists, and **what k1/b it was built with**. The index
    cache is keyed on ``sha256(config.as_dict())``, so a k1/b sweep is safe -
    but only if you know which config produced the file you are reading.
``dense_index``
    row count against the view, and the ``normalized`` flag. ``dense.py``
    encodes with ``normalize_embeddings=False`` and normalises downstream; if
    that step is skipped, inner-product search is not cosine and every dense
    score is wrong in a way no error message reports.
``fusion_depth``
    ``union_cap`` against the sum of per-leg ``top_k``. A cap below that
    silently truncates the fused list - the same class of defect as
    Elasticsearch's ``rank_window_size`` default of 10, and as
    ``evidence_top_k`` binding before ``max_total_chars``.
``pack_budget``
    ``evidence_top_k`` / ``candidate_window`` / ``max_total_chars`` /
    ``max_chunks_per_document`` against the champion (6 / 6000 / 3), and whether
    backfill is enabled at all.
``labels``
    that the label ids actually intersect the passage view. This is the check
    that catches a corpus migration: labels built against v3 ids score 0.0
    against a v4 view unless the ids were kept compatible.

Exit code is 0 when every check passes, 1 when any FAIL fires. WARN does not
fail the run; it is a thing to write down before reading the metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CHAMPION_PACK = {"evidence_top_k": 6, "max_total_chars": 6000, "max_chunks_per_document": 3}


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str, **extra: Any) -> None:
        self.rows.append({"check": name, "status": status, "detail": detail, **extra})

    @property
    def failed(self) -> bool:
        return any(row["status"] == "FAIL" for row in self.rows)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find(payload: Any, *keys: str) -> Any:
    """Best-effort lookup of a key anywhere in a nested manifest."""

    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                return payload[key]
        for value in payload.values():
            found = _find(value, *keys)
            if found is not None:
                return found
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--bm25-manifest", type=Path, default=None)
    parser.add_argument("--dense-manifest", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="Clean-warmup manifest.")
    parser.add_argument("--union-cap", type=int, default=None)
    parser.add_argument("--leg-top-k", type=int, action="append", default=None)
    parser.add_argument("--evidence-top-k", type=int, default=None)
    parser.add_argument("--candidate-window", type=int, default=None)
    parser.add_argument("--max-total-chars", type=int, default=None)
    parser.add_argument("--max-chunks-per-document", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checks = Checks()

    # --- passages view ---
    if not args.passages.is_file():
        checks.add("passages", "FAIL", f"missing: {args.passages}")
        passage_ids: set[str] = set()
    else:
        passage_ids = set()
        duplicates = 0
        levels: dict[str, int] = {}
        with args.passages.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pid = str(row.get("passage_id") or row.get("unit_id") or "")
                if pid in passage_ids:
                    duplicates += 1
                passage_ids.add(pid)
                level = str(row.get("retrieval_level") or row.get("level") or "?")
                levels[level] = levels.get(level, 0) + 1
        checks.add(
            "passages",
            "FAIL" if duplicates else "PASS",
            f"{len(passage_ids)} unique ids, {duplicates} duplicates",
            levels=levels,
        )

    # --- BM25 index ---
    if args.bm25_manifest:
        payload = _read_json(args.bm25_manifest)
        if payload is None:
            checks.add("bm25_index", "FAIL", f"unreadable: {args.bm25_manifest}")
        else:
            k1 = _find(payload, "k1")
            b = _find(payload, "b")
            count = _find(payload, "document_count", "documents", "row_count")
            status = "PASS"
            notes = [f"k1={k1} b={b} docs={count}"]
            if k1 is None or b is None:
                status = "WARN"
                notes.append("config k1/b not recorded in the manifest")
            if count is not None and passage_ids and int(count) != len(passage_ids):
                status = "FAIL"
                notes.append(f"index rows {count} != view rows {len(passage_ids)}")
            checks.add("bm25_index", status, "; ".join(notes))

    # --- dense index ---
    if args.dense_manifest:
        payload = _read_json(args.dense_manifest)
        if payload is None:
            checks.add("dense_index", "FAIL", f"unreadable: {args.dense_manifest}")
        else:
            normalized = _find(payload, "normalized")
            count = _find(payload, "row_count", "rows", "passage_count", "document_count")
            dim = _find(payload, "embedding_dim", "dim")
            status = "PASS"
            notes = [f"normalized={normalized} rows={count} dim={dim}"]
            if normalized is not True:
                status = "FAIL"
                notes.append(
                    "embeddings are not marked normalized: inner-product search "
                    "is then not cosine and every dense score is wrong"
                )
            if count is not None and passage_ids and int(count) != len(passage_ids):
                status = "FAIL"
                notes.append(f"index rows {count} != view rows {len(passage_ids)}")
            checks.add("dense_index", status, "; ".join(notes))

    # --- fusion depth ---
    if args.union_cap is not None and args.leg_top_k:
        needed = sum(args.leg_top_k)
        status = "PASS" if args.union_cap >= needed else "FAIL"
        checks.add(
            "fusion_depth",
            status,
            f"union_cap={args.union_cap} vs sum(leg top_k)={needed}"
            + ("" if status == "PASS" else " — the fused list is silently truncated"),
        )

    # --- pack budget ---
    if args.evidence_top_k is not None:
        notes = []
        status = "PASS"
        window = args.candidate_window or 0
        if window and window < args.evidence_top_k:
            status = "FAIL"
            notes.append("candidate_window < evidence_top_k")
        elif not window or window == args.evidence_top_k:
            status = "WARN"
            notes.append(
                "backfill disabled (candidate_window == evidence_top_k): anything "
                "the per-document cap or char budget drops is not replaced"
            )
        for key, value in (
            ("evidence_top_k", args.evidence_top_k),
            ("max_total_chars", args.max_total_chars),
            ("max_chunks_per_document", args.max_chunks_per_document),
        ):
            if value is not None and value != CHAMPION_PACK[key]:
                notes.append(f"{key}={value} differs from champion {CHAMPION_PACK[key]}")
        checks.add("pack_budget", status, "; ".join(notes) or "matches champion")

    # --- labels intersect the view ---
    if args.labels:
        if not args.labels.is_file():
            checks.add("labels", "FAIL", f"missing: {args.labels}")
        else:
            label_ids: set[str] = set()
            with args.labels.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    for item in row.get("relevant_ids") or []:
                        label_ids.add(str(item))
            article_ids: set[str] = set()
            if args.passages.is_file():
                with args.passages.open(encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        value = json.loads(line).get("article_id")
                        if value:
                            article_ids.add(str(value))
            hits = label_ids & (article_ids | passage_ids)
            coverage = len(hits) / max(1, len(label_ids))
            status = "PASS" if coverage >= 0.95 else ("WARN" if coverage >= 0.5 else "FAIL")
            checks.add(
                "labels",
                status,
                f"{len(hits)}/{len(label_ids)} label ids resolve into the view "
                f"(coverage {coverage:.4f})"
                + (
                    ""
                    if coverage >= 0.95
                    else " — a corpus migration usually causes this; re-export "
                    "with label-compatible article ids or re-project the labels"
                ),
            )

    # --- questions / scope ---
    if args.questions and args.questions.is_file():
        payload = _read_json(args.questions)
        if isinstance(payload, dict):
            checks.add("questions", "PASS", f"{len(payload)} questions")
    if args.manifest and args.manifest.is_file():
        checks.add("scope_manifest", "PASS", f"present: {args.manifest}")

    report = {
        "schema_version": "sedar-run-readiness-v1",
        "status": "FAIL" if checks.failed else "PASS",
        "checks": checks.rows,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
