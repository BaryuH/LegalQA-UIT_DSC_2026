#!/usr/bin/env python
"""Evidence-answerability audit: is the bottleneck retrieval or generation?

For each dev case, measure how much of the gold answer is actually present in the
retrieved evidence (verbatim 8-gram coverage, unigram recall, and gold legal-id
presence). Legal answers quote statutes, so low coverage means the right law was
never retrieved (retrieval-limited); high coverage but low oracle METEOR means
the model failed to use it (generation-limited).

Optionally cross with a run's per_case.jsonl oracle to show METEOR ceiling split
by coverage bucket -> tells you whether to spend effort on C (retrieval) or B
(generation).

Read-only; no model. Gold is used only for this offline audit.

Usage (from zip_for_huy/):
    python benchmark_meteor_matrix/scripts/evidence_answerability.py \
        --data processed/train_dev200.json \
        --evidence processed/train_dev200_rerank_evidence.jsonl \
        [--run-dir outputs/<run>] [--coverage-threshold 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import grounding, overlap  # noqa: E402
from benchmark.data import load_dataset  # noqa: E402

_LEGAL_ID_RE = grounding._LEGAL_ID_RE  # reuse the exact identifier shape


def _legal_id_presence(gold: str, evidence: str) -> float | None:
    ids = set(_LEGAL_ID_RE.findall(gold))
    if not ids:
        return None
    ev_ids = set(_LEGAL_ID_RE.findall(evidence))
    return len(ids & ev_ids) / len(ids)


def _load_oracle(run_dir: Path) -> dict[str, float]:
    oracle: dict[str, float] = {}
    per_case = run_dir / "per_case.jsonl"
    if not per_case.exists():
        return oracle
    with per_case.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mets = row.get("gold_meteor_per_candidate") or [0.0]
            oracle[str(row["id"])] = max(mets)
    return oracle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--shingle-size", type=int, default=overlap.DEFAULT_SHINGLE_SIZE)
    args = parser.parse_args()

    dataset = load_dataset(args.data, evidence_path=args.evidence)
    oracle = _load_oracle(Path(args.run_dir)) if args.run_dir else {}

    rows = []
    for case in dataset.cases:
        ev = case.evidence or ""
        cov = overlap.shingle_coverage(case.gold, ev, args.shingle_size)
        rows.append(
            {
                "id": case.id,
                "gold_shingle_coverage": cov,
                "gold_unigram_recall": overlap.unigram_recall(case.gold, ev),
                "gold_legal_id_presence": _legal_id_presence(case.gold, ev),
                "evidence_empty": not bool(ev.strip()),
                "oracle_meteor": oracle.get(case.id),
            }
        )

    n = len(rows)
    cov_vals = [r["gold_shingle_coverage"] for r in rows]
    uni_vals = [r["gold_unigram_recall"] for r in rows]
    id_vals = [r["gold_legal_id_presence"] for r in rows if r["gold_legal_id_presence"] is not None]
    low = [r for r in rows if r["gold_shingle_coverage"] < args.coverage_threshold]
    retrieval_limited_rate = len(low) / n if n else 0.0

    summary = {
        "cases": n,
        "shingle_size": args.shingle_size,
        "coverage_threshold": args.coverage_threshold,
        "mean_gold_shingle_coverage": round(overlap.mean(cov_vals), 4),
        "mean_gold_unigram_recall": round(overlap.mean(uni_vals), 4),
        "mean_gold_legal_id_presence": round(overlap.mean(id_vals), 4) if id_vals else None,
        "evidence_empty_rate": round(sum(r["evidence_empty"] for r in rows) / n, 4) if n else 0.0,
        "retrieval_limited_rate": round(retrieval_limited_rate, 4),
    }

    # If oracle is available, split the METEOR ceiling by coverage bucket.
    graded = [r for r in rows if r["oracle_meteor"] is not None]
    if graded:
        hi = [r for r in graded if r["gold_shingle_coverage"] >= args.coverage_threshold]
        lo = [r for r in graded if r["gold_shingle_coverage"] < args.coverage_threshold]
        summary["oracle_meteor_high_coverage"] = round(
            overlap.mean([r["oracle_meteor"] for r in hi]), 4
        ) if hi else None
        summary["oracle_meteor_low_coverage"] = round(
            overlap.mean([r["oracle_meteor"] for r in lo]), 4
        ) if lo else None
        summary["verdict"] = (
            "retrieval-limited: raise coverage (track C)"
            if retrieval_limited_rate >= 0.4
            else "generation-limited: evidence has the answer (track B)"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.run_dir:
        out = Path(args.run_dir) / "evidence_answerability.json"
        out.write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
