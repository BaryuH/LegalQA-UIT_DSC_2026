#!/usr/bin/env python
"""Precision / grounding guardrail over shipped answers.

METEOR rewards recall/length, so a longest-among-grounded selector can be gamed
toward verbose answers that drift from the evidence. This audits the *precision*
side of the final answers, independent of METEOR:

- unsupported_legal_id_rate: answers asserting a legal id / date not in evidence;
- mean_answer_grounding: fraction of each answer's 8-grams present in evidence
  (how much of the answer is actually backed by the retrieved text);
- refusal_rate.

Use as a secondary metric next to METEOR: a config that raises METEOR while
dropping answer_grounding or raising unsupported ids is metric-gaming, not
better law.

Read-only; no model; no gold. Joins final_answers.jsonl with the evidence file.

Usage (from zip_for_huy/):
    python benchmark_meteor_matrix/scripts/precision_audit.py \
        --final outputs/<run>/final_answers.jsonl \
        --evidence processed/train_dev200_rerank_evidence.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import grounding, overlap  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", required=True, help="final_answers.jsonl")
    parser.add_argument("--evidence", required=True, help="evidence jsonl {id, evidence}")
    parser.add_argument("--shingle-size", type=int, default=overlap.DEFAULT_SHINGLE_SIZE)
    args = parser.parse_args()

    final = _read_jsonl(Path(args.final))
    evidence = {str(r["id"]): str(r.get("evidence", "")) for r in _read_jsonl(Path(args.evidence))}

    rows = []
    for item in final:
        cid = str(item["id"])
        answer = str(item.get("answer", ""))
        ev = evidence.get(cid, "")
        decision = grounding.evaluate_candidate(answer, ev) if ev else grounding.GateDecision(True, ())
        unsupported_ids = [r for r in decision.reasons if r.startswith("unsupported_legal_id")]
        unsupported_dates = [r for r in decision.reasons if r.startswith("unsupported_date")]
        rows.append(
            {
                "id": cid,
                "answer_grounding": overlap.shingle_coverage(answer, ev, args.shingle_size) if ev else None,
                "unsupported_legal_ids": len(unsupported_ids),
                "unsupported_dates": len(unsupported_dates),
                "refusal": grounding.is_refusal(answer),
                "evidence_missing": not bool(ev.strip()),
            }
        )

    n = len(rows)
    grounded = [r["answer_grounding"] for r in rows if r["answer_grounding"] is not None]
    summary = {
        "answers": n,
        "shingle_size": args.shingle_size,
        "mean_answer_grounding": round(overlap.mean(grounded), 4) if grounded else None,
        "unsupported_legal_id_rate": round(sum(r["unsupported_legal_ids"] > 0 for r in rows) / n, 4) if n else 0.0,
        "unsupported_date_rate": round(sum(r["unsupported_dates"] > 0 for r in rows) / n, 4) if n else 0.0,
        "refusal_rate": round(sum(r["refusal"] for r in rows) / n, 4) if n else 0.0,
        "evidence_missing_rate": round(sum(r["evidence_missing"] for r in rows) / n, 4) if n else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = Path(args.final).with_name("precision_audit.json")
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
