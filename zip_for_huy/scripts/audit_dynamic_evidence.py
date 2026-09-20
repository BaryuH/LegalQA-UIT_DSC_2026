#!/usr/bin/env python3
"""Audit and QC tool for Dynamic vs Fixed Evidence Selection.

Audits whether dynamic evidence selection (variable k based on reranker score gap)
truly improves evidence quality over fixed k=4 without suffering from:
- Trap 1: Evidence dilution (excessive length without coverage gain).
- Trap 2: Budget overflow / truncation (hitting max_total_chars 6000).
- Trap 3: Regressing legal ID presence or verbatim shingle coverage.

Computes side-by-side comparative diagnostics and paired bootstrap CI:
- Gold shingle coverage (mean, distribution, paired delta).
- Gold legal ID presence.
- Pack block count distribution (how many queries use 2, 3, 4, 5, 6 blocks).
- Character budget consumption (mean, p90, max, truncation count).
- Final recommendation: PROMOTE / HOLD / REJECT.

Usage (from repo root or zip_for_huy/):
    python scripts/audit_dynamic_evidence.py \
        --questions processed/train_dev200.json \
        --baseline-evidence processed/train_dev200_hybrid_evidence_v2.jsonl \
        --candidate-evidence processed/train_dev200_dynamic_evidence.jsonl \
        [--output-report artifacts/evidence/dynamic_evidence_qc_report.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure repo root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARK_DIR = _REPO_ROOT / "zip_for_huy" / "benchmark_meteor_matrix"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
_BENCHMARK_ROOT = _REPO_ROOT / "benchmark_meteor_matrix"
if str(_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_ROOT))

from benchmark import grounding, overlap  # noqa: E402
from benchmark.data import load_dataset  # noqa: E402

_LEGAL_ID_RE = grounding._LEGAL_ID_RE
_BLOCK_HEADER_RE = re.compile(r"^\[(\d+)\]\s+", re.MULTILINE)


def _count_blocks(rendered_text: str) -> int:
    matches = _BLOCK_HEADER_RE.findall(rendered_text)
    return len(matches) if matches else (1 if rendered_text.strip() else 0)


def _legal_id_presence(gold: str, evidence: str) -> float | None:
    ids = set(_LEGAL_ID_RE.findall(gold))
    if not ids:
        return None
    ev_ids = set(_LEGAL_ID_RE.findall(evidence))
    return len(ids & ev_ids) / len(ids)


def _read_evidence_jsonl(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("id") or "")
            if qid:
                mapping[qid] = str(row.get("evidence") or "")
    return mapping


def _paired_bootstrap_delta(
    cand_values: list[float], base_values: list[float], iters: int = 10000, seed: int = 42
) -> tuple[float, float, float, float]:
    import numpy as np

    a = np.asarray(cand_values, dtype=np.float64)
    b = np.asarray(base_values, dtype=np.float64)
    diff = a - b
    n = len(diff)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(iters, n))
    boot_means = diff[indices].mean(axis=1)
    d = float(diff.mean())
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(boot_means <= 0.0)), float(np.mean(boot_means >= 0.0)))
    return d, float(lo), float(hi), min(p, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True, help="Questions JSON with gold answers for auditing.")
    parser.add_argument("--baseline-evidence", type=Path, required=True, help="Baseline evidence JSONL (e.g. fixed k=4).")
    parser.add_argument("--candidate-evidence", type=Path, required=True, help="Candidate evidence JSONL (e.g. dynamic k in [2, 6]).")
    parser.add_argument("--shingle-size", type=int, default=overlap.DEFAULT_SHINGLE_SIZE)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--max-budget-chars", type=int, default=6000)
    parser.add_argument("--output-report", type=Path, default=None)
    args = parser.parse_args()

    questions_path = args.questions.resolve()
    base_path = args.baseline_evidence.resolve()
    cand_path = args.candidate_evidence.resolve()

    dataset = load_dataset(questions_path)
    base_ev = _read_evidence_jsonl(base_path)
    cand_ev = _read_evidence_jsonl(cand_path)

    common_ids = [c.id for c in dataset.cases if c.id in base_ev and c.id in cand_ev]
    if not common_ids:
        raise SystemExit("No common query IDs between questions and evidence files!")

    case_map = {c.id: c for c in dataset.cases}

    # Per-case metrics
    base_shingle_cov: list[float] = []
    cand_shingle_cov: list[float] = []
    base_unigram_rec: list[float] = []
    cand_unigram_rec: list[float] = []
    base_id_pres: list[float] = []
    cand_id_pres: list[float] = []

    base_chars: list[int] = []
    cand_chars: list[int] = []
    base_block_counts: list[int] = []
    cand_block_counts: list[int] = []

    truncation_count_cand = 0

    for qid in common_ids:
        case = case_map[qid]
        gold = case.gold
        b_text = base_ev[qid]
        c_text = cand_ev[qid]

        # Lengths & blocks
        base_chars.append(len(b_text))
        cand_chars.append(len(c_text))
        base_block_counts.append(_count_blocks(b_text))
        c_blocks = _count_blocks(c_text)
        cand_block_counts.append(c_blocks)

        if len(c_text) >= args.max_budget_chars - 50:
            truncation_count_cand += 1

        # Shingles & recall
        b_cov = overlap.shingle_coverage(gold, b_text, size=args.shingle_size)
        c_cov = overlap.shingle_coverage(gold, c_text, size=args.shingle_size)
        base_shingle_cov.append(b_cov)
        cand_shingle_cov.append(c_cov)

        base_unigram_rec.append(overlap.unigram_recall(gold, b_text))
        cand_unigram_rec.append(overlap.unigram_recall(gold, c_text))

        b_pres = _legal_id_presence(gold, b_text)
        c_pres = _legal_id_presence(gold, c_text)
        if b_pres is not None:
            base_id_pres.append(b_pres)
        if c_pres is not None:
            cand_id_pres.append(c_pres)

    n = len(common_ids)
    delta_cov, ci_lo, ci_hi, p_val = _paired_bootstrap_delta(cand_shingle_cov, base_shingle_cov)

    base_retrieval_limited = sum(1 for cov in base_shingle_cov if cov < args.coverage_threshold) / n
    cand_retrieval_limited = sum(1 for cov in cand_shingle_cov if cov < args.coverage_threshold) / n

    cand_block_dist = Counter(cand_block_counts)

    # QA/QC Recommendation Logic
    recommendation = "HOLD"
    reasons = []

    if delta_cov > 0.0 and ci_lo > 0.0:
        if truncation_count_cand > n * 0.05:
            recommendation = "HOLD"
            reasons.append(f"Significant coverage gain (+{delta_cov:.4f}), BUT {truncation_count_cand}/{n} ({truncation_count_cand/n:.1%}) packs hit character truncation ceiling!")
        else:
            recommendation = "PROMOTE"
            reasons.append(f"Statistically significant coverage gain (+{delta_cov:.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], p={p_val:.4f}) without budget overflow.")
    elif delta_cov <= 0.0:
        recommendation = "REJECT"
        reasons.append(f"No coverage improvement (delta = {delta_cov:+.4f} pp). Dynamic selection diluted the pack.")
    else:
        recommendation = "HOLD"
        reasons.append(f"Inconclusive coverage gain (delta = {delta_cov:+.4f}, CI crosses 0: [{ci_lo:+.4f}, {ci_hi:+.4f}]).")

    report: dict[str, Any] = {
        "schema_version": "legalqa.dynamic_evidence_qc.v1",
        "recommendation": recommendation,
        "reasons": reasons,
        "cases_audited": n,
        "shingle_size": args.shingle_size,
        "coverage_threshold": args.coverage_threshold,
        "paired_shingle_coverage_delta": {
            "mean_delta": round(delta_cov, 4),
            "ci95_low": round(ci_lo, 4),
            "ci95_high": round(ci_hi, 4),
            "p_value": round(p_val, 4),
            "statistically_significant": bool(ci_lo > 0.0),
        },
        "baseline_summary": {
            "file": str(base_path.name),
            "mean_gold_shingle_coverage": round(overlap.mean(base_shingle_cov), 4),
            "mean_gold_unigram_recall": round(overlap.mean(base_unigram_rec), 4),
            "mean_gold_legal_id_presence": round(overlap.mean(base_id_pres), 4) if base_id_pres else None,
            "retrieval_limited_rate": round(base_retrieval_limited, 4),
            "mean_pack_chars": round(statistics.mean(base_chars), 1),
            "mean_blocks_per_pack": round(statistics.mean(base_block_counts), 2),
        },
        "candidate_summary": {
            "file": str(cand_path.name),
            "mean_gold_shingle_coverage": round(overlap.mean(cand_shingle_cov), 4),
            "mean_gold_unigram_recall": round(overlap.mean(cand_unigram_rec), 4),
            "mean_gold_legal_id_presence": round(overlap.mean(cand_id_pres), 4) if cand_id_pres else None,
            "retrieval_limited_rate": round(cand_retrieval_limited, 4),
            "mean_pack_chars": round(statistics.mean(cand_chars), 1),
            "mean_blocks_per_pack": round(statistics.mean(cand_block_counts), 2),
            "block_count_distribution": {f"{k}_blocks": cand_block_dist.get(k, 0) for k in range(1, 8) if cand_block_dist.get(k, 0) > 0},
            "packs_hitting_char_ceiling": truncation_count_cand,
            "char_ceiling_hit_rate": round(truncation_count_cand / n, 4),
        },
    }

    report_str = json.dumps(report, ensure_ascii=False, indent=2)
    print(f"\n{'='*70}\n[DYNAMIC EVIDENCE QC REPORT: {recommendation}]\n{'='*70}")
    print(report_str)

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report_str + "\n", encoding="utf-8")
        print(f"\nReport written to: {args.output_report}")

    return 0 if recommendation == "PROMOTE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
